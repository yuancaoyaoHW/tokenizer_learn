#!/usr/bin/env python3
"""并发 encode：单实例共享 vs deepcopy 池 vs 多进程。

可选演示 HF Fast 多线程共享同一 Rust tokenizer 时的
``RuntimeError: Already borrowed``（旧版 tokenizers / RefCell）。
较新的 tokenizers 用 RwLock 序列化，可能不再抛这个错，但仍会互相卡住；
此时对比池化/多进程的吞吐仍然有意义。对照：

- vLLM ``maybe_make_thread_pool``：深拷贝池
- SGLang ``--tokenizer-worker-num``：多进程 TokenizerManager
"""

from __future__ import annotations

import argparse
import copy
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import (  # noqa: E402
    add_common_args,
    load_corpus,
    load_hf_tokenizer,
    print_stats,
    rust_tokenizer,
    skip,
)

_MP_TOKENIZER = None
_MP_TEXT = None


def _mp_init(tokenizer_name: str, text: str, trust_remote_code: bool) -> None:
    global _MP_TOKENIZER, _MP_TEXT
    from transformers import AutoTokenizer

    _MP_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    _MP_TEXT = text


def _mp_encode(_: int) -> int:
    assert _MP_TOKENIZER is not None and _MP_TEXT is not None
    return len(_MP_TOKENIZER.encode(_MP_TEXT, add_special_tokens=False))


def _borrow_messages(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "already borrowed" in msg or "borrowed" in msg


def _run_jobs(fn, *, jobs: int, workers: int) -> tuple[list[float], list[str]]:
    """每个 job 一次 encode；返回单 job 延迟样本和异常摘要。"""
    errors: list[str] = []
    samples: list[float] = []

    def wrapped():
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(wrapped) for _ in range(jobs)]
        for fut in as_completed(futs):
            try:
                samples.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")
    return samples, errors


def demo_already_borrowed(tok: Any, text: str, *, workers: int, hits: int) -> None:
    print("\n--- Already borrowed 演示（共享同一 Fast tokenizer）---")
    print(
        f"threads={workers}  hits/thread={hits}  "
        "目标：并发 encode 同一底层 Rust tokenizer"
    )
    backend = rust_tokenizer(tok)

    def hammer_hf() -> None:
        for _ in range(hits):
            tok.encode(text, add_special_tokens=False)

    def hammer_rust() -> None:
        assert backend is not None
        for _ in range(hits):
            backend.encode(text)

    for label, target in (("hf.encode", hammer_hf), ("rust.encode", hammer_rust)):
        if label == "rust.encode" and backend is None:
            skip("rust.encode: 没有 _tokenizer backend")
            continue
        errors: list[str] = []
        borrowed = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(target) for _ in range(workers)]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {e}")
                    if _borrow_messages(e):
                        borrowed += 1
        elapsed = time.perf_counter() - t0
        if borrowed:
            print(
                f"  {label}: 捕获到 Already borrowed / borrowed 类错误 "
                f"{borrowed}/{len(errors)}  墙钟 {elapsed*1e3:.1f} ms"
            )
            for line in errors[:5]:
                print(f"    {line}")
        elif errors:
            print(f"  {label}: 抛错但不是 borrowed（{len(errors)}）墙钟 {elapsed*1e3:.1f} ms")
            for line in errors[:5]:
                print(f"    {line}")
        else:
            print(
                f"  {label}: 未抛错（墙钟 {elapsed*1e3:.1f} ms）。"
                "当前 tokenizers 可能已用 RwLock 序列化，而不是 RefCell。"
                "吞吐仍可能被同一把锁卡住，见下方单实例 vs 池。"
            )


def make_deepcopy_pool(tok: Any, copies: int):
    pool: queue.Queue = queue.Queue()
    prototype = copy.copy(tok)
    for _ in range(max(1, copies)):
        pool.put(copy.deepcopy(prototype))

    def encode(text: str) -> None:
        try:
            item = pool.get_nowait()
        except queue.Empty:
            item = copy.deepcopy(prototype)
        try:
            item.encode(text, add_special_tokens=False)
        finally:
            pool.put(item)

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU 微基准：共享实例 vs deepcopy 池 vs 多进程 encode"
    )
    add_common_args(parser)
    parser.add_argument("--workers", type=int, default=8, help="线程或进程数。")
    parser.add_argument("--jobs", type=int, default=64, help="并发 encode 总次数。")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=8,
        help="deepcopy 池初始副本数（对照 vLLM tokenizer pool）。",
    )
    parser.add_argument(
        "--demo-borrow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="尝试触发 Already borrowed（可用 --no-demo-borrow 关掉）。",
    )
    parser.add_argument(
        "--borrow-hits",
        type=int,
        default=200,
        help="每个线程在 borrow 演示里连续 encode 的次数。",
    )
    args = parser.parse_args()
    text = load_corpus(args)

    try:
        tok = load_hf_tokenizer(
            args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:  # noqa: BLE001
        skip(f"需要 HF fast tokenizer: {e}")
        return 1

    n_tokens = len(tok.encode(text, add_special_tokens=False))
    print(
        f"tokenizer={args.tokenizer!r} is_fast={getattr(tok, 'is_fast', None)}  "
        f"chars={len(text)} tokens={n_tokens}  workers={args.workers} jobs={args.jobs}"
    )
    print("说明: 测的是 tokenizer CPU 并发，不是 GPU TTFT。")
    print()

    # 预热
    for _ in range(args.warmup):
        tok.encode(text, add_special_tokens=False)

    def encode_shared() -> None:
        tok.encode(text, add_special_tokens=False)

    samples, errors = _run_jobs(encode_shared, jobs=args.jobs, workers=args.workers)
    if samples:
        print_stats("shared-instance", samples, extra=f"errors={len(errors)}")
    if errors:
        borrowed = sum(1 for e in errors if "borrow" in e.lower())
        print(f"  shared-instance 异常 {len(errors)}（borrow 类 {borrowed}）")
        for line in errors[:8]:
            print(f"    {line}")

    pool_encode = make_deepcopy_pool(tok, args.pool_size)
    for _ in range(args.warmup):
        pool_encode(text)
    samples, errors = _run_jobs(
        lambda: pool_encode(text), jobs=args.jobs, workers=args.workers
    )
    if samples:
        print_stats(
            "deepcopy-pool",
            samples,
            extra=f"pool={args.pool_size} errors={len(errors)}",
        )
    if errors:
        print(f"  deepcopy-pool 异常 {len(errors)}")
        for line in errors[:8]:
            print(f"    {line}")

    try:
        import multiprocessing as mp
    except ImportError:
        skip("multiprocess: 当前 Python 无法 import multiprocessing")
        mp = None  # type: ignore[assignment]

    if mp is not None:
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()
        with ctx.Pool(
            processes=args.workers,
            initializer=_mp_init,
            initargs=(args.tokenizer, text, args.trust_remote_code),
        ) as pool:
            for _ in range(args.warmup):
                pool.map(_mp_encode, range(args.workers))
            job_times: list[float] = []
            # 按批提交，使每批宽度 = workers，便于和线程池对比墙钟
            remaining = args.jobs
            while remaining > 0:
                batch = min(args.workers, remaining)
                bt = time.perf_counter()
                pool.map(_mp_encode, range(batch))
                job_times.append((time.perf_counter() - bt) / batch)
                remaining -= batch
        wall = time.perf_counter() - t0
        print_stats(
            "multiprocess-spawn",
            job_times,
            extra=f"wall={wall*1e3:.1f}ms (含进程启动，仅供对照)",
        )

    if args.demo_borrow:
        demo_already_borrowed(
            tok, text, workers=max(4, args.workers), hits=args.borrow_hits
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
