#!/usr/bin/env python3
"""对比 slow / HF-fast / tiktoken / fastokens 的 encode 与 decode 延迟。

缺可选依赖时打印 SKIP，不报错退出（transformers 为硬依赖）。
tiktoken 与 HF 不是同一套词表，只比延迟、不做 id 对齐。
fastokens 从同一份 tokenizer.json 构造，会做 encode id 对齐检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import (  # noqa: E402
    DEFAULT_TIKTOKEN_ENCODING,
    add_common_args,
    load_corpus,
    load_hf_tokenizer,
    print_stats,
    rust_tokenizer,
    skip,
    timed_loop,
)


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


def _encode_decode_fns(label: str, encode: Callable[[], list[int]], decode: Callable[[list[int]], str]):
    return label, encode, decode


def build_backends(args: argparse.Namespace, text: str) -> list[tuple[str, Callable[[], list[int]], Callable[[list[int]], str]]]:
    wanted = {x.strip() for x in args.backends.split(",") if x.strip()}
    if "all" in wanted:
        wanted = {"slow", "fast", "tiktoken", "fastokens"}
    backends: list[tuple[str, Callable[[], list[int]], Callable[[list[int]], str]]] = []
    hf_fast = None

    if "slow" in wanted:
        try:
            tok = load_hf_tokenizer(
                args.tokenizer, use_fast=False, trust_remote_code=args.trust_remote_code
            )
            if getattr(tok, "is_fast", False):
                skip(f"slow: {args.tokenizer} 仍是 is_fast=True，没有 Python tokenizer")
            else:
                backends.append(
                    _encode_decode_fns(
                        "hf-slow",
                        lambda t=tok: t.encode(text, add_special_tokens=False),
                        lambda ids, t=tok: t.decode(ids, skip_special_tokens=False),
                    )
                )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — 对照实验，缺失 slow 词表时跳过
            skip(f"slow: 无法加载 ({type(e).__name__}: {e})")

    if "fast" in wanted:
        try:
            hf_fast = load_hf_tokenizer(
                args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code
            )
            if not getattr(hf_fast, "is_fast", True):
                skip(f"fast: {args.tokenizer} is_fast=False")
            else:
                backends.append(
                    _encode_decode_fns(
                        "hf-fast",
                        lambda t=hf_fast: t.encode(text, add_special_tokens=False),
                        lambda ids, t=hf_fast: t.decode(ids, skip_special_tokens=False),
                    )
                )
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            skip(f"fast: 无法加载 ({type(e).__name__}: {e})")

    if "tiktoken" in wanted:
        tiktoken = _try_import("tiktoken")
        if tiktoken is None:
            skip("tiktoken: 未安装（pip install tiktoken）")
        else:
            try:
                enc = tiktoken.get_encoding(args.tiktoken_encoding)
                backends.append(
                    _encode_decode_fns(
                        f"tiktoken:{args.tiktoken_encoding}",
                        lambda e=enc: e.encode(text),
                        lambda ids, e=enc: e.decode(ids),
                    )
                )
            except Exception as e:  # noqa: BLE001
                skip(f"tiktoken: {type(e).__name__}: {e}")

    if "fastokens" in wanted:
        fastokens = _try_import("fastokens")
        if fastokens is None:
            skip("fastokens: 未安装（可选: pip install 'fastokens>=0.2.0'）")
        else:
            rust = None
            if hf_fast is None:
                try:
                    hf_fast = load_hf_tokenizer(
                        args.tokenizer,
                        use_fast=True,
                        trust_remote_code=args.trust_remote_code,
                    )
                except Exception as e:  # noqa: BLE001
                    skip(f"fastokens: 需要先加载 HF fast 以导出 tokenizer.json ({e})")
            if hf_fast is not None:
                rust = rust_tokenizer(hf_fast)
            if rust is None:
                skip("fastokens: 当前 HF tokenizer 没有可序列化的 Rust backend")
            else:
                try:
                    json_str = rust.to_str()
                    ft = fastokens.Tokenizer.from_json_str(json_str)
                    backends.append(
                        _encode_decode_fns(
                            "fastokens",
                            lambda t=ft: list(t.encode(text, add_special_tokens=False).ids),
                            lambda ids, t=ft: t.decode(ids, skip_special_tokens=False),
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    skip(f"fastokens: {type(e).__name__}: {e}")

    return backends


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU 微基准：slow / HF-fast / tiktoken / fastokens encode+decode"
    )
    add_common_args(parser)
    parser.add_argument(
        "--tiktoken-encoding",
        default=DEFAULT_TIKTOKEN_ENCODING,
        help="tiktoken encoding 名。默认 TIKTOKEN_ENCODING 或 gpt2。",
    )
    parser.add_argument(
        "--backends",
        default="all",
        help="逗号分隔: slow,fast,tiktoken,fastokens 或 all。缺依赖会 SKIP。",
    )
    args = parser.parse_args()
    text = load_corpus(args)
    print(f"tokenizer={args.tokenizer!r}  chars={len(text)}  warmup={args.warmup} iters={args.iters}")
    print("说明: encode_ms / decode 是 tokenizer CPU 时间，不是 GPU TTFT。")
    print()

    backends = build_backends(args, text)
    if not backends:
        print("没有可跑的 backend。安装 experiments/requirements.txt 后重试。")
        return 1

    encoded: dict[str, list[int]] = {}
    for label, encode, decode in backends:
        ids = encode()
        encoded[label] = ids
        extra = f"tokens={len(ids)}"
        print_stats(
            f"{label} encode",
            timed_loop(encode, args.warmup, args.iters),
            extra=extra,
        )
        print_stats(
            f"{label} decode",
            timed_loop(lambda d=decode, i=ids: d(i), args.warmup, args.iters),
            extra=extra,
        )

    fast_ids = encoded.get("hf-fast")
    ft_ids = encoded.get("fastokens")
    if fast_ids is not None and ft_ids is not None:
        if fast_ids == ft_ids:
            print(f"\nfastokens 与 hf-fast encode 逐 token 对齐  tokens={len(fast_ids)}")
        else:
            n = min(len(fast_ids), len(ft_ids))
            mismatch = sum(1 for a, b in zip(fast_ids, ft_ids) if a != b)
            print(
                "\nWARNING: fastokens 与 hf-fast encode 未对齐 "
                f"len={len(fast_ids)} vs {len(ft_ids)}  prefix_mismatch={mismatch}/{n}"
            )
            print("上线 fastokens 前必须先做正确性，不要只看延迟。")

    tik_keys = [k for k in encoded if k.startswith("tiktoken:")]
    if fast_ids is not None and tik_keys:
        print(
            f"\n注: {tik_keys[0]} 与 HF 不是同一词表，"
            "token 数不同是预期现象，不能当正确性检查。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
