"""本目录微基准的公共工具：计时、默认语料、HF tokenizer 加载。

各 ``bench_*.py`` 只依赖本文件 + ``experiments/requirements.txt``，
不 import vLLM / SGLang。
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_HF_TOKENIZER = os.environ.get("HF_TOKENIZER", "gpt2")
DEFAULT_TIKTOKEN_ENCODING = os.environ.get("TIKTOKEN_ENCODING", "gpt2")

# 中英混合长文本：plan 要求 backends 对比用这类语料。可通过 --repeat 拉长。
SAMPLE_TEXT = """Tokenizer serving is three CPU stages, not one encode() call.
Chat template (Jinja apply_chat_template) often dominates BPE itself.
Then encode turns text into token ids for the GPU scheduler.
Then incremental detokenize turns each new token into UTF-8 text for SSE.
请用中英混合句子覆盖 byte-level BPE 与 CJK：北京、上海、深圳的推理集群
在高 QPS 长 prompt 下，TTFT 变差不一定是 GPU kernel，也可能是 tokenizer
堵了 asyncio 主循环，或每步全量 decode 把 Python↔Rust 往返打满。
Special tokens, stop strings, tool parsers, and UTF-8 boundaries all sit
on the detok path. 不要把 encode_ms 和 GPU TTFT 混成一个数字。
""".strip()

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_HF_TOKENIZER,
        help=(
            "Hugging Face tokenizer id 或本地目录。"
            f"默认环境变量 HF_TOKENIZER 或 {DEFAULT_HF_TOKENIZER!r}。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="转发给 AutoTokenizer.from_pretrained。",
    )
    parser.add_argument("--warmup", type=int, default=3, help="不计时的预热次数。")
    parser.add_argument("--iters", type=int, default=20, help="计时迭代次数。")
    parser.add_argument(
        "--repeat",
        type=int,
        default=8,
        help="把内置 SAMPLE_TEXT 重复若干遍以拉长输入。",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help="用该文件内容替代内置语料（UTF-8）。",
    )
    return parser


def load_corpus(args: argparse.Namespace) -> str:
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
        if not text:
            raise SystemExit(f"--text-file 为空: {args.text_file}")
        return text
    n = max(1, int(args.repeat))
    return "\n".join([SAMPLE_TEXT] * n)


def percentile(sorted_xs: Sequence[float], p: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    k = (len(sorted_xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    if lo == hi:
        return float(sorted_xs[lo])
    w = k - lo
    return float(sorted_xs[lo]) * (1.0 - w) + float(sorted_xs[hi]) * w


def timed_loop(fn: Callable[[], Any], warmup: int, iters: int) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def print_stats(
    name: str,
    samples_s: Sequence[float],
    *,
    unit: str = "ms",
    extra: str = "",
) -> None:
    xs = sorted(float(x) for x in samples_s)
    if unit == "us":
        scale, suffix = 1e6, "us"
    else:
        scale, suffix = 1e3, "ms"
    scaled = [x * scale for x in xs]
    mean = statistics.fmean(scaled) if scaled else float("nan")
    line = (
        f"{name:28s} n={len(scaled):<4d}  "
        f"mean={mean:10.3f}{suffix}  "
        f"p50={percentile(scaled, 50):10.3f}{suffix}  "
        f"p95={percentile(scaled, 95):10.3f}{suffix}  "
        f"min={scaled[0] if scaled else float('nan'):10.3f}{suffix}  "
        f"max={scaled[-1] if scaled else float('nan'):10.3f}{suffix}"
    )
    if extra:
        line += f"  {extra}"
    print(line)


def skip(reason: str) -> None:
    print(f"SKIP  {reason}")


def load_hf_tokenizer(name: str, *, use_fast: bool, trust_remote_code: bool = False):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "缺少 transformers。请先: pip install -r experiments/requirements.txt"
        ) from e
    return AutoTokenizer.from_pretrained(
        name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )


def rust_tokenizer(hf_tok: Any):
    """HF Fast 的底层 ``tokenizers.Tokenizer``；slow 或缺失时返回 None。"""
    backend = getattr(hf_tok, "_tokenizer", None)
    if backend is None:
        return None
    encode = getattr(backend, "encode", None)
    if encode is None:
        return None
    return backend
