#!/usr/bin/env python3
"""全量 decode vs HuggingFace ``DecodeStream.step`` 增量 detokenize。

naive 流式：每来一个 token 就 ``decode(ids[:i])``，成本随序列变长。
增量：``tokenizers.decoders.DecodeStream.step``，只产出新 chunk。
对照 vLLM ``FastIncrementalDetokenizer``（tokenizers>=0.22）。
"""

from __future__ import annotations

import argparse
import sys
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
    timed_loop,
)


def _full_decode_once(tok: Any, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def _full_decode_per_token(tok: Any, ids: list[int]) -> str:
    text = ""
    for i in range(1, len(ids) + 1):
        text = tok.decode(ids[:i], skip_special_tokens=False)
    return text


def _decode_stream_step(rust, ids: list[int], DecodeStream) -> str:
    stream = DecodeStream(skip_special_tokens=False)
    chunks: list[str] = []
    for tid in ids:
        piece = stream.step(rust, tid)
        if piece:
            chunks.append(piece)
    return "".join(chunks)


def _us_per_token(samples_s: list[float], n_tokens: int) -> str:
    if not samples_s or n_tokens <= 0:
        return "detok_us_per_token=nan"
    mean_s = sum(samples_s) / len(samples_s)
    return f"detok_us_per_token={mean_s * 1e6 / n_tokens:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU 微基准：全量 decode vs DecodeStream.step"
    )
    add_common_args(parser)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="只对 encode 结果的前 N 个 token 做 decode 对比。",
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

    ids = tok.encode(text, add_special_tokens=False)
    if args.max_tokens > 0:
        ids = ids[: args.max_tokens]
    n = len(ids)
    print(
        f"tokenizer={args.tokenizer!r}  chars={len(text)}  "
        f"decode_tokens={n}  warmup={args.warmup} iters={args.iters}"
    )
    print("说明: detok_us_per_token 是 tokenizer CPU，不是 GPU ITL。")
    print()

    once_samples = timed_loop(
        lambda: _full_decode_once(tok, ids), args.warmup, args.iters
    )
    print_stats("full_decode_once", once_samples, extra=_us_per_token(once_samples, n))

    per_tok_samples = timed_loop(
        lambda: _full_decode_per_token(tok, ids), args.warmup, args.iters
    )
    print_stats(
        "full_decode_per_token",
        per_tok_samples,
        extra=_us_per_token(per_tok_samples, n) + "  (naive streaming)",
    )

    rust = rust_tokenizer(tok)
    try:
        from tokenizers.decoders import DecodeStream
    except ImportError:
        skip("DecodeStream: 未安装 tokenizers 或版本过旧")
        DecodeStream = None  # type: ignore[assignment]

    if rust is None:
        skip("DecodeStream: 当前 tokenizer 没有 Rust backend")
    elif DecodeStream is None:
        pass
    else:
        try:
            DecodeStream(skip_special_tokens=False).step(rust, ids[0] if ids else 0)
        except Exception as e:  # noqa: BLE001
            skip(f"DecodeStream.step 不可用 ({type(e).__name__}: {e})")
        else:
            stream_samples = timed_loop(
                lambda: _decode_stream_step(rust, ids, DecodeStream),
                args.warmup,
                args.iters,
            )
            print_stats(
                "DecodeStream.step",
                stream_samples,
                extra=_us_per_token(stream_samples, n),
            )

            once_text = _full_decode_once(tok, ids)
            stream_text = _decode_stream_step(rust, ids, DecodeStream)
            if once_text == stream_text:
                print("\nDecodeStream 拼接结果与一次 full decode 文本一致。")
            else:
                print(
                    "\nWARNING: DecodeStream 拼接与 full decode 文本不一致 "
                    f"(full_len={len(once_text)} stream_len={len(stream_text)})。"
                    "增量 decode 在 UTF-8 / special token 空格上可能与一次性 decode 有差异。"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
