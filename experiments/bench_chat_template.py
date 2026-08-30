#!/usr/bin/env python3
"""Chat template：``tokenize=True`` 一次完成 vs 先渲染字符串再 encode。

模板（Jinja）经常比 BPE 本身更慢；不要对同一 messages 重复
``apply_chat_template``。若 tokenizer 没有 ``chat_template``，注入一份
极简 fallback，并打印警告（真实 instruct 模型请设 HF_TOKENIZER）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import (  # noqa: E402
    FALLBACK_CHAT_TEMPLATE,
    add_common_args,
    load_corpus,
    load_hf_tokenizer,
    print_stats,
    skip,
    timed_loop,
)


def default_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant. 你是助理。"},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "收到，下面按 tokenizer / encode / detok 三段说明。"},
        {"role": "user", "content": "请把上面要点列成 checklist，中英都可以。"},
    ]


def ensure_chat_template(tok: Any) -> None:
    if getattr(tok, "chat_template", None):
        return
    tok.chat_template = FALLBACK_CHAT_TEMPLATE
    print(
        "WARNING: 该 tokenizer 没有 chat_template，已注入极简 fallback。"
        "换带模板的 instruct tokenizer 更接近 serving（export HF_TOKENIZER=...）。"
    )


def render_then_encode(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tok.encode(text, add_special_tokens=False)


def tokenize_true(tok: Any, messages: list[dict[str, str]]) -> list[int]:
    ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return list(ids)


def template_only(tok: Any, messages: list[dict[str, str]]) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU 微基准：apply_chat_template tokenize=True vs 渲染后再 encode"
    )
    add_common_args(parser)
    parser.add_argument(
        "--messages-file",
        default=None,
        help="JSON 文件，内容为 OpenAI-style messages 列表。",
    )
    args = parser.parse_args()

    if args.messages_file:
        with open(args.messages_file, encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise SystemExit("--messages-file 必须是 message 对象数组")
    else:
        messages = default_messages(load_corpus(args))

    try:
        tok = load_hf_tokenizer(
            args.tokenizer, use_fast=True, trust_remote_code=args.trust_remote_code
        )
    except Exception as e:  # noqa: BLE001
        skip(f"无法加载 tokenizer: {e}")
        return 1

    ensure_chat_template(tok)

    try:
        rendered = template_only(tok, messages)
        ids_once = tokenize_true(tok, messages)
        ids_two = render_then_encode(tok, messages)
    except Exception as e:  # noqa: BLE001
        skip(f"apply_chat_template 失败 ({type(e).__name__}: {e})")
        return 1

    print(
        f"tokenizer={args.tokenizer!r}  messages={len(messages)}  "
        f"rendered_chars={len(rendered)}  tokens_tokenize_true={len(ids_once)}  "
        f"tokens_render_then_encode={len(ids_two)}"
    )
    print("说明: chat_template_ms 是 CPU / Jinja，不是 GPU TTFT。")
    print()

    print_stats(
        "chat_template_ms (string)",
        timed_loop(lambda: template_only(tok, messages), args.warmup, args.iters),
        extra="tokenize=False",
    )
    print_stats(
        "encode_ms after render",
        timed_loop(
            lambda: tok.encode(rendered, add_special_tokens=False),
            args.warmup,
            args.iters,
        ),
        extra=f"tokens={len(ids_two)}",
    )
    print_stats(
        "render_then_encode",
        timed_loop(
            lambda: render_then_encode(tok, messages), args.warmup, args.iters
        ),
        extra="template + encode",
    )
    print_stats(
        "tokenize=True once",
        timed_loop(lambda: tokenize_true(tok, messages), args.warmup, args.iters),
        extra="apply_chat_template(tokenize=True)",
    )

    if ids_once == ids_two:
        print("\ntokenize=True 与「渲染字符串再 encode」的 token ids 一致。")
    else:
        n = min(len(ids_once), len(ids_two))
        mismatch = sum(1 for a, b in zip(ids_once, ids_two) if a != b)
        print(
            "\nWARNING: 两条路径 token ids 不一致 "
            f"len={len(ids_once)} vs {len(ids_two)} prefix_mismatch={mismatch}/{n}。"
            "常见原因是 encode(add_special_tokens=True) 又加了 BOS；本脚本已关。"
        )
    print(
        "\n不要对同一请求重复 apply_chat_template；"
        "工具 / reasoning parser 不用就关（见 notes/05-performance.md）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
