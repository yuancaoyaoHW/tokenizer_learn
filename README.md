# 推理框架 Tokenizer 学习仓库

本仓库按「算法基础 → 库实现 → 推理框架调用链 → 性能实验」四层对照源码学习 **LLM serving 里的 tokenizer**，而不是只收集链接。

心智模型先记住一件事：线上 tokenizer 不是一个 `encode()`，而是三段 CPU 工作——**chat template → encode → incremental detokenize**。BPE 本身往往不是最慢的那一段。展开见 [notes/00-fundamentals.md](notes/00-fundamentals.md)。

```
tokenizer/
  README.md              ← 本文件：阅读顺序与克隆 SHA
  notes/                 ← 中文导读（对照 third_party 行号）
  experiments/           ← encode / decode / 并发 / chat_template 微基准（其他笔记交付）
  third_party/           ← 浅克隆，不是 git submodule
```

---

## 克隆版本（勿擅自再 clone）

`third_party/` 已按下列 commit 就位。笔记里的行号相对这些 SHA。

| 目录 | 上游 | SHA |
| --- | --- | --- |
| `third_party/minbpe` | [karpathy/minbpe](https://github.com/karpathy/minbpe) | `1acefe89412b20245db5a22d2a02001e547dc602` |
| `third_party/tokenizers` | [huggingface/tokenizers](https://github.com/huggingface/tokenizers) | `d5827816baedcbf1cb5b452dea8048150b6872df` |
| `third_party/tiktoken` | [openai/tiktoken](https://github.com/openai/tiktoken) | `4e71bbe0c078468e00fefbf94b39849389f346e5` |
| `third_party/fastokens` | [crusoecloud/fastokens](https://github.com/crusoecloud/fastokens) | `6ceb6a8cb5da7ea057369886802e90837d197e1b` |
| `third_party/vllm` | [vllm-project/vllm](https://github.com/vllm-project/vllm) | `2c7d7dd64a2eaba0feedf42cab2f527486d7479c` |
| `third_party/sglang` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `e635577431cbdfb8ce5fafb0fcd8a4ac074062c6` |

未克隆：`transformers`（slow/fast 包装层在笔记里用 vLLM 的 `TokenizersBackend` / `PythonBackend` 对照）、TensorRT-LLM（接口与 vLLM 类似：HF wrap + `skip_tokenizer_init`）。

---

## 按周阅读顺序

跟一条请求走，遇到 CUDA graph / KV cache / attention backend **跳过**：

`HTTP messages → template → token ids →（不看 scheduler）→ 新 token → detokenize → 流式 chunk`

### 第 1 周 — 算法与库（先读小仓库）

目标：自己画出 HF Fast Tokenizer 的 pipeline，并能解释 `is_fast=False` 和 `RuntimeError: Already borrowed`。

| 顺序 | 读什么 | 笔记 |
| --- | --- | --- |
| 1 | `third_party/minbpe/minbpe/basic.py` → `regex.py`（可顺带 `gpt4.py`） | [notes/00-fundamentals.md](notes/00-fundamentals.md) |
| 2 | `third_party/tiktoken/tiktoken/_educational.py` 与 `tiktoken/core.py` 对照 | 同上 + [notes/01-hf-tokenizers.md](notes/01-hf-tokenizers.md) §5 |
| 3 | `third_party/tokenizers/tokenizers/src/tokenizer/mod.rs`：encode 五段 + `DecodeStream` | [notes/01-hf-tokenizers.md](notes/01-hf-tokenizers.md) |
| 4 | Python 绑定 `bindings/python/src/tokenizer.rs`（`Arc<RwLock>`）与 vLLM `maybe_make_thread_pool` | 01 §6 |
| 5 | `third_party/fastokens/README.md`（只建立「可替换 HF 内部 BPE」的印象） | 01 §7 |

本周实验脚本见 [experiments/README.md](experiments/README.md)（**此处没有跑数，不要把上游 README 的 3–6× / 10× 当成本仓库结果**）：

- [`experiments/bench_backends.py`](experiments/bench_backends.py) — 同一段中英混合长文本：slow / HF-fast / tiktoken / fastokens
- [`experiments/bench_concurrency.py`](experiments/bench_concurrency.py) — 单实例 vs deepcopy 池 vs 多进程；刻意触发 Fast tokenizer 的 `Already borrowed`
- [`experiments/bench_detok.py`](experiments/bench_detok.py) — 全量 `decode` vs `DecodeStream.step`
- [`experiments/bench_chat_template.py`](experiments/bench_chat_template.py) — `tokenize=True` 一次完成 vs 先渲字符串再 encode

### 第 2 周 — vLLM 调用链

笔记：[notes/02-vllm.md](notes/02-vllm.md)（独立交付）。优先文件：

- `third_party/vllm/vllm/tokenizers/registry.py` — `get_tokenizer`；slow 警告
- `third_party/vllm/vllm/tokenizers/hf.py` — `get_cached_tokenizer`、`maybe_make_thread_pool`
- `third_party/vllm/vllm/renderers/hf.py` — `safe_apply_chat_template` + 线程池
- `third_party/vllm/vllm/v1/engine/detokenizer.py` — `FastIncrementalDetokenizer`
- `third_party/vllm/vllm/tokenizers/fastokens.py` — `VLLM_USE_FASTOKENS=1`

第 1 周已经读过的交叉点：Renderer 线程池与三段式见 00 §1 / §7；副本池与 `Already borrowed` 见 01 §6。

### 第 3 周 — SGLang 调用链与对比

- [notes/03-sglang.md](notes/03-sglang.md)：`TokenizerManager` / `DetokenizerManager` / `--tokenizer-worker-num` / `--tokenizer-backend`
- [notes/04-compare.md](notes/04-compare.md)：线程池 vs 多进程、同进程 `DecodeStream` vs 独立 detokenizer 进程、chat template 落点

源码入口：

- `third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py`
- `third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py`（`DecodeStatus.surr_offset` / `read_offset`）
- `third_party/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py`（template 与 encode 拆开）

### 第 4 周 — 性能手册

[notes/05-performance.md](notes/05-performance.md) + [experiments/](experiments/README.md)。指标只记 tokenizer 自己的（`encode_ms`、`chat_template_ms`、`detok_us_per_token`），不要和 GPU TTFT 混为一谈。有 GPU 再可选 `vllm bench serve` / `sglang.bench_serving`，看 tokenizer 进程 CPU 是否先打满。

**尚未写入实验结果。** 跑完脚本后把数字记进 `experiments/` 的结果模板，不要把 fastokens / tiktoken 上游 README 的图表抄进本仓库当结论。

---

## 笔记索引

| 文件 | 状态 | 内容 |
| --- | --- | --- |
| [notes/00-fundamentals.md](notes/00-fundamentals.md) | 已写 | BPE/BBPE；minbpe `basic.py` → `regex.py`；special tokens；serving 三段式 |
| [notes/01-hf-tokenizers.md](notes/01-hf-tokenizers.md) | 已写 | HF pipeline；DecodeStream；slow vs fast；tiktoken vs HF；`Already borrowed`；fastokens |
| [notes/02-vllm.md](notes/02-vllm.md) | 已写 | Renderer 四步流水线 → tokenizer pool → Fast/Slow IncrementalDetokenizer |
| [notes/03-sglang.md](notes/03-sglang.md) | 已写 | TokenizerManager / DetokenizerManager / 多 worker IPC |
| [notes/04-compare.md](notes/04-compare.md) | 已写 | vLLM vs SGLang |
| [notes/05-performance.md](notes/05-performance.md) | 已写 | 指标、瓶颈、怎么跑到最好 |

---

## 阅读纪律

- 每个框架只跟一条请求；多模态、constrained decoding（xgrammar）放到第二轮。
- 行号以本 README 的 SHA 为准；上游 `main` 前进后以 blame 为准，不要假设行号永远不变。
- 引用代码时用仓库内路径（`third_party/...`），便于和笔记互跳。
