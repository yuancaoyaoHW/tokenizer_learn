# Tokenizer CPU 微基准

本目录只测 **tokenizer 自己的 CPU 时间**，不要和 GPU 的 TTFT / ITL 混在一张表里。指标含义见 [`notes/05-performance.md`](../notes/05-performance.md)。

脚本不 import vLLM / SGLang，无 GPU 也能跑。默认加载公开的 `gpt2` tokenizer（需要能访问 Hugging Face 或本地缓存）。可用环境变量换模型：

```bash
export HF_TOKENIZER=gpt2          # 默认；任意 AutoTokenizer 能加载的 id / 本地路径
export TIKTOKEN_ENCODING=gpt2     # 仅 bench_backends.py 的 tiktoken 路径
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r experiments/requirements.txt
# 可选
# pip install 'fastokens>=0.2.0'
```

`tiktoken` 未装：`bench_backends.py` 跳过该 backend。  
`fastokens` 未装：跳过 fastokens。  
某个模型没有 slow tokenizer：跳过 `hf-slow`。

## 运行

在仓库根目录：

```bash
python experiments/bench_backends.py
python experiments/bench_concurrency.py
python experiments/bench_detok.py
python experiments/bench_chat_template.py
```

常用参数（四个脚本都有）：

| 参数 | 含义 |
| --- | --- |
| `--tokenizer` | 覆盖 `HF_TOKENIZER` |
| `--warmup` / `--iters` | 预热与计次 |
| `--repeat` | 把内置中英混合语料重复 N 遍 |
| `--text-file` | 自定义 UTF-8 语料 |
| `--trust-remote-code` | 传给 `AutoTokenizer` |

各脚本额外参数见 `--help`。不要在这里跑长时间 GPU bench；可选 GPU 命令写在性能手册里。

`bench_lib.py` 只是计时 / 加载工具，不是实验。

## 结果记录模板

把命令行输出贴到下面。**不要抄别人的数字**；机器、tokenizer、`tokenizers` 版本不同，数字不可比。

### 环境

```
date:
host / CPU:
python:
transformers:
tokenizers:
tiktoken:
fastokens:          # 未安装就写 SKIP
HF_TOKENIZER:
命令:
```

### bench_backends.py

| backend | encode mean | encode p95 | decode mean | tokens | 备注 |
| --- | --- | --- | --- | --- | --- |
| hf-slow |  |  |  |  |  |
| hf-fast |  |  |  |  |  |
| tiktoken |  |  |  |  | 与 HF 不同词表 |
| fastokens |  |  |  |  | 是否与 hf-fast 逐 token 对齐： |

### bench_concurrency.py

| 模式 | mean | p95 | errors | 备注 |
| --- | --- | --- | --- | --- |
| shared-instance |  |  |  |  |
| deepcopy-pool |  |  |  |  |
| multiprocess-spawn |  |  |  | 含进程启动 |
| Already borrowed 演示 | 抛错 / 未抛错： |  |  |  |

### bench_detok.py

| 路径 | mean | detok_us_per_token | 与 full decode 文本是否一致 |
| --- | --- | --- | --- |
| full_decode_once |  |  | — |
| full_decode_per_token |  |  |  |
| DecodeStream.step |  |  |  |

### bench_chat_template.py

| 路径 | mean | tokens | ids 是否一致 |
| --- | --- | --- | --- |
| chat_template_ms (string) |  | — | — |
| encode_ms after render |  |  |  |
| render_then_encode |  |  |  |
| tokenize=True once |  |  |  |

观察（自己写，不要编数字）：

- template 是否明显大于 encode：
- 并发下单实例是否变慢或报 borrowed：
- naive 逐步 decode 是否随 token 数恶化：
