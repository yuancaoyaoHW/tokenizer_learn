## 第一节：什么事tokenizer
**Tokenizer** 是大语言模型的「文字门卫」：模型只吃整数，不吃字符串。它把人类可读的文本变成 token id 序列送进网络，再把模型吐出的 id 还原成文字。

一次完整往返是 **encode（文本 → ids）** 和 **decode（ids → 文本）**。今天主流 LLM（GPT、Llama、Mistral 等）几乎都用 **字节级 BPE（Byte-level Byte Pair Encoding）**：先把文本编成 UTF-8 字节（固定 256 个基础符号），再在语料上反复合并最高频的相邻字节对，得到一份有序 merge 表和词表。任意 Unicode 都能覆盖，不会出现「未知字符」。训练阶段按频次造规则；推理阶段按 merge 的先后顺序（rank）重放，不再重新数频次。

线上 serving 里，tokenizer 往往不止一个 `encode()`。一条 Chat 请求通常是三段 CPU 工作：
**chat template**（把 messages / tools 渲成字符串）→ **encode**（变成 token ids 进 GPU）→ **incremental detokenize**（每个新 token 增量还原成文本再流式返回）。BPE 合并本身常常不是最慢的那一段；真正容易卡住的是模板渲染、特殊 token、UTF-8 半字符缓冲，以及并发下的线程/进程隔离。

经典玩具例子：对 `"aaabdaaabac"` 做 3 次 merge（minbpe）。

先把字符看成字节：`a=97`, `b=98`, `c=99`, `d=100`。

```text
原文:  a  a  a  b  d  a  a  a  b  a  c
字节:  97 97 97 98 100 97 97 97 98 97 99
```

| 第几次 | 最高频相邻对 | 新 token | 序列 |
| --- | --- | --- | --- |
| 1 | `aa (97,97)` | `256`（叫它 `Z`） | `Z a b d Z a b a c` |
| 2 | `ab (97,98)` | `257`（叫它 `Y`） | `Z Y d Z Y a c` |
| 3 | `ZY (256,257)` | `258`（叫它 `X`） | `X d X a c` |

对应的 id 就是：

```text
encode("aaabdaaabac")  →  [258, 100, 258, 97, 99]
decode([258, 100, 258, 97, 99])  →  "aaabdaaabac"
```

代码上就是：

```python
from minbpe import BasicTokenizer
tok = BasicTokenizer()
tok.train("aaabdaaabac", vocab_size=256 + 3)  # 256 个字节 + 3 次合并
print(tok.encode("aaabdaaabac"))
# [258, 100, 258, 97, 99]
print(tok.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
```


vLLM 把 (1)(2) 放在 HfRenderer：safe_apply_chat_template 通过 make_async(..., executor=self._executor) 离开放 asyncio 主循环；tokenize 同样走线程池。

        # Thread pool executor for blocking tokenizer operations.  The
        # multimodal processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)
        ...
        self._tokenize_prompt_async = make_async(
            self._tokenize_prompt, executor=self._executor
        )

        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )

SGLang 更进一步做成 进程隔离：TokenizerManager 专门 tokenize，DetokenizerManager 专门增量 decode，中间用 ZMQ 传 token ids / 增量字符串。

"""TokenizerManager is a process that tokenizes the text."""

"""DetokenizerManager is a process that detokenizes the token ids."""

性能优化的核心是：别让 tokenizer 堵 event loop、别每步全量 decode、高并发时复制/多进程。