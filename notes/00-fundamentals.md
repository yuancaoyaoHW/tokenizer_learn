# Tokenizer 基础：BPE / BBPE、Special Tokens、Serving 三段式

本文对照 `third_party/minbpe`（教学实现）与推理框架里真正会卡住 CPU 的三段工作来建立心智模型。
HF 五段 pipeline、`DecodeStream`、tiktoken 对照见 [01-hf-tokenizers.md](01-hf-tokenizers.md)。

## 该打开的文件

| 文件 | 看什么 |
| --- | --- |
| `third_party/minbpe/minbpe/base.py` | `get_stats` / `merge`；vocab 由 256 字节 + merges + special tokens 派生 |
| `third_party/minbpe/minbpe/basic.py` | 纯 byte-level BPE：无 regex、无 special tokens |
| `third_party/minbpe/minbpe/regex.py` | GPT-2/4 regex 预切分；`allowed_special`；special 与 ordinary 分路 encode |
| `third_party/minbpe/minbpe/gpt4.py` | 从 tiktoken `cl100k_base` 还原 merges + byte shuffle |
| `third_party/tiktoken/tiktoken/_educational.py` | 与 minbpe 同构的教学 BPE（regex 切词 + 按 rank 合并） |
| `third_party/vllm/vllm/renderers/hf.py` | chat template 在 Renderer，不在 BPE |
| `third_party/vllm/vllm/renderers/base.py` | template / tokenize 丢进 `ThreadPoolExecutor` |
| `third_party/vllm/vllm/v1/engine/detokenizer.py` | 增量 detokenize（`DecodeStream.step`） |
| `third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py` | 独立 tokenizer 进程 |
| `third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py` | 独立 detokenizer 进程 + UTF-8 缓冲 |

---

## 1. Serving 里 tokenizer 不是「一个 encode 函数」

训练课上的 tokenizer 通常是 `text → ids` / `ids → text`。线上 Chat Completions 则是三段 **CPU** 工作，常常比 BPE 合并本身更慢：

1. **Chat template / 工具模板**：Jinja `apply_chat_template`，把 `messages` + tools + 多模态占位符渲成字符串（或直接渲成 ids）。
2. **Encode**：字符串 → token ids，送进 GPU scheduler。
3. **Incremental detokenize**：每个新 token → 增量文本。要处理 UTF-8 半字符、byte-fallback、special token 空格、stop string。

```
HTTP messages → apply_chat_template → encode → GPU
GPU 新 token → incremental detokenize → SSE / JSON chunk
```

vLLM 把 (1)(2) 放在 `HfRenderer`：`safe_apply_chat_template` 通过 `make_async(..., executor=self._executor)` 离开放 asyncio 主循环；tokenize 同样走线程池。

```82:98:third_party/vllm/vllm/renderers/base.py
        # Thread pool executor for blocking tokenizer operations.  The
        # multimodal processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)
        ...
        self._tokenize_prompt_async = make_async(
            self._tokenize_prompt, executor=self._executor
        )
```

```922:929:third_party/vllm/vllm/renderers/hf.py
        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )

        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )
```

SGLang 更进一步做成 **进程隔离**：`TokenizerManager` 专门 tokenize，`DetokenizerManager` 专门增量 decode，中间用 ZMQ 传 token ids / 增量字符串。

```14:14:third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py
"""TokenizerManager is a process that tokenizes the text."""
```

```14:14:third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py
"""DetokenizerManager is a process that detokenizes the token ids."""
```

**性能优化的核心不是「换一个更快的 BPE」**，而是：别让 tokenizer 堵 event loop、别每步全量 `decode`、高并发时复制/多进程、能跳过的就跳过。细节见后续 `notes/02-vllm.md` / `03-sglang.md` / `05-performance.md`。

---

## 2. BPE 与 Byte-level BPE（BBPE）

**BPE（Byte Pair Encoding）**：在符号序列上反复把最高频相邻对合并成新符号，得到一份有序 merge 表。训练得到 vocab；推理时按 **merge 优先级（越早合并 rank 越低）** 重放，而不是再数频次。

**Byte-level BPE（BBPE）**：先把 Unicode 编成 UTF-8 字节（0–255），再在字节上做 BPE。GPT-2 把这条路做成 LLM 标配。好处：

- 任意 Unicode 都能表示，没有「未知字符」；
- 单个 token 可以是半个 UTF-8 字符（多字节汉字/emoji 常被拆开）——这正是流式 decode 必须缓冲的原因。

minbpe 的 README 写得很直白：算法跑在 UTF-8 字节上，`BasicTokenizer` 最简，`RegexTokenizer` 加上 regex 预切分和 special tokens。

tiktoken 的教学模块把同一件事拆成两步：**regex 切成「词」→ 每个词内部做 byte pair merge**：

```23:37:third_party/tiktoken/tiktoken/_educational.py
    def encode(self, text: str, visualise: str | None = "colour") -> list[int]:
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
            tokens.extend(word_tokens)
        return tokens
```

合并时选 **rank 最小** 的 pair（训练时越早出现的 merge 优先级越高）：

```95:110:third_party/tiktoken/tiktoken/_educational.py
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        ...
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
```

这与 `BasicTokenizer.encode` 用 `self.merges.get(p, inf)` 取最小 merge index 是同一规则。

---

## 3. `base.py`：共享的 merge 原语

`get_stats` 统计相邻对频次；`merge` 把一对替换成新 id：

```13:41:third_party/minbpe/minbpe/base.py
def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    ...
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
```

Vocab 是确定性派生的：先 256 个单字节，再按 merge 拼接，最后把 special token 的 UTF-8 字节写进去。`.vocab` 文件用 `errors='replace'` 打印，**不能用来 load**——因为许多 token 是不完整 UTF-8。

```88:95:third_party/minbpe/minbpe/base.py
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
```

---

## 4. `basic.py`：先读完这个，再读 regex

`BasicTokenizer` **不做** regex 切分、**不处理** special tokens。整段文本 `encode("utf-8")` 成 `0..255` 的整数列表，然后循环合并。

**训练**：每轮 `get_stats` 找最高频 pair，铸新 id `256 + i`。

```20:49:third_party/minbpe/minbpe/basic.py
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        ...
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
```

**推理 encode**：不再数频次，而是在当前序列的所有相邻对里，选 **merge 表里 rank 最低** 的那一对；若没有任何 pair 在 merge 表里则停止。

```57:74:third_party/minbpe/minbpe/basic.py
    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
```

**decode**：把各 token 的 bytes 拼起来，再 `utf-8` decode，非法序列用 `errors="replace"`（`�`）。这就是流式场景里「半个汉字先变成 `�`、下一个 token 才拼出完整字符」的根源。

```51:55:third_party/minbpe/minbpe/basic.py
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
```

**缺了什么（所以才有 `regex.py`）**：

- merge 可以跨「单词 / 数字 / 标点」边界，`"hello world"` 可能把空格和字母焊在一起，泛化变差；
- 用户文本里出现 `<|endoftext|>` 时，没有「整段当一个 id」的通道。

---

## 5. `regex.py`：GPT-2/4 的预切分 + BBPE

GPT-2 引入、GPT-4 沿用的预处理：先用 regex 按 **类别**（字母、数字、标点、空白）切开，**BPE 只在 chunk 内部 merge，绝不跨边界**。

默认用 GPT-4 pattern（与 tiktoken `cl100k_base` 同源；tiktoken 侧还加了占有量词 `++` 做加速）：

```16:19:third_party/minbpe/minbpe/regex.py
# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

tiktoken 官方 `cl100k_base` 的 pattern：

```87:91:third_party/tiktoken/tiktoken_ext/openai_public.py
    return {
        "name": "cl100k_base",
        "pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
```

**训练**：`findall` 得到 chunks，每个 chunk 单独变成 byte ids；`get_stats` 在所有 chunk 上 **累加** 计数，merge 也按 chunk 分别应用——这保证新 token 不会跨类别。

```40:60:third_party/minbpe/minbpe/regex.py
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        ...
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
```

**普通 encode**：同样先切 chunk，再对每个 chunk 跑与 `basic.py` 相同的 `_encode_chunk`。

```111:121:third_party/minbpe/minbpe/regex.py
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
```

Rust 里 tiktoken 的热路径是同一结构：regex `find_iter` → 整段命中 encoder 就直接查表，否则 `byte_pair_encode`。

```360:371:third_party/tiktoken/src/lib.rs
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }
```

---

## 6. Special tokens：为什么默认要 raise

Special tokens 是 **人工插入的控制符**（EOT、FIM、chat 角色标记），不是语料里的普通子词。若用户 prompt 里的字面量 `<|endoftext|>` 被悄悄编成 EOT，模型会提前结束或被注入。

minbpe 与 tiktoken 对齐：`allowed_special="none_raise"`（默认）时，文本里出现已注册 special 就 `assert`；`"all"` 才按 special id 编码。实现上用 **捕获组 `re.split`**，把 special 片段从普通文本里剥出来，special 走查表，其余走 `encode_ordinary`。

```123:164:third_party/minbpe/minbpe/regex.py
    def encode(self, text, allowed_special="none_raise"):
        """
        ...
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        ...
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        ...
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
```

tiktoken 生产代码把「禁止」做成显式 `ValueError`，并区分 `allowed_special` / `disallowed_special`：

```82:124:third_party/tiktoken/tiktoken/core.py
    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes a string into tokens.
        ...
        Hence, by default, encode will raise an error if it encounters text that corresponds
        to a special token.
        """
        ...
            if match := _special_token_regex(disallowed_special).search(text):
                raise_disallowed_special_token(match.group())
```

GPT-4 / `cl100k_base` 的 special 集合（minbpe `gpt4.py` 与 tiktoken 一致）：

```49:55:third_party/minbpe/minbpe/gpt4.py
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
```

Chat 模型里还有 `<|im_start|>` / `<|im_end|>` 这类 **对话角色标记**。它们通常由 **chat template** 插入，而不是用户在 raw encode 里随手打开 `allowed_special="all"`。模板层与 BPE 层解耦，正是下一节的要点。

HF 侧 special 是 `AddedToken`（`special=True` 默认不 normalize，decode 时可 skip），在 Model 之前用 Aho-Corasick 从原文抽出。见 [01-hf-tokenizers.md](01-hf-tokenizers.md)。

---

## 7. 为什么线上必须拆成三段

### 7.1 Chat template ≠ BPE

`apply_chat_template` 是 Jinja：拼 system/user/assistant、工具 schema、generation prompt、多模态 placeholder。vLLM 把它放在 `HfRenderer.safe_apply_chat_template`，明确与 tokenizer 的 BPE 分离。

SGLang OpenAI serving 甚至把 `tokenize=True` **拆成 render 字符串 + encode**，避免 chat template 已经带了角色 special 之后，encode 再 `add_special_tokens` 造成双 BOS：

```1375:1396:third_party/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
            # Split apply_chat_template(tokenize=True) into render + encode so we
            # can skip add_special_tokens=False on tokenizers that don't auto-add
            # specials (Kimi-like, OpenAI-chat analogue of #25265). Chat
            # templates already include role/special tokens, so the encode must
            # avoid double BOS on tokenizers that would add it.
            ...
                rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    ...
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
```

长对话 + 工具定义时，**Jinja 渲染经常比 BPE 更贵**。所以测延迟时要单独记 `chat_template_ms` 和 `encode_ms`，不要混成一个「tokenize」。

### 7.2 Encode 必须离开 asyncio / 可被跳过

Encode 是 CPU 密集、可能持有 Rust tokenizer 内部锁。vLLM 用线程池；SGLang 用独立进程。

若网关已经 tokenize 好，engine 可以 `skip_tokenizer_init`，只吃 `prompt_token_ids`：延迟最优，但 **丢掉服务端模板一致性**（客户端必须自己渲对 chat template）。

```81:83:third_party/vllm/vllm/entrypoints/llm.py
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
```

```270:272:third_party/vllm/vllm/tokenizers/registry.py
def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None
```

### 7.3 Incremental detokenize：不能每步全量 decode

BBPE 的 token 边界 ≠ UTF-8 字符边界。逐步 `decode([id])` 会：

- 在半字符处吐 `�`，下一次又把前面重解一遍，流式文本抖动；
- Metaspace / 前导空格 decoder 会把「第一个 token 的前导标记」剥掉，导致 `"This"+"This"` 变成 `"ThisThis"` 而不是 `"This This"`（HF `DecodeStream` 文档里的反例，见 01 笔记）。

因此必须保留前缀状态，只发射 **新的合法 UTF-8 增量**。

vLLM fast 路径直接调 `tokenizers.decoders.DecodeStream`：

```168:187:third_party/vllm/vllm/v1/engine/detokenizer.py
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(self, tokenizer: TokenizersBackend, request: EngineCoreRequest):
        ...
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
```

`decode_next` 里还要处理 **连续 special token 之间不要多空格**，以及 `Invalid prefix` 时重置 stream（非单调 / 非法 UTF-8 会打坏内部 prefix 状态）。

SGLang 不用 DecodeStream，而用 `surr_offset` / `read_offset`：对「已确认前缀」和「当前可读后缀」分别 `batch_decode`，差集才是增量；遇到不完整 UTF-8 只发 `find_printable_text`，**不提交 offset**，下一轮带上更多 token 再试。

```75:81:third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py
class DecodeStatus:
    """Store the status of incremental decoding."""

    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
```

```394:404:third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    ...
                else:
                    # Incomplete UTF-8: emit the printable prefix only; do not
                    # commit (token offsets stay so the next iteration retries
                    # with more tokens).
                    printable = find_printable_text(new_text)
```

Stop string 往往在 **文本域** 匹配，所以 detokenize 还要和 stop / tool parser 绑在一起——又一个「不能只跑 BPE」的理由。

---

## 8. 读完 basic → regex 之后你应该能画的图

```
训练:  UTF-8 bytes → (optional regex chunks) → 反复 merge 高频 pair → merges + vocab
推理:  文本
         ├─ special tokens（精确匹配，可选 / 默认拒绝）
         └─ regex 预切分 → 每 chunk UTF-8 bytes → 按 merge rank 重放 → ids

Serving:
  messages ──Jinja template──► string ──encode──► ids ──GPU──► new ids
                                                              │
                                                              ▼
                                              incremental detokenize ──► chunk
```

下一篇把「regex 预切分 + BPE」放进 HF 的完整 pipeline（Normalizer → PreTokenizer → Model → PostProcessor → Decoder），并解释 `is_fast=False` 和 `Already borrowed`。
