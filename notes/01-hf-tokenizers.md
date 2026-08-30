# HuggingFace tokenizers：Pipeline、DecodeStream、Slow/Fast、tiktoken、线程安全

本文对照 `third_party/tokenizers`（Rust 核心 + Python 绑定）、`third_party/tiktoken`，并简要提到 `third_party/fastokens`。算法基础见 [00-fundamentals.md](00-fundamentals.md)。

本仓库克隆 SHA：`tokenizers @ d5827816baedcbf1cb5b452dea8048150b6872df`。这一版 Python `Tokenizer` 已用 `Arc<RwLock>` 包住内部状态；**线上 serving 仍普遍按「不能多线程共享同一个 Fast tokenizer 对象」来设计**（vLLM deepcopy 池、测试注释里的 `already borrowed`）。下文会把历史陷阱和当前源码对齐写清。

## 该打开的文件

| 文件 | 看什么 |
| --- | --- |
| `third_party/tokenizers/tokenizers/README.md` | 官方五段 pipeline 定义 |
| `third_party/tokenizers/tokenizers/src/tokenizer/mod.rs` | `TokenizerImpl` 字段；`encode_single_sequence` / `decode` / `DecodeStream` |
| `third_party/tokenizers/tokenizers/src/tokenizer/added_vocabulary.rs` | special / added tokens 在 Normalizer 前后抽出 |
| `third_party/tokenizers/tokenizers/src/pre_tokenizers/byte_level.rs` | GPT-2 byte-level：regex + bytes↔unicode 映射 |
| `third_party/tokenizers/tokenizers/src/models/bpe/model.rs` | BPE `tokenize`；thread-local `RefCell` 缓存 |
| `third_party/tokenizers/tokenizers/src/processors/bert.rs` | PostProcessor 加 `[CLS]`/`[SEP]` |
| `third_party/tokenizers/bindings/python/src/tokenizer.rs` | `PyTokenizer`：`Arc<RwLock<Tokenizer>>` |
| `third_party/tokenizers/bindings/python/src/decoders.rs` | Python `DecodeStream.step` |
| `third_party/tiktoken/tiktoken/core.py` + `src/lib.rs` | 推理专用：regex + BBPE，无 train |
| `third_party/fastokens/README.md` + `python/tests/test_thread_safety.py` | 更快的推理 BPE；显式提到 `"already borrowed"` |
| `third_party/vllm/vllm/tokenizers/hf.py` | `maybe_make_thread_pool` / `get_cached_tokenizer` |

---

## 1. Pipeline：Normalizer → PreTokenizer → Model → PostProcessor → Decoder

HF 自己的定义（Rust crate README）：

```24:36:third_party/tokenizers/tokenizers/README.md
A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
The various steps of the pipeline are:

1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
   the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
   More details about how to use the `Normalizers` are available on the
   [Hugging Face blog](https://huggingface.co/docs/tokenizers/components#normalizers)
2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
   splitting text is simply on whitespace.
3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
   `BPE` or `WordPiece`.
4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
   that, for example, a language model would need, such as special tokens.
```

结构体就是这五段（外加 added vocabulary、truncation、padding）：

```544:558:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
pub struct TokenizerImpl<M, N, PT, PP, D> {
    // Tokenizer parts
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    model: M,
    post_processor: Option<PP>,
    decoder: Option<D>,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,
    ...
}
```

Python 绑定把同一顺序写进 docstring：

```481:492:third_party/tokenizers/bindings/python/src/tokenizer.rs
/// A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
/// and outputs an :class:`~tokenizers.Encoding`.
///
/// The pipeline is structured as follows:
///
///     1. The :class:`~tokenizers.normalizers.Normalizer` normalizes the raw input text.
///     2. The :class:`~tokenizers.pre_tokenizers.PreTokenizer` splits the normalized text
///        into word-level tokens.
///     3. The :class:`~tokenizers.models.Model` tokenizes each word into subword tokens
///        and maps them to IDs.
///     4. The :class:`~tokenizers.processors.PostProcessor` applies any final
///        transformations (e.g., adding special tokens like ``[CLS]`` and ``[SEP]``).
```

**Encode 热路径**（`encode` → `encode_single_sequence` → `post_process`）：

```768:784:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
        let encode = |is_pre_tokenized, subseq_idx, subseq| -> Result<Encoding> {
            let normalized = self
                .added_vocabulary
                .extract_and_normalize(self.normalizer.as_ref(), subseq);
            let pre_tokenized = self.do_pre_tokenize(normalized)?;
            let subseq_encoding = self.do_tokenize(
                pre_tokenized,
                type_id,
                ...
            )?;
            Ok(subseq_encoding)
        };
```

对应关系（对照 minbpe）：

| HF 段 | minbpe / tiktoken 里的对应物 |
| --- | --- |
| Normalizer | minbpe 没有（GPT-2 BBPE 通常 NFC/不 lower）；BERT 会 lowercase + accent strip |
| PreTokenizer | `regex.py` 的 `findall(pattern)`；byte-level 还要做 GPT-2 的 bytes↔printable unicode |
| Model | `_encode_chunk` / tiktoken `byte_pair_encode`；也可以是 WordPiece / Unigram |
| PostProcessor | minbpe 没有；BERT 在这里包 `[CLS] … [SEP]` |
| Decoder | `decode` 时 bytes→UTF-8；ByteLevel 要把映射字符还原成真实字节 |
| AddedVocabulary | `register_special_tokens` + `re.split`，但 HF 用 trie，且分「是否 normalize」两路 |

---

## 2. 各段在源码里做什么

### 2.1 Added vocabulary + Normalizer

真正的 `do_normalize` 很薄：有 normalizer 就 `normalize(&mut NormalizedString)`。

但 encode 入口先走 `extract_and_normalize`：用两棵 Aho-Corasick trie，**先从不规范化原文抽出 `normalized=False` 的 added/special tokens，再对剩余片段 normalize，再抽 `normalized=True` 的 token**。这样 `<s>` 不会被 NFC/lowercase 拆碎，而 `"yesterday"` 这类普通 added token 仍能匹配规范化后的文本。

```523:551:third_party/tokenizers/tokenizers/src/tokenizer/added_vocabulary.rs
    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        normalizer: Option<&N>,
        sequence: &str,
    ) -> PreTokenizedString {
        let mut pretokenized: PreTokenizedString = sequence.into();

        // 1. We extract all the non-normalized tokens from the non-normalized string
        pretokenized
            .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.split_trie)))
            ...
        // 2. Then extract the normalized tokens from the normalized pieces of the string
        pretokenized
            .split(|_, mut sequence| {
                normalizer.map(|n| n.normalize(&mut sequence));
                Ok(self.split_with_indices(sequence, &self.split_normalized_trie))
            })
```

`AddedToken` 的语义：

```17:41:third_party/tokenizers/tokenizers/src/tokenizer/added_vocabulary.rs
pub struct AddedToken {
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}
impl AddedToken {
    pub fn from<S: Into<String>>(content: S, special: bool) -> Self {
        Self {
            content: content.into(),
            normalized: !special,
            special,
            ..Default::default()
        }
    }
```

`special=True` 默认 `normalized=False`，decode 时可 skip。这与 tiktoken「special 必须精确匹配、不走 BPE」一致，但 HF 把 special 嵌进 pipeline 前端，而不是 `encode(allowed_special=...)` 参数。

### 2.2 PreTokenizer（ByteLevel ≈ GPT-2 regex + byte 映射）

GPT-2 风格的 byte-level：可选加前导空格，用与 GPT-2 几乎相同的 regex 切开，再把每个 UTF-8 字节映射成「看起来像 unicode 字符」的码点（控制字节映射到 `U+0100+`），这样 BPE 的 vocab 全是合法 Unicode 字符串。

```41:46:third_party/tokenizers/tokenizers/src/pre_tokenizers/byte_level.rs
/// Regex that matches exactly one token.
/// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98
static RE: LazyLock<SysRegex> = LazyLock::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});
```

```120:146:third_party/tokenizers/tokenizers/src/pre_tokenizers/byte_level.rs
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            ...
                    s.as_bytes()[i..i + size]
                        .iter()
                        ...
                        .map(|(i, b)| (BYTES_CHAR[b], isize::from(i > 0))),
```

Llama / GPT-4 类 tokenizer.json 常用 `Split` 预分词（cl100k 风格 regex）而不是这条 GPT-2 `RE`；**Model 仍然是 BPE**。`use_regex=false` 是给自定义切分（BigScience 等）留的口子。

### 2.3 Model：BPE / WordPiece / Unigram

`do_tokenize` 对每个 pre-token 调 `Model::tokenize`：

```1176:1199:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
    fn do_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
        ...
    ) -> Result<Encoding> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();
        ...
        self.model
            .tokenize_in_pretokenized(&mut pretokenized, truncation)?;
        pretokenized.into_encoding(word_idx, type_id, offsets_type)
    }
```

BPE 实现带 **按实例 id 隔离的 thread-local 缓存**（`RefCell`，只在单线程内，不是跨线程共享状态）：

```82:89:third_party/tokenizers/tokenizers/src/models/bpe/model.rs
thread_local! {
    /// Per-thread BPE tokenization cache.  This is the only BPE cache
    /// on the hot path: there is no shared global map, so lookups and
    /// inserts need no atomic synchronization at all.
    static BPE_LOCAL_CACHE: RefCell<AHashMap<u64, AHashMap<String, Word>>> =
        RefCell::new(AHashMap::new());
}
```

```601:611:third_party/tokenizers/tokenizers/src/models/bpe/model.rs
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_with_cache(sequence)
        } else {
            let word = self.merge_word(sequence)?;
            Ok(self.word_to_tokens(&word).collect())
        }
    }
```

同一 crate 还实现 WordPiece（BERT）和 Unigram（SentencePiece 风格）。Unigram 的 lattice 也用 `Rc<RefCell<Node>>`，那是算法内部图结构，与 Python 绑定的「Already borrowed」不是同一件事。

### 2.4 PostProcessor

`post_process`：先按 `TruncationParams` 截断（预留将要插入的 special 数量），再调 PostProcessor，最后 padding。

BERT 的 PostProcessor 在单句上插入 `[CLS] + ids + [SEP]`：

```51:67:third_party/tokenizers/tokenizers/src/processors/bert.rs
    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if !add_special_tokens {
            return Ok(encodings);
        }
        ...
                if i == 0 {
                    let ids = [&[self.cls.1], encoding.get_ids(), &[self.sep.1]].concat();
```

GPT-2 ByteLevel 作为 PostProcessor **不加 special**，只 trim offsets。Chat 模型的 BOS/EOS 多半来自 **Jinja chat template** 或 `AddedToken`，不要和 BERT 的 `[CLS]` 混为一谈。Chat 场景 encode 时常 `add_special_tokens=False`，避免和模板里已经写死的角色标记叠两层（见 00 笔记里 SGLang serving_chat 的注释）。

### 2.5 Decoder

`TokenizerImpl::decode`：id → token 字符串（added vocab 优先），可选跳过 special，再交给 Decoder；没有 Decoder 就用空格 join。

```934:952:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .iter()
            .filter_map(|id| {
                self.added_vocabulary
                    .simple_id_to_token(*id)
                    .or_else(|| self.model.id_to_token(*id))
                    .filter(|token| {
                        !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                    })
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }
```

ByteLevel decoder 把映射字符还原成字节再 `from_utf8_lossy`——**一次消费全部 tokens 再拼**，避免单 token 不是合法 UTF-8：

```150:171:third_party/tokenizers/tokenizers/src/pre_tokenizers/byte_level.rs
/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as
/// as String.
impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| { ... })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}
```

---

## 3. DecodeStream：流式 decode 的真正实现

全量 `decode(ids)` 假设看到完整序列。流式生成每次只多一个 id，会遇到：

1. **UTF-8 / byte-fallback 半字符**：当前 id 还构不成合法字符串，应返回 `None`，等后续 id。
2. **依赖周围 token 的 decoder**（Metaspace 剥前导 `▁`、strip 空格）：逐步 `decode([t])` 与 `decode([t, t])` 的增量对不上。

`DecodeStream` 内部保留 `ids` / `prefix` / `prefix_index`，每步把新 id 拼进去做一次 decode，检查结果是否以旧 prefix 开头且不以 `�` 结尾，再切出增量并滑动窗口。

```962:990:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
/// DecodeStream will keep the state necessary to produce individual chunks of
/// strings given an input stream of token_ids.
///
/// This is necessary because decoding in general cannot achieve that since strings
/// depend on surrounding ids to provide a valid string. Typically stripping extra spaces
...
/// Returning `None` means the given id is not enough to produce a chunk.
/// This typically happens with `byte_fallback` options where some tokens do
/// not represent valid utf-8, and only follow-up token_ids will help produce
/// a valid chunk.
```

Metaspace 反例（逐步 decode 会丢掉第二次的空格）：

```1051:1059:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
/// // Strip decoder removes the extra initial space
/// assert_eq!(tokenizer.decode(&[0, 0], false).unwrap(), "This This");
/// // Decoding one token at a time would produce "ThisThis"
/// assert_eq!(tokenizer.decode(&[0], false).unwrap(), "This");
///
/// // Using a stream fixes it by keeping the necessary state.
```

`step_decode_stream` 的核心：

```1143:1170:third_party/tokenizers/tokenizers/src/tokenizer/mod.rs
    ids.extend(token_ids);
    let string = tokenizer.decode(ids.as_slice(), skip_special_tokens)?;
    if string.len() > prefix.len() && !string.ends_with('�') {
        if !(string.starts_with(&*prefix)) {
            return Err(Box::new(DecodeStreamError::InvalidPrefix { ... }));
        }
        let new_text = &string[prefix.len()..].to_string();
        ...
        Ok(Some(new_text.to_string()))
    } else {
        Ok(None)
    }
```

Python API 把状态放在 `DecodeStream` 对象上，`step(tokenizer, id)` 每次 **读锁** 取 tokenizer：

```846:879:third_party/tokenizers/bindings/python/src/decoders.rs
    fn step(&mut self, tokenizer: &PyTokenizer, id: StreamInput) -> PyResult<Option<String>> {
        let id: Vec<u32> = match id { ... };
        let tokenizer_guard = tokenizer.read_inner()?;
        ToPyResult(tk::tokenizer::step_decode_stream(
            &tokenizer_guard,
            id,
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        ))
        .into()
    }
```

vLLM `FastIncrementalDetokenizer` 用 prompt ids **prefill** stream，然后每个新 token `stream.step`；遇到 `Invalid prefix` 就重建 stream（见 00 笔记）。这就是 serving 第三段的库级实现。

---

## 4. Slow vs Fast

本仓库没有克隆 `transformers`，但 vLLM 的加载路径把区分写死了。

- **Fast**：`transformers.TokenizersBackend`，底层是本 crate 的 Rust `Tokenizer`（`tokenizer.json`）。`is_fast=True`，有 `offset_mapping`，能用 `DecodeStream`。
- **Slow**：`transformers.PythonBackend`，纯 Python（BERT WordPiece 的 `tokenize` 循环、或从 vocab 文件慢慢查）。同一模型慢一到两个数量级并不罕见。

vLLM 发现 `not tokenizer.is_fast` 会直接打警告：

```258:262:third_party/vllm/vllm/tokenizers/registry.py
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
```

增量 detokenize 也按 backend 分叉：Fast 走 `DecodeStream`，否则 Python 的 `SlowIncrementalDetokenizer`（自己维护 `prefix_offset` / `read_offset`）。

```61:66:third_party/vllm/vllm/v1/engine/detokenizer.py
        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            return FastIncrementalDetokenizer(tokenizer, request)
        return SlowIncrementalDetokenizer(tokenizer, request)
```

Renderer 只有 fast tokenizer 才声称能产 char offsets：

```931:934:third_party/vllm/vllm/renderers/hf.py
    def _can_produce_offsets(self) -> bool:
        # HF tokenizers may be slow (use_fast=False); only fast tokenizers
        # expose offset_mapping.
        return self.tokenizer is not None and self.tokenizer.is_fast
```

另外 HF 若干属性（`all_special_ids`、`get_vocab()`）每次访问会重算。vLLM `get_cached_tokenizer` 把它们 freeze 到闭包里：

```107:111:third_party/vllm/vllm/tokenizers/hf.py
def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
```

**Serving 纪律：永远用 fast；slow 只做对照实验。** `experiments/bench_backends.py`（其他 agent）应用同一段中英混合长文本对比 slow / fast / tiktoken。

---

## 5. tiktoken vs HF tokenizers

tiktoken 是 **推理专用 Encoding**：构造时给定 `pat_str` + `mergeable_ranks` + `special_tokens`，没有 train API，没有 Normalizer/PostProcessor 插件链。

```16:57:third_party/tiktoken/tiktoken/core.py
class Encoding:
    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        mergeable_ranks: dict[bytes, int],
        special_tokens: dict[str, int],
        explicit_n_vocab: int | None = None,
    ):
        ...
        self._core_bpe = _tiktoken.CoreBPE(mergeable_ranks, special_tokens, pat_str)
```

Python 的 `encode` 在释放 GIL 后进 Rust（`py.detach`）：

```34:48:third_party/tiktoken/src/py.rs
    fn py_encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> PyResult<Vec<Rank>> {
        py.detach(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();
            match self.encode(text, &allowed_special) {
                Ok((tokens, _)) => Ok(tokens),
                ...
            }
        })
    }
```

| | HF `tokenizers` | tiktoken |
| --- | --- | --- |
| 目标 | 训练 + 推理，多种算法 | 只推理 OpenAI BBPE |
| 配置 | `tokenizer.json` 五段插件 | `pat_str` + ranks + special 字典 |
| Special | `AddedToken` + PostProcessor + chat template | `allowed_special` / `disallowed_special`，默认拒绝 |
| 流式 decode | `DecodeStream` | 无官方 stream；`decode` 是 `decode_bytes` + UTF-8（默认 `errors="replace"`，有损） |
| 并发 | 见下一节；历史上 PyO3 borrow | `Encoding` 只读，`encode_batch` 直接 `ThreadPoolExecutor` |
| 覆盖模型 | BERT / Llama / Qwen / … | `r50k` / `p50k` / `cl100k` / `o200k` |

tiktoken README 自己的对照实验（**那是上游仓库里的历史数字，不是本仓库 `experiments/` 的结果**）：相对当时的 `GPT2TokenizerFast`，tiktoken 约 3–6×。不要把该数字当成当前 HF tokenizers 或 fastokens 的结论。

OpenAI 模型用 tiktoken；开源权重几乎都走 HF `tokenizer.json`。Llama-3 等是「HF 壳 + GPT-4 风格 regex BPE」，算法近、文件格式不同。

---

## 6. Rust / PyO3：「Already borrowed」与 RefCell

### 6.1 现象

多线程共享 **同一个** HF Fast tokenizer 对象做 `encode` / `decode`，经典报错：

```text
RuntimeError: Already borrowed
```

vLLM 测试在并发预处理前会显式套上池子，注释就是这句话：

```47:48:third_party/vllm/tests/models/multimodal/processing/test_llava_onevision.py
    # Avoid tokenizer already borrowed error
    maybe_make_thread_pool(ctx.tokenizer)
```

### 6.2 根因：PyO3 对 `#[pyclass]` 的运行时 borrow check

PyO3 把 Python 对象里的 Rust 结构放在类似 `RefCell` 的内部借用标志里：一次 `encode` 取得 `&self`/`&mut self` 直到调用返回。另一个线程同时进入同一对象 → 第二次 `borrow` 失败，变成 `RuntimeError: Already borrowed`。

这和「Rust BPE 算法能不能并行」不是一回事。BPE 热路径的 `BPE_LOCAL_CACHE` 是 **thread_local RefCell**，每个 OS 线程一份，专门避免共享 `RwLock` 缓存。跨线程炸的是 **Python 绑定对象**。

tiktoken 的 `CoreBPE` 在 `py.detach` 期间不持有 PyO3 对 Encoding 的可变借用去调回 Python，且内部数据只读，所以 `encode_batch` 可以裸线程池。

### 6.3 本 SHA 的 tokenizers：内部改成 `Arc<RwLock>`

```517:522:third_party/tokenizers/bindings/python/src/tokenizer.rs
pub struct PyTokenizer {
    /// `Arc` so cloning is a refcount bump (matches the pre-RwLock semantics
    /// where the inner tokenizer was shared across `PyTokenizer` clones).
    /// `RwLock` so concurrent setters and encoders don't race PyO3's
    /// per-pyclass borrow check on free-threaded Python.
    pub(crate) tokenizer: Arc<RwLock<Tokenizer>>,
}
```

README 写明：setter 拿写锁，并发 encode 拿读锁。

```77:81:third_party/tokenizers/bindings/python/README.md
The full mutable API works on 3.14t — the same as on regular CPython.
Setters are thread-safe: the inner tokenizer state is wrapped in a
`std::sync::RwLock`, so concurrent `tokenizer.X = …` from multiple threads
serialize correctly and concurrent encode operations take a read guard
that blocks writers only briefly.
```

`from_py_object` 仍可能让「嵌套调用同一 pyclass」碰到 PyO3 层 borrow；`decode` `allow_threads` 时另一线程若走可变 API，旧代码会 panic。fastokens 测试把这段历史写进了 docstring：

```1:8:third_party/fastokens/python/tests/test_thread_safety.py
"""Concurrency test: `decode` (read) and `enable_truncation` (write) on the
same `Tokenizer` instance must not panic the Rust side.

`decode` releases the GIL via `py.allow_threads`, so another Python thread
can call a mutator while it is running. Before the `RwLock` refactor, that
race could surface as a PyO3 borrow-check panic ("already borrowed"); with
the lock, the mutator simply waits its turn.
"""
```

### 6.4 Serving 为什么仍用 deepcopy 池

vLLM 不依赖「新版 RwLock 已经够用」，而是 **深拷贝出 N 份 tokenizer**，每次调用从队列借一份：

```25:57:third_party/vllm/vllm/tokenizers/hf.py
def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """
    If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.
    ...
    - Adjacent method calls could happen on different deep copies.
    """
    ...
    tokenizer_pool: queue.Queue[TokenizersBackend] = queue.Queue()
    for _ in range(copies):
        tokenizer_pool.put(copy.deepcopy(og_tokenizer))

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = copy.deepcopy(og_tokenizer)
            yield tok
        finally:
            tokenizer_pool.put(tok)
```

原因 recap：

1. 部署的 `tokenizers` wheel 版本不一，旧 wheel 仍是 RefCell 语义；
2. 相邻的 `apply_chat_template` 然后 `encode` 可能落在不同副本上——文档已声明；
3. `DecodeStream` 还要同时 borrow tokenizer（`step` 里 `read_inner`），和并发 encode 叠在同一对象上仍然别扭；
4. Transformers 包装层自己的可变状态不只 Rust 那一把锁。

**实验意图**（`experiments/bench_concurrency.py`，勿在此编造数字）：单实例多线程触发 `Already borrowed` 或锁等待；deepcopy 池 / 多进程对比吞吐。SGLang 用进程隔离从根本上避开「同一 pyclass 跨线程」。

---

## 7. fastokens（简要）

[fastokens](https://github.com/crusoecloud/fastokens) 是推理向 BPE 后端：读 HF `tokenizer.json` 或 tiktoken ranks，Rust 实现，API 故意做成 HF 子集。README 声称相对 `tokenizers` 平均 10x+（**上游宣传数字，本仓库尚未复现**）。不支持完整 Encoding 附加输出和全部 normalizer/pretokenizer。

```3:5:third_party/fastokens/README.md
fastokens is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer for use with
popular open-weight LLMs, built on top of a high-performance Rust backend. It loads both the
HuggingFace `tokenizer.json` format and [tiktoken](https://github.com/openai/tiktoken) model files.
```

```38:39:third_party/fastokens/README.md
Note that `fastokens` is focused on inference and does not support all features of `tokenizers`.
In particular, additional encoding outputs, and some normalizers/pretokenizers are not available.
```

Python 绑定同样用 `RwLock` 包状态：

```480:484:third_party/fastokens/python/src/lib.rs
/// An LLM tokenizer backed by `tokenizer.json`.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    state: RwLock<TokenizerState>,
}
```

接入 serving：

- vLLM：`VLLM_USE_FASTOKENS=1` 时 `fastokens.patch_transformers()`，换掉 HF fast tokenizer 内部 Rust，并 **rebind `tokenizers.decoders.DecodeStream`**，这样 `FastIncrementalDetokenizer` 不用改。

```3:10:third_party/vllm/vllm/tokenizers/fastokens.py
When ``VLLM_USE_FASTOKENS=1`` is set, ``fastokens.patch_transformers()`` swaps
the inner Rust tokenizer of every HF fast tokenizer loaded afterwards with the
fastokens shim and rebinds ``tokenizers.decoders.DecodeStream`` so the
streaming detokenizer accepts the shim.
```

- 用法示例：`fastokens.patch_transformers()` 后照常 `AutoTokenizer.from_pretrained`。

换后端前必须 **同一模型 encode 逐 token 对齐**（正确性），再谈 TTFT。对齐与微基准属于 `notes/05-performance.md` / `experiments/`。

---

## 8. 一张图串起来

```
raw text
  │
  ├─ AddedVocabulary.extract_and_normalize   (special / added tokens)
  ├─ Normalizer                              (NFC, lowercase, …)
  ├─ PreTokenizer                            (regex / ByteLevel / Whitespace)
  ├─ Model                                   (BPE | WordPiece | Unigram)
  ├─ PostProcessor                           ([CLS]/[SEP], trim offsets, …)
  └─ (padding / truncation)
        │
        ▼
     Encoding.ids  ──► GPU
        │
        ▼  (streaming)
     DecodeStream.step  ──► 增量 UTF-8 chunk
```

并发：不要假设「Fast tokenizer 对象是 `Sync` 的只读句柄」。旧绑定是 PyO3 `RefCell`；新绑定是 `RwLock`；vLLM 用副本池；SGLang 用进程。tiktoken `Encoding` 更接近只读。fastokens 想在不改 pipeline 形状的前提下把 Model 段换掉。
