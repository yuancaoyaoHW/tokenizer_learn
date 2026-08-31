# 07 · 启动期选 tokenizer / 运行期进 encode

对照源码钉：

- `third_party/vllm` @ `2c7d7dd64a2eaba0feedf42cab2f527486d7479c`
- `third_party/sglang` @ `e635577431cbdfb8ce5fafb0fcd8a4ac074062c6`

本仓库**没有** clone `transformers`（见 README「未克隆」）。HF 那一列只写接口层事实，以及两边注释里能钉死的 v5 行为，不给 `file:line`。

[02-vllm.md](02-vllm.md) / [03-sglang.md](03-sglang.md) 跟一条请求走调用链；[04-compare.md](04-compare.md) 比隔离模型。本篇只比 **分发**：启动时选哪个 tokenizer 类，运行时一条请求怎么进 encode。三方差异最大的是前者。

```
启动期  get_tokenizer(...)  →  某个 tokenizer 实例（进程里常驻）
运行期  HTTP 文本 / 已有 ids / embeds  →  要不要 encode、单条还是批量
```

## 该打开的文件

| 主题 | vLLM | SGLang |
| --- | --- | --- |
| 入口 | `vllm/tokenizers/registry.py` | `python/sglang/srt/utils/hf_transformers/tokenizer.py` |
| HF 包装 | `vllm/tokenizers/hf.py` `CachedHfTokenizer` | 同上文件里的 `AutoTokenizer` + 修复函数 |
| 运行期 encode | `vllm/renderers/base.py` | `python/sglang/srt/managers/tokenizer_manager.py` |
| Renderer 由 mode 选出 | `vllm/renderers/registry.py` | 无此层 |
| MM 侧补丁（非 get_tokenizer） | — | `python/sglang/srt/utils/hf_transformers/processor.py` `_fix_added_tokens_encoding` |

兼容 shim：`python/sglang/srt/utils/hf_transformers_utils.py` 只 re-export，不要在那里找实现。

---

## 1. 启动期：选哪个 tokenizer 实现

### 1.1 vLLM — 声明式注册表

入口 `get_tokenizer`（`registry.py:186`）。两段式：**先把 mode 收敛成表里的 key，再查表取类**。

```42:56:third_party/vllm/vllm/tokenizers/registry.py
_VLLM_TOKENIZERS = {
    # ``cohere`` mode uses the standard cached HF tokenizer; only the
    # renderer (template stage) is replaced with a melody-based one.
    "cohere": ("hf", "CachedHfTokenizer"),
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Tokenizer"),
    "deepseek_v4": ("deepseek_v4", "DeepseekV4Tokenizer"),
    "hf": ("hf", "CachedHfTokenizer"),
    "kimi_audio": ("kimi_audio", "KimiAudioTokenizer"),
    "kimi_k3": ("hf", "CachedHfTokenizer"),
    "mistral": ("mistral", "MistralTokenizer"),
    # Inkling uses the plain HF tokenizer for token operations; the "inkling"
    # mode exists to select the InklingRenderer, which renders chat to
    # token ids natively (Inkling has no Jinja chat template).
    "inkling": ("hf", "CachedHfTokenizer"),
}
```

`cohere` / `kimi_k3` / `inkling` 指向同一个 `CachedHfTokenizer`。它们存在的意义是**选 renderer 而不是选 tokenizer**：`renderer_from_config` 用同一套 `tokenizer_args_from_config` 得到 `renderer_mode`。

归一化是独立纯函数，被 `lru_cache` 包住（`cached_resolve_tokenizer_args`，`registry.py:169`）：

```141:164:third_party/vllm/vllm/tokenizers/registry.py
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")

        tokenizer_mode = "hf"
        kwargs["use_fast"] = False

    # Try to use official Mistral tokenizer if possible
    if (
        tokenizer_mode == "auto"
        and is_mistral_model_repo(
            model_name_or_path=str(tokenizer_name), revision=revision
        )
        and any_pattern_in_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=["tekken.json", "tokenizer.model.v*"],
            revision=revision,
        )
    ):
        tokenizer_mode = "mistral"

    # Fallback to HF tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"
```

同一函数还按 `runner_type` 设 `truncation_side`：generate / draft 截左，pooling 截右（`registry.py:133-139`）。

然后 `TokenizerRegistry.load_tokenizer_cls` 用 `resolve_obj_by_qualname` 动态导入（`registry.py:78-85`），统一 `cls.from_pretrained`（`registry.py:253`）。外部插件走 `register()`（`registry.py:64`）。

`get_tokenizer` 自己还有第二层 `lru_cache`（`cached_get_tokenizer`，`registry.py:267`）：同一组 `from_pretrained` 参数不要重复加载。这和 `get_cached_tokenizer` 的 **O(1) 属性缓存**不是一回事，见 02 §设计点 1。

### 1.2 SGLang — 命令式探测链

入口同一个名字 `get_tokenizer`（`tokenizer.py:470`），判定和加载写在一个函数里，顺序 if/elif。先拦非 HF 格式和 fastokens patch：

```470:487:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    tokenizer_backend: str = "huggingface",
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    # Tiktoken format has its own backend — no fastokens patching needed.
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_backend == "fastokens":
        _ensure_fastokens_patched()
```

`slow` 强制 `use_fast=False`；`auto` 显式 `use_fast=True`（给后面非 AutoTokenizer 的回退路径用，因为 v5 AutoTokenizer 会忽略这个参数）。然后 GGUF 无 sidecar 且有原生支持则直接 `build_gguf_tokenizer` 并 return。再 `_resolve_tokenizer_name`。裸 tekken（有 `tekken.json`、无 `tokenizer.json`）走 `MistralCommonTokenizer`，注释写明忽略 `tokenizer_backend`：

```520:540:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
    try:
        if is_bare_tekken_checkpoint(tokenizer_name, tokenizer_revision):
            from transformers.tokenization_mistral_common import (
                MistralCommonTokenizer,
            )

            logger.info(
                "Detected bare-tekken checkpoint %s (tekken.json, no "
                "tokenizer.json); loading via mistral-common MistralCommonTokenizer, "
                "ignoring tokenizer_backend=%r.",
                tokenizer_name,
                tokenizer_backend,
            )

            tokenizer = MistralCommonTokenizer.from_pretrained(
                tokenizer_name, revision=tokenizer_revision
            )
        else:
            tokenizer = _auto_tokenizer_from_pretrained(
                tokenizer_name, *args, **common_kwargs
            )
```

其余走 `AutoTokenizer`，**再看结果对不对**（下一节）。`get_tokenizer` 本身没有 `lru_cache`，实例由 TokenizerManager / DetokenizerManager / Scheduler 各持一份。

加载成功之后还有一整套修复，vLLM 几乎没有对等物：

```426:441:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
def _apply_post_load_fixes(tokenizer, tokenizer_name, revision):
    """Apply all post-load patches and return the final tokenizer."""
    _install_tokenizer_warnings_filter(tokenizer)
    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, revision)
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, revision)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    patch_mistral_common_tokenizer(tokenizer)
    _fix_special_tokens_pattern(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return patch_tokenizer(tokenizer)
```

| 补丁 | 修什么 |
| --- | --- |
| `_fix_v5_tokenizer_components`（`:266`） | Llama 类 `__init__` 用类默认 pre_tokenizer/decoder 盖掉 `tokenizer.json`（DeepSeek-V3.2 实际是 ByteLevel） |
| `_fix_v5_add_bos_eos_token`（`:317`） | v5 见到 `tokenizer.json` 就丢掉 `add_bos_token`；不少模型靠这个 flag 而不是 post-processor |
| `_fix_special_tokens_pattern` | 默认 `"cls_sep"` 会在没有 cls/sep 时往 ids 里插 `None` |
| `patch_tokenizer` | Kimi TikToken 缓存 `all_special_ids` |

`_fix_added_tokens_encoding`（`:570`）**不在**这条链上。它从 `processor.py` 调，修的是多模态 special token 在 v5 里被拆成 subword。不要把它算进 `get_tokenizer` 的容错。

vLLM 加载后几乎不做这套修复。`CachedHfTokenizer.from_pretrained` 只把部分 `ValueError` 翻译成「建议 `--trust-remote-code` / 升级 transformers」，再包 `get_cached_tokenizer`；顺带处理 sentence-transformer 的 `do_lower_case`（`hf.py:218-251`）。SGLang 的 `_auto_tokenizer_from_pretrained`（`:175`）也有同样的 `trust-remote-code` 提示，另外还有 MistralCommon 拒 HF kwargs 时的重试。

### 1.3 HF 自身（无本仓库行号）

`AutoTokenizer.from_pretrained` 读 checkpoint 的 `tokenizer_class` / `auto_map`，不认识就动态加载远程代码。v5 起，`model_type` 没有专用映射时统一落到 `TokenizersBackend`。这条能从两边注释钉死：

```51:62:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
# Class name used by transformers v5 when no tokenizer mapping exists for a model_type.
_TOKENIZERS_BACKEND = "TokenizersBackend"


def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):
    """Load tokenizer by the class declared in tokenizer_config.json.

    AutoTokenizer resolves to TokenizersBackend when the model's config
    model_type has no tokenizer class mapping (e.g. deepseek_vl_v2), even
    though tokenizer_config.json declares a standard class like
    LlamaTokenizerFast.  Returns None if it cannot improve on AutoTokenizer.
    """
```

```235:236:third_party/vllm/vllm/tokenizers/registry.py
    # Some models have an incorrect tokenizer_class on the hub.
    # For these model types, bypass AutoTokenizer and use TokenizersBackend directly.
```

也就是说：**HF 只按 checkpoint 元数据分发，不认识 tiktoken `.json` / GGUF / 裸 tekken。** 那部分正是 serving 框架在外面再套一层的原因。

---

## 2. 分歧点：同一个类名，相反的态度

`TokenizersBackend` 是 transformers v5 的通用 fast tokenizer。两边都在处理它，但 **一个把它当修正，一个把它当退化**。

| | vLLM | SGLang |
| --- | --- | --- |
| 时机 | 加载**前** | 加载**后** |
| 判据 | `model_type` 命中黑名单 `_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS`（`internlm2` / `step3_vl` / `step3p7` / `unlimited-ocr`，`registry.py:35-40`） | `type(tokenizer).__name__ == "TokenizersBackend"`（`tokenizer.py:545-548`），且 `tokenizer_backend != "fastokens"` |
| 对 TokenizersBackend 的态度 | Hub 上的 `tokenizer_class` **错了**，应改成通用 fast | AutoTokenizer **退化**到了通用类，应尽量找回声明的专用类 |
| 动作 | 直接把 `tokenizer_cls_` 换成 `TokenizersBackend`，加载后再 `get_cached_tokenizer` | `_resolve_tokenizers_backend`：先 `use_fast=False` 重载；若仍是该类，按 `tokenizer_config.json` 的 `tokenizer_class` 直载（可走 `auto_map` + `get_class_from_dynamic_module`） |
| 失败时 | 无这条旁路；`from_pretrained` 异常往外抛 | `use_fast=False` **抛错就中止**（不是退回第一次的结果）。声明类失败则留下当前实例并 warning |
| 代价 | 每个坏模型加一行黑名单；注释自认是临时 workaround，正解是改 transformers 或 Hub | 每次加载多一次类型嗅探，可能重载两次 |

```237:257:third_party/vllm/vllm/tokenizers/registry.py
    model_type = getattr(config, "model_type", None) if config else None
    if model_type in _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        logger.debug(
            "Overriding tokenizer_class to TokenizersBackend for model_type=%r",
            model_type,
        )
        tokenizer_cls_ = TokenizersBackend

    if config is not None and tokenizer_cls_ is CachedHfTokenizer:
        # AutoTokenizer otherwise reloads config.json internally. Reuse the
        # config that get_config just loaded successfully so a concurrent Hub
        # cache refresh cannot invalidate the file between the two reads.
        kwargs.setdefault("config", config)

    tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)
    if model_type in _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS:
        from vllm.tokenizers.hf import get_cached_tokenizer

        tokenizer = get_cached_tokenizer(tokenizer)
```

```545:551:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
            if (
                type(tokenizer).__name__ == _TOKENIZERS_BACKEND
                and tokenizer_backend != "fastokens"
            ):
                tokenizer = _resolve_tokenizers_backend(
                    tokenizer_name, *args, **common_kwargs
                )
```

fastokens 开着时 SGLang **故意不**走这条重解析：patched 的 `TokenizersBackend.from_pretrained` 已经是 fastokens shim，再按声明类加载会把 shim 丢掉。

---

## 3. 启动期总表

| 维度 | vLLM | SGLang | HF transformers |
| --- | --- | --- | --- |
| 入口 | `registry.py:186` `get_tokenizer` | `tokenizer.py:470` `get_tokenizer` | `AutoTokenizer.from_pretrained` |
| 分发形态 | 声明式注册表：`mode → (module, class)` | 命令式探测链：顺序 if/elif | 元数据驱动单一路径 |
| 归一化 | 独立纯函数 `resolve_tokenizer_args`（`:100`） | 无；判定与加载混在一个函数 | 无 |
| 类怎么来 | `resolve_obj_by_qualname`（`:78-85`） | 直接 import / `getattr(transformers, name)` / `get_class_from_dynamic_module`（`:88-105`） | `tokenizer_class` + `auto_map` |
| 非 HF 格式 | mode 表：`mistral`、`deepseek_v32/v4`、`kimi_audio` | 文件特征：`.json`→Tiktoken（`:481`）、GGUF（`:500`）、bare-tekken（`:521`，忽略 `tokenizer_backend`） | 不支持 |
| 加载失败 | 无回退；部分 `ValueError` 改写成 `--trust-remote-code` 提示（`hf.py:218-238`） | `trust-remote-code` 提示 + MistralCommon kwargs 重试；TokenizersBackend 见上节（不是无限回退） | 无 |
| 加载后修复 | 几乎没有（属性缓存、`do_lower_case`） | `_apply_post_load_fixes` | — |
| 缓存 | 两层 `lru_cache`：`:169`、`:267` | 无（上层持有实例） | 内部无（本仓看不到） |
| 扩展 | `TokenizerRegistry.register()`（`:64`） | 往判定链里插分支 | 改 Hub 元数据 |
| fastokens | 环境变量 `VLLM_USE_FASTOKENS`，`get_tokenizer` 开头打 patch（`:196-201`） | 参数 `tokenizer_backend="fastokens"`（`:476,486`）；失败明确叫你去掉该 flag（`:555-561`） | — |
| truncation_side | `runner_type`：generate/draft 截左，pooling 截右（`:133-139`） | 未在此层处理 | tokenizer 默认 |

**分发形态**：vLLM 数据驱动、可注册、可缓存、纯函数化归一；SGLang 顺序探测 + 失败后修复；HF 只认 checkpoint 元数据。

**容错位置**：vLLM 把知识前置成黑名单和 mode 表，出错倾向于报错让用户改参数；SGLang 把知识后置成嗅探和修复，倾向于尽力自愈。

**扩展成本**：vLLM 加一个新 tokenizer 家族 = `_VLLM_TOKENIZERS` 加一行 + 一个类，适合外部插件。SGLang = 往 `get_tokenizer` 判定链里插一段，对「格式靠文件特征识别」（`.json`、GGUF magic、tekken）更顺手。

---

## 4. 运行期：请求怎么进 encode

启动选好的是**实例**。请求来了还要决定：跳过、单条、还是真批量。

### 4.1 vLLM：renderer 四步，gather 单条 + 线程池

四个 Step 注释在 `renderers/base.py`：`:372` 渲 prompt、`:430` tokenize、`:650` extras、`:663` 转 `EngineInput`。细节见 02。

批量入口始终是「对每条 `asyncio.gather`」，没有把多条文本一次塞进 HF `batch_encode_plus`：

```641:648:third_party/vllm/vllm/renderers/base.py
    async def tokenize_prompts_async(
        self,
        prompts: Sequence[DictPrompt],
        params: TokenizeParams,
    ) -> list[TokPrompt]:
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )
```

真并行来自 `renderer_num_workers` 大小的 `ThreadPoolExecutor`（`base.py:85-98`）。单条内部已有 `prompt_token_ids` / `prompt_embeds` 就跳过 encode：

```557:564:third_party/vllm/vllm/renderers/base.py
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; "
                    "use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = await self._tokenize_prompt_async(prompt, params)
```

### 4.2 SGLang：默认顺序 await，批量是开关

单请求走 `_tokenize_one_request`（`:961`）：`input_embeds` / `input_ids` / 文本 三分支，文本再进 `_tokenize_texts`（`:889`）。

默认 **batch 并不是** `asyncio.gather` tokenize。`parallel_sample_num == 1` 且没开批量开关时，是顺序 `for` + `await _tokenize_one_request`（`:1837-1853`）。`asyncio.gather(*(_tokenize_one_request ...))` 只出现在 `parallel_sample_num != 1`（`:1865-1867`）。

真批量要显式 `enable_tokenizer_batch_encode`：`_batch_tokenize_and_process`（`:1467`）把多条 `text` 交给一次 `_tokenize_texts`。代价是多模态 / 预分词 `input_ids` / `input_embeds` 直接报错：

```1515:1527:third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )
```

另有可选 `enable_dynamic_batch_tokenizer`：把**并发的单条字符串**在 TokenizerManager 里攒成一批（03 §1.4）。那是跨请求 coalescing，不是 `enable_tokenizer_batch_encode` 那种「一个 HTTP batch 里一次 `tokenizer(list[str])`」。

### 4.3 HF（接口层）

`tokenizer(text)` / `tokenizer(list[str])` 分发到 `encode_plus` / `batch_encode_plus`。fast 版最终走 Rust `encode_batch`。serving 框架要不要调用这条批量 API，是上一小节的选择，不是 HF 替你做的。

### 4.4 运行期总表

| 维度 | vLLM | SGLang | HF |
| --- | --- | --- | --- |
| 批量入口 | `tokenize_prompts_async`（`base.py:641`） | `_handle_batch_request`（`tokenizer_manager.py:1814`） | `__call__` |
| 默认多条 | `asyncio.gather` 逐条 | **顺序** `await _tokenize_one_request`（`:1845-1853`） | 同步 |
| 真批量 encode | 无（始终单条 `__call__`） | 可选 `enable_tokenizer_batch_encode`（`:1541`、`_tokenize_texts` `:889`） | `batch_encode_plus` → Rust `encode_batch` |
| 并行来源 | `renderer_num_workers` 线程池（`base.py:85-98`） | 默认无；批量开关 = 一次 HF 调用；另可选动态攒批 | Rust 内部 |
| 已是 token | `prompt_token_ids` / `prompt_embeds` 跳过（`base.py:557`） | `input_ids` / `input_embeds` 跳过；批量路径直接拒绝这三种（`:1515-1527`） | — |
| 流程分段 | 四个 Step | `_tokenize_one_request` 单体 | — |

---

## 读完应能回答

1. vLLM 的 `slow` / `auto` / `mistral` 是在哪一层被收成表里的 key 的？`cohere` mode 换的是 tokenizer 还是 renderer？
2. SGLang 看到 `foo.json`、GGUF、裸 `tekken.json` 各走哪条？哪一条会忽略 `--tokenizer-backend`？
3. `TokenizersBackend` 在 vLLM 黑名单里是「要用的类」还是「要躲开的类」？SGLang 呢？fastokens 开着时 SGLang 为什么不重解析？
4. 两边加载失败时，哪些是硬报错、哪些会修完再用？`_fix_added_tokens_encoding` 算不算 `get_tokenizer` 的一步？
5. 一个 HTTP batch 里 8 条文本：vLLM 会不会一次 `batch_encode_plus`？SGLang 默认会不会 `asyncio.gather` tokenize？要真批量该开哪个开关、不能带什么输入？
6. 加一种新的非 HF tokenizer：vLLM 改表还是改 if？哪种更适合「看文件头识别格式」？
