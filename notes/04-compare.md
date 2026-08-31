# 04 · vLLM vs SGLang：Tokenizer serving 对照

对照源码钉：

- `third_party/vllm` @ `2c7d7dd64a2eaba0feedf42cab2f527486d7479c`
- `third_party/sglang` @ `e635577431cbdfb8ce5fafb0fcd8a4ac074062c6`

调用链细节分别在 [02-vllm.md](02-vllm.md) / [03-sglang.md](03-sglang.md)。本篇只比 **隔离模型**：线程池 vs 多进程、同进程 `DecodeStream` vs 独立 detokenizer、chat template 落点、skip 通路、fastokens 开关，以及何种负载下哪种模型更合适。启动时怎么选出 tokenizer 类、请求怎么进 encode，见 [07-tokenizer-dispatch.md](07-tokenizer-dispatch.md)。vLLM 的 ids 怎么跨 ZMQ 进 EngineCore，见 [08-vllm-enginecore-zmq.md](08-vllm-enginecore-zmq.md)。

**不发明数字。** 下面的「谁赢」是机制推演（GIL、IPC 跳数、能否扩核），不是端到端 QPS 表。要测的指标见后续 `notes/05-performance.md`。

---

## 该打开的文件

| 主题 | vLLM | SGLang |
| --- | --- | --- |
| 进程拓扑 | `vllm/v1/engine/async_llm.py`、`vllm/v1/engine/core_client.py` | `python/sglang/srt/entrypoints/engine.py` |
| Encode 隔离 | `vllm/renderers/base.py`（`ThreadPoolExecutor`）、`vllm/tokenizers/hf.py`（deepcopy 池） | `python/sglang/srt/managers/tokenizer_manager.py`、`multi_tokenizer_mixin.py` |
| Detok 隔离 | `vllm/v1/engine/detokenizer.py`、`output_processor.py` | `python/sglang/srt/managers/detokenizer_manager.py` |
| Chat template | `vllm/renderers/hf.py`、`online_renderer.py` | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| Skip | `vllm/config/model.py` `skip_tokenizer_init`、`registry.py` | `server_args.py`、`tokenizer_manager.py`、`ipc_channels.py` |
| fastokens | `vllm/tokenizers/fastokens.py`、`envs.py` `VLLM_USE_FASTOKENS` | `server_args.py` `--tokenizer-backend`、`hf_transformers/tokenizer.py` |
| 扩容旋钮 | `--renderer-num-workers` | `--tokenizer-worker-num` / `--detokenizer-worker-num` |

---

## 0. 先把进程图画对

两边 GPU 都不在 tokenizer 线程里。差别是 **encode / detok 跟 HTTP 的距离**。

**vLLM（`vllm serve` / AsyncLLM）**

```
API 进程（asyncio）
  ├─ Renderer：Jinja + encode（ThreadPoolExecutor + tokenizer deepcopy 池）
  ├─ OutputProcessor：FastIncrementalDetokenizer / DecodeStream.step
  └─ HTTP SSE 直接读 RequestOutput.text
        │  ZMQ：EngineCoreRequest.prompt_token_ids  →
        │  ZMQ：EngineCoreOutput.new_token_ids    ←
EngineCore 子进程：只吃 ids，做 GPU
```

`AsyncLLM` 把 Renderer / InputProcessor / OutputProcessor 放在前端，EngineCore 走后台进程：

```135:156:third_party/vllm/vllm/v1/engine/async_llm.py
        self.renderer = renderer = renderer_from_config(self.vllm_config)
        self.input_processor = InputProcessor(self.vllm_config, renderer)
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            ...
        )
        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
```

```503:510:third_party/vllm/vllm/v1/engine/core_client.py
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs
```

不要把 EngineCore 子进程当成 tokenizer 隔离。vLLM 的 tokenizer / detokenizer **仍在 API 进程**。这条河上有哪些帧、主线程为什么不碰 socket，见 [08-vllm-enginecore-zmq.md](08-vllm-enginecore-zmq.md)。

**SGLang**

```
主进程：HTTP + TokenizerManager（encode；OpenAI 层先做 chat template）
        │  ZMQ TokenizedGenerateReqInput
Scheduler 子进程：GPU
        │  ZMQ BatchTokenIDOutput
DetokenizerManager 子进程：增量 decode
        │  ZMQ BatchStrOutput
主进程 TokenizerManager.handle_loop → SSE
```

```211:223:third_party/sglang/python/sglang/srt/entrypoints/engine.py
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): ...
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.
    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC ... via the ZMQ library.
```

默认 `tokenizer_worker_num=1` 时，TokenizerManager 和 HTTP **同进程**（跟 vLLM 的 Renderer 一样占 API 进程）。真正的「encode 多进程」要 `--tokenizer-worker-num > 1`。

---

## 1. 线程池 vs 多进程 TokenizerManager

### vLLM：两套「池」，别混

| 池 | 解决什么 | 扩容旋钮 |
| --- | --- | --- |
| `ThreadPoolExecutor(max_workers=renderer_num_workers)` | Jinja / encode / 部分预处理不堵 asyncio | `--renderer-num-workers`（默认 1） |
| `maybe_make_thread_pool(..., copies=workers+1)` | HF fast 的 Rust `RefCell` 不能多线程同时 borrow | 副本数跟着 workers 走；池空再 `deepcopy` |

```82:90:third_party/vllm/vllm/renderers/base.py
        # Thread pool executor for blocking tokenizer operations.  The
        # multimodal processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)
        # Separate single-worker executor so tokenization never queues behind
        # MM preprocessing; must stay single-worker per #38418 (P0/P1 order).
        self._mm_executor: Executor = ThreadPoolExecutor(max_workers=1)
```

```25:36:third_party/vllm/vllm/tokenizers/hf.py
def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """
    If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.
    ...
    - Only ``TokenizerLike``'s public interface is thread-safe.
      This doesn't include ``_tokenizer`` property ...
```

`HfRenderer` 把 `safe_apply_chat_template` 和 tokenize 都丢进 `_executor`，并按 `renderer_num_workers + 1` 预拷 tokenizer：

```922:929:third_party/vllm/vllm/renderers/hf.py
        self._apply_chat_template_async = make_async(
            safe_apply_chat_template, executor=self._executor
        )
        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )
```

文档写明：这个池只服务 **async renderer**（`vllm serve`）。离线 `LLM.generate` 走同步路径，`--renderer-num-workers` 无效。

机制含义：

- HF **fast** encode 主体在 Rust，会放 GIL → 线程池对 BPE 本身能并行。
- Jinja `apply_chat_template` 是 Python，GIL 下线程扩核收益有限。
- 线程安全靠 **深拷贝**，不是「同一个 Rust tokenizer 加锁」。`Already borrowed` 是 RefCell，不是锁粒度问题。
- Detok 用的 `tokenizer._tokenizer` **不进池**（见第 2 节）。

### SGLang：进程隔离 + 可选多 worker

默认一条 TokenizerManager 在主进程里同步/异步 tokenize，再 ZMQ 发给 Scheduler：

```14:14:third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py
"""TokenizerManager is a process that tokenizes the text."""
```

`--tokenizer-worker-num N>1` 时主进程只留 `MultiTokenizerRouter`，每个 HTTP worker 进程自带一份 TokenizerWorker（自己 `get_tokenizer`、自己的回程 IPC）：

```1193:1201:third_party/sglang/python/sglang/srt/entrypoints/engine.py
        if server_args.tokenizer_worker_num == 1:
            tokenizer_manager, template_manager = init_tokenizer_manager_func(...)
        else:
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
            template_manager = None
```

```424:429:third_party/sglang/python/sglang/srt/server_args.py
    tokenizer_worker_num: A[
        int, "The worker num of the tokenizer manager.", NS("serving")
    ] = 1
    detokenizer_worker_num: A[
        int, "The worker num of the detokenizer manager.", NS("serving")
    ] = 1
```

单进程里还有两个「少堵 event loop」的开关，**不是**多进程：

- `--enable-dynamic-batch-tokenizer`：把并发单条 encode 攒进内部单线程 `ThreadPoolExecutor`
- `--enable-tokenizer-batch-encode`：一个 HTTP batch 做一次 batch encode（不要跟图像 / 已有 `input_ids` 混用）

多 worker 时正向路径是 worker → Router → Scheduler，反向按 `http_worker_ipc` 拆回 originating worker（`multi_tokenizer_mixin.py` 的 `router_worker_obj` / `handle_loop`）。

### 对照

| | vLLM | SGLang |
| --- | --- | --- |
| 默认 | 1 个线程 + 若干 tokenizer 副本，与 HTTP 同进程 | 1 个 TokenizerManager，与 HTTP 同进程 |
| 扩 encode | 加线程 + 深拷贝 | 加 **进程**（整条 HTTP+Tokenizer） |
| 为什么复制 | Rust `RefCell` / `Already borrowed` | 进程间不能共享 Rust 对象；顺便绕开 GIL |
| IPC | encode 结果只作为 ids 进 EngineCore（一跳；帧布局见 [08](08-vllm-enginecore-zmq.md)） | encode 结果 ZMQ 进 Scheduler（一跳）；多 worker 再加 Router |
| 多模态预处理 | 独立 `_mm_executor`，**固定 1 线程** | `mm_processor` 活在 TokenizerManager；多 worker 就有多份 processor |

---

## 2. 同进程 DecodeStream vs 独立 DetokenizerManager

这是两边增量 detok 最大的结构差。

### vLLM：API 进程里逐步 `DecodeStream.step`

OutputProcessor 拿的是 **同一个** `renderer.tokenizer`。每个请求一个 detokenizer；fast 路径直接握底层 Rust 对象：

```50:66:third_party/vllm/vllm/v1/engine/detokenizer.py
        if tokenizer is None:
            return IncrementalDetokenizer()  # 只攒 token ids，不解码
        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            return FastIncrementalDetokenizer(tokenizer, request)
        return SlowIncrementalDetokenizer(tokenizer, request)
```

```168:186:third_party/vllm/vllm/v1/engine/detokenizer.py
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):
    def __init__(...):
        self.tokenizer: Tokenizer = tokenizer._tokenizer  # 底层 Rust，不走 Python pool
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
```

要点：

1. **真增量**：每步一个新 token id → `DecodeStream.step`，不把已生成前缀再 `decode` 一遍。
2. 用 **prompt ids 预热**，跨 prompt/生成边界的 UTF-8 / BPE 续写才正确。
3. 跑在 `AsyncLLM.output_handler` 里，和 HTTP **同进程、同一条 Python 循环**。EngineCore 只回 ids。
4. `stream_interval`（engine 默认 1，请求级只能抬高）控制多久组一个 `RequestOutput`：少 Python↔Rust 往返和 SSE，ITL 可能出现周期性尖峰。首 token 和最终 chunk 总是立刻发。

```153:157:third_party/vllm/vllm/config/scheduler.py
    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length.
    A smaller value (1) makes streaming smoother by sending each token immediately,
    while a larger value (e.g., 10) reduces host overhead ...
```

Chat Completions：`stream=true` → `DELTA`；`stream=false` → `FINAL_ONLY`。`FINAL_ONLY` 仍逐步 `update`（stop string 要文本），只是不把中间 chunk 交给客户端。

另有请求级 `SamplingParams.detokenize=False`：tokenizer 仍加载，这一次不解码（OpenAI API 不暴露；内部 token-in/token-out 会用）。

### SGLang：独立进程 + surrounding 窗口 + `batch_decode`

DetokenizerManager 自己再 `get_tokenizer` 一份，和 TokenizerManager **不共享** Rust 对象。Scheduler 先裁 surrounding 窗口（prompt 末尾最多 5 个 token + 已生成 ids），再把 `decode_ids` / `read_offsets` 推进 `BatchTokenIDOutput`：

```155:155:third_party/sglang/python/sglang/srt/managers/schedule_batch.py
INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5
```

```1008:1018:third_party/sglang/python/sglang/srt/managers/schedule_batch.py
        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        ...
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
```

增量公式是减法，不是 `DecodeStream`：

```
new_text = decode(read_ids)[len(decode(surr_ids)):]
```

即 surrounding 打败 tokenizer cleanup，再从更长前缀里剥掉。fast 路径走 `_grouped_batch_decode`（按 `(skip_special_tokens, spaces_between_special_tokens)` 分组后 `tokenizer.batch_decode`）。不完整 UTF-8（结尾 `�`）不推进 offset，用 `find_printable_text` 只发可打印前缀。

`--skip-tokenizer-init` 时 Scheduler **绕过** detokenizer，直接把 `BatchTokenIDOutput` 推回 tokenizer 侧：

```55:65:third_party/sglang/python/sglang/srt/managers/scheduler_components/ipc_channels.py
            if skip_tokenizer_init:
                # No decode work: send outputs straight to the tokenizer side
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.tokenizer_ipc_name, False
                )
            else:
                send_to_detokenizer_raw = get_zmq_socket(
                    context, zmq.PUSH, port_args.detokenizer_ipc_name, False
                )
```

注意：启动路径仍会拉起 detokenizer 进程（`_launch_detokenizer_subprocesses` 不看 skip 开关）；skip 时它只是收不到业务流量。`detokenizer_worker_num > 1` 时用 `crc32(http_worker_ipc)` 钉死 worker，保证同一请求始终落在同一份 `decode_status`。

SGLang 也有 `--stream-interval`（默认 1）和 `--incremental-streaming-output`（chunk 的 `"text"` 用 delta，避免中间步 `O(n²)` 拼整串）。

### 对照

| | vLLM | SGLang |
| --- | --- | --- |
| 跑在哪 | API 进程 `output_handler` | 独立 `sglang::detokenizer` 进程 |
| 算法 | HF `DecodeStream.step`（O(1)/token） | surrounding 窗口 + 两次 `decode` 相减；fast 时整批 `batch_decode` |
| 每步 IPC | EngineCore → API：新 token ids | Scheduler → Detok：ids；Detok → TM：增量字符串（两跳） |
| 批处理 | Python 循环逐请求 `step` | 按 flag 分组后一次 `batch_decode` |
| 和 HTTP 抢 CPU | 会（同进程 GIL / asyncio） | detok CPU 不抢 GPU Scheduler；回程仍进 TM 的 asyncio |
| 扩容 | `stream_interval`；无独立 detok worker | `--detokenizer-worker-num` |
| skip 时 | 空 `IncrementalDetokenizer`，`text==""` | 不加载 tokenizer，Scheduler 直推 ids |

`DecodeStream` 更省每 token 的 CPU；独立进程 + `batch_decode` 更不怕「很多路并发流式」把 API/GPU 调度线程拖死。没有万能赢家，见第 6 节。

---

## 3. Chat template 落在哪一层

两边都把「模板 / 工具 / 多模态 placeholder」和 BPE **切开**。差别是类名和默认 `tokenize=`。

### vLLM：Renderer，进 EngineCore 之前

`OnlineRenderer.preprocess_chat` 是 HTTP JSON → `EngineInput` 的枢纽。对 HF tokenizer，默认 **`tokenize=False`**：先渲染字符串，再单独 encode。只有 Mistral tokenizer 或 `enable_prompt_embeds` 才 `tokenize=True` 一次出 ids。

```395:403:third_party/vllm/vllm/renderers/online_renderer.py
                tokenize=(
                    is_mistral_tokenizer(renderer.tokenizer)
                    or self.model_config.enable_prompt_embeds
                ),
```

`safe_apply_chat_template` 在 `HfRenderer`：解析模板（请求 → Processor → Tokenizer → 预置 fallback），然后 `tokenizer.apply_chat_template`。这是请求形状，不是 vocab 操作。registry 写得很直白：换 Renderer 不必换 tokenizer——

```42:55:third_party/vllm/vllm/tokenizers/registry.py
    # ``cohere`` mode uses the standard cached HF tokenizer; only the
    # renderer (template stage) is replaced with a melody-based one.
    ...
    # Inkling uses the plain HF tokenizer for token operations; the "inkling"
    # mode exists to select the InklingRenderer ...
```

模板和 encode 都在 renderer 线程池里，不进 EngineCore。

### SGLang：OpenAI serving 层，TokenizerManager 之前

TokenizerManager **不跑** Jinja。原生 `/generate` 吃的已经是 `text` 或 `input_ids`。

OpenAI `/v1/chat/completions` 在 **同一主进程** 的 `OpenAIServingChat` 里先渲染再 encode，再构造 `GenerateReqInput(input_ids=...)`。当前实现故意把 `apply_chat_template(tokenize=True)` 拆开，避免模板里已有的 special tokens 再被加一遍 BOS：

```1375:1396:third_party/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
            # Split apply_chat_template(tokenize=True) into render + encode so we
            # can skip add_special_tokens=False on tokenizers that don't auto-add
            # specials ...
                rendered_prompt = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=False,
                    ...
                )
                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )
```

这条路径上 TokenizerManager 的 encode 会被跳过（已有 `input_ids`）。`TemplateManager` 只做模板检测 / parser 建议，不跑 BPE。

`--tokenizer-worker-num > 1` 时主进程 `template_manager = None`，模板跟每个 HTTP worker 上的 tokenizer 走。

### 对照

| | vLLM | SGLang |
| --- | --- | --- |
| 默认 Chat Completions | Renderer（API 进程，线程池） | `OpenAIServingChat`（与 TokenizerManager 同进程） |
| 原生 completions / generate | Renderer `render_cmpl` | `/generate` 无模板，直接 text/ids |
| 默认 `tokenize=` | HF：`False`（渲染字符串 + 再 encode） | OpenAI 层同样拆成 `False` + `encode` |
| 和 BPE 解耦 | Renderer vs `CachedHfTokenizer` | Serving 层 vs TokenizerManager.encode |
| skip 之后 | 不能传 messages；`get_tokenizer()` 直接报错 | OpenAI 层没有 tokenizer 可用；`/generate` 必须 `input_ids` |

对外 Chat Completions 两边都建议 **服务端统一模板**。网关自己 `apply_chat_template` 再走 skip 通路，一致性由你保证。

---

## 4. `skip_tokenizer_init` / `prompt_token_ids`

两条正交开关：

1. **引擎级 skip**：根本不加载 tokenizer / detokenizer。
2. **请求级已有 ids**：tokenizer 仍在，这一次跳过 template + encode，下游仍要 detok。

### vLLM

```287:290:third_party/vllm/vllm/config/model.py
    skip_tokenizer_init: bool = False
    """Skip initialization of tokenizer and detokenizer. Expects valid
    `prompt_token_ids` and `None` for prompt from the input. The generated
    output will contain token ids."""
```

```270:272:third_party/vllm/vllm/tokenizers/registry.py
def cached_tokenizer_from_config(...):
    if model_config.skip_tokenizer_init:
        return None
```

效果：

- `Renderer.tokenizer is None`；`get_tokenizer()` 抛 `Tokenizer not available when skip_tokenizer_init=True`
- 文本 prompt / chat template 不可用；测试写死了契约：`generate("abc")` 失败，`{"prompt_token_ids": [1,2,3]}` 成功且 `text == ""`
- `--tokens-only` 会强制打开这个开关
- Structured outputs 与 skip **互斥**（`sampling_params.py`：「requires a tokenizer」）

请求里已经有 ids 时，即使不 skip 也会跳过这一次 encode：

```516:528:third_party/vllm/vllm/renderers/base.py
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            ...
            prompt = self._tokenize_prompt(prompt, params)
```

disagg decode 侧还会从 `kv_transfer_params.prompt_token_ids` 复用 ids，跳过模板和 tokenize（`online_renderer.py` `_reused_prompt_token_ids`）。那是「跳过这一次」，下游 OutputProcessor 仍要 tokenizer 做 DecodeStream。

### SGLang

```430:434:third_party/sglang/python/sglang/srt/server_args.py
    skip_tokenizer_init: A[
        bool,
        "If set, skip init tokenizer and pass input_ids in generate request.",
        NS("serving"),
    ] = False
```

`GenerateReqInput` 三个入口：`text` / `input_ids` / `input_embeds`。skip 时走 `text` 会炸：

```982:990:third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
```

多模态例外：skip 仍创建 `mm_processor`，以便只送图、由 processor 补 ids：

```496:508:third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(...)
            if get_serving().skip_tokenizer_init:
                self.tokenizer = self.processor = None
```

字段名：vLLM 热路径是 `prompt_token_ids`；SGLang 是 `input_ids`。语义一样——引擎只吃 ids。

### 对照

| | vLLM skip | SGLang skip |
| --- | --- | --- |
| CLI | `--skip-tokenizer-init`；`--tokens-only` 会强制打开 | `--skip-tokenizer-init` |
| 输入 | `prompt_token_ids` | `input_ids`（或 embeds） |
| 输出 | `token_ids`，`text==""` | `BatchTokenIDOutput`，不填 `"text"` |
| Chat Completions messages | 不可用 | 同样不可用（没 tokenizer 渲模板） |
| 请求级已有 ids、不 skip | 跳过 encode，仍 detok | 已有 `input_ids` 则跳过 encode，仍 detok |
| 延迟 | 最优（无 Jinja、无 BPE、无 DecodeStream） | 最优（再少一跳 detok IPC） |
| 代价 | 网关必须对齐同一份模板 + tokenizer 版本 | 同左 |

内部高吞吐、网关已预 tokenize：走 skip。对外 Chat Completions 仍建议服务端渲染。

---

## 5. fastokens 怎么打开

换的是 HF fast tokenizer **内部 Rust BPE**，不是换隔离模型。正确性前提：同一模型 encode 结果必须与 HF fast **逐 token 对齐**。瓶颈若在 GPU，端到端几乎看不见。

### vLLM：环境变量，没有 `--tokenizer-backend`

```1:10:third_party/vllm/vllm/tokenizers/fastokens.py
"""When ``VLLM_USE_FASTOKENS=1`` is set, ``fastokens.patch_transformers()`` swaps
the inner Rust tokenizer of every HF fast tokenizer loaded afterwards with the
fastokens shim and rebinds ``tokenizers.decoders.DecodeStream`` so the
streaming detokenizer accepts the shim. The patch is process-global and
idempotent ...
```

```196:201:third_party/vllm/vllm/tokenizers/registry.py
    if envs.VLLM_USE_FASTOKENS:
        from .fastokens import apply_fastokens_patch
        apply_fastokens_patch()
```

```717:722:third_party/vllm/vllm/envs.py
    # If true, replace the Rust BPE backend that powers HF fast tokenizers
    # with the `fastokens` ... The `fastokens` Python package must be installed.
    "VLLM_USE_FASTOKENS": lambda: bool(int(os.getenv("VLLM_USE_FASTOKENS", "0"))),
```

- 在 `get_tokenizer` **加载之前**打补丁；要求 `fastokens >= 0.2.0`，缺包直接 `ImportError`
- 对任何最终走到 HF fast 的 mode 生效（`hf`、`deepseek_v32`、`deepseek_v4`…）
- `mistral` / `kimi_audio` 不走 HF fast，flag 无效
- `FastIncrementalDetokenizer` 按 **模块名** 取 `tokenizers.decoders.DecodeStream`，所以 shim 替换能生效
- **不是** `--tokenizer-backend`（本树不存在该 CLI）

```bash
VLLM_USE_FASTOKENS=1 vllm serve Qwen/Qwen3-8B
```

### SGLang：`--tokenizer-backend huggingface|fastokens`

```414:423:third_party/sglang/python/sglang/srt/server_args.py
    tokenizer_backend: A[
        str,
        Arg(
            help="Tokenizer backend. 'huggingface' uses the default HuggingFace "
            "tokenizers library, and 'fastokens' uses the fastokens library "
            "for faster tokenization. Requires the fastokens package to be installed.",
            choices=["huggingface", "fastokens"],
        ),
        ...
    ] = "huggingface"
```

```486:487:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
    if tokenizer_backend == "fastokens":
        _ensure_fastokens_patched()
```

失败 **不会** 静默回退 huggingface：

```555:561:third_party/sglang/python/sglang/srt/utils/hf_transformers/tokenizer.py
        if tokenizer_backend == "fastokens":
            raise RuntimeError(
                f"fastokens failed to load tokenizer for {tokenizer_name!r}. "
                ...
                f"Re-run without --tokenizer-backend=fastokens to use the default backend."
            ) from e
```

`*.json` tiktoken 文件走自己的 `TiktokenTokenizer`，不打 patch。TokenizerManager / DetokenizerManager /（非 skip 时的）Scheduler 都会 `get_tokenizer`，三进程后端必须一致。

```bash
python -m sglang.launch_server --model-path ... --tokenizer-backend fastokens
```

### 对照

| | vLLM | SGLang |
| --- | --- | --- |
| 开关 | `VLLM_USE_FASTOKENS=1` | `--tokenizer-backend fastokens` |
| 默认 | 关 | `huggingface` |
| 机制 | `fastokens.patch_transformers()` | 同左 |
| 失败 | 缺包即报错 | 加载失败明确报错，不回退 |
| 与 detok | 必须同时换 `DecodeStream` | 各进程各自加载同一 backend |
| 和隔离无关 | 线程池还是线程池 | 三进程还是三进程 |

---

## 6. 哪种隔离模型何时更合适

下面只谈 **tokenizer CPU 路径**，不要和 GPU TTFT 混为一谈。没有本仓实测数字。

先记住两边默认其实很像：encode 都和 HTTP 同进程，GPU 都在子进程。SGLang 多出来的是 **专用 detok 进程** 和 **可把 encode 扩成多进程**。

### 高 QPS、短 prompt

**倾向 vLLM 线程池。**

短 prompt 上 BPE 和 Jinja 都轻。SGLang 每条请求多两次 detok 相关 ZMQ（ids 出 Scheduler、字符串回 TM），msgpack 开销占比更高。vLLM 的 deepcopy 池修的是 `Already borrowed`，短请求并发 encode 用线程 + Rust 放 GIL 就够。

若短请求的瓶颈其实是 GPU 调度，tokenizer 隔离模型几乎看不见。

### 长 prompt

**倾向 SGLang 多 tokenizer worker；vLLM 把 `--renderer-num-workers` 加到 2–4 是同方向、更弱的一步。**

长 prompt 上三段 CPU 都会变重，但重的位置不同：

- **Jinja**：Python，GIL。线程池很难把多核吃满；`--tokenizer-worker-num=2–4` 才能把模板从 API 循环里拆到别的核。
- **BPE（fast）**：Rust，放 GIL。vLLM 线程池 + 副本对 encode 有效；SGLang 单 worker 时可用 `--enable-dynamic-batch-tokenizer` 把 encode 挪出 asyncio，多 worker 则直接多进程。
- **超长字符**：vLLM 有 `max_chars_per_token` 粗切，避免一条 prompt 把 tokenizer 打爆。

默认两边都是 1 个 encode worker。长 prompt 高 QPS 时先看 tokenizer / HTTP 进程 CPU 是否先打满，再加 worker——不要一上来加到核数。

### 多模态

**倾向 SGLang 多 tokenizer worker。**

vLLM 把 MM 预处理放到 **单独的 1 线程** executor，避免和 tokenize 排队，但 MM 本身不能靠 `--renderer-num-workers` 并行。pooling 模型若打开 mm processor cache，`--renderer-num-workers > 1` 会被拒绝（cache 非线程安全）：

```829:840:third_party/vllm/vllm/config/model.py
            if (
                self.renderer_num_workers > 1
                and self.multimodal_config.mm_processor_cache_gb > 0
                and self.runner_type == "pooling"
            ):
                raise ValueError(
                    "Cannot use --renderer-num-workers > 1 with the "
                    "multimodal processor cache enabled for pooling models. "
```

SGLang 的 `mm_processor` 活在 TokenizerManager；`tokenizer_worker_num > 1` 等于每条 HTTP+Tokenizer 进程各有一份图像预处理。skip 仍保留 mm_processor，所以「网关已 tokenize 文本、服务端仍要编图」这条路走得通。

图像解码 / processor 往往是 Python + numpy，比 BPE 更吃 GIL 和内存。进程隔离比线程池对口。

### 流式

**逐 token 平滑、路数不多：倾向 vLLM 同进程 DecodeStream。**

**高并发多路流式、且 detok / stop 字符串已经能在 `top` 里看见 CPU：倾向 SGLang 独立 detok 进程。**

机制：

- `DecodeStream.step` 是 O(1)/token，没有 surrounding 窗口的重复 `decode`，也没有「本步 ids → 另一进程 → 增量字符串」的往返。`stream_interval=1` 时 ITL 更接近 GPU 产出节奏。
- SGLang 每步（或每 `stream_interval`）把窗口 ids 打到 detok 进程，再 `batch_decode` 两遍取差。单路更贵，但整批请求可以一次 `batch_decode`，且 **GPU Scheduler 不等 decode**。
- vLLM 的 `output_handler` 和 HTTP 同进程：很多路 `stream=true` 时，Python 循环 + stop 字符串匹配会和 SSE / 下一轮 Jinja 抢同一颗 GIL。`FINAL_ONLY` 或加大 `stream_interval` 是同进程模型的减负手段，不是换隔离。
- SGLang `--incremental-streaming-output` 只影响 chunk 里填 delta 还是整串，不改变 detok 在哪个进程。

stop **字符串**匹配两边都在文本上做，比 token-id stop 贵。能用 id 就不要每步扫字符串——这与隔离无关。

### 一张决策表

| 负载 | 更合适的隔离 | 原因（机制，非数字） |
| --- | --- | --- |
| 短 prompt、高 QPS | vLLM 线程池 | 少 IPC；fast encode 放 GIL，线程够用 |
| 长 prompt、模板重 | SGLang `--tokenizer-worker-num` 2–4 | Jinja 持 GIL，只有进程能扩核 |
| 长 prompt、纯 BPE | 两边都能加 worker | fast tokenizer 线程/进程都可并行 |
| 多模态预处理 | SGLang 多 tokenizer worker | vLLM MM executor 固定 1 线程 |
| 流式、路数少、要平滑 ITL | vLLM `DecodeStream` | 无窗口重 decode、无 detok IPC |
| 流式、路数很多、detok CPU 高 | SGLang 独立 detok（必要时 `detokenizer_worker_num`） | GPU 调度与 HTTP 不跟 `batch_decode` 抢核 |
| 网关已 tokenize | 两边都 skip | 去掉整段 CPU；对外 Chat 仍建议服务端模板 |
| 内存很紧 | vLLM 默认 | 少一份（或 N 份）完整 tokenizer 进程 |

实际组合（计划里的收益顺序，两边通用）：永远 fast tokenizer；阻塞的 template/encode 离开 asyncio；流式用增量 decode；能 skip 的内部流量 skip；fastokens 只在对齐正确后再开。隔离模型是「CPU 打满之后怎么扩」，不是换 BPE 算法。

---

## 7. 总表

| | vLLM | SGLang |
| --- | --- | --- |
| GPU | EngineCore 子进程，只吃 ids | Scheduler 子进程，只吃 ids |
| Encode 默认 | API 进程线程池 | 主进程 TokenizerManager |
| Encode 扩容 | `--renderer-num-workers` + deepcopy 池 | `--tokenizer-worker-num`（整进程复制） |
| 线程安全 | deepcopy 池修 `Already borrowed` | 进程不共享 Rust 对象 |
| Detok | 同进程 `DecodeStream.step` | 独立进程 surrounding + `batch_decode` |
| Chat template | Renderer | OpenAI serving 层（`/generate` 无模板） |
| 已有 ids | `prompt_token_ids` | `input_ids` |
| 引擎 skip | `--skip-tokenizer-init` / `--tokens-only` | `--skip-tokenizer-init` |
| fastokens | `VLLM_USE_FASTOKENS=1` | `--tokenizer-backend fastokens` |
| 流式减负 | `stream_interval`、`FINAL_ONLY`、`detokenize=False` | `stream_interval`、`incremental_streaming_output` |
| MM 预处理 | 1 线程 `_mm_executor` | TokenizerManager / 每 worker 一份 |

差别是 **进程边界 vs 线程池副本**，不是 BPE 算法。fastokens 只换 Rust 后端，不改变这张表的隔离列。

---

## 读完应能回答

1. vLLM 的 EngineCore 子进程和 SGLang 的 TokenizerManager 子进程，各隔离的是哪一段 CPU？
2. vLLM 为什么要 **两套**池（`ThreadPoolExecutor` vs `maybe_make_thread_pool`）？SGLang 默认 `tokenizer_worker_num=1` 时，encode 还在不在 HTTP 进程里？
3. `DecodeStream.step` 和 `decode(read)[len(decode(surr)):]` 各付什么 CPU？多一次 ZMQ 换来了什么？
4. chat template 在 vLLM 的哪一个类、SGLang 的哪一个类？Tokenizer 模块负责模板吗？
5. `skip_tokenizer_init` 之后还能把 messages 丢给 `/v1/chat/completions` 吗？请求里已有 `prompt_token_ids` / `input_ids`、但 **没** skip，还会不会 detokenize？
6. 打开 fastokens：vLLM 设哪个环境变量？SGLang 加哪条 CLI？失败会不会静默回退？
7. 长 prompt、多模态、多路流式，各该拧哪颗隔离旋钮？为什么（不引用任何编造的 QPS）？
