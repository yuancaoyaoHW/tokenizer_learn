# 08 · vLLM V1 进程边界：`EngineCoreRequest` 如何跨越 ZMQ

对照 `third_party/vllm` @ `2c7d7dd64a2eaba0feedf42cab2f527486d7479c`。

[02-vllm.md](02-vllm.md) 在 `InputProcessor` 之后写「EngineCore GPU 跳过」。本篇把跳过的那一跳拆开：前端把已经 tokenize 好的 ids 送过 ZMQ，EngineCore 只吐新 token ids 回来。文本、chat template、detokenize **全程不进这条河**。

覆盖范围：`input_processor.py:281 process_inputs` 打包 `EngineCoreRequest` → ZMQ → EngineCore 进程 → ZMQ → `async_llm.py:666 _run_output_handler`。

单进程对照：`InprocClient`（`core_client.py:306`）是直接函数调用、零 ZMQ。想看清「ZMQ 到底加了什么」，对着这个类读最快。

---

## 该打开的文件

| 角色 | 路径 |
| --- | --- |
| 过河结构体 | `vllm/v1/engine/__init__.py` |
| 打包请求 | `vllm/v1/engine/input_processor.py` |
| 前端：先注册再发送 | `vllm/v1/engine/async_llm.py` |
| 前端 socket / 编解码 | `vllm/v1/engine/core_client.py` `AsyncMPClient` |
| EngineCore 主循环与 IO 线程 | `vllm/v1/engine/core.py` `EngineCoreProc` |
| msgpack + 大 tensor 零拷贝 | `vllm/v1/serial_utils.py` `MsgpackEncoder` |
| 无 ZMQ 对照 | `vllm/v1/engine/core_client.py` `InprocClient` |
| 文本只活在这边 | `vllm/v1/engine/output_processor.py` `RequestState.prompt` |
| stop string（前端判定） | `vllm/v1/engine/detokenizer.py` |

---

## 1. 拓扑：2 个进程，EngineCore 主线程不碰 socket

```
API server 进程（一条 asyncio 循环）          EngineCore 进程
─────────────────────────────────            ──────────────────────────────
_add_request                                 [输入 IO 线程] process_input_sockets
  async_llm.py:425                             core.py:1694  DEALER
  core_client.py:1149 add_request_async            │ msgspec.decode
         │                                         │ preprocess_add_request
    input_socket ROUTER(bind) ──── ZMQ ──→         ▼
         │                                      input_queue (queue.Queue)
process_outputs_socket (asyncio task)              │
  core_client.py:1041                           [主线程] run_busy_loop
         ▲                                        core.py:1411
         │  output_socket PULL ←──── ZMQ ────     │ step() 调度 / forward / sample
    outputs_queue (asyncio.Queue)                 ▼
         │                                      output_queue
_run_output_handler (asyncio task)             [输出 IO 线程] process_output_sockets
  async_llm.py:666                              core.py:1797  PUSH
```

EngineCore 主线程只跟两个 `queue.Queue` 打交道，一次 socket 都不碰。动机写在 IO 线程启动处：

```1111:1115:third_party/vllm/vllm/v1/engine/core.py
            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
```

即：ZMQ IO 与 msgpack 编解码放到独立线程，和 GPU forward 重叠。API 侧没有再开线程：发送、`process_outputs_socket`、`_run_output_handler` 都在同一条 asyncio 循环上（后两个是独立 task）。同步 `LLMEngine` 走 `MPClient` 时，接收端才是真正的后台线程（`core_client.py:835`）。

Socket 选型：

- 入边 `ROUTER(bind)` ↔ `DEALER`：按 identity **定向**投到指定 engine（DP 多 engine）。外部管 engine 时走 `client_addresses`（`:554-560`），本进程拉起 engine 时走 `get_engine_zmq_addresses`（`:589-595`）。
- 出边 `PUSH` → `PULL`：只需多对一**汇聚**。

```554:563:third_party/vllm/vllm/v1/engine/core_client.py
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )
```

```1710:1717:third_party/vllm/vllm/v1/engine/core.py
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
```

DEALER 起来后必须先主动 send 一份 ready payload，ROUTER 见过对端 identity 才能寻址：

```1738:1741:third_party/vllm/vllm/v1/engine/core.py
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
```

---

## 2. 过河的两种消息

定义都在 `vllm/v1/engine/__init__.py`。两个方向都是 `msgspec.Struct`，三个选项都是为跨进程省开销：

- `array_like=True` — 序列化成数组而非 map，省掉字段名
- `omit_defaults=True` — 默认值不上线
- `gc=False` — 不参与 GC 环检测

```107:122:third_party/vllm/vllm/v1/engine/__init__.py
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None
```

没有 `prompt: str`。打包点是 `InputProcessor.process_inputs` 末尾，只填 ids / embeds / sampling：

```418:434:third_party/vllm/vllm/v1/engine/input_processor.py
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_input.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
            session_id=session_id,
        )
```

`client_index` 默认 0，多 API server 时用来把输出送回同一个前端：

```130:132:third_party/vllm/vllm/v1/engine/__init__.py
    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0
```

```196:211:third_party/vllm/vllm/v1/engine/__init__.py
class EngineCoreOutput(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    pooling_output: torch.Tensor | None = None

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
```

**「EngineCore 只吐 token id」是类型级保证**——两个结构体都没有能装字符串 prompt / 增量生成文本的字段。`stop_reason` 可以是匹配到的 stop token id 或 stop string，不是生成文本流。prompt 只活在前端 `RequestState`：

```163:164:third_party/vllm/vllm/v1/engine/output_processor.py
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
```

类型标签本身就是字节串，可以单独成帧、接收侧不解 payload 就能分派：

```284:297:third_party/vllm/vllm/v1/engine/__init__.py
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"
```

本篇只跟 `ADD` / `ABORT`。`UTILITY` 是 RPC（`get_supported_tasks` 等），返回值走 future，不进普通 output queue。

---

## 3. 逐跳拆解

### 跳 1 — 本地先注册，再过河

```425:437:third_party/vllm/vllm/v1/engine/async_llm.py
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)
```

顺序不能反：`RequestState` + detokenizer 必须先建好。输出回来时 `process_outputs` 用 `request_states.get`；找不到就当「已经 abort」直接 `continue`，等于静默丢弃：

```640:642:third_party/vllm/vllm/v1/engine/output_processor.py
            if req_state is None:
                # Ignore output for already-aborted request.
                continue
```

### 跳 2 — 编码与发送

```1149:1152:third_party/vllm/vllm/v1/engine/core_client.py
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()
```

`client_index` 告诉 engine 结果回给哪个前端。`_ensure_output_queue_task` 懒启动接收 task。

```1108:1127:third_party/vllm/vllm/v1/engine/core_client.py
    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity
    ) -> Awaitable[Any]:
        self.ensure_alive()
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        return self.input_socket.send_multipart((engine,) + message, copy=False)
```

入边帧布局：

```
[engine_identity] [b"\x00"] [msgpack 主体] [tensor 裸内存 1] ...
   ROUTER 寻址用    类型标签   EngineCoreRequest    仅大张量单独成帧
```

- 类型标签独立成帧，本身就是 `b"\x00"` 这类字节，接收侧不解 payload 就能分派。
- `MsgpackEncoder.encode` 用 `aux_buffers` 收集额外 buffer：

```166:176:third_party/vllm/vllm/v1/serial_utils.py
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            if self.oob_tensor_consumer is not None:
                self.oob_tensor_consumer.new_message()
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
```

- `_encode_tensor`：`nbytes < VLLM_MSGPACK_ZERO_COPY_THRESHOLD`（默认 256B）且在 CPU 上才内联进 msgpack，否则把 backing memory 挂成独立帧 → 大 tensor（如 `prompt_embeds`）零拷贝：

```257:271:third_party/vllm/vllm/v1/serial_utils.py
    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        oob_consumer = self.oob_tensor_consumer
        # view the tensor as a contiguous 1D array of bytes
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        elif oob_consumer is not None and (data := oob_consumer(obj)) is not None:
            assert isinstance(data, dict)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
```

- `copy=False`：zmq 发完之前靠 memoryview 引用链把原始 tensor 保住。

### 跳 3 — 落地并顺手做预处理

```1751:1765:third_party/vllm/vllm/v1/engine/core.py
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    # NOTE(yongji): ignore READY message sent by DP coordinator
                    # that is used to notify newly started engines
                    if type_frame.buffer == b"READY":
                        assert input_socket == coord_socket
                        continue
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
```

DEALER 收到时 identity 帧已被 ROUTER 剥掉，所以第一帧直接是 `type_frame`。

```1766:1795:third_party/vllm/vllm/v1/engine/core.py
                        try:
                            request = self.preprocess_add_request(req)
                        except MultiModalCacheMissError as e:
                            # P0/P1 shadow drift -- return a retryable signal (P0
                            # drops the stale entry, client resends with data).
                            self._handle_mm_cache_miss(req, e)
                            continue
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    elif request_type == EngineCoreRequestType.UTILITY:
                        request = generic_decoder.decode(data_frames)
                        client_idx, call_id, method, args = request
                        if method == FT_UTILITY_METHOD:
                            self.ft_sentinel.handle_command(
                                client_idx, call_id, args[0]
                            )
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
```

`preprocess_add_request` 跑在 **IO 线程**，不在主循环。docstring 写明可以和 Model forward 并行。纯 CPU：MM receiver cache 查表 → `Request.from_engine_core_request`（含 prefix cache 的 block hash）→ 结构化输出的 `grammar_init`。两段 "Note on thread safety" 解释为何无竞争：

```988:1010:third_party/vllm/vllm/v1/engine/core.py
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave
```

预处理失败不会把坏请求丢进主循环：MM cache miss 走 `_handle_mm_cache_miss`（前端丢掉 shadow、客户端带数据重发）；其它异常走 `_handle_request_preproc_error`。

ABORT 进 **双队列**（`aborts_queue` 求即时响应 + `input_queue` 保顺序）。合法性依据：scheduler 里 abort 是幂等的。

### 跳 4 — 主循环只碰 queue

```1411:1420:third_party/vllm/vllm/v1/engine/core.py
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
            self._maybe_publish_request_counts()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
            self._maybe_publish_request_counts()
```

`_process_input_queue`（`:1437`）：没活干时在 `input_queue.get(block=...)` 上阻塞（默认 `process_input_queue_block=True`），不空转；有活时 `get_nowait` 把队列排空立刻返回，不耽误 GPU。空闲还会清 `aborts_queue`（abort 已经经由 `input_queue` 处理过）。Elastic EP 扩缩容会把 `process_input_queue_block` 打成 `False`，避免主循环卡在 `get` 上。

`_process_engine_step` 调 `self.step_fn()`（无 batch queue 时就是 `step`），再把 `dict[int, EngineCoreOutputs]` **按 item** `put_nowait` 进 `output_queue`。key 是 `client_index`：

```1468:1475:third_party/vllm/vllm/v1/engine/core.py
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
```

```597:609:third_party/vllm/vllm/v1/engine/core.py
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

本篇到 `execute_model` 为止，不跟进 CUDA graph / KV cache。

### 跳 5 — 回程编码（buffer 池 + MessageTracker）

PUSH socket 设 `linger=4000`，保证 `ENGINE_CORE_DEAD` 死讯发完再关；tensor 帧不进 `pending`，zmq 自己持有到原始 tensor 的引用链：

```1802:1818:third_party/vllm/vllm/v1/engine/core.py
        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Payload buffers that can't be reused yet because zmq may still be
        # sending them.
        # Buffers of the zero-copy tensor/ndarray frames don't need tracking
        # here: zmq itself holds a reference to each until it's done with it.
        pending = deque[tuple[zmq.MessageTracker, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
```

热路径：回收 zmq 用完的 buffer → `encode_into` 复用 `bytearray` → `copy=False` 发出后用 `MessageTracker` 盯着，`tracker.done` 才能回池。池上限 `len(sockets) + 1`。`sockets[client_index]` 闭合跳 2 设的 `request.client_index`。

```1849:1864:third_party/vllm/vllm/v1/engine/core.py
                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reclaimed = pending.pop()[1]
                    if len(reuse_buffers) < max_reuse_bufs:
                        reuse_buffers.append(reclaimed)

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = self._send_msg_tracking_payload(
                    sockets[client_index], buffers
                )
                if not tracker.done:
                    pending.appendleft((tracker, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)
```

### 跳 6 — 前端接收 task

```1025:1033:third_party/vllm/vllm/v1/engine/core_client.py
        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        _self_ref = weakref.ref(self)
```

```1041:1047:third_party/vllm/vllm/v1/engine/core_client.py
        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
```

```1086:1106:third_party/vllm/vllm/v1/engine/core_client.py
                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs
```

1. **独立 asyncio task**，与消费者只通过 `asyncio.Queue` 通信。用 `weakref.ref(self)` + 局部变量，防止循环引用让客户端无法被 GC。
2. **异常不抛而是入队**（含 `CancelledError` → `EngineDeadError`），由 `get_output_async` 在消费侧 re-raise。错误能顺着 `_run_output_handler` 走到 `propagate_error`，一次性打给所有在等的请求，而不是死在后台 task 里无人知晓。`validate_alive` 检 `ENGINE_CORE_DEAD` 哨兵。UTILITY 返回值走 future，不进普通 output queue。

### 跳 7 — 终点 `_run_output_handler`

防 GC：`engine_core` / `output_processor` / `renderer` 拷成局部变量，`logger_manager` 用可变 list 包一层，闭包不持有 `self`：

```666:685:third_party/vllm/vllm/v1/engine/async_llm.py
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        # We use a mutable list for logger_manager so that it can be updated
        # during elastic EP scaling (see scale_elastic_ep) without creating
        # a circular reference via self.
        self._logger_ref = [self.logger_manager]
        logger_ref = self._logger_ref
        renderer = self.renderer
        # P0 multi-modal sender ("shadow") cache; None for text-only models.
        mm_processor_cache = renderer.mm_processor_cache
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
```

```698:730:third_party/vllm/vllm/v1/engine/async_llm.py
                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # 2b) Recover from P0/P1 cache drift: the engine flags hashes
                        # it couldn't find (mm_cache_miss_hashes); drop them from the
                        # P0 shadow so the client's retry resends the data and
                        # repopulates P1. Hot-path no-op (field is None otherwise).
                        if mm_processor_cache is not None:
                            for eco in outputs_slice:
                                if eco.mm_cache_miss_hashes:
                                    for mm_hash in eco.mm_cache_miss_hashes:
                                        mm_processor_cache.invalidate(mm_hash)

                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )
```

- **`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`（默认 128）分块 + `await asyncio.sleep(0)`**：detokenize 是同步 CPU，一个 batch 上百条会卡死事件循环、拖慢所有 SSE。
- **`assert not processed_outputs.request_outputs`**：AsyncLLM 路径下结果全部 `queue.put` 进各请求的 `RequestOutputCollector`；返回列表那条路给同步 `LLMEngine`：

```704:709:third_party/vllm/vllm/v1/engine/output_processor.py
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)
```

- **`reqs_to_abort` 要反向再发一次 ABORT**：engine 只认 token 不认 stop string。字符串级停止在前端判定，判定出来后必须通知对面停下：

```129:132:third_party/vllm/vllm/v1/engine/detokenizer.py
        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
```

```721:724:third_party/vllm/vllm/v1/engine/output_processor.py
                    if not engine_core_output.finished:
                        # If req not finished in EngineCore, but Detokenizer
                        # detected stop string, abort needed in EngineCore.
                        reqs_to_abort.append(req_id)
```

- 跳 3 的 MM cache miss 闭环也在这里：engine 在 `EngineCoreOutput.mm_cache_miss_hashes` 上报找不到的 hash，handler 从 P0 shadow `invalidate`，客户端带数据重发。

---

## 4. 每跳的「为什么」

| 跳 | 手法 | 为了什么 |
| --- | --- | --- |
| 2 | 类型标签独立成帧 | 不解 payload 就能路由 |
| 2 | `copy=False` + `aux_buffers` | 大 tensor 零拷贝过河 |
| 3 | `preprocess_add_request` 放 IO 线程 | 请求初始化与 GPU forward 重叠 |
| 4 | 主循环只碰 queue | GIL 让给 IO 线程 |
| 5 | buffer 池 + MessageTracker | decode 阶段高频小包不反复分配 |
| 6 | 独立 task + 异常入队 | IO 与消费解耦，错误可传播 |
| 7 | 分块 + `sleep(0)` | detokenize 不阻塞事件循环 |

和 tokenizer 笔记的衔接：过河前后都是 ids。encode 在过河前（Renderer），detokenize 在过河后（OutputProcessor）。SGLang 的对照见 [04-compare.md](04-compare.md) —— 它把增量**文本**再 ZMQ 一次；vLLM 这一跳只传 ids。

---

## 5. 对照阅读 / 动手（可选）

不要改 `third_party/`。用阅读代替打补丁：

- `InprocClient.add_request` 直接 `preprocess_add_request` + `engine_core.add_request`，把跳 2–6 整段折叠成两次函数调用：

```327:329:third_party/vllm/vllm/v1/engine/core_client.py
    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)
```

- 同步 `MPClient._send_input` 同样带 identity 前缀（ROUTER 寻址），并不是「同步就没有 identity 帧」。两边入边布局一样。异步路径多出来的是：`client_index`、可指定/按负载选择 `engine` identity，以及接收端从后台线程换成 asyncio task。

```888:895:third_party/vllm/vllm/v1/engine/core_client.py
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        self.input_socket.send_multipart(msg, copy=False)
```

- 纯文本请求 `encoder.encode` 通常只有 1 个 buffer；带 `prompt_embeds` 且超过 256B 时 `aux_buffers` 变长。阈值是 `VLLM_MSGPACK_ZERO_COPY_THRESHOLD`。
- 高并发下看 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`：`num_outputs` 大于 chunk 时 `_run_output_handler` 会 `sleep(0)`。默认 128。

---

## 读完应能回答

1. `EngineCoreRequest` / `EngineCoreOutput` 里有没有字符串 prompt / 生成文本？文本存在哪两个对象上？
2. 为什么 `_add_request` 必须先 `output_processor.add_request` 再 `add_request_async`？
3. 入边 multipart 第一帧和第二帧各是什么？DEALER 侧为什么看不到 identity 帧？
4. `preprocess_add_request` 为什么不放在 `run_busy_loop` 里？ABORT 为什么进两个 queue？
5. `copy=False` 发出去的 bytearray 什么时候才能进 `reuse_buffers`？tensor 帧呢？
6. stop string 在哪个进程判定？判定之后还要不要再过一次河？
7. `InprocClient` 删掉了上面哪几跳？
