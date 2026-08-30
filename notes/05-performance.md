# Tokenizer 性能手册

Tokenizer 在 serving 里是三段 **CPU** 工作，常常比 BPE 本身更慢：

1. **Chat template / 工具模板**（Jinja `apply_chat_template`）
2. **Encode**（text → token ids，进 GPU）
3. **Incremental detokenize**（每个新 token → 增量文本：UTF-8 边界、special token 空格、stop string、tool parser）

优化核心不是「换一个更快的 BPE」，而是：**别让 tokenizer 堵 event loop、别每步全量 decode、高并发时复制/多进程、能跳过的就跳过。**

本篇不写具体延迟数字。把脚本跑在你自己的机器上，填 [`experiments/README.md`](../experiments/README.md) 里的空表。

## 该测的指标（tokenizer 自己的）

| 指标 | 含义 | 本仓库怎么测 |
| --- | --- | --- |
| `encode_ms` | 一段文本（或已渲染 prompt）变成 token ids 的 CPU 时间 | [`experiments/bench_backends.py`](../experiments/bench_backends.py) |
| `chat_template_ms` | `apply_chat_template`（Jinja 渲染）的 CPU 时间 | [`experiments/bench_chat_template.py`](../experiments/bench_chat_template.py) |
| `detok_us_per_token` | 每个生成 token 换成文本的 CPU 时间 | [`experiments/bench_detok.py`](../experiments/bench_detok.py) |
| 并发下 tokenizer CPU / 卡顿 | 多线程共享一把 Rust tokenizer、还是池/多进程 | [`experiments/bench_concurrency.py`](../experiments/bench_concurrency.py) |

打印出来的 `mean` / `p50` / `p95` 都是 **进程内 `time.perf_counter()`**，不含 GPU kernel、不含网络。

### 不要和 GPU TTFT 混为一谈

端到端 serving 里还能看到：

- **TTFT**（time to first token）：从请求进站到第一个可展示 token。里面叠了排队、chat template、encode、prefill、以及把首 token detok 出去。
- **ITL**（inter-token latency）：相邻输出 token 的间隔，decode kernel + 调度 + 增量 detok + 流式发送。

tokenizer 只解释其中的 CPU 切片：

- 长 prompt、高 QPS 时 **TTFT 变差**，且 tokenizer 线程/进程先打满、GPU SM 还空着 → 更像 encode / template 堵在入口。
- 流式 **ITL 周期性尖峰** → 更像逐步全量 decode、stop 字符串扫描、或 `stream_interval=1` 时 Python↔Rust 往返过密。

用隔离脚本先得到 `encode_ms`、`chat_template_ms`、`detok_us_per_token`，再拿它们去和 TTFT/ITL **对照量级**，而不是用 TTFT 反推「BPE 慢了多少」。

## 「最好性能」清单（按收益排序）

与学习计划一致，实现落点对照 `third_party/`。

1. **永远用 fast tokenizer；slow 只做对照实验。**  
   vLLM 加载到 slow 会打 warning（`third_party/vllm/vllm/tokenizers/registry.py` 里 `not tokenizer.is_fast`）。SGLang `--tokenizer-mode slow` 同理。`bench_backends.py` 把 `hf-slow` 和 `hf-fast` 放在同一段中英混合文本上比。

2. **打开属性缓存 + tokenizer 副本池。**  
   HF 若干属性每次访问会重算，热路径不可接受：`get_cached_tokenizer` 缓存 `all_special_ids` / vocab / `is_fast` 等：

```107:118:third_party/vllm/vllm/tokenizers/hf.py
def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)
```

   Fast tokenizer 不能多线程同时安全借用同一块 Rust 状态。vLLM 用深拷贝池把 `encode` / `apply_chat_template` / `decode` 借出副本（池空则再 `deepcopy`）：

```48:57:third_party/vllm/vllm/tokenizers/hf.py
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

   副本数跟 Renderer 线程数走：`maybe_make_thread_pool(self.tokenizer, config.model_config.renderer_num_workers + 1)`（`third_party/vllm/vllm/renderers/hf.py`）。  
   SGLang 是进程隔离：长 prompt / 多模态把 `--tokenizer-worker-num` 从 1 加到 2–4，用 CPU 打满 encode（`third_party/sglang/python/sglang/srt/server_args.py` 默认 `tokenizer_worker_num=1`）。  
   对照实验：`bench_concurrency.py` 的 `shared-instance` / `deepcopy-pool` / `multiprocess-spawn`。

3. **阻塞的 encode / `apply_chat_template` 必须离开 asyncio 主循环。**  
   vLLM Renderer 把 tokenize 和 `safe_apply_chat_template` 丢进 `ThreadPoolExecutor`：

```82:98:third_party/vllm/vllm/renderers/base.py
        # Thread pool executor for blocking tokenizer operations.  The
        # multimodal processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)

        # Separate single-worker executor so tokenization never queues behind
        # MM preprocessing; must stay single-worker per #38418 (P0/P1 order).
        self._mm_executor: Executor = ThreadPoolExecutor(max_workers=1)

        # Offload tokenization to the thread pool. The sync
        # ``_tokenize_prompt`` already encapsulates the unified ``__call__``
        # path and char-offset extraction, so the async variant is just it
        # offloaded (mirrors ``_process_multimodal_async`` below).
        self._tokenize_prompt_async = make_async(
            self._tokenize_prompt, executor=self._executor
        )
```

   自己写网关时：不要在 `async def` 里直接 `tokenizer.encode` / `apply_chat_template`。

4. **流式用 incremental detokenize + `batch_decode`；能 `FINAL_ONLY` 就不要逐步 decode；`stream_interval>1` 降低 Python↔Rust 往返。**  
   HuggingFace `DecodeStream` 存在的原因：一次性 `decode` 依赖周围 token（空格、不完整 UTF-8），逐步全量 `decode(ids[:i])` 既慢又可能错。Python API：`tokenizers.decoders.DecodeStream.step`（`third_party/tokenizers/bindings/python/src/decoders.rs`）。  
   vLLM 在 `tokenizers>=0.22` 上走 `FastIncrementalDetokenizer`，用 prompt ids 预填 stream，再对每个新 id `stream.step`：

```23:26:third_party/vllm/vllm/v1/engine/detokenizer.py
# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")
```

```184:187:third_party/vllm/vllm/v1/engine/detokenizer.py
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
```

   非流式 / 只要最终文本：`RequestOutputKind.FINAL_ONLY` 在未结束时直接不组 output（`third_party/vllm/vllm/v1/engine/output_processor.py`）。`stream_interval>1` 时未完成请求每隔 N 个 token 才发一次。  
   SGLang 在独立 Detokenizer 进程里用 `DecodeStatus.surr_offset` / `read_offset` 做增量，并按 `(skip_special_tokens, spaces_between_special_tokens)` 分组 `batch_decode`（`third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py`）。  
   对照实验：`bench_detok.py` 的 `full_decode_per_token` vs `DecodeStream.step`。

5. **stop 尽量用 token id，少用每步字符串匹配。**  
   `FastIncrementalDetokenizer` 仍会在 `self.stop` 非空时扫 `output_text`（同文件 `BaseIncrementalDetokenizer.update`）。字符串 stop 越长、越密，detok 路径越重。能在采样侧用 eos / stop token id 就不要每步 `in text`。

6. **BPE 模型试 `fastokens`；先做正确性：同一模型 encode 必须逐 token 对齐。**  
   vLLM：`VLLM_USE_FASTOKENS=1`（或文档中的 tokenizer backend），`apply_fastokens_patch()` 在加载 HF fast 之前替换 Rust BPE，并重绑 `DecodeStream`（`third_party/vllm/vllm/tokenizers/fastokens.py`，`registry.py`）。要求 `fastokens>=0.2.0`。  
   SGLang：`--tokenizer-backend huggingface|fastokens`。  
   `bench_backends.py` 从同一份 `tokenizer.json` 构 fastokens，对齐失败会打 WARNING。对不齐就不要上 serving。

7. **网关预 tokenize + engine `skip_tokenizer_init`（内部高吞吐通路）；对外 Chat Completions 仍建议服务端统一模板。**  
   vLLM：`skip_tokenizer_init=True` 时期望输入已是 `prompt_token_ids`，输出也是 ids（`third_party/vllm/vllm/config/model.py`）；`cached_tokenizer_from_config` 直接返回 `None`。SGLang 同样有 `--skip-tokenizer-init`。  
   延迟最优，但丢掉服务端 chat template / special token 一致性。对外 API 不要把「各客户端自己 render 模板」当成默认。

8. **不要重复 `apply_chat_template`；工具 / reasoning parser 不用就关。**  
   `bench_chat_template.py` 对比 `tokenize=True` 一次走完 vs 先渲染字符串再 `encode`。多调用一次 Jinja 就是多一截 `chat_template_ms`。parser 挂在 detok/流式路径上，不用等于少一次 CPU 与字符串扫描。

## 怎么判断 tokenizer 是不是瓶颈

隔离指标（先跑 `experiments/`）：

- `chat_template_ms` 经常 **大于** `encode_ms`。若只优化 BPE、不看模板，收益会偏。
- `full_decode_per_token` 的 `detok_us_per_token` 随生成长度恶化，而 `DecodeStream.step` 应接近按 token 摊还。若 serving 行为像前者，流式 ITL 会被 detok 拖住。
- `shared-instance` 在多线程下变慢、报错或吞吐不再随 `--workers` 涨，而 `deepcopy-pool` / 多进程还能涨 → 入口被同一把 tokenizer 锁住。

端到端（有 GPU 时，见下一节；**本仓库脚本不跑这些**）：

- 提高 QPS、加长 prompt：TTFT 变差，但 GPU util / SM 占用仍低，同时 **tokenizer 进程或 API 进程 CPU 先打满**。
- SGLang：`ps`/`htop` 里 TokenizerManager / DetokenizerManager 相对 Scheduler 先满。`--tokenizer-worker-num` 加上去之后 TTFT 降、GPU 才开始满，更像 tokenizer 入口不够。
- vLLM：Renderer 线程池（`renderer_num_workers`）打满、engine 侧还在等 ids。
- 关掉流式（`FINAL_ONLY`）或增大 `stream_interval` 后 ITL 尖峰消失，更像 detok / 流式发送而不是 decode kernel。
- `skip_tokenizer_init` + 客户端直接传 `prompt_token_ids` 后 TTFT 明显下降（内部通路），而 Chat Completions（服务端模板）没有 → 瓶颈在 template+encode，不在 GPU。

以上都是 **对照实验设计**，不是保证会出现的数字。

历史陷阱：多线程共享同一个 HF Fast tokenizer 曾抛 `RuntimeError: Already borrowed`（Rust 侧不可重入借用）。较新的 `tokenizers` Python 绑定改用 `RwLock`（`third_party/tokenizers/bindings/python/src/tokenizer.rs`），可能改为阻塞而不是抛错，但吞吐仍上不去。`bench_concurrency.py --demo-borrow` 两种结果都会打印。

## 可选：本机有 GPU 时的 serving bench

不要用这些命令的 TTFT 代替上面的 `encode_ms`。用途只有一个：看 **tokenizer CPU 是否先于 GPU 打满**。

vLLM（文档示例在 `third_party/vllm/docs/cli/README.md`）：

```bash
vllm bench serve \
    --model <model> \
    --host 127.0.0.1 \
    --port 8000 \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 100
```

对照：同一模型开 `--skip-tokenizer-init` 且客户端传 ids（仅内部通路）、或把 `--tokenizer-mode slow` 当反例。看 API 进程 CPU vs GPU util，不要只看请求完成时间。

SGLang（`python -m sglang.bench_serving` 仍可用，实现已迁到 `sglang.benchmark.serving`）：

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --dataset-name random \
    --random-input 1024 \
    --random-output 128 \
    --num-prompts 100
```

对照：`--tokenizer-worker-num 1` vs `2`/`4`，以及 `--tokenizer-backend fastokens`（先跑 `bench_backends.py` 确认 id 对齐）。观察 TokenizerManager CPU 是否先满。

这些命令会拉模型、占 GPU、跑很久。**本学习仓库的 experiments 默认不要跑它们。**

## 该打开的文件

跟一条请求：`HTTP messages → template → token ids →（跳过 scheduler）→ 新 token → detokenize → 流式 chunk`。

| 主题 | 路径 |
| --- | --- |
| 属性缓存、深拷贝池 | `third_party/vllm/vllm/tokenizers/hf.py` |
| 加载、slow 警告、`VLLM_USE_FASTOKENS`、`skip_tokenizer_init` | `third_party/vllm/vllm/tokenizers/registry.py` |
| Renderer 线程池、`apply_chat_template` 异步化 | `third_party/vllm/vllm/renderers/base.py`、`third_party/vllm/vllm/renderers/hf.py` |
| `DecodeStream` 增量 detok | `third_party/vllm/vllm/v1/engine/detokenizer.py` |
| `FINAL_ONLY`、`stream_interval` | `third_party/vllm/vllm/v1/engine/output_processor.py` |
| fastokens patch | `third_party/vllm/vllm/tokenizers/fastokens.py` |
| SGLang tokenize 入口 | `third_party/sglang/python/sglang/srt/managers/tokenizer_manager.py` |
| SGLang 增量 detok、`batch_decode` | `third_party/sglang/python/sglang/srt/managers/detokenizer_manager.py` |
| `--tokenizer-worker-num` / `--tokenizer-backend` / `--skip-tokenizer-init` | `third_party/sglang/python/sglang/srt/server_args.py` |
| `DecodeStream` 为何必须存在 | `third_party/tokenizers/tokenizers/src/tokenizer/mod.rs`（`DecodeStream` 文档）、Python `bindings/python/src/decoders.rs` |

算法与调用链细节分别在 `notes/00`–`04`；本篇只回答「测什么、怎么排优先级、如何判断是不是 tokenizer 的锅」。
