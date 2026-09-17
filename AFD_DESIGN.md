# SGLang Attention–FFN 分离（AFD）架构设计

| 项 | 内容 |
|---|---|
| 范围 | 当前仓库正在运行的实现：decode farm over cuda_ipc，含 AfPool MxN 实验路径 |
| 状态 | 已实现并验证（P0–P9） |
| 代码入口 | `python/sglang/srt/afd/` |
| 时点 | `HEAD = 9b6135cdcb` |
| 配套文档 | 分层文字说明 `AFD_ARCHITECTURE.md`；实测量化数据 `progress.md` |

> **本文定位**：讲**架构设计**——职责边界、接口契约、状态机、设计不变量与决策记录。
> 量化结果只保留"会影响设计选择"的结论，完整的基准测试数据见 `progress.md`。

---

## 1. 背景与设计目标

### 1.1 问题

Decode 阶段，Attention 与 FFN 是两种性质相反的负载：

| 维度 | Attention（MLA） | FFN / MoE |
|---|---|---|
| 瓶颈资源 | **显存**：KV Cache 随并发线性膨胀 | **算力**：权重读取主导 |
| 访存特征 | 访存密集，对 batch 不敏感 | 计算密集，需要大 M 才划算 |
| 扩容诉求 | 加显存 / 加实例 | 加算力 |

把两者**绑定在同一张卡上**，它们的扩容诉求会互相绑架：KV 想吃显存，MoE 权重也想吃显存。

**AFD 的核心主张**：把 Attention 与 FFN 拆到**两个独立实例池**，中间只交换 activation，
使两侧可以**各自独立扩容**。

### 1.2 与 PD 分离的区别

这是设计上必须首先澄清的一点，否则接口会串味：

| | 搬运的内容 | 语义 |
|---|---|---|
| **PD 分离**（Prefill/Decode） | **KV Cache** | 阶段间状态交接 |
| **AFD**（Attention/FFN） | **Activation**（hidden + routing） | 层内计算接力 |

代码中用独立的 `AfdMode` 枚举表达，**不与 `DisaggregationMode` 混用**（`mode.py`）。

### 1.3 设计目标 / 非目标

**目标**

1. 传输 API 与底层链路解耦（fake / cuda_ipc / stepmesh 可替换）。
2. 明确且稳定的 Attn ↔ FFN tensor 契约，启动时预注册缓冲。
3. 在 `DeepseekV2DecoderLayer` 的 `prepare_mlp` 之后插入切分点。
4. FFN 侧服务循环：消费 A2F、计算、回应 F2A。
5. 与 CUDA Graph、PD 组合、权重裁剪、路由方案 A/B、FP8 A2F 共存。

**非目标**

- 不替代 FFN 域内部的 DeepEP —— DeepEP 仍然**留在 FFN 侧本地**。
- 不做跨 decode step 的 full CUDA Graph 覆盖。
- 不含多机集群调度器（仅有 bring-up 示例脚本）。
- 生产级 straggler 策略不在范围内；AfPool 只提供最小可用的 credit + least-inflight 路由。

---

## 2. 总体架构

### 2.1 角色切分：按角色，不按层区间

这是 AFD 最重要的一个设计决策。模型不是"前 13 层在 A、后 14 层在 B"，
而是**两侧各自持有全部 27 层的不同部分**。

```
   ══ Attn Worker ═══ GPU A ═══ SGLANG_AFD_MODE=attn ═══════════════════

      · embedding / 输入
      · MLA Attention                    × 27 层
      ·   └── KV Cache ── 全系统唯一持有者
      · MoE gate + topk                  × 27 层   ← Scheme A
      · final norm / lm_head / 采样
      · 对外提供 OpenAI 兼容 API ── 唯一服务端

                    │                              ▲
                    │  A2F hop                     │  F2A hop
                    │  hidden + topk_ids/weights   │  mlp_out
                    │  + layer_id                  │
                    ▼                              │

   ══ FFN Worker ════ GPU B ═══ SGLANG_AFD_MODE=ffn ════════════════════

      · MoE experts + shared experts     × 27 层
      · 无 attention、无 KV Cache、无 embedding
      · 后台 poll 线程驱动：get_batch → compute → respond
      · 不接客户端请求
```

**为什么按角色而不是按层区间切？**

| 方案 | 问题 |
|---|---|
| 按层区间切（A 跑前 K 层，B 跑后 K 层） | B 侧仍需 KV Cache → 显存问题没解决；且 B 侧需要 attention 模块，权重裁剪收益消失 |
| **按角色切（当前方案）** | FFN 侧**完全没有 KV Cache 和 attention 模块**，显存收益最大化；两侧的模块集合互不重叠 |

**代价**：每层都要过一次网络 —— 一次 decode step 要走 27 个 hop。这个代价是本设计的核心约束，
详见 §5。

### 2.2 进程与通信拓扑

```
        客户端
          │  HTTP（OpenAI 兼容）
          ▼

   进程 1 · launch_server · SGLANG_AFD_MODE=attn · CUDA_VISIBLE_DEVICES=A
          │
          │◄──── Unix socket：SGLANG_AFD_IPC_ENDPOINT ────►
          │      交换 CUDA IPC handle 与 eventfd
          │
   进程 2 · launch_server · SGLANG_AFD_MODE=ffn  · CUDA_VISIBLE_DEVICES=B

        [ GPU A ]  ◄──── cuda_ipc 数据面（NVLink / P2P）────►  [ GPU B ]
```

| 事实 | 说明 |
|---|---|
| 真跑起来是**两个 `launch_server`** | 各自独立进程、独立 GPU 可见性 |
| FFN 是**被动服务端** | 由后台 poll 线程驱动，永不接请求 |
| 两侧共享同一份权重文件，但只加载各自需要的部分 | `SGLANG_AFD_RELEASE_UNUSED_PARAMS=1` 可把用不到的权重挪到 CPU |
| 同机走 `cuda_ipc` | 跨机可切 `stepmesh`（RDMA） |

### 2.3 单层职责边界

切分点定在 `prepare_mlp` 之后。下图标出每段代码归属哪一侧：

```
   ── Attn 侧本地执行 ──────────────────────────────────────────────
      prepare_attn
          │
      self_attn (MLA)
          │  └── 写 KV Cache
      prepare_mlp   (post-attention norm)
          │
      MoE gate + topk            ← Scheme A 默认在这里
          │
          ├─── A2F ────────────────────────────────────────────────►
          │    hidden + topk_ids / topk_weights + layer_id
          │
   ── FFN 侧远端执行 ──────────────────────────────────────────────
          │
          │                            experts + shared_experts
          │                                 │   (fused MoE)
          │◄── F2A ─────────────────────────┘
          │    mlp_out
          │
   ── 回到 Attn ───────────────────────────────────────────────────
      scatter 回 residual
          │
      postprocess_layer
```

**关键边界约定**：

| 约定 | 内容 |
|---|---|
| **residual 只在 Attn** | residual stream 从不上 A2F（仅 layer-merge 模式例外） |
| **KV Cache 只在 Attn** | FFN 侧无任何 attention 状态 |
| **routing 归属可变** | Scheme A 在 Attn 算 topk；Scheme B 在 FFN 算 |
| **dense MLP** | `SGLANG_AFD_REMOTE_MOE_ONLY=1` 时 dense MLP 可留在 Attn（不走 RPC） |

### 2.4 模块裁剪（module stubs）

`module_stubs.py` 让两侧只构建自己需要的模块，直接转化为显存收益：

| 角色 | 构建 | stub 掉（不构建） |
|---|---|---|
| **Attn**（Scheme A） | `self_attn` (MLA) + `gate` + `topk` | MoE `experts` / `shared_experts` / dense MLP body |
| **Attn**（Scheme B） | `self_attn` (MLA) | 整个 `mlp`（连 gate/topk 都在 FFN） |
| **FFN** | MoE `experts` / `shared_experts`（+ Scheme B 的 gate/topk） | `self_attn` —— **省下的正是 KV 显存** |

被 stub 的模块替换为 `AfdMissingModule`，一旦在热路径被调用**立即抛错**——
这是一个故意设计的"快速失败"机制，用来暴露接线错误，而不是静默降级。

> FFN worker 因此**没有任何 KV Cache**，`SGLANG_AFD_MAX_NUM_TOKEN` 也无需按 CG bucket 放大。

---

## 3. 数据面设计

### 3.1 数据契约总览

数据面由三部分组成，各自职责清晰：

```
   ① 语义契约 ── protocol.py
      定义 payload 的字段、形状、dtype、槽位编号

   ② 存储契约 ── buffers.py
      AfdBufferPool：启动时一次性分配，热路径只 copy、不 alloc

   ③ 传输契约 ── cuda_ipc_transport.py / transport.py
      如何把 ① 的数据搬过 ② 的缓冲、如何通知对端

   ────────────────────────────────────────────────────────────────
   三层解耦：换 transport 不需要改语义契约与缓冲布局
```

三层解耦的好处：换 transport（cuda_ipc → stepmesh）不需要改语义契约和缓冲布局。

### 3.2 32-bit key 位域

沿用 StepMesh 兼容的 key 打包方式（`protocol.py`）：

```
   bit 31 ─ 24   方向 direction
                 0 = A2F (push)  /  1 = F2A (pull)

   bit 23 ─ 16   worker_rank
                 Attn worker 序号（为多 Attn 预留，单 A 下也正确寻址）

   bit 15 ─  8   microbatch
                 mb_id；同一层可有多个 mb 并发在飞，互不冲突

   bit  7 ─  0   private_key
                 tensor 槽位号（最多 256 个）
```

设计要点：

- `private_key` 只占 8 bit → **最多 256 个张量槽位**，足够表达 A2F 的全字段。
- `microbatch` 独立成段 → 同一层可以有多个 mb 并发在飞，互不冲突。
- `worker_rank` 独立成段 → 为多 Attn worker（AfPool MxN）预留，**单 A 场景下也能正确寻址**。
- 方向位放在最高位 → 收发两侧可用同一个 key 空间做匹配，不必维护两套表。

### 3.3 A2F / F2A 载荷定义

**A2F（Attn → FFN）**

| 槽位 | 名称 | 形状 | dtype | 出现条件 |
|--:|---|---|---|---|
| 0 | `hidden` | `[T, H]` | bf16 / fp16 / **fp8** | 总是 |
| 1 | `num_tokens` | `[1]` | int32 | 总是（真实长度，T 可能被 pad） |
| 2 | `layer_id` | `[1]` | int32 | 总是（layer-merge 时高 8 位放 `merge_k`） |
| 3 | `topk_ids` | `[T, K]` | int32 | **Scheme A MoE** |
| 4 | `topk_weights` | `[T, K]` | fp32 | **Scheme A MoE** |
| 5 | `hidden_scale` | `[1]` | fp32 | wire dtype = fp8 |
| 6 | `residual` | `[T, H]` | compute | 仅 layer-merge |
| 7 | `positions` | `[T]` | int64 | 仅 layer-merge |

**F2A（FFN → Attn）**

| 槽位 | 名称 | 形状 | dtype | 出现条件 |
|--:|---|---|---|---|
| 0 | `mlp_out` | `[T, H]` | compute dtype | 总是 |
| 1 | `residual` | `[T, H]` | compute dtype | 仅 layer-merge |

**设计要点**

- `num_tokens` 与 pad 分离：`T` 按 decode CUDA Graph bucket 对齐，`num_tokens` 记真实长度。
  这让 buffer 形状恒定，**避免热路径动态分配**。
- `layer_id` 与 `merge_k` 打包进同一个 int32：layer-merge 场景下不增加字段。
- Scheme A/B 的差异只体现在**槽位 3/4 是否存在**，传输层无需分支。

### 3.4 缓冲池设计

`AfdBufferPool` 在启动时一次性分配：

```
   NUM_MB × ( max_num_token × hidden_size )      ← A2F hidden  (wire dtype)
 + NUM_MB × ( max_num_token × hidden_size )      ← F2A mlp_out (compute dtype)
 + 每槽位：num_tokens / layer_id / topk_ids / topk_weights / hidden_scale
 + layer-merge 时额外：residual / positions
```

**设计原则：热路径零分配。**

`fill_a2f()` 只做 `copy_`，绝不 `torch.empty`。若 `hidden` 超过注册容量则**立即抛错**，
并在错误信息里提示抬高 `SGLANG_AFD_MAX_NUM_TOKEN`。

> `SGLANG_AFD_MAX_NUM_TOKEN` 是**唯一的 OOM 开关** —— 刻意做成单一旋钮，
> 避免"多发一个显存参数"带来的配置复杂度。

### 3.5 传输通道设计

```
  启动阶段（只做一次）
  ────────────────────
     FFN  : _ffn_export_and_accept()        导出 CUDA IPC handle
              │
              ▼  Unix socket 交换
     Attn : _attn_connect_and_import()      导入并映射同一块显存
              │
              ▼
            _HostMailbox 建立（POSIX shared memory）

  每个 hop（零拷贝）
  ────────────────
     Attn : fill_a2f()         写入共享 slot
              │
            set_posted() + _signal_wake()    eventfd / 可选 GPU doorbell
              │
              ▼
     FFN  : get_batch()        扫描 NUM_MB 个槽位，取 posted > ffn_seen
              │
            (按 layer_id 分组 → fused MoE)
              │
            respond()         F2A copy + set_done()
              │
              ▼
     Attn : wait()            读 done
```

**核心机制**

| 机制 | 作用 | 设计理由 |
|---|---|---|
| **预注册 + IPC 映射** | 张量内存在启动时共享 | 在零拷贝路径下每 hop **没有 tensor 拷贝** |
| **`NUM_MB` 个 mailbox 槽位** | 提供多个**独立在飞**槽位 | 单槽位会强制串行；多槽位是重叠的前提 |
| **`_HostMailbox`** | 承载 `posted` / `done` / `meta(num_tokens, layer_id)` | host 侧直接读 meta，省掉 GPU `.item()` 同步 |
| **eventfd 唤醒** | `_signal_wake` / `_park_until_wake` | 避免 `sleep(0)` 空转烧 CPU |
| **GPU doorbell（可选）** | `cuStreamWriteValue64` / `cuStreamWaitValue64` | 让 GPU 自己在流上等，适合 in-graph 场景 |

**`NUM_MB` 会随 `MAX_INFLIGHT` 自动抬升**（`farm/env.py`），保证"调度器认为可以同时在飞的 hop 数"
与"传输层实际能承载的槽位数"始终一致 —— 这是避免"调度器以为发出去了、其实在排队"的关键一致性约束。

### 3.6 三种 transport 的定位

| 实现 | 用途 | 特点 |
|---|---|---|
| `FakeAfdTransport` | CI / 单进程 smoke | 同进程队列，可挂本地 FFN 回调 |
| `StepMeshAfdTransport` | 跨机 RDMA | 基于 `fserver_lib` |
| `CudaIpcAfdTransport` | 同机 | CUDA IPC，NVLink / P2P 可用时走 P2P |

三者实现同一套语义契约，因此**上层 farm 逻辑与 transport 无关**。

---

## 4. 控制面设计

### 4.1 一次 decode step 的控制流

入口在 `deepseek_v2.py`：原本的 lockstep 循环

```python
for layer in layers:
    layer(...)
```

被替换成对模型层区间调用 `run_farm_layers(...)`：

```
model.forward()
│
└─ run_farm_layers()
   │
   ├─ scheduler.begin_batch(range(n_seq))          所有 seq 一起入 layer 0
   │
   ├─ while finished < n_seq:
   │  │
   │  ├─ _issue_round()
   │  │  ├─ _start_next_context()                  按需注入下一个 context
   │  │  ├─ _drain(block=False)                    先无阻塞收一遍
   │  │  ├─ while 有空闲槽位:
   │  │  │  ├─ scheduler.reserve(b_step, ...)      选一个 window
   │  │  │  ├─ slice_decode_window(...)            切出该 window 的行
   │  │  │  ├─ run_layer_forward_pre_ffn(...)      本地 MLA + gate + topk
   │  │  │  ├─ try_issue_hop(...)                  A2F 发给 FFN
   │  │  │  └─ scheduler.commit(ticket)            reserved → running
   │  │  └─ flush_hop_queues()                     批量刷出
   │  │
   │  ├─ _drain()                                  completion-first 收割
   │  └─ _block_one()                              无可发射时等一个 hop 再重扫
   │
   └─ 抽干所有 pending → scheduler.finish_batch()
   │
└─ self.norm() → lm_head → logits → 采样
```

**两个关键设计性质**

| 性质 | 含义 | 设计理由 |
|---|---|---|
| **completion-first 收割** | `_drain` 收割**最先完成**的 hop，而不是 `pending[0]` | 单个慢 hop 不会堵住整个波前（head-of-line blocking） |
| **credit 门控，不是定时器** | 只有下游 credit 允许时才取出一组发送；发不出去就 `rollback()` 还给调度器 | 用背压而非超时来适配下游速度，避免死等与猜测超时 |

流动的 window 形态：

```
batch（全部 seq）
   → [layer L]  Attn（本地 MLA）+ MoE gate/topk（本地）
   → A2F hop → FFN（远端 experts）→ F2A
   → [layer L+1] ...
```

### 4.2 调度器状态机

`FarmContinuousScheduler`（`farm/scheduler.py`）+ `LayerReadyQueues`（`farm/token_queue.py`）。
状态是**进程内持久**的；`PERSISTENT=1` 时可跨 decode step 持久。

```
                  reserve()                    commit()                 complete()
   waiting ──────────────────► reserved ──────────────────► running ────────────────► layer+1 的 waiting
      ▲                           │                            │
      │                           │                            │  已是最后一层
      │        rollback()         │                            ▼
      └───────────────────────────┘                         finished
         （未拿到 A2F credit）                                 │
                                                              ▼
                                                            [*]
```

| 状态 | 含义 | 占用 credit |
|---|---|---|
| `waiting`（`ready[]`） | 积压，尚未占用 | 否 |
| `reserved` | 已占槽位，尚未上线 | 是（占 in-flight 名额） |
| `running` | F2A 在飞 | 是 |
| `finished` | 已过最后一层，转为等待采样 | — |

**关键设计点：`commit()` 推迟到"确认拿到 A2F credit"之后。**
`reserve()` 只做逻辑占位，只有当发送队列真的把 hop 推出去时，才 `mark_running`。
这样"调度器认为在飞的 hop 数"与"传输层真实的在飞数"不会漂移。
发不出去的 reservation 通过 `rollback()` 原样还给队列。

### 4.3 三类并发约束

设计上刻意把它们**拆成互相独立的三个旋钮**：

| 约束 | 控制什么 | 默认 | 典型用途 |
|---|---|---|---|
| `MAX_INFLIGHT` | 全局在飞 hop 数 | 4 | 总的并发上限 |
| `MAX_INFLIGHT_PER_LAYER` | **单层**发出的 hop 上限 | `0 → min(max_inf, 4)` | 防止"第一个就绪的层吃光所有 MB 槽位"而退回 lockstep |
| `GLOBAL_TOKEN_BUDGET` | 持有 credit 的 token 总数 | `max_inf × b_step` | token 维度的背压 |

外加：

- **aging**（`MAX_AGE_STEPS`）：防饿死，避免某层长期不被选中。
- **层选择策略** `sched ∈ {max, oldest, deepest}`：决定从哪个就绪层取活。
- **`B_WIN_K` / `sticky_layer`**：换层前在同一层停留的 window 数，用于摊薄换层开销。

> `MAX_INFLIGHT_PER_LAYER` 的存在理由值得单列：若没有它，第一个就绪的层会占满所有槽位，
> 后面的层永远没有机会进入在飞状态 —— 那就退化成了 lockstep。它是**保证重叠可能发生**的约束。

### 4.4 多 context 波前

`plan_context_ranges()` 把一个 decode batch 切成若干**连续区间**，每个区间是一个独立 context：

```
   一个 decode batch（16 seq）
   ┌────┬────┬────┬────┐
   │ c0 │ c1 │ c2 │ c3 │     NUM_CONTEXTS = 4
   └────┴────┴────┴────┘
     │    │    │    │
     │    │    │    └──► 深度 3 × STAGGER_LAYERS 后注入
     │    │    └───────► 深度 2 × STAGGER_LAYERS 后注入
     │    └────────────► 深度 1 × STAGGER_LAYERS 后注入
     └─────────────────► 立即进入 layer 0
```

- 每个 context 是**一条独立的依赖链**，可以处在不同层。
- `context_stage_ready()` 用 `deepest_active_layer` 作为波前探针，保证错峰。
- `CONTEXTS_PER_STAGE` 允许若干 context **同批进入**，从而让它们的同层工作能被合并成一次 A2F。

**同一个 context 内部仍然严格按层有序**；跨 context 才允许乱序 —— 这是正确性的边界。

---

## 5. 关键设计不变量

这一节是本文的核心：**三条不变量刻画了各优化方向的收益边界与成立条件**。

### 5.1 不变量 I1：波前形态（单 batch 时）

在三行代码的作用下，同一个 batch 的所有 token 永远共享一个层：

```
   begin_batch(range(n_seq))  →  queues.enqueue(0, idxs)          全部一起进 layer 0
   pick()                     →  take_contiguous_run(ready[li])   只从一个层取
   complete()                 →  queues.enqueue(layer+1, 同一个 window)   原样推到下一层
```

于是：

```
   {A,B,C,D} @ L0 ──hop──► {A,B,C,D} @ L1 ──hop──► {A,B,C,D} @ L2 ──► ... ──► @ L26
```

**可观测后果**（在所有 farm 配置下都成立）：

| 指标 | 值 |
|---|---|
| `layers_peak` | **1 / 27** |
| `hops/fwd` | **27**（= 层数） |
| `pending_peak` | 1 |
| `mean_win_len` | 1.00 |
| `issue_nocredit` / `layer_cap_blocks` / `gl_cap_blocks` | 全为 **0**（三类约束从未触发） |

**设计推论**：单 batch 场景下，调度器的并发能力**不会被触发** ——
这不是调度器的缺陷，而是"进度单位是 hop，不是 request"这一事实的必然结果。
`reserve/commit/rollback`、三层约束、aging 会在**多 context** 场景下才真正生效。

### 5.2 不变量 I2：per-hop 存在固定成本

hop 的成本可以分解为一个**与 token 数无关的固定项** `f` 和一个随 token 增长的边际项：

```
   TPOT(G) ≈ base + 27 · G · f + 432 · v

   G    = context 组数
   f    = 每次 FFN 调用的固定成本（host 侧 dispatch 主导）
   v    = 每 token 的边际成本
```

三点拟合的实测结果：`f ≈ 1.5 ms`，而 16-token hop 的边际 GPU 工作只有 `≈ 1.1 ms`。

**为什么会有固定项**：FFN 进程的调用路径是 **host/dispatch 受限**，不是 GPU 受限 ——
每 hop 的 GPU kernel 总时间远小于 host 侧 CPU 时间，且**与 token 数无关**。

**设计推论（重要）**：任何"把一个大 batch 拆成 G 份"的方案都要付 `G × 27 × f`。
当 `f` 相对每 hop 的真实工作量还不够小时，**拆分换重叠的净收益为负**。
由此可以给出三类方案的**收益边界**：

| 方案 | 收益边界 / 成立条件 |
|---|---|
| 步内拆 microbatch（`STAGGER_MB`） | hop 数 × G，需"重叠省下的量"大于 `G × f` 的增量 |
| persistent 多 context（`GROUPS > 1`） | 同上，且平均每 hop 只携带约 1 个 token，需先压缩 `f` |
| "攒更大的 FFN batch" | 合并键是 `layer_id`，需要**同一瞬间同一层**；而重叠要求上下文处在**不同层**，在现有点分组策略下互斥 —— 需要新的分组策略（见 §9.2） |

### 5.3 不变量 I3：TPOT 的数据依赖下界

对**固定 batch**，单个 token 的计算链是严格串行的：

```
   attn(L) ──► hop(L) ──► attn(L+1) ──► hop(L+1) ──► ...
                 └── hop(L) 的输出正是 attn(L+1) 的输入
```

因此：

```
   TPOT ≥ N_layers × (attn + hop) + 采样与调度开销
```

**设计推论**：

- 跨 token / 跨 batch 的重叠能提升**吞吐**，**无法降低单条序列的 TPOT** —— 这是数据依赖，不是实现问题。
- 要降 TPOT 只有三条路：**更少/更便宜的 hop**、**更便宜的 attn**、或**一次遍历产出多个 token**（投机解码，`TPOT ≈ 链长 / K`）。

### 5.4 三条不变量共同指向的设计结论

```
   I1（单 batch 波前形态）  ─┐
                             ├──► 并发调度机制的收益在多 context 下才展开
   I2（per-hop 固定成本）   ─┤
                             ├──► 拆分 batch 换重叠，需先压缩 f 才转正
   I3（TPOT 数据依赖）      ─┘
                             └──► 重叠提升吞吐；降单序列延迟要靠更少/更便宜的 hop
```

**因此当前阶段的架构重心应放在：**
1. 降低 per-hop 固定成本 `f`（它既是"每 hop 的代价"，也是"拆分策略能否转正"的开关）；
2. 降低 Attn 侧每 hop 的 host 侧开销；
3. 在 `f` 压下来之后，再评估"制造并发"类设计的收益。

---

## 6. 可扩展维度与变体

架构在五个维度上预留了扩展点：

```
   AFD 扩展维度
   │
   ├─ ① 路由方案      Scheme A（Attn 算 topk） / Scheme B（FFN 算 topk）
   ├─ ② 精度          bf16 / fp16 / FP8 A2F
   ├─ ③ 层粒度        layer-merge K（K 层共用一次 A2F/F2A）
   ├─ ④ 时间粒度      one-shot / persistent（跨 decode step）
   └─ ⑤ 拓扑规模      1A1F 经典 / AfPool MxN
```

### 6.1 路由方案 A / B

| | Scheme A（默认） | Scheme B |
|---|---|---|
| gate + topk 在哪 | **Attn** | **FFN** |
| A2F 载荷 | hidden + `topk_ids` + `topk_weights` | 只有 hidden |
| Attn 侧 MoE 模块 | 保留 gate/topk，stub 掉 experts | 整个 `mlp` 都 stub |
| FFN 侧 | 只有 experts | gate + experts |

**权衡**：Scheme A 把 routing 放在 Attn，A2F 多传 2 个小张量，但 Attn 侧能省掉 experts 权重；
Scheme B 让 A2F 更瘦，但 FFN 侧要加载 gate 权重。**默认选 A**：因为 Attn 侧显存更紧张。

### 6.2 精度与量化

- `SGLANG_AFD_A2F_DTYPE=fp8`：Attn 侧做 per-tensor absmax 量化，
  写入 `float8_e4m3fn` + scale（槽位 5），FFN 侧反量化后再计算。
- 收益：A2F 带宽减半。代价：一次量化 + 一次反量化。
- **设计约束**：`hidden_scale` 槽位**存在与否取决于 wire dtype**，
  所以 `AfdServerBatch.hidden_scale` 用"扫一遍张量找"的方式定位，而不是硬编码槽位号 ——
  这让 wire dtype 可以在启动时自由切换。

### 6.3 层合并（layer-merge）

`SGLANG_AFD_LAYER_MERGE_K = K`：**K 层共用一次 A2F/F2A**。

```
   merge_k = 1（默认）                 merge_k = 3
   ────────────────────                ──────────────
   L0  ──A2F/F2A──►                     L0  ─┐
   L1  ──A2F/F2A──►                     L1   ├── 一次 A2F/F2A
   L2  ──A2F/F2A──►                     L2  ─┘
   L3  ──A2F/F2A──►                     L3  ─┐
                                        ...
```

由于组内**中间层需要跑 attention**，而这些层的 attention 被放到了 FFN 侧执行，
所以 A2F 需要额外携带 `residual` + `positions`（槽位 6/7），F2A 需要回传 residual。
这是 residual 唯一一次离开 Attn 侧的例外。

**副作用**：`apply_layer_merge_env()` 会强制 `REMOTE_FROM_LAYER=0`、
关闭 in-graph wait、并把 `NUM_MB` 抬到至少 2 —— 这些是层合并的**必要前置条件**，不是可选项。

### 6.4 跨 step 持久化（persistent）

| | one-shot（`PERSISTENT=0`） | persistent（`PERSISTENT=1`） |
|---|---|---|
| context 生命周期 | 一次 `model.forward()` 内 | **跨 forward**，直到采样完成 |
| 未完成的 hop | 必须在本 forward 内抽干 | 可以留在队列里跨 forward |
| 单元 | 一个 window | **一个 context（= N 行）** |
| 已知限制 | — | 只有 `GROUPS=1`（context 尽可能肥）时才划算 |

**设计要点**：persistent 的价值**不来自重叠**，而来自**取消了 forward 内的
`while finished < n_seq` 屏障**。真正做重叠（`GROUPS > 1`）反而更慢，原因见 I2。

> `GROUPS=1` 时等价于"one-shot 但状态跨 forward 存活"，是当前 CG 关闭下的最优形态之一。

### 6.5 拓扑扩展：AfPool MxN

经典 1A1F 的推广，目标从"单流 TPOT"转为**吞吐 + FFN 利用率**：

```
     Router  (least_inflight / rr)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      FFN 0       FFN 1       FFN 2
        │           │           │
        └───────────┼───────────┘
                    │  per-pair socket: a{i}_f{j}.sock
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Attn 0      Attn 1      Attn 2
```

| 组件 | 职责 |
|---|---|
| `topology.py` | env → Na × Nf endpoint 矩阵 |
| `router.py` / `credit.py` | least-inflight 路由 + credit 窗口 |
| `attn_client.py` / `ffn_worker.py` | 多链路 cuda_ipc |
| `bootstrap_pool.py` | `POOL=1` 时的 ModelRunner hook |

**关键设计问题（已修）**：FFN 侧轮询多个 Attn link 时，若**逐 link 阻塞**扫描，
空闲的 link 0 会阻塞满超时，导致 link 1 上已就绪的 hop 被压在后面（队头阻塞）。
正确做法是**先 drain 所有 link 再统一 park**。

**由此得到一条拓扑设计原则**：多个 Attn worker 的**负载均衡度**是**一阶变量**。
`round_robin` 下两个 Attn 大致同步，队头阻塞几乎不出现；
真实部署（带 mini-LB、或 prefill/decode 混合不均）下**总有一个 Attn 领先**，此时该问题影响显著。

### 6.6 CUDA Graph 策略矩阵

| 角色 | 策略 | 原因 |
|---|---|---|
| **Attn** | decode CG 强制 **`breakable`** | remote FFN 是 eager 图断点；in-graph wait 已被 `TRUE_OVERLAP` 取代 |
| **FFN** | model CG **`disabled`** | 模型 CG 会撞到 attn stub |
| **FFN compute CG** | `SGLANG_AFD_FFN_CUDA_GRAPH=1` 可开 | per-`(layer, token-bucket)` 捕获 |
| **两者关系** | 互斥（in-graph wait 与 layer pipeline 不可同时启用） | 由 `apply_afd_cuda_graph_policy()` 统一裁决 |

**FFN compute CG 的设计局限**：图按固定 `bs` 捕获成桶，
而 farm hop 的 **token 数是变化的** → 多数 hop 命中不了桶，要付图查找 + 回退代价。
这是"CG 概念正确、但集成方式不适配 farm"的典型案例。

---

## 7. 启动与生命周期

### 7.1 启动时序

```
   ① 启动 FFN 进程（先起）
      SGLANG_AFD_MODE=ffn
      CUDA_VISIBLE_DEVICES=<GPU B>
      python3 -m sglang.launch_server --skip-server-warmup
         │
         ├─ load 权重（仅 experts，attn 被 stub）
         ├─ 分配 AfdBufferPool
         ├─ _ffn_export_and_accept()   导出 CUDA IPC handle
         ├─ bind_ffn_shared_io()       （可选）绑定 CG staging
         └─ _start_ffn_poll_loop()     启动后台 poll 线程
               │
               └──► 日志出现 "AFD cuda_ipc FFN waiting"

   ② 等待 FFN 就绪
      轮询日志，最长约 240 s

   ③ 启动 Attn 进程
      SGLANG_AFD_MODE=attn
      CUDA_VISIBLE_DEVICES=<GPU A>
      python3 -m sglang.launch_server
         │
         ├─ load 权重（仅 MLA + gate/topk，experts 被 stub）
         ├─ _attn_connect_and_import()  导入 FFN 的 handle
         ├─ 建立 _HostMailbox
         └─ （可选）enable_wait_flag_sync()
               │
               └──► /health 可用 → 对外服务
```

**为什么 FFN 必须先起**：handle 的导出方是 FFN（它拥有那块显存），
Attn 是导入方。顺序反了就没有可导入的 handle。

### 7.2 运行时职责

| | Attn 进程 | FFN 进程 |
|---|---|---|
| 主驱动 | `run_farm_layers`（在 forward 内） | 后台 poll 线程 |
| 是否接请求 | 是 | 否 |
| 是否持有 KV | 是 | 否 |
| 触发方式 | 客户端请求驱动 | mailbox `posted` 驱动 |
| 空闲时行为 | 正常调度 | park 在 zmq poll 上（需 `--sleep-on-idle`） |

> **`--sleep-on-idle` 是必要的**：FFN 进程的 scheduler 若不 park，
> 会持续运行纯 Python 热路径并**持有 GIL**，与计算线程争抢。
> 由于该 scheduler 不承接真实请求，这是一处比较隐蔽的资源竞争。

### 7.3 关闭

- `cleanup` 逐进程 `TERM` → 等待 → `KILL`，并清理子进程组。
- 清理端口与 `SGLANG_AFD_IPC_ENDPOINT` socket。
- `_HostMailbox.close(unlink=True)` 释放 POSIX 共享内存。

---

## 8. 设计决策记录（ADR）

| # | 决策 | 理由 | 代价 | 状态 |
|--:|---|---|---|---|
| D1 | **按角色切分**，不按层区间切分 | FFN 侧彻底去掉 KV Cache 与 attention 模块，显存收益最大化 | 每层都要过一次网络（27 hop/step） | 已采纳 |
| D2 | **residual 只留在 Attn** | 避免每 hop 多传一个 `[T,H]`；residual 是层内状态，语义上属于 Attn | layer-merge 需要例外处理 | 已采纳 |
| D3 | **Scheme A 为默认** | Attn 侧显存更紧张，把 experts 全部移走收益更大 | A2F 多传 topk 两个小张量 | 已采纳 |
| D4 | 启动时**预注册缓冲**，热路径零分配 | 消除热路径 `torch.empty` 与随之而来的抖动 | 需要提前定 `MAX_NUM_TOKEN` | 已采纳 |
| D5 | 用 **credit 门控**而非定时器 | 用背压适配下游真实速度，避免猜测超时 | reservation 需要 `rollback` 路径 | 已采纳 |
| D6 | **completion-first** 收割 | 单个慢 hop 不阻塞波前 | 需要遍历 pending 而非只取队首 | 已采纳 |
| D7 | `commit()` **推迟到拿到 credit 之后** | 保证"调度器在飞数"与"传输层在飞数"不漂移 | 多一次状态检查 | 已采纳 |
| D8 | 拆出 `MAX_INFLIGHT_PER_LAYER` | 防止第一层吃光所有槽位而退回 lockstep | 多一个旋钮 | 已采纳 |
| D9 | **步内拆 microbatch 默认关闭** | I2：拆分付 `G × 27 × f`，实测更慢 | 放弃了"制造重叠"的简易路径 | 已采纳并实测验证 |
| D10 | **persistent 默认关闭**，仅 `GROUPS=1` 时启用 | 真重叠（`GROUPS>1`）更慢；`GROUPS=1` 的收益来自取消屏障 | 两个形态需要分别维护 | 已采纳 |
| D11 | 模块 stub 失败时**立即抛错** | 快速暴露接线错误，不静默降级 | 无 | 已采纳 |
| D12 | Attn 强制 **breakable** CG | remote FFN 本质是 eager 图断点 | 失去 full CG 的部分收益 | 已采纳 |
| D13 | FFN 侧轮询**先 drain 再 park** | 消除多 Attn 场景的队头阻塞 | 无 | 已修正 |
| D14 | `mean_ffn_util` 改为 **FFN 侧自报** | 旧的 attn 侧 round-trip 累加会 clamp 到 1.0，无法区分半闲与饱和 | KPI 语义变更 | 已修正 |

---

## 9. 演进路线

### 9.1 已交付阶段

| 阶段 | 范围 | 状态 |
|---|---|---|
| P0–P0.5 | RFC、Fake transport、DeepSeek hook、自动初始化、breakable CG | 完成 |
| P1 | MoE Scheme A（topk 走 A2F） | 完成 |
| P2 | wait_flag | 完成 |
| P3 | pipeline / NUM_MB 重叠 | 完成 |
| P4 | 权重裁剪 + PD 策略 | 完成 |
| P5 | 模块 stub + Linear parity | 完成 |
| P6 | Scheme B、FP8 A2F、residual 策略、smoke/bring-up | 完成 |
| P7 | 错峰双/三 mb 层流水（`LAYER_PIPELINE`） | 完成 |
| P8 | in-graph wait + full decode CG | 完成（后被 `TRUE_OVERLAP` 取代） |
| P9 | AfPool MxN 工作池 | 实验 |

### 9.2 架构层面的下一步

依据 §5.4 的推论，重心应放在**降低 per-hop 固定成本与 Attn 侧 host 开销**；
并发类设计的收益待 `f` 压下来后再评估：

| 优先级 | 方向 | 理由 |
|---|---|---|
| 高 | **压缩每 hop 的 host 侧编组开销** | Attn 侧 farm 时间中相当大一部分是"每个 forward 重复数十次"的张量切分与载荷拼装；这部分不随 batch 增大而摊薄 |
| 高 | **让窗口级结构（child batch）在层间复用** | 同一 window 走 27 层，行元数据不变；每层重建属于重复劳动 |
| 中 | **降低 FFN 侧每次调用的固定成本** | `f` 既是"每 hop 的代价"，也是"拆分策略能否转正"的开关；压缩它对两条路线都有利 |
| 中 | **FFN compute CG 适配变长 hop** | CG 概念上正确（去掉 dispatch 开销），但当前按固定 bucket 捕获不适配 farm 的变长 token |
| 低 | FFN 侧 kernel 微优化 | 收益被 hop 的隐藏性吸收；可待 hop 成本下降后重估 |
| 低 | 更多 context / 更强重叠 | 收益取决于 I2 中 `f` 的压缩进度，`f` 降下来后重估 |

### 9.3 判定新方案是否值得做的准则

任何新方案在做之前，先用这三问过滤：

1. 它是否**减少** hop 次数，或减少 per-hop 固定成本？（若增加 hop 数，需先有 `f` 的压缩作为对冲，见 I2）
2. 它是否降低了**单条序列**的串行链长度？（若只提升并发度，则改善吞吐；改善 TPOT 需另设路径，见 I3）
3. 它是否作用在**关键路径**上？（FFN 侧计算目前被 hop 隐藏吸收，收益需待 hop 成本下降后重估）

---

## 10. 附录

### 10.1 核心配置项

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `SGLANG_AFD_MODE` | `null` | `null` / `attn` / `ffn` |
| `SGLANG_AFD_TRANSPORT` | `fake` | `fake` / `stepmesh` / `cuda_ipc`（别名 `nvlink`） |
| `SGLANG_AFD_IPC_ENDPOINT` | `/tmp/afd_cuda_ipc.sock` | handle 交换用 Unix socket |
| `SGLANG_AFD_NUM_MB` | 1（farm 下抬到 `MAX_INFLIGHT`） | mailbox 槽位数 |
| `SGLANG_AFD_MAX_NUM_TOKEN` | 256 | A2F pad 尺寸，**唯一 OOM 开关** |
| `SGLANG_AFD_MODULE_STUBS` | true（mode≠null） | 模块裁剪开关 |
| `SGLANG_AFD_ROUTING_SCHEME` | `a` | `a` / `b` |
| `SGLANG_AFD_A2F_DTYPE` | `auto` | `auto` / `bf16` / `fp16` / `fp8` |
| `SGLANG_AFD_LAYER_MERGE_K` | 1 | K 层共用一次 A2F/F2A |
| `SGLANG_AFD_RELEASE_UNUSED_PARAMS` | false | 未用权重移到 CPU |
| `SGLANG_AFD_FFN_CUDA_GRAPH` | true | FFN compute CUDA Graph |

### 10.2 farm 调度配置项

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `SGLANG_AFD_FARM` | off | 启用 decode farm |
| `SGLANG_AFD_FARM_B_STEP` | 16 | 每个 window 的 token 数 |
| `SGLANG_AFD_FARM_B_WIN_K` | 8 | 换层前停留层内的 window 数 |
| `SGLANG_AFD_FARM_COALESCE_K` | 1 | 把 K 个 window 打包成一次发射 |
| `SGLANG_AFD_FARM_MAX_INFLIGHT` | 4 | 全局在飞 hop 数 |
| `..._MAX_INFLIGHT_PER_LAYER` | `0 → min(max_inf, 4)` | 单层发出 hop 上限 |
| `..._GLOBAL_TOKEN_BUDGET` | `max_inf × b_step` | 持有 credit 的 token 上限 |
| `..._MAX_AGE_STEPS` | `B_WIN_K` | 防饿死 |
| `SGLANG_AFD_FARM_SCHED` | `max` | `max` / `oldest` / `deepest` |
| `SGLANG_AFD_FARM_NUM_CONTEXTS` | 1 | 独立 context 波前数 |
| `..._CONTEXT_STAGGER_LAYERS` | 0 | 下一 context 进入前需领先的层数 |
| `..._CONTEXTS_PER_STAGE` | 1 | 同批进入的 context 数 |
| `SGLANG_AFD_FARM_STAGGER_MB` | **0（关）** | 步内拆 microbatch（见 I2） |
| `SGLANG_AFD_FARM_PERSISTENT` | 0 | 跨 decode step 保留状态 |
| `..._PERSISTENT_GROUPS` | 0 | `0` = 一行一 context；`N` = 每 context N 行 |

### 10.3 诊断开关

| 开关 | 输出 |
|---|---|
| `SGLANG_AFD_FARM_STAGE_STATS_EVERY` | `hops/fwd`、`layers_peak`、`span_peak`、`ctxs_peak` |
| `SGLANG_AFD_FARM_PHASE_TIMING` | `pre` / `issue` / `drain` / `consume` 分解 |
| `SGLANG_AFD_PROFILE_DETAIL` | `mla_*` / `moe_*` 细粒度 span |
| `SGLANG_AFD_TIMELINE` | 跨进程 6 时间戳 ring（POSIX shm） |
| `SGLANG_AFD_FFN_PROBE` | 一次性 kernel/CPU census |
| `SGLANG_AFD_FARM_LPU_STATS_EVERY` | FFN serve loop 的 `group_hist` |

> **读诊断数据的前提**：`PROFILE_DETAIL` 会给 Attn 侧插入十余个 span，
> 而 Attn 的 scheduler 在关键路径上 —— 一旦 Attn 被自己的插桩拖慢，TPOT 就变成 attn-bound，
> FFN 侧的真实改善会被掩盖。**profiling 运行只适合在同一臂内做归因，不适合跨臂比 tok/s。**

### 10.4 代码索引

| 模块 | 职责 |
|---|---|
| `mode.py` | `AfdMode` 角色枚举（与 PD 的 `DisaggregationMode` 分离） |
| `protocol.py` | A2F/F2A 语义契约、key 位域打包 |
| `buffers.py` | `AfdBufferPool`，预注册缓冲与 `fill_a2f` |
| `cuda_ipc_transport.py` | `CudaIpcAfdTransport`，mailbox、唤醒、零拷贝 |
| `transport.py` | transport 抽象基类 |
| `module_stubs.py` | 模块裁剪与 `AfdMissingModule` |
| `remote_policy.py` | 哪些层的 FFN 走远端、layer-merge 策略 |
| `routing_scheme.py` | Scheme A/B 判定 |
| `bootstrap.py` | 从 ModelRunner 初始化 AFD、启动 FFN poll 线程、CG 策略 |
| `ffn_compute.py` | FFN 侧计算：单 batch / 同层融合 |
| `ffn_cuda_graph.py` | FFN compute CUDA Graph（按 bucket） |
| `farm/decode_farm_loop.py` | farm 主循环 `run_farm_layers` |
| `farm/scheduler.py` | `FarmContinuousScheduler` 状态机 |
| `farm/token_queue.py` | `LayerReadyQueues`：sticky / coalesce / 层选择 |
| `farm/attn_farm.py` | `AttnSendQueue`、`try_issue_hop`、credit 门控 |
| `farm/env.py` | farm 环境变量与初始化 |
| `farm/persistent_runtime.py` | 跨 forward 的持久 context |
| `pool/` | AfPool MxN：topology / router / credit / attn_client / ffn_worker |

---

*本文档聚焦架构设计；量化数据与逐条实验记录见 `progress.md`，分层文字说明见 `AFD_ARCHITECTURE.md`。*
