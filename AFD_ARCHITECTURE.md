# AFD Decode Farm — 架构（精简版）

范围：**当前正在运行**的实现（decode farm over cuda_ipc）。
实测数字在 `progress.md`；本文只讲结构和流程。

> 📊 **图文架构设计版见 [`AFD_DESIGN.md`](AFD_DESIGN.md)** —— 含拓扑图、职责边界、数据契约、
> 状态机、设计不变量与 ADR，纯文本示意图，可直接复制到飞书等文档。
> 另有面向汇报的 `AFD_REPORT.md`（架构 + 开销定位 + 优化杠杆，含 Mermaid 图，
> 适合 GitHub / IDE 渲染）。

---

## 1. 拓扑

两个进程、两张 GPU，按**角色**切分模型（不是按层区间切）：

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│  ATTN worker   (GPU A)      │         │  FFN worker    (GPU B)      │
│  SGLANG_AFD_MODE=attn       │         │  SGLANG_AFD_MODE=ffn        │
│                             │         │                             │
│  • embedding / norm /       │  A2F →  │  • MoE experts + shared     │
│    lm_head / 采样            │  ← F2A  │  • 没有 attention，没有 KV   │
│  • MLA attention（所有层）   │         │                             │
│  • MoE gate + topk（scheme A）│        │                             │
│  • 每一层的 attn             │         │  • 每一层的 FFN             │
└─────────────────────────────┘         └─────────────────────────────┘
        ▲                                                            │
        └──────────────── cuda_ipc（零拷贝）───────────────────────────┘
```

* `SGLANG_AFD_MODULE_STUBS=1` 让两侧只构建自己需要的模块（`module_stubs.py`）：
  Attn 保留 attention + gate/topk，FFN 保留 experts。
* FFN worker 跑一个**后台 poll 线程**（`bootstrap.py`），所以它是服务端：
  主动拉请求、计算、把结果写回。
* KV cache **只在 Attn** worker 上。FFN worker 从不做 attention。

## 2. 单个 decode step 的控制流（farm 路径）

入口：`deepseek_v2.py:2797` 对 `layers[normal_start_layer:normal_end_layer]`
调用 `run_farm_layers(...)`，取代原先 lockstep 的
`for layer in layers: layer(...)` 循环。

```
model.forward()
  └─ run_farm_layers()                     [decode_farm_loop.py:326]
       ├─ scheduler.begin_batch(range(n_seq))   → 所有 seq 一起入 layer 0
       ├─ while finished < n_seq:
       │    ├─ _issue_round()                    ← 取活、跑本地 Attn、发 hop
       │    │     for 每个空闲 in-flight 槽位:
       │    │       ticket = scheduler.reserve(b_step, ...)
       │    │       slice_decode_window(...)     ← 切出这个 window 的行
       │    │       run_layer_forward_pre_ffn()  ← 本地 Attn（MLA + gate + topk）
       │    │       try_issue_hop(...)           ← A2F 发给 FFN
       │    │       scheduler.commit(ticket)     ← 转为 "running"
       │    ├─ _drain()                          ← completion-first：收割任意完成的 hop
       │    │     for hop in pending: if poll_hop(hop): _consume_hop(...)
       │    └─ 若无可发射：_block_one()          ← 等一个 hop，然后重新扫描
       └─ 抽干所有 pending，然后 finish_batch()
  └─ self.norm() → lm_head → logits → 采样    [deepseek_v2.py:2899]
```

关键性质：主循环是 **completion-first**。`_drain` 收割**最先完成**的那个 hop
（而不是 `pending[0]`），所以单个慢 hop 不会堵住整个波前。

流动的 window 形如：

```
batch（全部 seq）
   → [layer L]  Attn（本地）  +  MoE gate/topk（本地）
   → A2F hop →  FFN（远端）   → F2A
   → [layer L+1] ...
```

## 3. hop：线上到底传了什么

定义在 `protocol.py`，由 `buffers.py` 中预注册的池承载。

| 方向 | 载荷（`AfdA2FPayload`） | 说明 |
|-----------|---------------------------|-------|
| Attn → FFN | `hidden [T,H]`、`num_tokens [1]`、`layer_id [1]`、`topk_ids [T,K]`、`topk_weights [T,K]` | `hidden` 可为 fp8/bf16；量化时额外带 `hidden_scale` |
| （仅 layer-merge） | `+ residual [T,H]`、`positions [T]` | interior 层也在 FFN 上跑 attention 时用 |

| 方向 | 载荷（`AfdF2APayload`） | 说明 |
|-----------|---------------------------|-------|
| FFN → Attn | `mlp_out [T,H]` | layer-merge 时 `+ residual` |

传输层（`cuda_ipc_transport.py`）：

* **张量内存在启动时一次性预注册并共享**
  （`_ffn_export_and_accept` / `_attn_connect_and_import`）；在 `_zero_copy`
  路径下每 hop **没有拷贝** —— `push_pull` 只 record 一个 CUDA event 并写
  mailbox 槽位。
* **`NUM_MB` 个 mailbox 槽位**（默认 2，会自动抬到 `MAX_INFLIGHT`）提供多个
  独立的在飞槽位；`mb_id` 选择其一。
* `_HostMailbox` 承载 `posted` / `done` / `meta(num_tokens, layer_id)`。
* 唤醒：eventfd（`_signal_wake` / `_park_until_wake`），可选 GPU doorbell。
* `get_batch()`（FFN 侧）轮询所有槽位，并可选地**攒批** `gather_us`，把同层到达
  的多个请求融成一次计算调用。

## 4. Attn 侧：发送队列与发射

`AttnSendQueue`（`farm/attn_farm.py`）是 Attn 侧的出站队列：

* `enqueue` → `flush_due` → `_issue_layer` 构造 `_SendGroup`。
* 只有在**下游 credit** 允许时才取出一组（`_issue_capacity`），被阻塞的组会被
  记账（`_account_blocked`）。这是 credit 门控，不是定时器。
* `try_issue_hop`（`attn_farm.py:573`）是每 hop 的入口：切片、发送、返回
  `FarmHop`。`flush_hop_queues()` 批量刷出。
* `poll_hop` / `wait_hop` 读取 F2A 结果。

`FarmHop` 是在飞记录：层号、seq 索引、token 区间、residual/meta 引用、
`sched_ticket`（调度器预留凭证）。

## 5. FFN 侧：服务循环

`bootstrap.py` 为每个 FFN worker 起一个后台线程：

```
while not stop:
    batches = transport.get_batch(timeout_s=0.05)     # 轮询所有 mb 槽位（+攒批）
    if not batches: continue
    by_layer = group(batches, key=layer_id)           # ← 混合层是安全的
    for layer_id, group in by_layer.items():
        fused = try_compute_same_layer_group(compute, group)   # 同层 MoE 融合
        for batch, outs in ...:
            transport.respond(batch, outs)            # F2A 拷贝 + set_done
```

有两点值得注意：

* **同层分组是显式的**（`by_layer`）。即使一次攒批混了不同层，也是正确的，
  因为计算前会按 `layer_id` 重新分组。
* `try_compute_same_layer_group` 把**同一层**的多个槽位融成一次 MoE 调用——
  这才是 `NUM_MB>1` 带来吞吐收益（而非重复计算）的原因。

## 6. 调度器状态机

`FarmContinuousScheduler`（`farm/scheduler.py`）+ `LayerReadyQueues`
（`farm/token_queue.py`）。状态是**进程内**且**跨循环持久**的，可选跨 decode
step 持久（`PERSISTENT=1`）。

每层状态：

```
waiting (ready[])          积压，不占 credit
   │  reserve()             ← 经 _choose_layer 选取（sticky / oldest / deepest）
   ▼
reserved (_reserved_hops)  已占槽位，尚未上线
   │  commit()              ← 拿到 A2F credit
   ▼
running (running[])        F2A 在飞
   │  complete()            ← 把同一个 window 重新入队到 layer+1
   ▼
layer+1 的 waiting（若 layer == 最后一层则 finished）
```

* `rollback()` 归还未拿到 credit 的预留。
* `_choose_layer` 遵循 `sticky_layer` / `sticky_left`（`B_WIN_K`），并支持
  aging（`MAX_AGE_STEPS`）与 `sched ∈ {max, oldest, deepest}`。
* 存在三个互相独立的约束：`MAX_INFLIGHT`（全局槽位）、
  `MAX_INFLIGHT_PER_LAYER`、`GLOBAL_TOKEN_BUDGET`。

## 7. 结构性不变量（最重要）

`begin_batch` 把**所有** sequence 一起入 layer 0，而 `complete` 又把**同一个
window** 重新入队到 `layer+1`。当 `b_step ≥ batch` 时，`pick()` 一次取走整层。
因此：

```
{A,B,C,D}@L0 → hop → {A,B,C,D}@L1 → hop → {A,B,C,D}@L2 → ...
```

* 每层**恰好一个 window / 一个 hop 在飞** → `layers_peak = 1`。
* `picks == layer_switches == 3456`、`mean_win_len = 1.00`、`pending_peak = 1`。
* 三个约束**从未被触发**（`issue_nocredit=0`、`*_blocks=0`）。

队列机制是完整的，但处于**空转**：永远不存在第二个有活的层可供重叠。
完整论证见 `progress.md` §9；以及 §4 解释为什么"拆 batch 制造重叠"必亏
（它会乘上一个固定的每 hop 成本）。

## 8. 配置对照表

| 环境变量 | 默认 | 含义 |
|-----|---------|---------|
| `SGLANG_AFD_FARM` | off | 启用 decode farm |
| `SGLANG_AFD_TRANSPORT` | `fake` | 真实运行需设为 `cuda_ipc` |
| `SGLANG_AFD_FARM_B_STEP` | 16 | 每个 window 的 token 数 |
| `SGLANG_AFD_FARM_B_WIN_K` | 8 | 换层前在同一层停留的 window 数 |
| `SGLANG_AFD_FARM_COALESCE_K` | 1 | 把 K 个 window 打包成一次发射（更大的 M） |
| `SGLANG_AFD_FARM_NATURAL_BATCH` | 0 | 让发送队列自然攒批到达 |
| `SGLANG_AFD_FARM_MAX_INFLIGHT` | 4 | 全局在飞 hop 数（会抬 `NUM_MB`） |
| `..._MAX_INFLIGHT_PER_LAYER` | 0→`min(max_inf,4)` | 单层发出的 hop 上限 |
| `..._GLOBAL_TOKEN_BUDGET` | 0→`max_inf × b_step` | 持有 credit 的 token 上限 |
| `..._MAX_AGE_STEPS` | 0→`B_WIN_K` | 防饿死 aging |
| `..._SCHED` | `max` | `max` / `oldest` / `deepest` |
| `..._STAGGER_MB` | **0（关）** | 把 batch 拆成 G 个 microbatch（见下） |
| `..._STAGGER_SPAN` | 2 | 领头 microbatch 前进几层后注入下一个 |
| `..._PERSISTENT` | 0 | 跨 decode step 保留调度器状态 |
| `SGLANG_AFD_FFN_GATHER_US` | 0（farm 下经 LPU 为 50） | 同层融合的攒批窗口 |
| `SGLANG_AFD_NUM_MB` | 2 | mailbox 槽位数（自动 ≥ `MAX_INFLIGHT`） |
| `SGLANG_AFD_MAX_NUM_TOKEN` | — | A2F pad 尺寸；**OOM 的开关** |

诊断：`..._FARM_STAGE_STATS_EVERY`、`..._FARM_PHASE_TIMING`、
`..._FARM_LOG_EVERY`、`SGLANG_AFD_PROFILE_DETAIL`、
`SGLANG_AFD_FARM_FFN_SECTION_TIME`、`SGLANG_AFD_FARM_FFN_PROBE`（一次性）。

## 9. 错峰注入（`STAGGER_MB`）—— 已实现，默认关闭

`enqueue_entry()` 把后续 microbatch 注入到活着的 batch 中，`deepest_active_layer()`
是波前探针。设 `STAGGER_MB=G` 时，sequence 按轮转切成 G 组，在不同时刻进入
layer 0。

它**确实**能产生真正的重叠（`layers_peak = 2`）——但它把 hop 数乘了 G
（`27 → 61–92`），从而乘上了固定的每 hop 成本。实测：TPOT 142 → 268 ms。
在 hop 变便宜之前保持关闭。（`progress.md` §4。）

## 10. 时间花在哪

每 hop 成本分解与 batch 缩放判别实验见 `progress.md` §1–§3。要点：

* `block`（等 FFN hop）≈ **2.97 ms/层 × 27 = 80 ms = TPOT 的 55%**。
* hop 内部，`ffn_routed` ≈ **1.6 ms**，且**与 batch 无关**
  （token 数 ×2.3 → 仅 −1.8%）：是 host/dispatch 受限，不是 GPU 受限。
* 不是 sleep 受限（`a2f_sync` = 4 µs），不是 poll 受限（`post_poll` 从不触发）。
* `pre`（本地 Attn）≈ 1.6 ms/层，主要是 `attn_core`（MLA）。

## 11. 跨 step 重叠（P1.4）的已知阻塞

见 `README.md` 与 `progress.md` §10 —— 概述：

1. **队列条目身份** —— 条目是指向某一次 forward 张量的裸行索引；两个并存的
   batch 会切错行。需要 `(batch, index)` + per-batch 张量上下文。
2. **`run_farm_layers` 是屏障** —— 直到每个 token 都走出最后一层才返回，
   而紧接着就是 `norm → lm_head → 采样`。不存在"step N 的尾巴与 step N+1 的
   头部并存"的时刻，也没有地方寄存未完成的 hop（缺失的 "OutputBarrier"）。
3. **采样依赖** —— step N 的 logits 需要 layer 26 的输出；step N+1 需要
   step N 采样出的 token。硬依赖，缝隙为零。

`MAX_ACTIVE_BATCHES` 被钉死为 1。

## 12. TPOT 的硬下界

```
TPOT = N_layers × (attn + hop) + 其他开销
     = 27 × (1.60 + 2.97) + ~12 ms  = 141 ms      （实测 141.9 ms）
```

`hop(L)` 产出的正是 `attn(L+1)` 的输入，所以这条链是数据依赖。
**不同 token 之间的重叠只能提升吞吐，无法降低单条序列的 TPOT。**
要降 TPOT：更少/更便宜的 hop、更便宜的 attn，或一次遍历产出多个 token
（投机解码，`≈ 链长 / K`）。
