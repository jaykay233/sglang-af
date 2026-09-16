# RFC: SGLang Decode × StepMesh Attention–FFN Disaggregation (AFD)

| Field | Value |
|-------|-------|
| Status | Implemented (P0–P6) |
| Authors | SGLang + StepMesh integration |
| Created | 2026-08-29 |
| Tracking | `python/sglang/srt/afd/` |

## 1. Motivation

Decode often couples **KV-heavy Attention** and **compute-heavy FFN/MoE** on the same GPUs.
Attention–FFN Disaggregation (AFD) places them on different instance pools and exchanges
activations over RDMA (StepMesh), so each side can scale independently.

This RFC defines the **interfaces** between SGLang decode and StepMesh, the eager path,
breakable CUDA Graph sync, selective load/stubs, routing schemes, and optional FP8 A2F.

## 2. Goals / Non-goals

**Goals**

1. Stable transport API independent of RDMA (FakeTransport for CI / same-process).
2. Clear Attn↔FFN tensor contract (A2F / F2A) with pre-registered buffers.
3. Hook in `DeepseekV2DecoderLayer` after `prepare_mlp`.
4. FFN server loop that consumes A2F and responds F2A.
5. Breakable CUDA Graph + optional wait_flag; pipeline overlap; PD composition;
   weight filter + module stubs; scheme A/B routing; optional FP8 A2F.

**Non-goals (historical P0–P6; P9 AfPool is experimental)**

- Replacing DeepEP inside the FFN domain (DeepEP stays local to FFN ranks).
- Full decode `cuda_graph_backend=full` spanning StepMesh.
- Multi-node cluster scheduler (example bring-up script only).

> Production-grade straggler policy remains out of scope; AfPool provides a
> minimal credit + least-inflight router for same-host cuda_ipc MxN.

## 3. Roles

```
AfdMode.NULL   – collocated (default)
AfdMode.ATTN   – Attention worker (holds KV, StepMesh worker)
AfdMode.FFN    – FFN server (StepMesh server, no KV)
```

Distinct from PD `DisaggregationMode` (KV transfer). Do not overload those enums.

## 4. Layer cut

```
prepare_attn → self_attn → prepare_mlp
                              │
                              ▼  A2F (hidden [+ routing meta])
                         FFN / MoE (remote)
                              │
                              ▼  F2A (mlp output)
                         postprocess_layer
```

- **Residual** stays on Attn only (never on A2F).
- **Scheme A (default):** Attn runs gate+topk; A2F carries routing tensors.
- **Scheme B:** FFN runs gate+topk+experts; A2F is hidden (+ meta) only.
- Dense MLP: A2F is post-norm hidden only.

## 5. Wire protocol

### 5.1 Keys

```
bit 0–7   : private_key (tensor slot id)
bit 8–15  : microbatch_id
bit 16–23 : worker_rank
bit 24    : direction (0 = A2F/push, 1 = F2A/pull)
```

### 5.2 A2F tensors (Attn → FFN)

| Slot | Name | Shape | Dtype | Notes |
|------|------|-------|-------|-------|
| 0 | `hidden` | `[T, H]` | bf16/fp16/fp8 | Post-`prepare_mlp` |
| 1 | `num_tokens` | `[1]` | int32 | |
| 2 | `layer_id` | `[1]` | int32 | |
| 3 | `topk_ids` | `[T, K]` | int32 | Scheme A MoE only |
| 4 | `topk_weights` | `[T, K]` | fp32 | Scheme A MoE only |
| last | `hidden_scale` | `[1]` | fp32 | When `SGLANG_AFD_A2F_DTYPE=fp8` |

### 5.3 F2A tensors (FFN → Attn)

| Slot | Name | Shape | Dtype |
|------|------|-------|-------|
| 0 | `mlp_out` | `[T, H]` | compute dtype (bf16/fp16) |

### 5.4 Padding

`T` padded to decode CUDA-graph bucket; `num_tokens` is the true length.

## 6. Transport

| Class | Use |
|-------|-----|
| `FakeAfdTransport` | Same-process queue + optional local FFN callback (CI) |
| `StepMeshAfdTransport` | `fserver_lib` RDMA path |
| `CudaIpcAfdTransport` | Same-host CUDA IPC (NVLink/P2P when available) |

Bring-up: `python -m sglang.srt.afd.smoke` and `bringup_stepmesh_example.sh`.

## 7. Buffer pool

Preallocates `num_mb × slots × max_num_token × hidden` (wire dtype for A2F hidden,
compute dtype for F2A). No decode-hot-path `torch.empty` for A2F/F2A.

## 8. CUDA Graph

Decode **Attn** uses `breakable` (not `full`). Optional `SGLANG_AFD_USE_WAIT_FLAG=1` for
StepMesh `write_flag` / `wait_flag` around CPU `push_pull`.

Decode **FFN** must keep model-level decode CG **disabled** (attn modules are stubs).
FFN speed uses `SGLANG_AFD_FFN_CUDA_GRAPH=1` (default): per-(layer, token-bucket) CUDA
graphs around MLP / MoE experts only (`ffn_cuda_graph.py`).

## 9. Runtime configuration

| Env | Meaning | Default |
|-----|---------|---------|
| `SGLANG_AFD_MODE` | `null` / `attn` / `ffn` | `null` |
| `SGLANG_AFD_TRANSPORT` | `fake` / `stepmesh` / `cuda_ipc` (alias `nvlink`) | `fake` |
| `SGLANG_AFD_IPC_ENDPOINT` | Unix socket for cuda_ipc handle exchange | `/tmp/afd_cuda_ipc.sock` |
| `SGLANG_AFD_NUM_MB` | microbatch slots | `1` |
| `SGLANG_AFD_MAX_NUM_TOKEN` | pad length | `256` |
| `SGLANG_AFD_USE_WAIT_FLAG` | GPU flag sync | `false` |
| `SGLANG_AFD_PIPELINE` | overlap A2F/F2A across mb | `false` |
| `SGLANG_AFD_LAYER_PIPELINE` | P7 stagger dual/triple-mb Attn↔FFN across layers (`NUM_MB>=2`) | `false` |
| `SGLANG_AFD_TRUE_OVERLAP` | Breakable CG + dual-mb + deferred wait_flag (issue≠wait); keeps CG segments | `false` |
| `SGLANG_AFD_IN_GRAPH_WAIT` | P8 GPU write/wait_flag inside **full** decode CG (no eager break) | `false` |
| `SGLANG_AFD_RELEASE_UNUSED_PARAMS` | move unused params to CPU | `false` |
| `SGLANG_AFD_ALLOW_PREFILL_ATTN` | allow attn+PD prefill | `false` |
| `SGLANG_AFD_MODULE_STUBS` | skip unused module construct | `true` |
| `SGLANG_AFD_ROUTING_SCHEME` | `a` (Attn topk) / `b` (FFN topk) | `a` |
| `SGLANG_AFD_A2F_DTYPE` | `auto` / `bf16` / `fp16` / `fp8` | `auto` |
| `SGLANG_AFD_WORKER_RANK` | Attn worker rank in key space | `-1` (→ tp_rank) |
| `SGLANG_AFD_FFN_CUDA_GRAPH` | FFN MLP/MoE compute CUDA graphs | `true` |
| `SGLANG_AFD_TIMELINE` | emit cross-process RT timeline (post/seen/compute/respond/done) | `false` |

## 10. Phased delivery

| Phase | Scope | Status |
|-------|-------|--------|
| P0–P0.5 | RFC, Fake, DeepSeek hook, auto-init, breakable CG | done |
| P1 | MoE scheme A topk on A2F | done |
| P2 | wait_flag | done |
| P3 | pipeline / NUM_MB overlap | done |
| P4 | weight filter + PD policy | done |
| P5 | module stubs + Linear parity | done |
| P6 | scheme B, FP8 A2F, residual policy, smoke/bring-up | **done** |
| P7 | staggered dual/triple-mb layer pipeline (`SGLANG_AFD_LAYER_PIPELINE`) | **done** |
| P8 | in-graph wait_flag + full decode CG (`SGLANG_AFD_IN_GRAPH_WAIT`) | **done** |
| P9 | AfPool MxN work pool (`SGLANG_AFD_POOL`) over cuda_ipc | **experimental** |

## 11. Design decisions (closed)

1. **Residual** — Attn-only; never on A2F.
2. **FP8 A2F** — Attn-side per-tensor absmax → `float8_e4m3fn` + scale; FFN dequants
   before compute (`SGLANG_AFD_A2F_DTYPE=fp8`).
3. **Scheme B** — gate weights load on FFN; Attn stubs entire MoE module when stubs on.
4. **P7 layer pipeline** — split decode batch into 2–3 mbs; issue FFN(mb_i,L) then run
   Attn(mb_j) so remote FFN overlaps the other mb's Attn. Disabled when TBO is active
   or `SGLANG_AFD_PIPELINE=1`. Forces decode CUDA Graph **disabled**. Best when FFN RTT
   is large; on low-latency `cuda_ipc` leave off by default.
5. **P8 in-graph wait** — `SGLANG_AFD_IN_GRAPH_WAIT=1` forces decode CG **full**, drops
   `@eager_on_graph` breaks, and uses fserver `write_flag`/`wait_flag` inside the graph;
   a CPU thread runs transport `push_pull`/`wait` (cuda_ipc or stepmesh). Mutually
   exclusive with P7.
6. **P9 AfPool** — optional MxN Attn↔FFN work pool (`SGLANG_AFD_POOL=1`) with per-pair
   cuda_ipc endpoints, credit windows, and least-inflight routing. KPI is tok/s + FFN
   util under multi-request load. Default off; classic 1A1F TRUE_OVERLAP unchanged.
   See `python/sglang/srt/afd/pool/README.md`.

## 12. Testing

1. Unit: FakeTransport, buffers, pipeline, layer pipeline, PD policy, stubs, scheme B, FP8 roundtrip, AfPool credit/router.
2. `python -m sglang.srt.afd.smoke`
3. StepMesh: `tests/fserver` + `bringup_stepmesh_example.sh` on RDMA hosts.
4. AfPool: `python -m sglang.srt.afd.bench_af_pool --compare-1a1f` (real cuda_ipc).
