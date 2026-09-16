# AFD decode farm — measured results (do not re-measure)

Scope: **only** results from the current farm implementation. Older numbers from
removed/incorrect implementations are explicitly marked as void at the bottom.

Harness: `bench_farm_e2e.sh`, DeepSeek-V2-Lite-Chat, 1A+1F cuda_ipc,
`MODES=sticky`, CG off (`--cuda-graph-backend-decode disabled`,
`SGLANG_AFD_FFN_CUDA_GRAPH=0`), conc=8, in=128/out=64, mem-fraction=0.75.

---

## 1. Top-level TPOT breakdown (2 reps, B_STEP=8, MAX_INFLIGHT=2)

`SGLANG_AFD_FARM_STAGE_STATS_EVERY=1` + `SGLANG_AFD_FARM_PHASE_TIMING=1`

```
STAGE  forwards=1 hops/fwd=27.0 issue_ok=27.0 issue_nocredit=0.0
       pending_peak=1 layers_peak=1/27 span_peak=0 nseq_max=8
       avg_inflight=0.54-0.58  hop_lat_p50=2.97-3.08ms p90=3.29-3.35ms
       | wall=141.4-150.1ms/head  pre=1.51-1.84ms/call(27)  block=2.92-3.08ms/call(27)
PHASE  loop=146.9ms  cpu_sum=135.0ms  unattributed=11.9ms
       pre=1.68ms/call  issue=0.13ms/call  drain_poll=0.00  drain_block=2.97ms/call  consume=0.10ms/call
```

| Component | per layer | ×27 | share |
|-----------|----------:|----:|------:|
| `block` (wait for FFN hop) | 2.97 ms | **80 ms** | **55%** |
| `pre` (local Attn) | 1.60 ms | 43 ms | 30% |
| unattributed (Python) | — | 12 ms | 8% |
| `issue` + `consume` | 0.23 ms | 6 ms | 4% |
| **total** | | **~141 ms** | |

`hops/fwd = 27` (= layer count) and `layers_peak=1/27` in **every** farm config
measured. `issue_nocredit=0`, `layer_cap_blocks=0`, `gl_cap_blocks=0`,
`tok_blocks=0`, `pending_peak=1`, `avg_inflight≈0.57`, `mean_win_len=1.00`,
`picks == layer_switches == 3456` — no bound ever fired, the scheduler's
concurrency machinery is idle.

---

## 2. Where the hop cost actually goes (the key measurement)

`SGLANG_AFD_PROFILE_DETAIL=1` + `SGLANG_AFD_FARM_FFN_SECTION_TIME=1`,
`FFN` role log, 2 reps. All values µs, p50.

| metric | rep1 | rep2 | note |
|--------|-----:|-----:|------|
| `a2f_sync_us` | **4** | **4** | event sync is free → **not sleep-bound** |
| `post_poll_us` | **0** | **0** | never recorded → **not poll-bound** |
| `compute_wall_us` | 2306 | 2325 | host wall around FFN compute |
| `compute_cuda_us` | 2454 | 2483 | wall + cuda synchronize |
| `ffn_prep_us` | 144 | 145 | topk/tensor prep |
| **`ffn_routed_us`** | **1616** | **1607** | **dispatch + core + combine — 70% of the hop** |
| `ffn_shared_us` | 0 | 0 | fused into routed |
| `ffn_fuse_us` | 8 | 9 | |
| `ffn_post_us` | 1 | 1 | |
| `respond_us` | 172 | 175 | F2A copy + set_done |

`ffn_compute_split: wall_mean=2358us cuda_sync_mean=2512us launch_overhead≈155us`
→ `verdict: COMPUTE_OR_LAUNCH`.

So of the ~2.8 ms hop: **`ffn_routed` ≈ 1.61 ms (57%)**, `respond` ≈ 0.18 ms,
`compute_cuda − compute_wall` ≈ 0.155 ms. The rest is elsewhere in the hop.

### 2b. Discriminator: is `ffn_routed` host-bound or GPU-bound?

Same harness, `B_STEP` 2 vs 16 (2.3× tokens per hop), `MAX_INFLIGHT` 8 vs 2.

| | `B_STEP=2` | `B_STEP=16` |
|---|---:|---:|
| `mean_tok/launch` | **1.9** | **4.4** |
| `ffn_routed_us` p50 | **1657** | **1628** |
| `ffn_prep_us` p50 | 145 | 145 |
| `compute_wall_us` p50 | 2350 | 2319 |
| `compute_cuda_us` p50 | 2503 | 2469 |
| `respond_us` p50 | 174 | 173 |
| `hop_lat_p50` | 3.02 ms | 3.04 ms |
| `wall` | 165.2 ms | 139.9 ms |

**2.3× more tokens per hop changes nothing (−1.8%).** `ffn_routed` is therefore
**host/dispatch-bound and batch-independent**, not GPU-bound. Combined with
§2: the hop is not sleep-bound, not poll-bound, not GPU-bound — it is
host-side dispatch cost. This is why every batching/scheduling knob measured
flat, and why pipelining loses (it multiplies this fixed cost).

---

## 3. Attn-side breakdown (`pre` = 1.6-1.8 ms/layer)

`SGLANG_AFD_PROFILE_DETAIL=1`, attn log, rep2, p50 µs.

| sub-span | µs | note |
|----------|----:|------|
| `attn_core_us` (MLA) | **1016** | |
| `mla_attn_block_us` | **311** | 2×`torch.cat` + `attn_mqa` + KV store |
| `mla_qproj_us` | **217** | (= q 74 + kv 40 + knorm 76) |
| `mla_prepare_us` (bracket) | 417 | forward_absorb_prepare |
| `mla_core_us` (bracket) | 480 | forward_absorb_core |
| `attn_route_us` | 345 | gate + topk |
| ├ `attn_gate_us` | 51 | MoE gate GEMM |
| └ `attn_topk_us` | 166 | after the fused-topk fix (was 495) |
| `mla_vbmm_us` | 80 | |
| `mla_knorm_us` | 76 | kv_a_layernorm alone |
| `mla_qnope_bmm_us` | 61 | |
| `mla_oproj_us` | 57 | |
| `mla_split_pe_us` | 36 | |
| `attn_prep_us` | 43 | |
| `attn_prepmlp_us` | 36 | |

`mla_qk_norm_us` / `mla_qb_proj_us` are 0 — not on the no-LoRA path.
Note `attn_core` 1016 µs + `attn_route` 345 µs + prep ≈ 1.6 ms/`pre` call,
consistent with §1.

---

## 4. Pipelining (staggered 1F1B entry)

`SGLANG_AFD_FARM_STAGGER_MB` / `..._SPAN`, B_STEP=8, MAX_INFLIGHT=8, CG off.

| Config | hops/fwd | `layers_peak` | wall | tok/s | TPOT |
|--------|---------:|--------------:|-----:|------:|-----:|
| no stagger | 27 | **1**/27 | 131 ms | 38.6 | **141.9 ms** |
| `STAGGER_MB=4`, span 2 | **61–92** | **2**/27 | 189–460 ms | 22.0 | **267.6 ms** |

Overlap is real but loses: the hop count multiplies the fixed per-hop cost from
§2. Keep `STAGGER_MB=0` until the hop is cheap.

### Why overlap was structurally impossible before

`begin_batch()` enqueued **every** sequence at layer 0; `complete()` re-enqueued
that **same set** at `layer+1`. The wavefront is flat by construction — all
tokens share one layer at all times — so `layers_peak` can never exceed 1
regardless of `B_STEP`, per-layer cap or `SCHED`. Fixed by `enqueue_entry()` +
`deepest_active_layer()`.

---

## 5. Config sweeps — all flat

| Config | hops/fwd | `layers_peak` | tok/s | TPOT |
|--------|---------:|--------------:|------:|-----:|
| base (B=8, mi=2) | 27 | 1/27 | 37.3 | 147.9 ms |
| B=4, mi=4 | 27 | 1/27 | 30.7 | 196.2 ms |
| B=4, mi=4, natural_batch | 27 | 1/27 | 31.5 | 189.7 ms |
| B=2, mi=8, natural_batch | 27 | 1/27 | 16.4 | 428.6 ms |
| B=1, cap=1, `sched=deepest` | 27 | 1/27 | 12.9 | 578.0 ms |
| B=1, cap=1, `sched=max` | 27 | 1/27 | 12.7 | 591.2 ms |
| B=2, cap=1, `sched=deepest` | 27 | 1/27 | 21.8 | 317.3 ms |

`hops/fwd` is pinned at 27 and `layers_peak` at 1 in **all** of them.

---

## 6. Attn-side wins already landed

| Change | Effect | Where |
|--------|--------|-------|
| degenerate grouped-topk fused | **+2.8% tok/s / −3.3% TPOT** (3 reps, CG off) | `SGLANG_OPT_DEGENERATE_GROUPED_TOPK_FUSED`, default on |
| QKV no-LoRA fusion | +1.8% tok/s / −2.2% TPOT, **under-sampled** (on-arm ±4.4 ms vs off ±0.25 ms) | `SGLANG_OPT_FUSE_QKV_A_PROJ_NOLORA`, default off |

---

## 7. Void — do NOT cite

Removed or incorrect implementations; numbers from these are not comparable:

- `/tmp/afd_layer_merge_bench/*` (Aug 30): the `k1` and `k2` JSONs are
  byte-identical, so the `merge_k=2` arm never ran; the whole run predates
  current farm code. `LAYER_MERGE_K` is **not** part of the farm path.
- `/tmp/afd_farm_e2e_high` `109 tok/s`: no concurrency recorded, incompatible
  with its own `MAX_CONCURRENCY=8`.
- Any claim that `IN_GRAPH_WAIT` / `FFN_CUDA_GRAPH=1` gives 142→30 ms: that
  came from the void Aug-30 config.

---

## 8. Open question (not yet measured)

> **RESOLVED in §13.** The census was blocked by the probe profiling its first,
> cold call; it now warms up first. Answer: 6 kernels, only ~6.6% of the cost is
> launch-related, and the FFN process is **CPU-bound at 2.2x** (738us GPU vs
> 1628us CPU per hop). Also note the CG flag was hardcoded to 0 in
> `bench_pool_e2e.sh`, so the "FFN_CUDA_GRAPH=1 does not help" line below was
> never actually measured; genuinely enabled it is *worse* (TPOT 319.7 ms).

`ffn_routed` (~1.6 ms) is host-bound and batch-independent, but the *specific*
host cost is not yet split between kernel-launch overhead (CUDA-graph-fixable)
and Python/grouped-GEMM dispatch (not graph-fixable). The existing
`FFN_CUDA_GRAPH=1` was measured **not** to reduce hop cost, so it does not
currently cover this path. One more probe (launch-count / nsys on `ffn_routed`)
decides which fix applies.

Harnesses: `bench_hop_prof.sh` (§1-3), `bench_hop_scale.sh` (§2b),
`bench_pipe_stagger.sh` (§4), `bench_pipe_sweep.sh` (§5).

---

## 9. Why only ONE layer is ever busy (the structural answer)

Not a scheduler deficiency — the **unit of progress is a hop, not a request**.

1. `begin_batch(range(n_seq))` → `queues.enqueue(0, idxs)` (`scheduler.py:239`):
   **all** sequences enter layer 0 together.
2. `pick()` → `run = take_contiguous_run(self.ready[li], max_tok)`
   (`token_queue.py:319`): takes from **one layer only**, up to
   `max_tok = b_step × coalesce_k`. With `b_step=8` and a real batch of ~4,
   this takes **everything on that layer**.
3. `complete()` → `queues.enqueue(nxt, ticket.seq_idxs)` (`scheduler.py:386`):
   re-enqueues **the same window** at `layer+1`.

So the whole batch is welded into one hop that moves layer to layer:

```
{A,B,C,D}@L0 → hop → {A,B,C,D}@L1 → hop → {A,B,C,D}@L2 → ...
```

Every layer therefore has exactly one hop in flight. This is the source of
`picks == layer_switches == 3456`, `mean_win_len=1.00`, `pending_peak=1`,
`hops/fwd=27`. "Request B waits at the previous layer" never happens because
request B is never *left behind* — it is inside the same hop as everyone else.

Implicit premise: within one `model.forward()` the batch is fixed; no new
request arrives mid-forward.

### Is the user's idea (let B enter the previous layer) possible?

Yes, in two forms. **They are NOT equivalent: (a) multiplies hops, (b) does not.**

| | (a) intra-step split (`STAGGER_MB`, implemented) | (b) cross-decode-step (P1.4, not done) |
|---|---|---|
| mechanism | split ONE step's batch into G groups, inject at staggered layers | keep each batch whole; let step N's tail coexist with step N+1's head |
| batch internal state | **fragmented** into G pieces | **stays fused** (all tokens share one hop) |
| hops per step | **G × 27** (measured 61–92) | **27 — unchanged** |
| hop count over 2 steps | 2 × G × 27 | 2 × 27 = 54 (same as sequential) |
| overlap source | groups at different layers *within* a step | different *batches* at different layers |
| measured | TPOT 142 → 268 ms (**worse**) | — |

Why (b) does not multiply hops: at any instant batch N occupies layer ~20 and
batch N+1 occupies layer ~0, each still emitting **one** hop per layer carrying
its **whole** batch. Over two steps the aggregate hop count is unchanged; the
only difference is that FFN work and Attn work now overlap instead of
alternating.

**Do not conflate these.** Only (a) pays the fixed-per-hop multiplication.

### What (b) would buy

Today's step is serial: `27 × (attn + hop) = 27 × (1.60 + 2.97) ≈ 123 ms`, with
the Attn GPU idle ~65% of the time. Perfectly overlapped, the period becomes

```
max(27 × attn, 27 × hop) = max(43 ms, 80 ms) = 80 ms     (~1.5× throughput)
```

This does **not** require a cheaper hop — it only requires two batches to be
alive at once, i.e. exactly the runner change (see §10). Contention may raise
hop latency (measured 2.97 → 4.5 ms under concurrency), so the gain is
empirical, not guaranteed.

### Stronger consequence for TPOT

For a **fixed** batch, a single token's chain is strictly serial:
`attn(L) → hop(L) → attn(L+1) → hop(L+1) → …` (hop L produces the input of
attn L+1). And token N+1 needs token N sampled first. So

```
TPOT = 27 × (attn + hop) + overhead      (a hard, serial lower bound)
     = 27 × (1.60 + 2.97) + ~12 ms = 141 ms   ← matches measured 141.9 ms
```

**Pipeline overlap between *different* tokens raises throughput; it cannot lower
per-sequence TPOT**, because the per-token chain is a data dependency. To lower
TPOT you must shorten the chain (fewer/cheaper hops, cheaper attn) or produce
more tokens per traversal (speculative decoding, `TPOT ≈ chain / K`).

---

## 10. Why P1.4 (cross-step overlap) is blocked by the *runner*, not the scheduler

`run_farm_layers` is called from `deepseek_v2.py:2797`; the three blockers:

**(1) Queue entries are bare row indices into one forward's tensors.**
An entry identifies rows of `hidden_states`/`residual` for *the* forward it was
sliced from. If two forwards' work coexist in the queues, an index is
ambiguous — it would slice the wrong rows. Needs a `(batch, index)` tag plus a
per-batch tensor context.

**(2) `run_farm_layers` is a barrier, so there is no "between two steps" moment.**
`decode_farm_loop.py` runs `while finished < n_seq:` and then, before returning:

```python
while pending:
    _block_one()
if not scheduler.finish_batch():
    raise RuntimeError(...)
```

It returns only once **every** token has exited layer 26. Immediately after it
returns, `self.norm(hidden_states, residual)` runs (`deepseek_v2.py:2899`) and
the runner computes logits and samples. Control flow is one straight line:

```
farm(barrier) ─→ norm ─→ lm_head ─→ logits ─→ sample ─→ build next step
```

There is **nowhere to park** step N's unfinished hops, and no point at which
step N's tail can stay in flight while step N+1 begins. The "OutputBarrier /
deferred consumer" is the missing place to park them.

**(3) Sampling for step N must wait for step N to drain.**
A direct consequence of (2): logits need layer 26's output for the token, so
sampling cannot run until that token is complete; and step N+1 cannot start
until sampling produced its input token. Both are hard dependencies, so the
gap is zero — today's behaviour.

`MAX_ACTIVE_BATCHES` is pinned to 1.

---

## 11. Persistent farm runtime (P1.4) — implemented, validated, and *losing*

Design implemented as specified: process-level `PersistentFarmRuntime`, one
context per `req_pool_idx`, contexts and pending A2F hops survive across
`model.forward()`, a forward may sample a subset or zero rows, plus the
`ready/deferred_req_pool_indices` protocol through `GenerationBatchResult`,
`tp_worker`, `prepare_for_decode` / `alloc_for_decode` (ready-only KV),
`batch_result_processor`, and `deepseek_v2`. Gated by
`SGLANG_AFD_FARM_PERSISTENT`; `=0` keeps the one-shot farm byte-for-byte.
Unit tests: `sglang/srt/afd/farm/test_persistent_runtime.py` (14 pass).

### The mechanism works — the design's targets are met

From `AFD farm STAGE` (new fields: `rows`, `live_peak`, `reserve_none`,
`hop_none`, `sched_peaks=run/res/sctx`, `capblk/fwd`), steady state at
`rows≈10-16`, `SGLANG_AFD_FARM_MAX_INFLIGHT=32`:

- **Contexts live across forwards.** `live_peak=11` with `rows` oscillating
  8→16→3; contexts are adopted, sampled and dropped across many forwards
  instead of all dying inside one forward.
- **Different contexts sit at different layers at the same time.**
  `layers_peak=4`, `span_peak=26` (i.e. layer 0 and layer 26 pending in the
  same instant) with `ctx_span_peak=26`. This is the thing the one-shot farm
  can never do.
- **Zero-ready forwards happen.** `hops/fwd=0.4-0.5` during ramp = a forward
  that issues/consumes nothing and returns an empty ready set, contexts kept.
- **Pending hops cross the forward boundary.** `pending_peak=7-8` with
  `reserve_none=1.0` (the issue loop stops because no context has an eligible
  layer, not because of the cap) and `hop_none=0.0`.
- `capblk/fwd=0.00/0.00` — neither the global nor the per-layer cap blocks.

### But it is 2.8x slower — matched A/B, same host/config

16 prompts, `max_concurrency=16`, in 128 / out 32, `1a1f` (attn gpu7, ffn gpu4),
`MAX_INFLIGHT=32`, `MAX_INFLIGHT_PER_LAYER=32`, CG off:

| mode | out tok/s | med TPOT | p99 TPOT | med E2E |
|---|---|---|---|---|
| one-shot (`PERSISTENT=0`) | **44.5** | **153.9 ms** | 206.8 | 3074 |
| persistent (`PERSISTENT=1`) | 20.7 | 435.9 ms | 644.4 | 7499 |

Same picture at 24 prompts / out 64: 39.7 tok/s / 167.8 ms vs 19.1 tok/s / 402.2 ms.

### Root cause: the FFN *call rate* is constant, and per-context hops carry 1 token

`hops/fwd` is pinned at **4.1-4.2 regardless of `rows` (8→16) and
regardless of `max_inflight` (8→32)**. That is a hard completion ceiling, not a
credit ceiling (`reserve_none=1.0`, `capblk/fwd=0`). With ~7.3 ms/forward it is
≈**560 A2F calls/s**.

- one-shot: 27 calls per token-step carrying the whole batch (16 tokens)
  → `27 / 153.9 ms = 175 calls/s x 16 tok = 2808 token-layers/s`.
- persistent: ≈560 calls/s x **1 token per call** → `560 token-layers/s`.

The gap is the *payload*, not the overlap. The one-shot farm already coalesces
all rows that sit at a layer into one A2F (`_send_queue_key` keys by
`layer_id`); in persistent mode each context is at its own layer, so the
average group is ~0.6 tokens. Per-call cost is ~1.8 ms host dispatch in the FFN
process (the `inplace_fused_experts` CPU-side op the CG-off probe already
identified) — i.e. **constant in token count**. Spreading contexts over 27
layers multiplies the number of calls by ~27x per token-step while shrinking
each call to 1 token.

Corollary that kills the obvious "fixes":

- More in-flight credit does **not** help (`max_inf=32` → `hops/fwd` unchanged).
- Staggering/2-group pipelining also loses: with `cost ≈ 2.8 ms + 0.21 ms/token`,
  two groups of 8 cost `54 x 4.5 ms / 2 = 121 ms` vs one group's `27 x 6.2 ms
  = 81 ms`. Splitting a batched call is a net loss as long as the fixed
  per-call term dominates.

### What this implies

1. `PERSISTENT=0` (one-shot farm) stays the default; the persistent path is
   correct but uneconomic under CG-off and should not be enabled for perf.
   *(Superseded by §12: with `PERSISTENT_GROUPS=1` the persistent path is the
   faster one. Only the one-row-per-context form is uneconomic.)*
2. The persistent design only wins if hops stay **fat**: contexts must own a
   *group* of rows (not one row), so a hop carries many tokens. That is a
   hybrid — cross-forward state as implemented, but `Context = N rows at one
   layer` instead of `Context = 1 row`. Overlap then comes from having 2-4 such
   groups at different layers, keeping call size ≥ 4-8.
3. Alternatively, attack the ~1.8 ms/call FFN host dispatch — the same CG
   target already identified — because that term is the whole gap.

<a id="s12"></a>
## 12. Hybrid grouping (`Context = N rows`) — implemented and measured

Implements the recommendation at the end of §11: the persistent context now
owns a **group of rows** at one layer instead of a single row, so an A2F hop
carries many tokens while several groups can still sit at different layers.
Gated by `SGLANG_AFD_FARM_PERSISTENT_GROUPS` (`G`): `G=0` keeps the one-row
behaviour of §11, `G=N` targets `ceil(rows/N)` groups.

### Bug found and fixed while landing it

`FarmQueue.take_contiguous_run` identifies a run by **consecutive** indices, but
adoption passed `req_pool_idx` values (sparse: 3, 9, 17...). Every multi-row
group was therefore split into several tickets, so one context's rows advanced
on independent tickets and its `hidden` got overwritten with a different row
count — the "sampler row count does not match ready tokens" / server-crash
symptom of the first sweep.

Fix, in `persistent_runtime.py` + `decode_farm_loop.py`:

- `runtime.alloc_queue_idxs(n)` hands out a dense, monotonically growing block
  of scheduler indices, independent of the sparse `req_pool_idx` used as the
  cross-forward identity.
- `_issue_round` reserves `max(group_size, max(ctx.n_tokens for live ctxs))` so
  a ticket can never truncate a live group.
- new test `test_queue_idxs_are_dense_and_never_split_a_group` (18 pass total).

### Measured: fat contexts win, and *any* splitting loses

16 prompts, `max_concurrency=16`, in 128 / out 32, `1a1f`, `MAX_INFLIGHT=32`,
`MAX_INFLIGHT_PER_LAYER=32`, B_step=16, coalesce_k=4, CG off, same host.

| mode | groups | rows/ctx | out tok/s | med TPOT | p99 TPOT |
|---|---|---|---|---|---|
| persistent | 16 (G=0, 1 row) | 1 | 20.7 | 435.9 ms | 644.4 |
| persistent | 4 | 4 | 30.0 | 257.3 ms | 267.9 |
| persistent | 2 | 8 | 40.2 | 183.8 ms | 194.5 |
| **persistent** | **1** | **16** | **47.9-48.5** | **141.8-145.9 ms** | 153.8-194.5 |
| one-shot (`PERSISTENT=0`) | – | 16 | 41.4-44.9 | 151.8-166.8 ms | 205.9-208.0 |

Two independent A/B pairs, same script/config, run back to back:

| pair | one-shot TPOT | persistent G=1 TPOT | Δ |
|---|---|---|---|
| 1 | 166.8 ms / 41.4 tok/s | 141.8 ms / 48.5 tok/s | **-15.0% / +17%** |
| 2 | 151.8 ms / 44.9 tok/s | 145.9 ms / 47.9 tok/s | **-3.9% / +6.7%** |
| mean of 3 baseline vs 2 G=1 runs | 157.5 ms | 143.9 ms | -8.7% |

### Overlap *is* achieved — and it is what makes it slower

`AFD farm STAGE`, steady state:

| G | `ctxs_peak` | `layers_peak` | `ctx_span_peak` | `hops/fwd` |
|---|---|---|---|---|
| 1 | 1 | 1/27 | 0 | 0.4-0.5 |
| 2 | 2 | 2/27 | 0-1 | 1.0 |
| 4 | 4 | 2/27 | 3 | 2.0-2.5 |

`G=4` genuinely puts 4 contexts in flight, two of them busy at different layers
in the same instant, with a 3-layer span between oldest and newest — exactly the
structure §11 said was missing. It is also **1.8x slower** than `G=1`.

### The cost model, now measured end to end

Fitting `TPOT ≈ a + b x (#hops)` with `#hops = 27 x groups`:

- 27 hops (G=1) → 143.9 ms
- 54 hops (G=2) → 183.8 ms
- 108 hops (G=4) → 257.3 ms
- `b ≈ 1.5 ms/hop`, `a ≈ 100 ms` (139.8/183.8/262.6 predicted vs 143.9/183.8/257.3
  measured)

> **Naming.** `b` here is the *per-hop fixed cost* — the slope of
> TPOT-vs-hop-count. It is **not** batch size and **not** the wall time of a
> single hop (~2.8 ms, §10/§11; that figure includes queueing/round-trip that
> interleaves when a second group is added, so it overstates the marginal
> cost). Written with an explicit per-token term it is clearer as `f`:
> `TPOT(G) ≈ base + 27·G·f + 432·v`, where `f` is the fixed cost per FFN call
> and `v` the marginal cost per token. The 3-point sweep identifies only the
> combination `a = base + 432·v ≈ 100 ms` and `f ≈ 1.5 ms`; splitting `a`
> further needs an independent estimate of `v`.

So TPOT is a **linear function of the hop count**, and the hop price is
`~1.5 ms` — the constant FFN-side host dispatch already isolated in §10/§11
(`inplace_fused_experts` is host-bound and batch-independent). Overlap buys
`max(C_ffn, C_attn)` while paying `G x (#hops)`, so it can only win once the
per-hop constant is small relative to per-hop *work*. At `~1.5 ms` of pure
dispatch against `~1.1 ms` of marginal GPU work for a 16-token hop, splitting
loses for every `G >= 2`, and the measurement agrees.

### Conclusions

1. **Persistent `G=1` is the new best CG-off farm config**: ~144 ms vs ~157 ms
   one-shot (-8.7%), consistent in sign across all runs. It wins *without*
   overlap — by removing the intra-forward `while finished < n_seq` barrier, not
   by pipelining.
2. **True pipelining is implemented and verified, and it is the wrong trade
   under CG-off.** `layers_peak>1` is real; it just costs more hops than it
   saves. §11's structural complaint was correct, but fixing it does not pay.
3. The remaining lever is unchanged and is the same one §10/§11 pointed at: the
   ~1.5 ms/call FFN host dispatch is both the per-hop price and the reason
   splitting loses. Note that shrinking `b` helps `G=1` too, so it does not by
   itself make pipelining pay — `G=2` only wins if overlap removes part of the
   **base** term `a` (i.e. if part of the 1.5 ms is idle round-trip rather than
   busy FFN dispatch). The probe in §10 says it is busy dispatch
   (`inplace_fused_experts` on the CPU), which is why overlap loses here.
4. Reproduce: `PERSISTENT=1 PERSISTENT_GROUPS=1` is the recommended setting;
   `PERSISTENT=0` still reproduces §11 byte-for-byte.

<a id="s13"></a>
## 13. Answering §8: the FFN hop is **CPU-bound inside the FFN process** (census done)

`SGLANG_AFD_FARM_FFN_PROBE=1` + new `SGLANG_AFD_FARM_FFN_PROBE_WARMUP=300`
(the probe used to census the very first, cold call — that was §8's blocker; it
now skips N calls first). 16 prompts, in 128 / out 32, 1a1f, CG off.

```
[ffn-probe] ffn_routed wall=45006us kernels=6 gpu_kernel_total=738us
            cpu_total=1628us cpu_per_launch=271.4us
[ffn-probe] gpu fused_moe_kernel                     473.5us
[ffn-probe] gpu fused_moe_kernel                     243.4us
[ffn-probe] gpu act_and_mul_kernel<...bf16>            7.0us
[ffn-probe] gpu moe_sum_reduce_kernel<...>             5.8us
[ffn-probe] gpu moe_align_block_size_kernel<int>       5.2us
[ffn-probe] gpu count_and_sort_expert_tokens_kernel    3.1us
[ffn-probe] cpu sglang::inplace_fused_experts        840.1us x1 (cuda 716.9)
[ffn-probe] cpu sglang::_run_activation_inplace      263.8us x1 (cuda   7.0)
[ffn-probe] cpu aten::view                           243.4us x4 (cuda   0.0)
[ffn-probe] cpu sgl_kernel::moe_align_block_size      98.2us x1 (cuda   8.3)
[ffn-probe] cpu aten::empty                           46.3us x7 (cuda   0.0)
[ffn-probe] cpu cudaLaunchKernel                      35.6us x3
[ffn-probe] cpu cudaDeviceSynchronize                 25.5us x2
[ffn-probe] cpu cuLaunchKernelEx                      19.1us x2
[ffn-probe] cpu cudaStreamWaitEvent                   13.2us x11
[ffn-probe] cpu cudaEventRecordWithFlags              12.4us x10
```

`wall` is inflated ~28x by the profiler itself — ignore it. `cpu_total=1628us`
is trustworthy because it matches the independent, profiler-free
`ffn_routed_us` = 1616/1607us (§2) within 1%. Two methods, same number.

### What the census settles

1. **It is not launch-syscall-bound.** Only **6 kernels**; all launch-related CPU
   (`cudaLaunchKernel` + `cuLaunchKernelEx` + event/stream ops) is **~107us =
   6.6%**. The kernels are 3 Triton (`fused_moe_kernel` x2,
   `moe_align_block_size`, `count_and_sort_expert_tokens`) + sgl C++ elementwise.
2. **GPU work per hop is only 738us while CPU is 1628us** → the FFN process is
   **CPU-bound at 2.2x**. This is the single cleanest explanation of every
   negative result so far: an overlapped, CPU-bound stage cannot be sped up by
   more overlap, and splitting a hop only adds more of the *binding* resource.
   `max(C_ffn, C_attn) = C_ffn` ⇒ pipelining can only add hops, never win.
3. The CPU splits as: **`inplace_fused_experts` 840us (52%)**,
   `_run_activation_inplace` 264us (16%), `aten::view` x4 243us (15%),
   `moe_align_block_size` 98us (6%), `aten::empty` x7 46us (3%).
   `aten::view` costing **61us each** and `_run_activation_inplace` costing
   **264us of CPU for 7us of GPU** are both pathological.

### §8's "CUDA graph doesn't help" was measured against a flag hardcoded to 0

`bench_pool_e2e.sh` did `export SGLANG_AFD_FFN_CUDA_GRAPH=0` unconditionally,
clobbering any external override — so the FFN CG path was never exercised.
Fixed to `"${SGLANG_AFD_FFN_CUDA_GRAPH:-0}"`.

With it genuinely on, the graph **does** engage:

```
AFD FFN: FFN compute graphs via SGLANG_AFD_FFN_CUDA_GRAPH=True
AFD FFN CUDA graph captured layer=0 bs=8 mb=0 topk=False mode=private
AFD FFN CUDA graph captured layer=1 bs=8 mb=1 topk=True  mode=private
```

and it is **much worse**: out 21.2 tok/s, med TPOT **319.7 ms** (vs 166.8 ms
eager), med TTFT 1926 ms (vs 691 ms), p99 TPOT 573 ms. Graphs are captured
per `(layer, mb)` at a fixed `bs=8`, but farm hops have a *variable* token count,
so most hops miss the bucket and pay graph lookup + fallback, and the capture
warms 27x2 graphs. Conclusion: the CG *concept* targets the right cost (it
removes exactly the 52%+16%+15% Python/dispatcher items) but this integration is
not usable for the farm.

### Ranked options to attack `f`

| # | lever | covers | risk |
|---|---|---|---|
| 1 | Fix the FFN CG integration for variable-size hops (single bucket padded to `max_num_token`, or per-`ceil(tokens)` buckets; keep the graph resident) | up to ~85% of `f` | medium; must not fall back to eager |
| 2 | Fold `_run_activation_inplace` into the MoE epilogue — `fused_experts_impl` already takes `activation`/`is_gated` | 16% | low |
| 3 | Kill the 4 `aten::view` / 7 `aten::empty` per call (pre-allocated persistent buffers, no reshapes) | ~18% (243+46us) | low |
| 4 | Batch/hide `moe_align_block_size` | 6% | low |
| 5 | ~~CUDA graph for launch overhead~~ | **6.6% ceiling** | not the lever |

Do **not** expect scheduling work (more groups, staggering, caps) to recover any
of this: the stage is CPU-bound, so extra hops are pure addition. Reproduce with
`bench_ffn_probe.sh` + `SGLANG_AFD_FARM_FFN_PROBE_WARMUP=300`.

## 14. Acting on §13 items #2/#3: the census over-attributed, but the FFN is GIL-bound

§13 ranked the levers from a `torch.profiler` census (self-CPU per op). This
section re-derives them with **per-call wall timers placed inside the MoE call
path**, isolates each item, and then settles whether host work matters at all.
Everything below is CG-off, 1A1F, `PERSISTENT=0`, `B_STEP=16`, 16 prompts.

### 14.1 A wall-clock breakdown of `ffn_routed_us`

New sub-section timers (`moe_*` keys, gated by `SGLANG_AFD_PROFILE_DETAIL=1`
plus `SGLANG_AFD_FARM_FFN_SECTION_TIME=1`) give, per hop, p50:

| level | p50 us |
|---|---|
| `ffn_routed_us` | 1698 |
| `moe_fx_us` (`fused_experts` + custom-op dispatch + impl) | 1650 |
| `moe_apply_us` (`quant_method.apply`) | 1691 |
| `moe_core_us` | 1705 |
| `moe_disp_us` (dispatcher) | 5 |
| `moe_comb_us` | 24 |

and *inside* the kernel sequence:

| item | p50 us | note |
|---|---|---|
| `moe_cfgsel_us` (config pick) | **11** | `get_moe_configs` is already `lru_cache`d, so §13's `b` was wrong here |
| `moe_align_us` (`moe_align_block_size`) | **300** | 4× `torch.empty` + one custom op |
| `moe_alloc_us` (cache1/out) | 17 | |
| `moe_k1_us` (gate_up launch) | 94 | |
| `moe_act_us` (activation) | **406** | |
| `moe_k2_us` (down launch) | 95 | |

`moe_cfg_us` = 336 = `moe_cfgsel_us` + `moe_align_us` + asserts, so the config
*file lookup* that §13 blamed is not a cost at all; `moe_align_block_size` is.

### 14.2 Items #2 and #3 are not worth the code

Isolated microbenchmarks on an idle GPU (`bench_act_micro.py`,
`bench_align_micro.py`):

```
  14.0 us  jit silu_and_mul(x.view, out)      <- #2 whole call, incl. 2 views
  12.1 us  jit silu_and_mul(preflat, out)
  11.4 us  jit raw _run_activation_inplace
   6.9 us  sgl_kernel silu_and_mul(x, out)
   1.3 us  noop view
  28.9 us  moe_align_block_size rows=16 topk=6 E=64
  28.8 us  moe_align_block_size rows=96 topk=6 E=64
  15.3 us  4x torch.empty (floor)
```

So §13's numbers were profiler-inflated: the whole activation dispatch is
**14 us**, and it is *batch-independent* (28.8→29.0 us as rows go 16→128),
confirming §2b. `sgl_kernel` would save 7 us. The `aten::empty` item (#3) is
worth ~15 us. Against a 1700 us hop these are 1% each — implemented only as the
free part (19 redundant `intermediate_cache1.view(-1, N)` reshapes replaced by a
`_flat2d` no-op guard).

### 14.3 What the in-situ numbers really are: GIL contention, not work

Every §14.1 item that **launches a CUDA kernel** reads ~10-30x its isolated
cost, while every item that does **not** (`moe_cfgsel_us` 11 us, `moe_disp_us`
5 us, `moe_comb_us` 24 us) reads normal. Two independent checks say the process
is the constraint:

- `top -H` during decode: the FFN process sits at **99% of one core** (128 cores,
  load ~37, 70% idle — not OS starvation). A pure-Python process cannot exceed
  100% because of the GIL, so the FFN host path is **GIL-serialised onto one
  core**.
- `ffn_routed_cpu_us` (`time.process_time`, added to `moe_bridge.py`) vs
  `ffn_routed_us` (`perf_counter`): **1930 us vs 1698 us**. Process CPU *exceeds*
  wall, i.e. another thread (the transport poll loop) is burning CPU in the same
  window and competing for the GIL.

So the hop's wall time is not the sum of the ops' isolated costs; it is
single-core CPU time plus **GIL handoff delay around every GIL-releasing CUDA
call**. That is why folding one 14 us activation cannot fix a 1700 us hop, and
why the lever is *how much Python holds the GIL per hop*, not which kernel is
50 us faster.

### 14.4 A lever that does pay: the MoE combine

`_fused_moe_kernel_sequence` reduces the down-projection output with
`moe_sum_reduce_torch_compile` whenever `num_tokens <= 32` — which the farm
always is (~4-16 tokens/hop). That is a `@torch.compile` function, and even with
`TORCHDYNAMO_DISABLE=1` (already set by `bench_pool_e2e.sh`) its compiled stub
lowers to `torch.sum` + `out.mul_` + reshapes instead of one fused kernel.

`bench_combine_micro.py`:

```
rows=  4 topk=6 dim=2048: torch.compile= 45.7us  sgl_kernel= 5.4us  ratio= 8.5x
rows=  8 topk=6 dim=2048: torch.compile= 49.9us  sgl_kernel= 5.3us  ratio= 9.4x
rows= 16 topk=6 dim=2048: torch.compile= 49.8us  sgl_kernel= 5.3us  ratio= 9.3x
rows= 64 topk=6 dim=2048: torch.compile= 49.6us  sgl_kernel= 5.4us  ratio= 9.2x
```

New knob `SGLANG_AFD_MOE_SUM_REDUCE_COMPILE` (default `1` = stock upstream) makes
`_use_moe_sum_reduce_torch_compile` return `False`, selecting the fused
`sgl_kernel.moe_sum_reduce`. The farm harness sets it to `0`.

Interleaved A/B, 3 reps each, medians (`NUM_PROMPTS=16`, run order `1,0,1,0,1,0`;
within-arm spread <1%):

| `SGLANG_AFD_MOE_SUM_REDUCE_COMPILE` | out tok/s | med TPOT ms |
|---|---|---|
| `1` (stock) | 43.4 / 43.4 / 43.1 → **43.4** | 157.9 / 157.1 / 158.4 → **157.9** |
| `0` (fused combine) | 46.0 / 45.3 / 45.7 → **45.7** | 148.1 / 148.8 / 148.7 → **148.7** |

= **+5.3% tok/s / −5.8% TPOT**. Note this is far larger than the 45 us/call the
microbenchmark predicts (45 us × 27 hops ≈ 1.2 ms ≈ 0.8% of TPOT), which is the
clearest single confirmation of §14.3: removing a GIL-holding Python path buys
more than its own runtime, because it also stops starving the poll thread.

### 14.5 Revised ranking (supersedes §13's)

| # | lever | expected | note |
|---|---|---|---|
| 1 | Cut GIL-holding Python per hop / stop the poll thread competing (poll without re-entering Python, or raise its interval when compute is in flight) | **large, structural** | this is what §14.3 identifies; unmeasured |
| 2 | Fused combine — `SGLANG_AFD_MOE_SUM_REDUCE_COMPILE=0` | +5.3% tok/s, −5.8% TPOT | **landed & measured** |
| 3 | `moe_align_block_size` (300 us in-situ / 29 us isolated) — persistent buffers, or hide it | medium | next-biggest in-situ item |
| 4 | Fold activation into the epilogue (§13 #2) | ~14 us | not worth it |
| 5 | Pre-allocated buffers for the `aten::empty`s (§13 #3) | ~15 us | not worth it |
| 6 | ~~CUDA graph for launch overhead~~ | 6.6% ceiling | §13 |

Reproduce the breakdown with `SGLANG_AFD_PROFILE_DETAIL=1
SGLANG_AFD_FARM_FFN_SECTION_TIME=1`; isolate with `bench_act_micro.py`,
`bench_align_micro.py`, `bench_combine_micro.py`.


## 15. GIL contention found and fixed: the FFN scheduler's idle housekeeping

§14.5 ranked "stop the poll thread competing for the GIL" as lever #1 but left it
unmeasured. It is now identified, fixed and measured — **+32.6% tok/s / −23.2%
TPOT**, the largest single win in this document.

### 15.1 The FFN hop is not CPU-starved by its own work

Adding a third clock to the routed region (`bench_ffn_threads.py`,
`ffn_routed_thr_us`) splits the hop three ways:

| metric | p50 |
|---|---|
| `ffn_routed_us` (wall) | 1497 us |
| `ffn_routed_thr_us` (**calling thread** CPU) | **681 us** |
| `ffn_routed_cpu_us` (**process** CPU, all threads) | 1705 us |

Two things follow. The FFN compute thread is busy only ~45% of the hop — the
rest it is blocked, not working. And process CPU *exceeds* the hop's wall clock
(1.14 cores), so a **second thread in the same process is burning ~1 core** for
the whole hop. `nvidia-smi` puts GPU 4 (FFN) at **10%** and GPU 7 (ATTN) at
**8%**, so neither GPU is the constraint.

### 15.2 Naming the second thread

Two traps when attributing this by hand:

- The FFN compute is **not** in the `launch_server` process. It is in the
  `sglang::scheduler` child on GPU 4 (find it with `nvidia-smi
  --query-compute-apps`; matching on `launch_server` finds the idle parent).
- The pool thread's name (`af-pool-ffn0-serve`) is a *Python* thread name and is
  **not** written to `/proc/<tid>/comm`, so `/proc` scans for it find nothing.
  Only `py-spy` shows it.

`py-spy dump --native` on that pid gives the answer directly:

```
Thread 33476 (active+gil): "MainThread"
    _get_token_info (scheduler_components/pool_stats_observer.py:221)
    get_pool_stats (scheduler_components/pool_stats_observer.py:200)
    on_idle (scheduler.py:3613)
    event_loop_overlap (scheduler.py:1622)
Thread 35871 (active): "af-pool-ffn0-serve"
    pthread_cond_timedwait
    get_batch (afd/cuda_ipc_transport.py:1116)
    _poll_ready (afd/pool/ffn_worker.py:152)
```

The scheduler `MainThread` is **holding the GIL** inside `on_idle`, while the
AFD FFN compute thread waits. Per-thread CPU confirmed it: 1.13 cores split as
0.68 (MainThread, in `on_idle`) + 0.45 (`af-pool-ffn0-serve`).

### 15.3 Root cause

The AFD FFN server is launched without `--sleep-on-idle`, so
`Scheduler.maybe_sleep_on_idle()` is a no-op and `event_loop_overlap` calls
`on_idle()` on **every** loop iteration. `on_idle` is pure Python that runs, each
time:

- `invariant_checker._check_all_pools(get_pool_stats())`
- `invariant_checker._check_tree_cache()`
- `publish_load_snapshot(force=True)` — which calls `get_pool_stats()` a *second*
  time

That is thousands of Python passes per second on a process that will never serve
a real request, and every bytecode holds the GIL away from the pool thread. This
is the concrete answer to "why does overlap not happen": the FFN side was
spinning its own scheduler in Python, so the FFN compute thread could not run
continuously.

### 15.4 Fix

`SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS` (new, `EnvInt`, default `0` = upstream
behaviour of running the full pass every iteration). When > 0, `on_idle` runs the
housekeeping at most that often and otherwise goes straight to
`maybe_sleep_on_idle()`. The farm harness defaults it to `250` ms. Nothing is
disabled, only throttled; `/get_loads` stays fresh to within 250 ms.

### 15.5 Measured (`1a1f`, 3 interleaved reps, CG off, `MOE_SUM_REDUCE_COMPILE=0`)

| interval (ms) | tok/s | med TPOT | p99 TPOT | med TTFT | med e2e |
|---|---|---|---|---|---|
| 0 (before) | 45.5 / 41.8 / 44.6 | 149.2 / 158.5 / 151.8 | 201.8 / 263.5 / 203.7 | 698 / 919 / 697 | 3012 / 3403 / 3029 |
| 250 (after) | 59.1 / 57.5 / 58.3 | 116.5 / 118.2 / 118.2 | 159.0 / 165.2 / 159.8 | 502 / 572 / 506 | 2312 / 2413 / 2341 |
| **delta** | **+32.6%** | **−23.2%** | −21.6% | −28% | −23.3% |

Ranges do not overlap on any of the three reps.

### 15.6 Mechanism confirmed after the fix

Same live measurement, throttle on:

| thread | before | after |
|---|---|---|
| `sglang::scheduler` MainThread | **0.68 cores**, in `on_idle` → `get_pool_stats` (GIL held) | 0.41 cores, in `_apply_war_barrier` → `wait_stream` (GIL released) |
| `af-pool-ffn0-serve` | 0.45 cores | **0.73 cores**, in `run_same_layer_fused` (real FFN work) |

The compute thread now dominates and is doing FFN work instead of waiting.

### 15.7 Follow-up landed: park the FFN scheduler entirely

After §15.4 the compute thread dominates, but the MainThread still spun ~0.41
cores in `_apply_war_barrier` → `torch.cuda.Stream.wait_stream`. The FFN
scheduler cannot avoid that barrier: it runs at the top of every
`event_loop_overlap` iteration, before the "is there a batch?" test.

The real cure is to stop the FFN scheduler from looping at all. It never serves
a request, and SGLang already has the mechanism: `--sleep-on-idle`, which parks
the loop in `IdleSleeper.maybe_sleep()` → `zmq.Poller.poll(1000)` with the GIL
released. The farm harness now passes it via
`SGLANG_AFD_FFN_SLEEP_ON_IDLE=1`. (`SGLANG_EMPTY_CACHE_INTERVAL` defaults to
`-1`, so parking does not trigger `empty_cache`.)

Measured (`1a1f`, 3 interleaved reps, throttle 250 ms on both arms):

| `SGLANG_AFD_FFN_SLEEP_ON_IDLE` | tok/s | med TPOT | p99 TPOT | med TTFT | med e2e |
|---|---|---|---|---|---|
| 0 | 57.9 / 58.0 / 58.6 | 119.0 / 117.8 / 117.5 | 164.9 / 171.2 / 159.9 | 516 / 538 / 509 | 2367 / 2366 / 2341 |
| 1 | 62.6 / 62.8 / 62.9 | 112.2 / 112.3 / 112.1 | 155.1 / 153.4 / 153.4 | 431 / 425 / 427 | 2177 / 2171 / 2171 |
| **delta** | **+7.9%** | **−5.0%** | −8.0% | **−19%** | −8.1% |

A second clean causal chain, this time with total CPU also *falling*:

| stage | scheduler MainThread | `af-pool-ffn0-serve` | process total |
|---|---|---|---|
| baseline | 0.68 cores, `on_idle` (GIL **held**) | 0.45 cores | 1.13 cores |
| + throttle (§15.4) | 0.41 cores, WAR barrier (GIL released) | 0.73 cores | 1.14 cores |
| + `--sleep-on-idle` | **parked**, `Thread 201588 (idle)` in `poll (zmq/sugar/poll.py:106)` | **0.99 cores** | 0.99 cores |

### 15.8 Cumulative

`1a1f` decode, CG off, `MOE_SUM_REDUCE_COMPILE=0`:

| config | tok/s | med TPOT |
|---|---|---|
| baseline (§14) | 43.97 | 153.2 |
| + `IDLE_HOUSEKEEPING=250` | 58.30 | 117.6 |
| + `--sleep-on-idle` | **62.77** | **112.2** |
| **total** | **+42.8%** | **−26.8%** |

Settings: `SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS=250`,
`SGLANG_AFD_FFN_SLEEP_ON_IDLE=1`, `SGLANG_AFD_MOE_SUM_REDUCE_COMPILE=0`.
Scripts: `bench_ffn_threads.py`, `bench_top_threads.py`. §14.5's items 2–6 are
unchanged.

## 16. Post-fix re-measurement: the §14 "10–30x inflation" was the GIL

§14.3 hypothesised that the in-situ sub-section timers read far above their
isolated costs because of "GIL handoff delay around every GIL-releasing CUDA
call", and §14.5 listed the items but could not size them. With §15's two knobs
landed, the same-config A/B can now size them directly.

Both arms identical (64 prompts, conc 16, in 128 / out 192, ctx 4096,
profile detail + FFN section timing on, `MOE_SUM_REDUCE_COMPILE=0`); the only
difference is `IDLE_HOUSEKEEPING_INTERVAL_MS` + `FFN_SLEEP_ON_IDLE`.

### 16.1 The breakdown, same config, before → after (p50)

| key | before | after | factor | isolated (§14.2) |
|---|---|---|---|---|
| `ffn_routed_us` (wall) | 1500 | 520 | **2.9x** | — |
| `ffn_routed_thr_us` (thread CPU) | 678 | 521 | 1.3x | — |
| `ffn_routed_cpu_us` (process CPU) | 1709 | 540 | **3.2x** | — |
| `serve_us` | 2457 | 875 | 2.8x | — |
| `moe_cfg_us` | 335 | 95 | 3.5x | — |
| `moe_core_us` | 1445 | 469 | 3.1x | — |
| `moe_apply_us` | 1430 | 456 | 3.1x | — |
| `moe_fx_us` | 1390 | 420 | 3.3x | — |
| `moe_align_us` | 299 | 63 | **4.7x** | 29 |
| `moe_act_us` | 288 | 46 | **6.3x** | 14 |
| `moe_k1_us` | 90 | 82 | 1.1x | — |
| `moe_k2_us` | 203 | 79 | 2.6x | — |
| `moe_alloc_us` | 15 | 9 | 1.7x | — |
| **`moe_cfgsel_us`** | **10** | **10** | **1.0x** | — |
| **`moe_disp_us`** | **5** | **4** | **1.2x** | — |
| **`moe_comb_us`** | **23** | **21** | **1.1x** | 5 |

### 16.2 The split is exactly along the kernel boundary

Sort the table by whether the item launches CUDA work:

- **Launches a kernel** — `moe_core`/`apply`/`fx`, `moe_align`, `moe_act`,
  `moe_k1`, `moe_k2`: drop **2.6–6.3x**.
- **Pure Python / already-fused** — `moe_cfgsel` (config pick), `moe_disp`
  (dispatcher), `moe_comb` (`sgl_kernel` combine): **1.0–1.2x, unchanged**.

That is §14.3's prediction confirmed with no free parameters: every GIL-releasing
kernel launch was paying a handoff penalty per call, and pure-Python items never
paid it. §14's "10–30x vs isolated" is now **1.6–4.5x vs isolated** — the
residual is normal in-situ dispatch cost, not a pathology.

Corollary: `moe_align` (299→63) and `moe_act` (288→46) were **never** the levers
§14.2 thought they were. They looked like 300–400 us of work; ~80% of that was
the GIL artifact. §14.5 items #3/#4 are now worth *tens* of microseconds, not
hundreds.

### 16.3 Do not read TPOT from the profiling runs

Same A/B, TPOT: 309.3 → 298.1 ms (**−3.6%**). Profiling **off**, §15.5/15.7:
153.2 → 112.2 ms (**−26.8%**). The profiling run understates the win ~7x.

Reason: `SGLANG_AFD_PROFILE_DETAIL=1` instruments ~16 `moe_*` keys per hop *and*
~10 `mla_*` keys per layer on the **attn** side, whose scheduler is on the
critical path. Once attn is slowed by its own instrumentation, TPOT becomes
attn-bound and the FFN-side improvement is masked. Profiling runs are valid for
*attribution within one arm*, not for tok/s or TPOT across arms.

### 16.4 Where the remaining TPOT actually is

From the post-fix attn log (same run):

| term | p50 us |
|---|---|
| `attn_core_us` | 770 |
| `attn_route_us` (gate + topk) | 275 |
| `attn_prep_us` + `attn_prepmlp_us` | 69 |
| **attn work / layer** | **1114** |
| `rtt_us` (push → FFN result back) | **2297** (was 3287 before the fix) |

So per layer the farm still spends `1114 + 2297 = 3411 us` **serially**, and
`27 x 3411 us = 92 ms` accounts for the bulk of the 112 ms measured TPOT. The FFN
compute inside that 2297 us round trip is only **520 us (23%)**; the other ~1.8 ms
is transport, queueing and handoff.

Two consequences for the roadmap:

1. **FFN-side micro-optimisation is exhausted.** `moe_align` (63 us) is 1.8% of
   the 3411 us/layer, and it sits *inside* a round trip that is queue-dominated,
   so shrinking it moves TPOT far less than its wall share. §14.5's table should
   be read as closed.
2. **The remaining lever is the serialisation itself** — attn work and the
   FFN round trip still do not overlap (`peak_layers_busy` behaviour from §11 is
   unchanged). Overlapping them is worth up to `27 x min(1114, 2297) = 30 ms`.
   That is the persistent-farm/pipelining question again, now with a measured
   ceiling.

   > **Partly retracted by §19.** Treating the `2297 us` `rtt_us` as serial
   > latency is wrong: a sixth timeline stamp (`attn_wait_enter`) shows the hop
   > is *fully hidden*, and that 2297 us is the Attn-side inter-issue interval.
   > The 30 ms "overlap ceiling" here is therefore not a real ceiling. The
   > Attn-side serial work — and in particular ~39% of it being per-hop tensor
   > marshalling — is the actual lever; §19 measures two fixes worth +16.9%.

## 17. The missing reference: replica (no AFD) vs 1A1F, identical config

Every AF measurement in this document was relative to other AF configurations.
The plain single-GPU server was never measured on the same workload, so there
was no absolute answer to "is AFD winning?". `bench_replica_vs_1a1f.sh` runs both
arms interleaved with the same model args and the same client.

Verified identical between arms (from each server's `server_args` log):
`attention_backend='triton'`, `sampling_backend='flashinfer'`,
`cuda_graph_backend_decode='disabled'`, `cuda_graph_backend_prefill='disabled'`,
`tp_size=1`, `mem_fraction_static=0.75`, `max_running_requests=16`,
`context_length=2048`, `disable_overlap_schedule=False`, and
`TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1` in both. The only difference is
that one arm splits attn/FFN across 2 GPUs and the other runs the whole model on
1 GPU.

### 17.1 Results

Small (n=16, in 128 / out 32, conc 16), 3 reps interleaved, CG off:

| arm | tok/s | med TPOT | p99 TPOT | med TTFT | med e2e |
|---|---|---|---|---|---|
| 1A1F | 61.98 / 59.82 / 61.92 | 113.8 / 118.5 / 114.0 | 155 / 161 / 156 | 434 / 432 / 429 | 2201 / 2273 / 2202 |
| **replica** (no AFD) | 155.64 / 154.14 / 150.30 | **44.0 / 46.1 / 47.3** | 68 / 72 / 56 | 179 / 151 / 140 | 860 / 881 / 903 |
| ratio | **2.51x** | **2.52x** | 2.7x | 2.8x | 2.6x |

Standard (n=64, in 128 / out 192, conc 16), 2 reps interleaved, CG off:

| arm | tok/s | med TPOT | p99 TPOT | med TTFT | med e2e |
|---|---|---|---|---|---|
| 1A1F | 106.08 / 104.22 | 120.1 / 122.8 | 124 / 127 | 281 / 289 | 10887 / 11045 |
| **replica** (no AFD) | 273.18 / 275.10 | **46.8 / 46.1** | 51 / 49 | 124 / 123 | 4186 / 4111 |
| ratio | **2.61x** | **2.61x** | 2.5x | 2.3x | 2.6x |

The ratios agree across both workloads, so this is a topology/mechanism result,
not a load artifact.

**1A1F is 2.5–2.6x slower than not using AFD at all, while consuming 2 GPUs
instead of 1.** Per-GPU efficiency is therefore ~5x worse.

### 17.2 Why, in one line of arithmetic

Same-workload GPU work per token (all 27 layers):

```
attn  27 x 1114 us = 30.1 ms      (§16.4, measured on the attn side)
ffn   27 x  520 us = 14.0 ms      (§16.1, measured on the ffn side)
                      --------
total GPU work      = 44.1 ms
```

- **replica**: computes those 44.1 ms back-to-back on one GPU. Measured TPOT
  46.5 ms → **95% of wall time is useful compute.** No hop, no offload.
- **1A1F**: same 44.1 ms of work, but it is *serialised* as
  `27 x (attn 1114 us + rtt 2297 us)` = `30.1 + 62.0 = 92 ms`, plus ~29 ms of
  sampling/prefill/scheduling → 121 ms.

The hop costs **2297 us to deliver 520 us of FFN work** — 4.4x overhead. Exposed
once per layer per token, that is `27 x 1.78 ms = 48 ms` of pure added latency
per token, which is the entire deficit (121.4 − 46.5 = 74.9 ms, of which 48 ms is
hop overhead and the rest is the attn/FFN split no longer being batched together
on one device).

Note batching does **not** rescue this: hops are amortised across ~12 token-layers
each (`rtt_us` n=26802 vs 12288 tokens x 27 layers), which is good for
*throughput*, but TPOT is a *latency* metric — a token must still traverse 27
sequential hops, and each one's latency is exposed regardless of how many other
tokens share it. This is the same conclusion as §11's `peak_layers_busy=1`, now
priced in milliseconds.

### 17.3 What this implies for the roadmap

The overlap prize is real but bounded, and it is smaller than the current loss:

| regime | per-token TPOT | vs replica |
|---|---|---|
| replica (no AFD, 1 GPU) | 46.5 ms | 1.00x |
| **1A1F today** (serialised) | 121.4 ms | **0.38x** |
| 1A1F, attn<->hop perfectly overlapped | `max(30.1, 14.0) + fill ≈ 32 ms` | ~1.45x |
| 1A1F, overlap + hop overhead fully removed | `max(30.1, 14.0) ≈ 30 ms` | ~1.55x |

So:

1. **There is no configuration of the current design that beats replica without
   cross-layer overlap.** Even a zero-cost hop leaves 30.1 + 14.0 = 44.1 ms of
   serial work, i.e. exactly replica's time. Offload only pays if attn and FFN
   run *concurrently*.
2. **The ceiling is ~1.5x over replica**, and it requires both (a) real
   attn/FFN overlap and (b) the 1.78 ms/hop overhead largely gone. Today's
   121 ms is 4x away from that ceiling.
3. **This reframes §16.4's 30 ms estimate.** Overlapping attn with the hop has a
   30 ms ceiling *within* the current serial structure; the larger prize is
   turning the 92 ms serial chain into a ~32 ms overlapped one, worth ~89 ms.
4. **Any future AFD work should be justified against the 46.5 ms replica
   number**, not against other AF configurations. On this box, CG-off, at these
   batch sizes, AFD is currently a 2.6x regression that spends an extra GPU.

Reproduce: `REPS=3 G=small bash bench_replica_vs_1a1f.sh` (or `G=std`).

## 18. FFN same-layer batching: the queue exists, is disabled, and is a no-op

Before spending effort on "gather a bigger FFN batch", the batch size and the
gather machinery were measured directly. Config for every arm: 64 prompts,
conc 16, in 128 / out 192, ctx 4096, `MOE_SUM_REDUCE_COMPILE=0`,
`IDLE_HOUSEKEEPING_INTERVAL_MS=250`, `FFN_SLEEP_ON_IDLE=1`, persistent farm off.
`SGLANG_AFD_FARM_LPU_STATS_EVERY=3000` prints the FFN serve loop's
`group_hist` — the number of hops the FFN actually concatenated per MoE call.

### 18.1 What is actually enabled today

Two same-layer batching paths exist. **Both default to off and the harness sets
neither:**

| path | flag | default | what it does |
|---|---|---|---|
| `extra_gather` | `SGLANG_AFD_FFN_GATHER_US` / `_MAX` | active (50 us / 8) | spin-poll the links for more hops after the first arrives |
| `PerLayerBatchQueue` | `SGLANG_AFD_FFN_QUEUE_ENABLE` | **`False`** | per-`layer_id` deadline batching: `target_tokens=64`, `max_wait_us=300`, `max_hops_per_layer=4`, `max_global_hops=16` |
| attn-side queue | `SGLANG_AFD_ATTN_QUEUE_ENABLE` | **`False`** | coalesce same-layer hops before consuming an A2F slot |

`PerLayerBatchQueue` is exactly the FFN-side deadline batching (gather up to
`target_tokens`, bounded in-flight via `max_global_hops`) that the design above
calls for. It is not wired up; `apply_farm_env` only back-fills
`FFN_GATHER_US`/`_MAX` from `LPU_GATHER_US=50` and `max_inflight`.

### 18.2 Measured: the FFN *never* merges, in any configuration

| arm | change vs baseline | tok/s | med TPOT ms | `max_group` | tokens/hop |
|---|---|---|---|---|---|
| base | — | 105.3 | 120.9 | **1** | 10.5 |
| `biggather` | `GATHER_US` 50→1000, `_MAX` 8→32 | **75.2** | **167.6** | **1** | 10.5 |
| `attnq` | `ATTN_QUEUE_ENABLE=1` | 106.5 | 119.3 | **1** | 10.5 |
| `ffnq` | `FFN_QUEUE_ENABLE=1` | 104.3 | 122.2 | **1** | 10.5 |
| `bothq` | both queues on | 107.0 | 118.8 | **1** | 10.5 |
| `c2s0` | `NUM_CONTEXTS=2`, `STAGGER_LAYERS=0` | 85.0 | 150.3 | **1** | 10.5 |
| `c4s0` | `NUM_CONTEXTS=4`, `STAGGER_LAYERS=0` | 58.3 | 225.3 | **1** | 7.9 |
| `c4s0big` | as `c4s0` + big gather | 53.1 | 243.0 | **1** | 10.4 |

`group_hist={1: N}`, `singleton=N`, `multi=0`, `max_group=1` in **all eight
arms**. The FFN serve loop has never once seen two same-layer hops at the same
time. Enabling `FFN_QUEUE_ENABLE=1` is therefore a pure no-op — the queue never
holds more than one item.

Also note **tokens/hop is constant at 10.5 across every arm** (`attnq` included).
A hop already carries the whole row-window of a context; there is no
fragmentation to coalesce and the attn queue changes nothing.

### 18.3 Why it can never merge: layer diversity vs layer coincidence

The merge key is `layer_id`. For two hops to fuse they must be **in flight at
the same instant and at the same layer**. The farm deliberately holds contexts
at *different* layers — that is what overlap means, and
`CONTEXT_STAGGER_LAYERS=1` enforces it. Layer diversity (what overlap needs) and
layer coincidence (what FFN batching needs) are mutually exclusive.

Forcing coincidence makes it strictly worse, not better: `c2s0` 85.0 tok/s and
`c4s0` 58.3 tok/s vs 105.3 baseline. With no layer spread the contexts convoy —
all of them wait on the same hop — so the attn/rtt serialization gets *worse*.

### 18.4 And there is no headroom: the FFN is under half utilised

- `calls = 27 000`, `serve_us` = 875 us (§16.1), run duration 56.7 s
  → **41.6% FFN utilisation**.
- routed MoE core only: `27 000 x 520 us = 14.0 s` → **24.8%**.

TPOT is latency-bound, not FFN-throughput-bound. Merging hops reduces FFN
*thread time*, which is not scarce, and pays for it in *per-hop latency*, which
is what TPOT measures. Even a zero-cost FFN would save only `27 x 520 us =
14 ms` of the 112 ms TPOT.

### 18.5 Conclusion

1. **"Accumulate a bigger FFN batch" has no headroom and measurably hurts.** The
   only knob that increases the gather window (`GATHER_US`) costs 29% tok/s and
   39% TPOT, because it adds up to 1 ms of dead spin per FFN call and returns
   zero extra hops.
2. **The proposed mechanism is already implemented and correct** — but its
   precondition (multiple same-layer hops in flight) is never satisfied by the
   current scheduler, so it cannot engage. `gather_hold_slots < total_slots`
   never binds either, because in-flight hops per layer is already 1.
3. **The lever remains §16.4/§17: the serialization.** Per layer the farm spends
   `1114 us` attn + `2297 us` rtt = `3411 us` serially; only 875 us of the rtt is
   FFN serve and only 520 us is FFN compute. The ~1.4 ms of transport/poll/hop
   overhead inside every rtt is worth `27 x 1.4 ms = 38 ms` — an order of
   magnitude more than the entire FFN dispatch budget.

Reproduce: arms as in the table, `SGLANG_AFD_FARM_LPU_STATS_EVERY=3000`, then
read `group_hist` from `$OUT_DIR/1a1f/ffn0.log`.

## 19. Chewing on the "1.4 ms hop" — it is not a hop cost at all

§16.4 concluded that each layer costs `1114 us` attn + `2297 us` rtt = `3411 us`
**serially**, and pointed at the ~1.8 ms of "transport, queueing and handoff"
inside the round trip. That model was wrong, and §19.2 shows why.

### 19.1 Extending the timeline to see who is late

The transport already carries a per-hop cross-process timeline
(`SGLANG_AFD_TIMELINE=1`, shm ring). It records `attn_post / ffn_seen /
ffn_compute / ffn_respond / attn_done`, so `respond_to_attn` had no way to tell
"Attn was busy elsewhere" from "Attn was parked spinning". A sixth stamp,
`attn_wait_enter`, was added at the top of the Attn-side `wait()` in
`cuda_ipc_transport.py`, and `_SLOT` grew `<qii5d` → `<qii6d`.

### 19.2 The hop is completely hidden

Same config as §18 (1A1F, 64 prompts, conc 16, in 128 / out 192,
`MAX_INFLIGHT`/`NUM_MB` = 8, `num_contexts=2`, `stagger=1`,
`max_inflight_per_layer=1`, persistent off), p50 us:

| phase | p50 | who |
|---|---|---|
| `post_to_ffn_us` | **14** | Attn post → FFN observes |
| `ffn_compute_us` | 733 | FFN dispatch + MoE |
| `ffn_to_respond_us` | **19** | respond copy + doorbell |
| `respond_to_wait_enter_us` | **1348** | result ready → Attn comes back for it |
| `wait_enter_to_done_us` | **17** | Attn parks → sees done |
| `total_rt_us` | 2135 | `attn_post → attn_done` |

Read it as a story: Attn posts the hop, then spends **2094 us** (`wait_enter_from_post_us`)
doing other work before it comes back to collect. By then the result has been
sitting ready for 1348 us, and it is collected in 17 us.

So:

1. **The FFN hop is not on the critical path.** Removing the FFN entirely would
   save nothing — 1.3 ms of slack sits between "FFN done" and "Attn wants it".
   Every FFN-side lever in §14–§18 was therefore worth zero TPOT, which is
   consistent with §18's measurement that the FFN is only 41.6% utilised.
2. **Transport is free.** `post_to_ffn` 14 us and `ffn_to_respond` 19 us say the
   cuda_ipc A2F, the eventfd wake, the GPU doorbell and the F2A copy are all
   already negligible. There is nothing to win there.
3. **§16.4's `1114 + 2297 serial` is retracted.** The 2297 us `rtt_us` is not
   latency; it is the Attn-side *inter-issue* interval for that hop. Summing it
   with attn work double-counted the Attn side and ignored that the hop overlaps.

### 19.3 py-spy: the Attn side is the bottleneck, and 39% of it is marshalling

`py-spy record` on the Attn `sglang::scheduler` during steady-state decode
(30 s, ~2900 samples). 84% of samples are inside `run_farm_layers`, 93.5% of
those inside `_issue_round`:

| category | share of farm |
|---|---|
| attention (MLA: `forward_absorb_*`, `forward_decode`) | 35.4% |
| **send/credit A2F** (`fill_a2f`, `wait_group`, `_issue_layer`) | **15.7%** |
| **window slice** (`slice_decode_window`, `filter_batch`, clone/cat) | **11.9%** |
| **sampling-info slice** (`slice_sampling_info`) | **11.1%** |
| sched reserve/commit | 6.1% |
| routing (`attn_compute_routing`, `grouped_topk`) | 4.8% |
| wait for hop | 4.4% |
| misc torch dispatch | 2.5% |

Only ~46% is actual compute (attention + routing). **~39% is per-hop host-side
tensor marshalling** — slicing the parent batch into a child `ForwardBatch`,
building an A2F payload — repeated 51 times per forward. That is the §16.4
"unattributed" wall time, and it is where the effort belongs.

The same PHASE instrument from §16.4 localises it independently: steady state
was `loop=79 ms/forward, unattributed=22.2 ms`, i.e. 28% of the loop outside
`pre/issue/drain/consume`.

### 19.4 Fix 1 — do not slice `SamplingBatchInfo` on intermediate hops

`slice_decode_window` builds a child `ForwardBatch` per hop and attaches a
sliced `SamplingBatchInfo` (`dataclasses.replace` + `torch.arange` + ~11 tensor
slices). But `filter_batch` already yields `sampling_info=None`, and the only
readers are `model_runner.sample()` and `logprob.py` — i.e. **only the hop that
finishes a sequence is ever handed to the sampler** (`_on_output_ready` uses
`hop.meta["forward_batch"]`). The other 26 layers' slices were dead work.

`slice_decode_window(..., with_sampling_info=)` now gates it, and the one-shot
`_issue_round` passes `li == n_layers - 1`. The persistent path keeps it (its
child batch is adopted once per group and reused at sampling).
`SGLANG_AFD_FARM_MID_SAMPLING_INFO=1` restores the old behaviour.

A/B, 2 reps interleaved, profiling off:

| arm | tok/s | med TPOT ms | mean TPOT ms |
|---|---|---|---|
| old (build on every hop) | 103.4 | 122.5 | 127.4 |
| gated to final hop | 118.0 | 107.6 | 111.9 |
| **delta** | **+14.1%** | **−12.1%** | **−12.2%** |

Mechanism validated by PHASE, same run shape, profiling on:

| | old | new |
|---|---|---|
| `unattributed` | 22.2 ms | **15.0 ms** (−32%) |
| `wall` | 116.0 ms/head | **96.6 ms/head** |
| `hop_lat_p50` | 2.27 ms | 1.94 ms |
| `hops/fwd` | 51.2 | 51.2 (unchanged) |

The 7 ms/forward drop lands exactly in the `unattributed` bucket — which is
where `slice_decode_window` sits (it is called before `_t_pre` is stamped, so it
is in neither `pre` nor `issue`). `hops/fwd` unchanged confirms no work was
skipped, only redone less.

### 19.5 Fix 2 — no `torch.cat` for a single-hop group

`AttnSendQueue._issue_layer` unconditionally did
`torch.cat([item.hidden for item in items])` plus `_cat_optional` for topk.
With `max_hops=1`-style groups that is a one-element cat: a fresh allocation and
a copy kernel per hop, for nothing. `fill_a2f` copies into the A2F slot anyway,
so the single-item path now passes the view straight through.

| arm | tok/s | med TPOT ms |
|---|---|---|
| gate only | 118.0 | 107.6 |
| gate + cat short-circuit | **120.9** | **104.9** |
| **delta** | **+2.5%** | **−2.6%** |

### 19.6 Cumulative, and the revised lever ranking

Both fixes, vs the §18 baseline, profiling off, 2 reps:

| | tok/s | med TPOT ms |
|---|---|---|
| §18 baseline | 103.4 | 122.5 |
| §19 both fixes | **120.9** | **104.9** |
| **delta** | **+16.9%** | **−14.4%** |

Zero errors in either arm. Remaining Attn-side host work, by the §19.3 shares:

1. **window slice, 11.9%** — `filter_batch` per hop does ~13 row-indexed tensor
   slices, builds a ~30-key dict, computes `extend_num_tokens`, and runs a full
   `dataclasses.fields(ForwardBatch)` validation loop. None of it depends on the
   layer: within one forward, a given seq window has identical row metadata. The
   obvious next step is to build the child batch **once per (window, forward)**
   and reuse it across that window's 27 layers, keeping only `hidden_states` /
   `residual` / `positions` per layer. Needs care: anything the attention
   backend mutates on the child (planned metadata, `forward_metadata_ready`)
   must be verified to be layer-invariant — it is for a fixed token range, but
   this is exactly the kind of assumption that has produced crashes here before.
2. **send/credit A2F, 15.7%** — now that the one-element cat is gone, the rest
   is `fill_a2f`'s copy (necessary) plus queue/credit bookkeeping
   (`enqueue`, `_take_group`, `wait_group`). Worth profiling at the same
   granularity as `slice_sampling_info` before touching.
3. **sched reserve/commit, 6.1%** — `token_queue.pick` is a listcomp with per-item
   hashing; suspiciously expensive for a queue peek.
4. **attention 35.4% is now the floor**, and it is real work, not overhead.

Reproduce §19.2: add `SGLANG_AFD_TIMELINE=1` plus
`SGLANG_AFD_TIMELINE_OUT=/tmp/tl.txt`; the summary is rewritten by the Attn
process every 512 completes and on exit.
