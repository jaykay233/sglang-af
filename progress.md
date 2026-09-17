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

## 20. NA1F FFN poll: a real head-of-line block (fixed), but no throughput win

§18 closed the FFN-batching line and §19 moved the lever to the Attn host path.
This section tests the *other* lever suggested for the FFN: multiple independent
Attn workers feeding one FFN (**2A1F**). Before building anything, two bugs in
the existing AfPool NA1F poll path were found and fixed.

### 20.1 Fix — `get_batch(timeout_s=0)` was a silent no-op

`CudaIpcAfdTransport.get_batch` guarded its scan with
`while time.time() < deadline:`. For `timeout_s == 0` that is immediately
false, so it returned `[]` **without scanning**. But callers use `timeout_s=0`
to mean "one non-blocking pass":

- `AfFfnWorker._poll_ready(n, 0.0)` — used when the per-layer queue is non-empty;
- `extra_gather(lambda: self._poll_ready(n, 0.0), ...)` — so **`extra_gather`
  had always been a no-op in the pool path**.

`get_batch` now always scans at least once, and takes `nonblocking=True` to mean
exactly one pass regardless of the clock.

### 20.2 Fix — per-link blocking starves later links (head-of-line)

`_poll_ready` called `tr.get_batch(timeout_s=timeout_s)` **per link, in link
order**. An idle link 0 blocks for the full timeout before link 1 is even
looked at, so a hop already ready on link 1 waits behind it. Measured directly
(fake links, one idle + one ready, `timeout_s=2ms`, 20 trials, p50):

| poll path | latency to reach link1's ready hop |
|---|---:|
| legacy per-link blocking | **2.056 ms** |
| drain-all (`SGLANG_AFD_FFN_POLL_DRAIN_ALL=1`, default) | **0.002 ms** |

`_poll_ready` now drains every link non-blocking, then parks **once** and
re-drains. Idle cost is O(1) in `num_attn` instead of O(num_attn).
`SGLANG_AFD_FFN_POLL_DRAIN_ALL=0` restores the legacy scan for A/B.
Regression tests: `test/registered/unit/afd/test_afd_ffn_poll.py` (6 tests).

### 20.3 Measured: no throughput difference in the synthetic pool bench

`bench_af_pool`, 2A1F vs 1A1F, 4 reps interleaved, `SKIP_EXTRA_GATHER=1` to
isolate the HOL fix, tree frozen (unlike a first, rejected run):

| regime | arm | 2A1F tok/s | 2A1F / 1A1F |
|---|---|---:|---:|
| attn-bound (`attn_us=1100`, `ffn_us=520`) | legacy | 12628 | 1.92x |
| attn-bound | drain-all | **12883** | **1.96x** |
| FFN-bound (`attn_us=200`, `ffn_us=400`, run 1) | legacy | 16362 | 1.07x |
| FFN-bound | drain-all | ~22600 (noisy) | 1.22–1.98x |

The attn-bound numbers are stable (wall 0.505–0.555 s across 8 runs) and show
**+0.6% (within noise)**. So the HOL delay is real (20.2) but is **not on the
critical path** in this bench: 2A1F already reaches ~1.95x, i.e. the FFN keeps
up at 2x Attn supply. The FFN-bound run 1 was rejected — 0.1–0.3 s walls,
one 30255 tok/s outlier, and the source tree was edited mid-run.

### 20.4 The pool KPI was broken: `mean_ffn_util` was clamped round-trip time

`bench_af_pool` reports `mean_ffn_util`, and `pool/README.md` said to read it.
It was computed from `compute_s = time.perf_counter() - t0` spanning **post →
wait**, i.e. the full *round trip*, accumulated per FFN rank and then
`min(1.0, busy_s / wall_s)`. Those round trips overlap, so whenever the link
saturates the sum exceeds wall time and it clamps — it read **1.000 for both
1A1F and 2A1F**, i.e. it could not distinguish a half-idle FFN (real
utilisation 0.503) from a saturated one (0.956). It only dips below 1.0 when
the link is under-filled (1A2F read 0.513). It is therefore an aggregate
round-trip occupancy, not a utilisation, and it cannot show the one quantity
the 2A1F hypothesis is about.

Fixed. `AfFfnWorker` now self-reports its own serve-loop occupancy
(`busy_s / elapsed_s`, where `busy_s` includes the post-serve
`torch.cuda.synchronize()` that `SGLANG_AFD_POOL_SERVE_SYNC` enables) to
`SGLANG_AFD_POOL_UTIL_FILE`; the bench reads it before terminating the FFN
children and reports it as `mean_ffn_util`. The old round-trip number is kept
as `mean_ffn_rtt_frac` and documented as *not* utilisation.

**This is the measurement §18.4 could only estimate.** Same config as §20.3
(attn-bound, `attn_us=1100` / `ffn_us=520`), one run each:

| arm | tok/s | 2A1F/1A1F | `mean_ffn_util` (new, FFN-side) | `mean_ffn_rtt_frac` (old) |
|---|---:|---:|---:|---:|
| 1A1F | 6589.1 | — | **0.503** | 1.000 |
| 2A1F | 12882.7 | **1.96x** | **0.956** | 1.000 |

FFN-side self-report (`ffn0_util.csv`, `busy_s,elapsed_s,frac,tasks`):

```
1A1F: 0.254387, 0.505262, 0.503477, 414
2A1F: 0.481435, 0.503657, 0.955880, 808
```

So the second Attn worker **doubles the hops served (414 → 808) and takes FFN
serve occupancy from 50.3% to 95.6%** — the "FFN was idle, more Attn supply
fills it" story, now measured rather than inferred. Two consequences:

1. **It also predicts the ceiling.** At 2A1F the FFN is at 95.6%, i.e. ~1.9x
   the 1A1F load, so the FFN is the next bottleneck. A third Attn worker would
   push demand past FFN capacity and add queueing, not throughput — consistent
   with §20.5 and with the 2A4F regression (0.73x) seen in E2E.
2. **It does not change §20.3's verdict.** The HOL fix still moves no
   throughput, because with drain-all vs legacy the hop count and wall are the
   same; utilisation is derived and therefore identical.

### 20.5 Conclusion

1. Both poll bugs were real and are fixed, with the HOL delay cut 1000x and a
   previously-dead `extra_gather` revived. Keep the fix.
2. **In the synthetic bench neither fix moves throughput** (12628 → 12883
   tok/s, within noise) because that bench cannot create the asymmetry — see
   item 3. **§21 then shows the fix is worth 1.33–1.42x tok/s end-to-end once
   the asymmetry is real.** Per §17.3, 2A1F should still be judged against the
   46.5 ms replica baseline, not against 1A1F.
3. The synthetic bench **cannot create the asymmetry** the HOL fix addresses:
   both Attn clients run the same lockstep workload, so link 0 is rarely idle
   while link 1 is ready. Testing the fix properly needs independent request
   streams (one Attn idle/light) — done in §21.
4. **`mean_ffn_util` is now trustworthy** (20.4) and confirms the 2A1F
   mechanism: FFN serve occupancy 50.3% → 95.6%. It also shows the FFN becomes
   the bottleneck at 2A1F, so adding a 3rd Attn worker is not the next move.

Reproduce §20.2: `python3 -m pytest test/registered/unit/afd/test_afd_ffn_poll.py -q`.
Reproduce §20.3: `REPS=4 ATTN_US=1100 FFN_US=520 REQS=16 bash python/sglang/srt/afd/pool/bench_poll_ab.sh`
(raw results in `/tmp/afd_poll_ab2/results.txt`).
Reproduce §20.4: same bench with `--compare-1a1f`; read the `mean_ffn_util`
column, or the raw self-report in `$ENDPOINT_DIR/ffn0_util.csv`.

## 21. The head-of-line fix *does* pay — under asymmetric load (2A1F E2E)

§20.3 could not show a throughput effect because `bench_af_pool` drives both
Attn clients with the same lockstep workload, so FFN link0 is rarely idle while
link1 holds a ready hop. The HOL condition never arises. This section creates it
deliberately, end-to-end with the real model.

### 21.1 Setup

Attn self-prefill (PD=null), DeepSeek-V2-Lite-Chat, **2A1F**:
Attn0 → GPU5, Attn1 → GPU6, FFN → GPU7, `mem-fraction-static 0.82`,
`in=1 / out=64 / conc=8 / 32 prompts`, breakable decode CG.

**All requests are sent straight to Attn1**; Attn0 is brought up and connected
but receives no traffic, so the FFN's link0 is permanently idle while link1 is
busy. Only `SGLANG_AFD_FFN_POLL_DRAIN_ALL` differs between arms. Two reps, with
the arm order reversed in rep2 to cancel drift. Reproduce with
`ARM=drainall|legacy bash python/sglang/srt/afd/pool/bench_asym_ab.sh`.

### 21.2 Result

| metric | rep | drain-all | legacy | legacy/drain-all |
|---|---|---:|---:|---:|
| output tok/s | 1 | **50.2** | 37.7 | 0.751x |
| output tok/s | 2 | **57.1** | 40.1 | 0.703x |
| median TPOT ms | 1 | **115.4** | 162.8 | 1.411x |
| median TPOT ms | 2 | **108.4** | 158.1 | 1.459x |
| FFN util | 1 | 0.392 | 0.300 | 0.767x |
| FFN util | 2 | 0.337 | 0.282 | 0.836x |
| hops served | 1 | 6963 | 7006 | — |
| hops served | 2 | 6982 | 6983 | — |
| completed | both | 32 | 32 | — |

**drain-all is 1.33–1.42x tok/s faster and 29–31% lower TPOT** (median TPOT
115–108 ms vs 158–163 ms), with no errors in either arm.

### 21.3 The mechanism is confirmed by the FFN's own counter

This is the part that makes it conclusive rather than a timing coincidence:

- **Hops served are identical** — 6963 vs 7006 (rep1), 6982 vs 6983 (rep2).
  Legacy did not do less work, and did not drop requests (32/32 both arms).
- **Yet legacy's FFN utilisation is *lower*** — 0.300 vs 0.392, and 0.282 vs
  0.337. Same work in the same wall window, but the FFN spent more of it *not
  serving*.

That is exactly the HOL signature: `_poll_ready` blocks on the idle link0 for
the full poll timeout before it ever looks at link1, so each serve iteration
absorbed dead time while a hop sat ready on link1. The §20.4 utilisation
self-report is what makes this visible; the old attn-side round-trip number
clamped at 1.000 for both arms and would have hidden it.

### 21.4 Why §20.3 saw nothing, and what that means

Both statements are true, and they are consistent:

| regime | link0 vs link1 | HOL delay | effect of fix |
|---|---|---|---|
| synthetic §20.3 | both equally busy (lockstep) | rare | none (within noise) |
| asymmetric §21.2 | link0 idle, link1 busy | every poll | **1.33–1.42x** |

So the fix is not a micro-optimisation — it removes a serialisation that is
*invisible when the two Attn workers happen to be balanced and severe when they
are not*. Balanced load is the special case: any real deployment behind a
round-robin router, a mini-LB, or with unequal prefill/decode mixes will have
one Attn worker ahead of the other most of the time, which is precisely the
regime measured here.

This also reframes the 2A1F E2E numbers on record. `bench_multi_attn.sh` uses
`round_robin`, i.e. approximately the balanced case, so its historical 1.30x
over 1A1F was measured in the regime where this bug costs least. The 2A1F
ceiling under imbalance was previously suppressed by the poll path itself.

### 21.5 Conclusion

1. Keep the fix; it is now justified by an end-to-end measurement (1.33–1.42x
   tok/s, 29–31% lower TPOT) rather than by mechanism alone.
2. `SGLANG_AFD_FFN_POLL_DRAIN_ALL=1` should stay the default, which it is.
3. Any future NA1F comparison must state the load balance between Attn workers,
   because it is now a first-order variable. A `round_robin`-only benchmark
   cannot detect this class of bug.
4. The §20.4 utilisation KPI is what made the mechanism provable (same hops,
   lower busy fraction). Keep it.

<a id="s22"></a>
## 22. The overlap lever is already spent; the one-shot window-slice cache

### 22.1 Do not rebuild "true pipelining" as a TPOT lever

§16.4 and §17.3 priced the serial chain at `27 x (attn 1114 + rtt 2297) = 92 ms`
and put an overlap prize of 30–89 ms on it. **§19.2 retracts that model**, and
§19.6's own retraction note is easy to miss when reading §17 first:

- The hop is **fully hidden**. Attn posts it, does 2094 us of other work, comes
  back, and the result has been ready for **1348 us**. Removing the FFN entirely
  would save nothing. So there is no exposed serial `rtt` to overlap with.
- The 2297 us is the Attn-side **inter-issue interval**, not latency. Summing it
  with attn work double-counted the Attn side.
- §9.4 still holds independently: a token's chain is a data dependency
  (`attn(L) -> hop(L) -> attn(L+1)`), so overlap across tokens/batches raises
  **throughput**, it does not shorten one sequence's latency.

And it has been built twice already, both times verified and both times slower:

| attempt | overlap achieved | result |
|---|---|---|
| §11 persistent runtime | `layers_peak=4`, `span_peak=26` | **2.8x slower** (44.5 -> 20.7 tok/s) |
| §12 grouped contexts | 4 contexts at 4 layers, `span_peak=3` | **1.8x slower** than `G=1` |

Both lose for the same reason: splitting a coalesced window into `G` chains
multiplies *concurrent FFN calls* by `G` while the per-call host dispatch is
fixed (`f ~ 1.5 ms`, §12.4). Overlap only pays if the thing being overlapped were
exposed — and §19.2 says it is not.

**Consequence for the roadmap:** the lever is the Attn-side inter-issue interval
(per-hop host work), not concurrency. An earlier "cross-layer overlap, ~89 ms"
framing derived from §17.3 must not be used to justify new pipelining work.

### 22.2 Window slice: build the child once per (window, forward)

§19.6 item 1: in the one-shot farm every hop called `slice_decode_window`, which
rebuilds a child `ForwardBatch` via `filter_batch` — ~13 tensor slices, a ~30-key
dict, and a full `dataclasses.fields(ForwardBatch)` validation loop. None of it
depends on the layer, yet a window walks all 27 of them.

Implemented as `_build_child_fb` + a `child_cache` keyed by `(seq_lo, seq_hi)`,
local to one `run_farm_layers` call, gated by `SGLANG_AFD_FARM_SLICE_CACHE`
(default 1; `0` restores build-per-hop). Only `hidden_states` / `residual` /
`positions` are re-sliced per layer, because `residual` can be reallocated
mid-forward.

This is **not a new assumption**: `_run_persistent` already reuses one child
(`ctx.child_fb`) across all 27 layers *and across forwards*, and §12 measured that
path as the fastest farm config. The one-shot path was simply rebuilding a
structure the persistent path already treats as layer-invariant.

Unit tests: `test/registered/unit/afd/test_afd_farm_slice_cache.py` (5 pass) —
one build per window, distinct windows distinct children, `child_cache=None`
preserves the old behaviour, an intermediate hop can never inherit a stale
`sampling_info`, and views are re-sliced rather than cached.

### 22.3 It pays only at the host-bound operating point

Two configs, 2 reps interleaved (order reversed on rep 2), CG off, 1A1F,
profiling off:

| config | OFF tok/s | ON tok/s | delta | OFF TPOT | ON TPOT |
|---|---|---|---|---|---|
| `MAX_INFLIGHT=2`, conc 8, 32 prompts | 41.34 / 41.62 | 41.99 / 41.25 | **+0.3%** (noise) | 168.3 / 168.7 | 166.4 / 168.9 |
| `MAX_INFLIGHT=8`, per-layer cap 1, conc 16, 64 prompts | 72.28 / 72.16 | 75.51 / 75.34 | **+4.4%** | 174.8 / 174.1 | 167.2 / 169.2 |

`picks` is **identical** in both arms (10125 at cfg 1, 27000 at cfg 2), so no
work was skipped — this is recomputation removed, not work deferred.

The split is the same lesson as §16.4/§18/§20.3: at low in-flight credit the farm
is not host-bound, so trimming host work measures flat; at the §18/§19 operating
point (`MAX_INFLIGHT=8`, per-layer cap 1 — where §19's fixes also paid) it is
worth +4.4% / −3.6%. **Any future Attn-host-path A/B must state `MAX_INFLIGHT`
and the per-layer cap**, or it will report a null result for a real win.

Note the cfg-2 baseline (72 tok/s) is not §19's 103.4 tok/s, so B_STEP / coalesce
differ from §19's run; the A/B is internally consistent, the absolute level is
not comparable across sections.

Harness change: `bench_farm_e2e.sh` now honours `SGLANG_AFD_FARM_MAX_INFLIGHT`,
`SGLANG_AFD_NUM_MB` and `SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER` overrides
(previously hardcoded to 2/2), which is what made the cfg-2 reproduction possible.

Reproduce: `ATTN_GPU=6 FFN_GPU=7 MODES=sticky NUM_PROMPTS=64 MAX_CONCURRENCY=16
RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=192 CONTEXT_LENGTH=4096
SGLANG_AFD_FARM_MAX_INFLIGHT=8 SGLANG_AFD_NUM_MB=8
SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1 SGLANG_AFD_FARM_SLICE_CACHE=<0|1>
bash bench_farm_e2e.sh`

## 23. The farm harness was GIL-starving the FFN: +44% tok/s, −29% TPOT

§15 found and fixed the FFN process's GIL contention — but **only
`bench_pool_e2e.sh` ever applied the fix**. `bench_farm_e2e.sh`, the harness that
produced all of §19 and §22, set neither
`SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS` nor `--sleep-on-idle`. Every farm A/B in
this document therefore ran against a crippled FFN.

### 23.1 How it surfaced

Profiling the A2F send path (§19.6 lever #2) turned out to be the wrong lead: at
cfg-2 the remaining A2F host work is ~12% of farm time, and most of it is
irreducible copies. Pulling the per-hop cross-process timeline instead showed the
hop is **not** hidden — `respond_to_wait_enter_us` p50 is only 71 us, i.e. Attn
gets back to the wait *before* the result is ready:

| per-hop (p50, cfg-2, TIMELINE only) | us |
|---|---|
| post → ffn | 640 |
| **ffn compute** | **2661** |
| ffn → respond | 23 |
| respond → wait_enter | 71 |
| wait_enter → done | 21 |
| **total_rt** | **3446** |

93 ms/token of the ~180 ms TPOT is hop time, and 77% of the hop is FFN. So the
FFN, not the Attn host path, is the critical path.

`py-spy record` on the FFN `sglang::scheduler` at that point:

| thread | samples | where |
|---|---|---|
| **MainThread** | **56%** | `_apply_war_barrier` → `Stream.wait_stream` / `record_event` |
| `afd-ffn-serve` (compute) | 44% | `run_same_layer_fused` → `ffn_apply_experts` |

That is §15's exact signature, from a harness that was supposed to have fixed it.

### 23.2 Fix

`bench_farm_e2e.sh` now mirrors `bench_pool_e2e.sh`:
`SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS` defaults to `250` and the FFN server is
launched with `--sleep-on-idle`. Both are overridable, so setting them to `0`
reproduces the starved behaviour for A/B.

### 23.3 Measured (cfg-2: 64 prompts, conc 16, MAX_INFLIGHT=8, per-layer cap 1)

2 reps, order reversed on rep 2, CG off, `SLICE_CACHE=1` in both arms:

| arm | tok/s | med TPOT ms |
|---|---|---|
| old (throttle 0, no `--sleep-on-idle`) | 67.83 / 73.97 | 182.9 / 170.7 |
| new (throttle 250 ms + `--sleep-on-idle`) | **102.54 / 101.95** | **125.2 / 125.6** |
| **delta** | **+44%** | **−29%** |

Ranges do not overlap. The corrected baseline (102.3) lands on §18's numbers
(105.3 base / 107.0 bothq), which is the cross-check that this restores the
operating point §18 and §19 were actually measured at — §18's config line does
list `IDLE_HOUSEKEEPING_INTERVAL_MS=250`, `FFN_SLEEP_ON_IDLE=1`. §19 is therefore
unaffected; **§22.3 is not** (see §23.4).

### 23.4 Consequence: §22.3's slice-cache number is invalid as stated

§22.3's A/B ran at 72.2 → 75.3 tok/s, i.e. entirely inside the starved regime.
Its "+4.4% / −3.6%" is a real measurement of a *GIL-bound* farm and must not be
quoted as the slice cache's value at the intended operating point. Re-run
under §23.2 follows in §23.5.

The general lesson, and the reason this section exists: **a config that is
host-bound enough to show a host-path win is often also a config whose FFN is
GIL-starved.** `MAX_INFLIGHT=8` + per-layer cap 1 is exactly such a point. Any
farm A/B from here on must state the throttle and `--sleep-on-idle` settings, the
same way §22.3 required stating `MAX_INFLIGHT` and the per-layer cap.

### 23.5 Re-check: the window-slice cache is still a win, and bigger

§22.3's A/B re-run with the §23.2 harness fix, same cfg-2 point, 2 reps,
order reversed on rep 2:

| arm | tok/s | med TPOT ms |
|---|---|---|
| `SLICE_CACHE=0` | 93.87 / 94.45 | 136.5 / 135.5 |
| `SLICE_CACHE=1` | **101.43 / 101.71** | **126.5 / 125.9** |
| **delta** | **+7.9%** | **−7.2%** |

Ranges do not overlap, and `on` is the tighter of the two. So:

- **§22.3's conclusion survives** — build the child `ForwardBatch` once per
  `(window, forward)`, not once per hop. The mechanism ("recomputation removed,
  not work deferred") is unchanged; `picks` was identical in both arms there.
- **Its magnitude was understated**, because 72 tok/s was deep in the starved
  regime where the FFN, not the Attn host, set the pace. At the corrected point
  the same host-side saving is worth ~1.8x what §22.3 recorded.
- The `MAX_INFLIGHT=8` + per-layer cap 1 config is host-bound *and* was
  GIL-starved. Those two properties are easy to conflate; §23.4's warning stands.

Reproduce: `ATTN_GPU=6 FFN_GPU=7 MODES=sticky NUM_PROMPTS=64 MAX_CONCURRENCY=16
RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=192 CONTEXT_LENGTH=4096
SGLANG_AFD_FARM_MAX_INFLIGHT=8 SGLANG_AFD_NUM_MB=8
SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1 SGLANG_AFD_FARM_SLICE_CACHE=<0|1>
bash bench_farm_e2e.sh` — with `SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS` and
`SGLANG_AFD_FFN_SLEEP_ON_IDLE` left at their new defaults.

### 23.6 Next lever flagged: the FFN F2A output pad

With the harness fixed, attention turns back to the hop. §23's clean timeline run
put `ffn_compute_us` at 2.66 ms p50 — 77% of the 3.45 ms round trip, with Attn
spending only 71 us between the FFN reply and re-entering the wait. So the FFN
hop, not the Attn host path, sets the pace, and arithmetic on it gives
`317 hops/s * 2.66 ms ≈ 84%` FFN occupancy.

Profiling the FFN process for where that 2.66 ms goes showed the F2A output path
allocating a full `max_num_token` (256) row slot per hop, writing only ~7 live
rows, zeroing the rest and handing the whole slot to `respond` to copy — ~42x the
live bytes. That is the §24 candidate.

> **Resolved in §24: real waste, but not a lever. Measured null (ranges overlap),
> reverted.**
>
> **⚠ Superseded by §25.** Every number in this subsection — the 2.66 ms
> `ffn_compute_us`, the 71 us `respond_to_wait_enter_us`, and the
> `317 hops/s * 2.66 ms ≈ 84%` occupancy — was measured on the GIL-starved
> harness. §25 re-measures on the corrected baseline: the FFN computes in
> **746 us**, Attn returns to the wait in **1416 us**, and FFN occupancy is
> **~19–31%**. The conclusion "the FFN hop, not the Attn host path, sets the
> pace" is **reversed** — Attn is the pace-setter. Read §25 before acting here.

## 24. FFN output padding is real waste but not a lever (reverted)

§23.6 flagged the FFN F2A output path as the next candidate: `_pack_group_outputs`
/ `run_same_layer_fused` allocated `torch.empty(max_num_token, H)`, wrote `t`
live rows, `zero_()`d the `max_num_token - t` pad, and returned it; `respond`
then copied the whole `max_num_token` slot across. At the §23.5 operating point
a hop is ~7 rows against a 256-row slot, i.e. ~36x the live bytes.

### 24.1 The change

`_emit_f2a` writes `slot[:t]` into the batch's exported `_f2a_bufs[index]` and
returns that view. `respond` already short-circuits on
`dst.data_ptr() == src.data_ptr()`, so the whole-slot copy disappears; the pad is
left stale, which is safe because consumers read `[:num_tokens]` only — the same
argument that let `fill_a2f` stop zeroing its pad (§19.2). Gated by
`SGLANG_AFD_FFN_F2A_SLOT` so the old allocate-and-copy path stays A/B-able
(`=0` reproduces it exactly). 8 unit tests pinned the aliasing contract and the
fallback. Verified live, not just in unit tests — a one-shot probe in the taken
branch logged `F2A_SLOT_TAKEN t=7 slot_shape=(256, 2048)` from a real farm run,
so the A/B below exercised the new path rather than silently falling back.

### 24.2 Result: null, and the ceiling says it had to be

| arm | tok/s | med TPOT ms |
|---|---|---|
| `F2A_SLOT=0` | 103.37 / 101.92 | 124.1 / 125.8 |
| `F2A_SLOT=1` | 100.21 / 102.08 | 128.7 / 125.1 |
| **delta** | **−1.5%** | **+1.6%** |

Ranges overlap; this is a null, not a regression. **Reverted** (`ffn_compute.py`
and `environ.py` restored, test deleted) — a change with no measured upside
should not add an env var and a code path.

The arithmetic that predicted it, and the reason not to retry:

- The slot is `(256, 2048)` — `hidden_size=2048`, not 7168. Whole slot is
  `256*2048*2 B ≈ 1.0 MB`; a 7-row hop is `≈ 28 KB`. On a ~2 TB/s part the copy
  being removed is `≈ 0.5 us`.
- The hop it sits in is `2660 us` (§23). Ceiling `≈ 0.02%`, i.e. two orders of
  magnitude below the measurement noise this harness can resolve.
- Per hop the change removes 1 alloc + 1 pad-wide `zero_()` + replaces a 256-row
  copy with a 7-row copy: ~2 kernel launches saved. That the result is *neutral*
  is itself the finding — **the hop is not host-launch-bound on the F2A output
  path**, so the FFN's cost is genuine MoE GPU compute. (The parenthetical
  "consistent with §23's 84% occupancy" is void; §25 corrects occupancy to
  ~19–31%. The null result itself does not depend on it.)

### 24.3 Scope note for the "padding waste" framing

The 42x figure that motivated this (§23.6) was a *ratio* on a path with a tiny
absolute denominator. It is worth restating as a rule: multiply the ratio by the
absolute bytes before ranking it. On the A2F input side the same ratio argument
is still open (there `max_num_token` padding interacts with routing/GEMM shapes,
not just a copy), but the F2A output side is now measured dead.

Reproduce: `ATTN_GPU=6 FFN_GPU=7 MODES=sticky NUM_PROMPTS=64 MAX_CONCURRENCY=16
RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=192 CONTEXT_LENGTH=4096
SGLANG_AFD_FARM_MAX_INFLIGHT=8 SGLANG_AFD_NUM_MB=8
SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1 bash bench_farm_e2e.sh` with
`SGLANG_AFD_FFN_F2A_SLOT` applied per arm (the knob no longer exists in tree;
re-apply §24.1 to reproduce).

## 25. Re-measured timeline: the decomposition inverts, FFN was never the bottleneck

§23.6's "FFN sets the pace" and the `84%` occupancy behind §24's ceiling both
came from the **GIL-starved** harness: §23.1's clean timeline (`ffn_compute_us`
2.66 ms, Attn back at the wait in 71 us) was taken *before* §23.2 applied the
throttle + `--sleep-on-idle`. Since §23.2 moved cfg-2 from ~72 to ~102 tok/s,
every conclusion drawn from that snapshot had to be re-derived before ranking
further levers.

### 25.1 Method

Same operating point as §23.5 (`SLICE_CACHE=1`, cfg-2), `SGLANG_AFD_TIMELINE=1`
only for the first arm; a second arm adds `SGLANG_AFD_PROFILE_DETAIL=1` for FFN
sub-phase attribution (it costs throughput — 92.5 → 53.8 tok/s — so it is used
for attribution, never for tok/s). Note the timeline is a **ring buffer**, so
`AFD_RT_TIMELINE n=` is the ring size, not the hop count; hop counts below come
from the farm's own `picks` counter and from `AFD_TIMELINE layer=` log lines.

### 25.2 The hop decomposition, before and after the harness fix

| phase (p50, us) | §23.1 starved | §25 fixed |
|---|---|---|
| `post_to_ffn_us` | ~15 | 15 |
| `ffn_compute_us` | **2660** | **746** |
| `ffn_to_respond_us` | ~16 | 16 |
| `respond_to_wait_enter_us` | **71** | **1416** |
| `wait_enter_to_done_us` | ~20 | 20 |
| `total_rt_us` | ~3450 | **2218** |

The two dominant phases **swap places**. Fixing the GIL cut the FFN's per-hop
compute 3.6x, and Attn's "time to get back to the wait" rose 20x. Post-fix
`share: fixed=79% compute=21%` (detail arm) and `verdict: MERGE_HANDSHAKE`.

Read carefully, this says the opposite of §23.6: Attn takes 1416 us to return to
a wait whose result is already there, so the FFN's 746 us is **hidden inside
Attn's own per-hop work**. FFN sub-phases agree — `a2f_sync_us` p50 = 3 us (no
wait for A2F), `compute_wall_us` 621 us, `compute_cuda_us` 721 us,
`respond_us` 64 us: the hop is ~0.8 ms of mostly GPU time and one FFN worker
serves 1 hop at a time.

### 25.3 `84%` is retracted: FFN occupancy is ~31%

`mean_tok/launch = 6.0` (B_step=8, coalesce_k=1), so needed hops =
`tokens * 27 / tok_per_launch`:

- clean arm: `5973 * 27 / 6.0 ≈ 26880` hops (farm `picks` 27000, `AFD_TIMELINE`
  lines 27243 — all three agree). FFN busy `= 27000 * 0.746 ms = 20.1 s`. Decode
  window `= 5973 / 92.54 = 64.5 s` → **31%**.
- detail arm: 28625 computes, `compute_cuda_us` p50 = 721 us → 20.6 s busy over
  `5973 / 53.75 = 111.1 s` → **18.6%**.

So the honest range is **~19–31% FFN occupancy**, and the pace-setter is Attn's
per-hop work. §23.6's arithmetic used 317 hops/s; the measured rate is 418
(`27000 / 64.5`), and 418 × 2.66 ms was itself only accidentally near 1.0
because the starved compute time was inflated. Two errors compounding in the
same direction — the ratio looked like a saturated resource when it was not.

### 25.4 Where the Attn scheduler's time goes

`py-spy record` on the Attn `sglang::scheduler` at the corrected point (40 s,
250 Hz, 7755 samples). Rollup:

| bucket | share |
|---|---|
| attention (MLA) | 40.2% |
| a2f (send/credit path) | 21.2% |
| routing (gate + topk) | 13.2% |
| attn_compute (`run_layer_forward_pre_ffn`) | 9.2% |
| sched | 5.2% |
| slice (window/sampling) | 4.4% |
| unattributed | 6.7% |

`slice` at **4.4% is the §22.3 slice cache visibly working** (it was 11.9% when
§19.6 flagged it). Inside `a2f`, the leaves are flat — `unquant.apply` 4.7%
(A2F quantization), `_clone_optional` 3.8%, `fill_a2f` 5.8% across its four
branches, `scatter_rows` 2.7%, `wait_group` 1.9%, `synchronize` 1.7%:

```
367  4.73%  apply (quantization/unquant.py:155)
291  3.75%  _clone_optional (afd/farm/attn_farm.py:618)
206  2.66%  scatter_rows (afd/farm/batch_slice.py:299)
179  2.31%  fill_a2f (afd/buffers.py:180)
147  1.90%  wait_group (afd/farm/attn_farm.py:482)
130  1.68%  synchronize (torch/cuda/streams.py:108)
```

No single dominant leaf: an eighth of the loss is spread over ~6 sub-3% items,
which is why the previous micro-optimizations each bought only a few percent.

**Caveat, stated because it bounds what this can support:** py-spy samples
Python frames and cannot cleanly separate "CPU-busy" from "blocked in a CUDA
sync that released the GIL". The 40% `attention` bucket is *where* the thread
was, not proof that MLA is CPU-bound; some of it is GPU wait. §25.5 ranks by
structure, not by this split.

### 25.5 Consequence: the lever is Attn, and the structural one is amortisation

With the FFN at ~31% and its 746 us hidden, per-hop *Attn* work (~1.4 ms, and
spread across many small leaves) is what to attack. Micro-optimising one 3%
leaf buys ~3%. The structural lever is **tokens per hop**: `mean_tok/launch = 6`
means Attn's large fixed per-hop cost is amortised over only 6 tokens.

`SGLANG_AFD_FARM_COALESCE_K` (default 1, "dequeue up to `COALESCE_K * B_step`
tokens into one Attn launch") raises exactly that, and it was never A/B'd — no
`COALESCE_K` experiment exists anywhere in this document. What blocked it
before was §18's premise that the FFN was the saturated resource (a bigger hop
would only queue behind it). §25.3 removes that premise: the FFN has ~69%
headroom, so folding 2–4 hops into one should amortise Attn's per-hop cost
while the FFN absorbs the extra compute in the idle it already has.

This does **not** contradict §18/§20 — those measured FFN-side batching (bigger
*FFN* batches for the FFN's benefit, `PerLayerBatchQueue`/`extra_gather`) and
found it worthless. Here the beneficiary is the *other* process, which is only
visible now that FFN occupancy is known to be 31% and not 84%.

Ranked, with the §24 lesson (multiply the ratio by the absolute) applied:

1. **`COALESCE_K` 1 → 2 → 4** at cfg-2. Highest expected value, zero new code,
   directly targets the 79%-of-hop Attn share. Watch for the hop growing enough
   that FFN compute stops being hidden (~2.5–3 ms/hop at K=4).

   > **Refined in §26.** `COALESCE_K` alone is inert: the take is owner-bounded,
   > and at `num_contexts=2` the owner is only 8 rows (`B_STEP=8`). It pays only
   > once the owner is widened — `num_contexts=1` + `COALESCE_K=2` measured
   > **+32.9% tok/s / −26.8% TPOT**. Read §26 before running the sweep above.
2. A2F `fill_a2f` + `unquant` (~10% of Attn together): fuse the quantise/copy,
   or skip quantisation when the scale path is inactive.
3. `_clone_optional` in `enqueue` (3.8%) — §19's ~2.7% item, still there.

### 25.6 Corrections this section makes

- **§23.6 / §24's framing**: `ffn_compute_us` 2.66 ms and `317 hops/s * 2.66 ms
  ≈ 84%` occupancy are void; the corrected values are 746 us and ~19–31%.
  §24's *result* (F2A slot A/B = null) stands on its own A/B and is unaffected.
- **§23.6's "the FFN hop, not the Attn host path, sets the pace"** is reversed.
  §19/§22 were attacking the right process all along; the harness bug had
  temporarily hidden that.
- **§18's "no FFN-batching headroom"** was a statement about the FFN's *own*
  benefit. It does not bound `COALESCE_K`, whose beneficiary is Attn.

Reproduce: `ATTN_GPU=6 FFN_GPU=7 MODES=sticky NUM_PROMPTS=64 MAX_CONCURRENCY=16
RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=192 CONTEXT_LENGTH=4096
SGLANG_AFD_FARM_MAX_INFLIGHT=8 SGLANG_AFD_NUM_MB=8
SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1 SGLANG_AFD_FARM_SLICE_CACHE=1
SGLANG_AFD_TIMELINE=1 [SGLANG_AFD_PROFILE_DETAIL=1] bash bench_farm_e2e.sh`
(the harness supplies the §23.2 throttle + `--sleep-on-idle` by default).

## 26. Hop width: `num_contexts=1` + `COALESCE_K=2` = +33% tok/s, −27% TPOT

§25.5 predicted `COALESCE_K` would help because Attn's per-hop cost is amortised
over only ~6 tokens. It does — but **only in combination with a wider owner**,
and the first sweep on its own was a measurement of a dead knob. Both halves are
recorded because the failed half is what explains the mechanism.

### 26.1 Sweep 1: `COALESCE_K` alone is inert (and why)

`COALESCE_K` 1/2/4 at the §25 baseline (`num_contexts=2`, concurrency 16),
2 reps, order reversed:

| `COALESCE_K` | tok/s | med TPOT | `picks` | `tok/launch` |
|---|---|---|---|---|
| 1 | 100.96 / 100.88 | 126.7 / 127.5 | 27000 / 27000 | 6.0 / 6.0 |
| 2 | 102.23 / 103.41 | 124.8 / 123.2 | 27000 / 27000 | 6.0 / 6.0 |
| 4 | 101.74 / 100.88 | 126.0 / 127.4 | 27000 / 27000 | 6.0 / 6.0 |

The farm's own counters are **bit-identical** across all three (`picks`,
`tok/launch`), so the ±2% is noise and the knob did nothing. `coalesce_k=2` was
verified present in the FFN's startup log, so this is not a config-plumbing
failure.

**Why:** `pick()` computes `max_tok = b_step * ck` and calls
`_filtered_run` → `take_contiguous_run`, which takes the longest prefix of
*token keys that belong to the same run* — same owner, consecutive `seq_idx`.
The take is therefore `min(b_step * ck, owner_width)`, and with
`num_contexts=2` at concurrency 16 each owner is only **8 rows**: for `ck >= 1`
the take is owner-bounded at 8 and `ck` can never bind. `B_STEP=8` already
equals the owner width, which is exactly why nothing moved.

> Correction to §25.5 as written: it said the lever was "tokens per hop,
> `mean_tok/launch = 6`". That is right, but the binding constraint is not the
> `COALESCE_K` cap — it is the **owner width**, and `COALESCE_K` is a no-op until
> the owner is wider than `b_step`.

### 26.2 Sweep 2: widen the owner, then the cap binds

Disentangling the two changes at `num_contexts=1` (concurrency 16 → one 16-row
owner), 2 reps, order reversed:

| arm | ctx | K | tok/s | med TPOT | `tok/launch` | `picks` |
|---|---|---|---|---|---|---|
| `base` | 2 | 1 | 103.03 / 100.98 | 124.7 / 127.1 | 6.0 | 27000 |
| `c1k1` | 1 | 1 | 96.50 / 96.00 | 145.8 / 148.6 | 6.9 | 23706 |
| `c1k2` | 1 | 2 | 133.45 / 133.02 | 94.2 / 94.0 | 10.0 | 16254 |
| `c1k4` | 1 | 4 | **135.80 / 135.40** | **92.1 / 92.3** | 10.0 | 16254 |

- **`c1k4` vs `base`: +32.9% tok/s, −26.8% TPOT.** `c1k2` is +30.6% / −25.2%.
- `peak_layers_busy=2` in **every** arm, including all the `ctx=1` ones — the
  win is not bought by giving up cross-layer overlap. (`mean_win_len` rises
  1.00 → 1.47 and `B_win switches` falls 27000 → 16164, so windows are wider,
  not fewer layers busy.)
- `c1k4` and `c1k2` have **identical** `picks` (16254) and `tok/launch` (10.0):
  raising the cap from 16 to 32 changes nothing because the 16-row owner binds.
  That is the §26.1 mechanism confirmed from the other direction, and it is why
  `k=2` is the sensible setting — `k=4` buys nothing but the 0.5–2% it shows is
  within noise.
- `c1k1` (owner widened, cap still 8) is **worse** than `base` (−5.6% / +17.0%)
  despite 12% *fewer* hops. Fewer, heavier hops at an 8-row take is a bad trade;
  the win needs the cap raised to match the owner. Not chased further — the
  useful arms are unambiguous.
- `contexts_per_stage=2` (contexts enter together "to share same-layer A2F
  groups") was also tested with `ctx=2,K=2`: **84.8 / 85.0 tok/s = −16.9%**.
  Entering together does not produce the shared A2F group the docstring
  promises; do not use it.

### 26.3 What this is

Attn's per-hop cost is fixed and was being paid 27000 times for 6 tokens each.
Paying it 16254 times for 10 tokens each is worth a third of the throughput.
This is the §25.5 "amortise the fixed Attn cost" lever, now measured, and it is
the largest single win since §19/§22's host-path fixes.

Note this is **not** the FFN-side batching §18/§20 rejected. The FFN still serves
one hop at a time; the change is that each hop carries more tokens. §25.3's 31%
FFN occupancy is what makes it safe — but see the caveat below.

**Caveat / scope.** Owner width is `n_seq / num_contexts`, so the optimal
`(num_contexts, COALESCE_K)` pair is **concurrency-dependent**: at concurrency
32, `ctx=2` already yields 16-row owners and `ctx=1` would yield 32. These
numbers are for 1A1F at concurrency 16 only. Also unmeasured here: whether
`ctx=1` still wins under the 2A1F asymmetric topology of §21, and whether a
hop of 10 tokens grows FFN compute enough to stop being hidden (the §25.5
watch item; `tok/launch` is 10, not the 16 the owner could supply, so the ready
queue is the next thing to look at if this line continues).

The in-tree defaults are still `num_contexts=2`, `COALESCE_K=1` — i.e. the
configuration measured here as leaving ~33% on the table. Changing a default is
a separate call; the measurement stands either way.

Reproduce: `ATTN_GPU=6 FFN_GPU=7 MODES=sticky NUM_PROMPTS=64 MAX_CONCURRENCY=16
RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=192 CONTEXT_LENGTH=4096
SGLANG_AFD_FARM_MAX_INFLIGHT=8 SGLANG_AFD_NUM_MB=8
SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1 SGLANG_AFD_FARM_SLICE_CACHE=1
SGLANG_AFD_FARM_NUM_CONTEXTS=<1|2> STICKY_COALESCE_K=<1|2|4>
bash bench_farm_e2e.sh` (`STICKY_COALESCE_K` overrides `COALESCE_K` in the
sticky mode; both are read at farm startup and echoed in the FFN log).

## 27. Correctness: a real cross-sequence leak in the AFD split path (open)

While generalising §26's hop-width win, a correctness probe (`parity_e2e.py`,
new) found output corruption at concurrency. This section records what is
established, what is **ruled out**, the method corrections forced along the way,
and the environment blocker that stopped the investigation. **No perf result in
§19–§26 is invalidated**: both the `base` (ctx2 K1) and candidate (ctx1 K2)
configurations are affected, so this predates the hop-width change.

### 27.1 The leak, stated precisely

The probe sends 16 greedy prompts at concurrency 16 and compares runs. Greedy
output is a function of the prompt, so two *different* prompts producing the
*identical* 32-token string is a leak.

- **AFD, conc=16**: 8 of 19 recorded runs contain such a duplicate pair, and the
  pair is essentially always the same: **`(row 0, row 9)`** (7x), once
  `(row 0, row 5)`. Row 9's prompt asks for the first five primes; the leaked
  text is `1, 2, 3, 4, 5, 6, 7, 8, 9, 10,` — **verbatim row 0's answer** to
  "list the numbers 1 to 20".
- **Plain SGLang, no AFD**: the same row-9 prompt is correct in **all 6** runs
  (conc=1 and conc=16), and across 4 runs there are **zero** within-run duplicate
  pairs.

So the leak is real, AFD-specific, and its **source is always row 0** — the first
hop to occupy a slot. That fingerprint is consistent with a per-request
index/slot falling back to a default of 0, not with a static row offset.

### 27.2 Method correction: the "vs conc=1 oracle" rate is confounded

The obvious test — compare a conc=16 run against a conc=1 oracle of the same
prompts — **overstates the damage**, because batching itself is not
bit-reproducible upstream of AFD:

| plain (no AFD), conc=16 | result |
|---|---|
| conc=1 vs conc=16 (same server) | 13–14 / 16 |
| conc=16 self-consistency | 15, 16, 15 / 16 |
| within-run duplicate outputs | **0** (4 runs) |

Its divergences are all "same prompt, different plausible continuation" — no
cross-row match. So an exact-match test against a single oracle run can fail for
a perfectly correct server. **The trustworthy discriminator is the within-run
duplicate pair** (two different prompts → identical text), which plain never
produces and AFD does. All the "x/16 vs oracle" numbers below are kept for the
record but carry that caveat; the duplicate-pair canary is the load-bearing one.

### 27.3 Ruled out (each by a direct A/B)

Trigger is **concurrency + unequal input lengths**. At decode every sequence
contributes exactly one token, so the batch is uniform; unequal lengths only
exist at prefill. `ignore_eos` (equal finish times) still reproduces it, so it is
admission-time, not mid-flight batch shrink.

| hypothesis | arm | outcome |
|---|---|---|
| farm / persistent runtime | `SGLANG_AFD_FARM=0` (lockstep) | **still leaks**; wrong idx `[3,8..15]`, identical to farm-on |
| `num_contexts` / owner split | `ctx1_b16` (single 16-row owner) | still wrong (56% vs oracle), different idx |
| `B_STEP` micro-batch boundary | `B_STEP=16` (single sub-batch) | wrong idx `[3,8..15]`, identical to `B_STEP=8` |
| radix / prefix cache | `--disable-radix-cache` | still leaks; two runs give the *same* wrong set |
| batched ragged prefill | `--prefill-max-requests 1` | still leaks (`row0 == row9` in one of three) |
| A2F stale pad (§19.2) | `SGLANG_AFD_A2F_ZERO_PAD=1` A/B | no effect; 2 leaks in both arms. Diagnostic reverted |
| upstream SGLang | plain monolithic server | no leak signature |

`afd/` was added wholesale in `934933b592` (no earlier AFD to bisect against),
and neither `farm/README.md` nor `RFC.md` documents a correctness limitation, so
this is an **undocumented, pre-existing** bug.

### 27.4 Leading hypothesis, not yet confirmed: slot reuse (`NUM_MB`)

The only mechanism shared by the farm path and `farm0` is the A2F/F2A **slot
pool**. Preliminary sweep (before the environment failed):

| `NUM_MB` | runs | leak pairs |
|---|---|---|
| 1 | 2 | **0** |
| 2 | 1 | **0** |
| 8 | 0 | — (all earlier leak observations were at 8) |

This is suggestive but **not established** — the confirming sweep was cut short
(§27.5). Note the transport's generation check is sound on inspection
(`push_pull` allocates a monotonic `hid`; `respond` writes `done[mb]=hid`; `wait`
spins until `done[mb]==hid`), so a stale-F2A explanation would have to come from
elsewhere — e.g. cross-GPU visibility of `fill_a2f`'s direct write into shared
IPC memory versus the `a2f_ready` event, or the deferred `AfdHandle(id=-1-mb)`
path (not taken here: `USE_WAIT_FLAG=0`, `PIPELINE=0`).

### 27.5 Environment blocker (paused here)

The confirming sweep could not run: `/data/share/models` is an **NFS** mount and
NFS hung. `bench_farm_e2e.sh` sits in **D state** in `nfs3_proc_getattr`; every
server launch blocks reading weights off NFS. Observed: load average **287**,
**244** D-state processes (242 stuck `runc init`), `/tmp` at 94%.

This also **explains two earlier "failures" that must not be read as config
evidence**: parity15's `mb2_b` "FAIL bring-up" and `mb8_a`'s 68-minute hang were
the NFS outage, not `NUM_MB`. The `NUM_MB>=4` hypothesis therefore rests only on
"all historical leaks happened at 8", which is weak on its own.

### 27.6 State of the tree

- `bench_farm_e2e.sh`: added an `EXTRA_SERVER_ARGS` passthrough (needed for the
  prefill/radix arms; harmless otherwise). Kept.
- `python/sglang/srt/afd/parity_e2e.py`: the probe. New file. Kept.
- `SGLANG_AFD_A2F_ZERO_PAD` (in `buffers.py` + `environ.py`): A/B measured
  **null**, so it was reverted, per §24's rule.

### 27.7 Next step when NFS recovers

Re-run the `NUM_MB` sweep (1/2/4/8, 3 reps, per-arm timeout + retry). If only
`NUM_MB>=4` leaks, instrument the `respond`/`get_batch` generation handshake and
`fill_a2f`'s cross-GPU visibility; the "source is always row 0" fingerprint
predicts the corrupting write is the *first* hop's, so log per-`mb_id`
`handler`/`hid` around the recycle point. Longer term this gates flipping the
default to `num_contexts=1 + COALESCE_K=2` (§26), even though that config's
throughput win is unaffected.
