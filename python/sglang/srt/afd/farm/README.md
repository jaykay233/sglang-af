# Decode farm (P0–P3)

Tokens at **different layers** in one decode step. Replaces lockstep
`model.forward()` all-layers gather. Transport: **cuda_ipc** (no StepMesh).

## Enable (measured defaults)

```bash
export SGLANG_AFD_FARM=1
export SGLANG_AFD_TRANSPORT=cuda_ipc
export SGLANG_AFD_FARM_B_STEP=16
export SGLANG_AFD_FARM_B_WIN_K=8
# Independent sequence-context wavefronts. Context 1 starts at layer 0
# while context 0 is already advancing through later layers.
export SGLANG_AFD_FARM_NUM_CONTEXTS=2
export SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS=1
export SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=1
# This is the overlap-validation configuration, not yet the throughput
# default. Splitting a coalesced decode window adds per-hop fixed cost;
# use NUM_CONTEXTS=1 for the current throughput baseline until same-layer
# hops can be coalesced across independent contexts.
# Print per-forward overlap counters (layers_peak, span_peak, ctxs_peak).
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=32
# Throughput / amortize: pack windows (stock cuBLAS sees large M)
export SGLANG_AFD_FARM_COALESCE_K=4
# Low TPOT: don't wait to coalesce — resident-W spin-wait session
# export SGLANG_AFD_FARM_COALESCE_K=1
# export SGLANG_AFD_FARM_SPIN_WAIT=1
export SGLANG_AFD_FARM_LAYER_CG=1          # optional launch amortize
# export SGLANG_AFD_FARM_PERSISTENT_LINEAR=1  # Triton weight-outer (slower than cuBLAS)
```

## Measure first (plan gate)

```bash
CUDA_VISIBLE_DEVICES=7 python -m sglang.srt.afd.farm.bench_farm_amortize \
  --b-steps 8,16 --mbs 1,2,4,8 --k 2048 --n 2112 --json-out /tmp/farm_amortize_kpi.json
```

## E2E serve (tok/s · TPOT)

True AF 1A+1F + farm, concurrent load; compares sticky / coalesce / spin_wait.

**HBM-safe defaults** (A800 + ComfyUI ~40GB free): `MAX_NUM_TOKEN=256`,
`FFN_CUDA_GRAPH=0`, `mem-fraction-static=0.75`, `context-length=2048`.

```bash
source /root/.cuda/afd_env.sh
ATTN_GPU=2 FFN_GPU=3 OUT_DIR=/tmp/afd_farm_e2e \
  NUM_PROMPTS=32 MAX_CONCURRENCY=8 \
  bash python/sglang/srt/afd/farm/bench_farm_e2e.sh
```

Stream e2e (in=128 / out=64 / conc=8), `MAX_NUM_TOKEN=512`, mem=0.78:

| mode | out tok/s | med TPOT (ms) | vs sticky tok/s | vs sticky TPOT |
|------|----------:|--------------:|----------------:|---------------:|
| sticky | 43.9 | 158.4 | — | — |
| coalesce (K=4) | 44.4 | 157.7 | 1.01× | ~1.00× |
| spin_wait | 41.0 | 172.2 | 0.94× | 1.09× |

Farm log still showed `amortize_factor≈1.0` (load did not fill coalesce windows).

Observed on A800 (projection microbench, T_arr=50µs proxy):

| Finding | Implication |
|---------|-------------|
| fused cuBLAS ≫ separate tiny GEMMs (~5–11× at mb=8) | **COALESCE_K** is the real HBM amortize win |
| Triton weight-outer < separate wall time | structural proof only; keep off for perf |
| coalesce TPOT proxy ~**1.76×** sticky | waiting for K windows hurts latency |
| spin-wait session beats coalesce_proxy when T_arr matters | enable **SPIN_WAIT** + `COALESCE_K=1` for low TPOT |

## KPI / honesty

| Knob | Effect |
|------|--------|
| sticky `B_win` | scheduling orifice |
| **`COALESCE_K>1`** | larger M → stock-kernel weight amortize (best tok/s) |
| **`SPIN_WAIT`** | resident W, push `B_step` mbs without coalesce wait (best TPOT proxy) |
| `LAYER_CG` | launch amortize; weights still re-read |
| `PERSISTENT_LINEAR` | Triton weight-outer; usually slower than cuBLAS |
| FlashInfer MLA / SM-resident poll grid | **not** done |

## Layout

| Module | Role |
|--------|------|
| `token_queue.py` | sticky + coalesce |
| `persistent_linear.py` | Triton weight-outer + MLA wrap |
| `spin_wait_linear.py` | resident-W session (spin-wait MVP) |
| `bench_farm_amortize.py` | coalesce × linear × TPOT gate |
| `decode_farm_loop.py` | farm state machine |
