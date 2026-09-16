#!/usr/bin/env bash
# Sweep farm batching knobs on 1A1F only, CUDA graph disabled everywhere.
set -uo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT=/tmp/afd_sweep_20260915
mkdir -p "$ROOT"

# name:ENV overrides (space separated)
CONFIGS=(
  "ref_b16c4:SGLANG_AFD_FARM_B_STEP=16 SGLANG_AFD_FARM_COALESCE_K=4 SGLANG_AFD_FARM_LAYER_BURST=0 SGLANG_AFD_FARM_SCHED=max"
  "wf_b8c1:SGLANG_AFD_FARM_B_STEP=8 SGLANG_AFD_FARM_COALESCE_K=1 SGLANG_AFD_FARM_LAYER_BURST=0 SGLANG_AFD_FARM_SCHED=max"
  "wf_b4c1:SGLANG_AFD_FARM_B_STEP=4 SGLANG_AFD_FARM_COALESCE_K=1 SGLANG_AFD_FARM_LAYER_BURST=0 SGLANG_AFD_FARM_SCHED=max"
  "wf_b16c2:SGLANG_AFD_FARM_B_STEP=16 SGLANG_AFD_FARM_COALESCE_K=2 SGLANG_AFD_FARM_LAYER_BURST=0 SGLANG_AFD_FARM_SCHED=max"
  "wf_b8c1_burst:SGLANG_AFD_FARM_B_STEP=8 SGLANG_AFD_FARM_COALESCE_K=4 SGLANG_AFD_FARM_LAYER_BURST=8 SGLANG_AFD_FARM_SCHED=max"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  overrides="${entry#*:}"
  echo "############################ $name ($overrides) $(date)"
  # shellcheck disable=SC2086
  env $overrides \
    CASES=1a1f \
    OUT_DIR="$ROOT/$name" \
    BASE_PORT=33910 \
    SGLANG_AFD_NUM_MB=16 \
    SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16 \
    bash farm/bench_pool_e2e.sh
  echo "############################ $name done rc=$? $(date)"
  sleep 5
done

echo "ALL SWEEP DONE $(date)"
