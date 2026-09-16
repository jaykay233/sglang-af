#!/usr/bin/env bash
# Ref config with per-phase wall-clock breakdown of the farm loop.
set -uo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT=/tmp/afd_phase_20260915
mkdir -p "$ROOT"

env \
  SGLANG_AFD_FARM_PHASE_TIMING=1 \
  SGLANG_AFD_FARM_B_STEP=16 \
  SGLANG_AFD_FARM_COALESCE_K=4 \
  SGLANG_AFD_FARM_LAYER_BURST=0 \
  SGLANG_AFD_FARM_SCHED=max \
  CASES=1a1f \
  OUT_DIR="$ROOT/ref" \
  BASE_PORT=35910 \
  SGLANG_AFD_NUM_MB=16 \
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16 \
  bash farm/bench_pool_e2e.sh

echo "PHASE DONE $(date)"
