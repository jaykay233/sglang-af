#!/usr/bin/env bash
# 1A1F diagnostic: timeline + detail profile + farm phase timing in one run.
set -uo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT=/tmp/afd_diag2_20260915
mkdir -p "$ROOT"
rm -f "$ROOT"/timeline.txt "$ROOT"/detail.txt

env \
  SGLANG_AFD_TIMELINE=1 \
  SGLANG_AFD_TIMELINE_OUT="$ROOT/timeline.txt" \
  SGLANG_AFD_PROFILE_DETAIL=1 \
  SGLANG_AFD_PROFILE_DETAIL_EVERY=512 \
  SGLANG_AFD_PROFILE_DETAIL_OUT="$ROOT/detail.txt" \
  SGLANG_AFD_FARM_PHASE_TIMING=1 \
  SGLANG_AFD_FARM_B_STEP=16 \
  SGLANG_AFD_FARM_COALESCE_K=4 \
  SGLANG_AFD_FARM_LAYER_BURST=0 \
  SGLANG_AFD_FARM_SCHED=max \
  CASES=1a1f \
  OUT_DIR="$ROOT/ref" \
  BASE_PORT=36910 \
  SGLANG_AFD_NUM_MB=16 \
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16 \
  bash farm/bench_pool_e2e.sh

echo "DIAG2 DONE $(date)"
