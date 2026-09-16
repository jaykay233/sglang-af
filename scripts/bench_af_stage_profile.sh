#!/usr/bin/env bash
# Stage/concurrency profile of the 1A1F decode farm, CUDA graph disabled.
# Goal: find whether A2F hops overlap across layers or run lockstep.
set -uo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT="${ROOT:-/tmp/afd_stage_20260915}"
mkdir -p "$ROOT"

COMMON_ENV=(
  SGLANG_AFD_FARM_PHASE_TIMING=1
  SGLANG_AFD_FARM_STAGE_STATS_EVERY=16
  SGLANG_AFD_FARM_LPU_STATS_EVERY=512
  SGLANG_AFD_FARM_LPU_GATHER_US=50
  CASES=1a1f
  ATTN_GPU_1A1F=4
  FFN_GPUS_1A1F=5
  NUM_PROMPTS=64
  MAX_CONCURRENCY=32
  RANDOM_INPUT_LEN=128
  RANDOM_OUTPUT_LEN=128
)

run_case() {
  local tag=$1
  shift
  echo "############################ $tag $(date)"
  env "${COMMON_ENV[@]}" "$@" \
    OUT_DIR="$ROOT/$tag" \
    BASE_PORT="${BASE_PORT:-43910}" \
    bash farm/bench_pool_e2e.sh
  echo "############################ $tag done rc=$? $(date)"
  sleep 5
}

run_case best_b16c2 env \
  SGLANG_AFD_FARM_B_STEP=16 \
  SGLANG_AFD_FARM_B_WIN_K=1 \
  SGLANG_AFD_FARM_COALESCE_K=2 \
  SGLANG_AFD_FARM_LAYER_BURST=0 \
  SGLANG_AFD_FARM_SCHED=max \
  SGLANG_AFD_NUM_MB=16 \
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16

run_case b32c1 env \
  SGLANG_AFD_FARM_B_STEP=32 \
  SGLANG_AFD_FARM_B_WIN_K=1 \
  SGLANG_AFD_FARM_COALESCE_K=1 \
  SGLANG_AFD_FARM_LAYER_BURST=0 \
  SGLANG_AFD_FARM_SCHED=max \
  SGLANG_AFD_NUM_MB=16 \
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16

echo "ALL STAGE PROFILE DONE $(date)"
