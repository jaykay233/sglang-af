#!/usr/bin/env bash
# Decompose the 1A1F farm hop: attn-GPU wait vs MoE host launch vs MoE GPU.
set -uo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT="${ROOT:-/tmp/afd_hop_split_20260915}"
mkdir -p "$ROOT"

COMMON_ENV=(
  SGLANG_AFD_FARM_PHASE_TIMING=1
  SGLANG_AFD_FARM_STAGE_STATS_EVERY=16
  SGLANG_AFD_PROFILE_DETAIL=1
  SGLANG_AFD_PROFILE_DETAIL_EVERY=256
  SGLANG_AFD_PROFILE_DETAIL_OUT="$ROOT/detail_common.txt"
  SGLANG_AFD_FARM_FFN_TIME_SPLIT=1
  SGLANG_AFD_FARM_FFN_SECTION_TIME=1
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
    SGLANG_AFD_PROFILE_DETAIL_OUT="$ROOT/detail_${tag}.txt" \
    OUT_DIR="$ROOT/$tag" \
    BASE_PORT="${BASE_PORT:-44910}" \
    bash farm/bench_pool_e2e.sh
  echo "############################ $tag done rc=$? $(date)"
  sleep 5
}

run_case b32c1 env \
  SGLANG_AFD_FARM_B_STEP=32 \
  SGLANG_AFD_FARM_B_WIN_K=1 \
  SGLANG_AFD_FARM_COALESCE_K=1 \
  SGLANG_AFD_FARM_LAYER_BURST=0 \
  SGLANG_AFD_FARM_SCHED=max \
  SGLANG_AFD_NUM_MB=16 \
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16

echo "HOP SPLIT DONE $(date)"
