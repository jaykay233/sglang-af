#!/usr/bin/env bash
# True 1F1B pipelining A/B: stagger microbatch entry so different microbatches
# occupy different layers concurrently. Judged by layers_peak (>1 = overlap).
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1
export SGLANG_AFD_FARM_PHASE_TIMING=1

port=35000
# name : STAGGER_MB : STAGGER_SPAN
CONFIGS=(
  "base:0:2"
  "st2s2:2:2"
  "st4s2:4:2"
  "st4s4:4:4"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name smb sspan <<< "$cfg"
  echo "########## $name (stagger_mb=$smb span=$sspan) ##########"
  env SGLANG_AFD_FARM_STAGGER_MB=$smb \
      SGLANG_AFD_FARM_STAGGER_SPAN=$sspan \
      SGLANG_AFD_FARM_MAX_INFLIGHT=8 \
      SGLANG_AFD_FARM_B_STEP=8 \
      OUT_DIR=/tmp/afd_pipe_stag/$name BASE_PORT=$port \
      bash bench_farm_e2e.sh 2>&1 | tail -3
  echo "--- $name STAGE ---"
  rg -o "AFD farm STAGE.*" /tmp/afd_pipe_stag/$name/sticky/attn.log 2>/dev/null | tail -1
  port=$((port + 10))
done
echo "PIPE_STAG_DONE"
