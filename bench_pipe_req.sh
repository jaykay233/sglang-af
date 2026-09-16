#!/usr/bin/env bash
# Per-request granularity (B_step=1) + deepest-first scheduling to break lockstep.
# Hypothesis: cap=1 forces tokens onto DIFFERENT layers -> real cross-layer overlap.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1
export SGLANG_AFD_FARM_PHASE_TIMING=1

port=34900
# name : B_STEP : SCHED : PER_LAYER_CAP : MAX_INFLIGHT
CONFIGS=(
  "deep_cap1:1:deepest:1:8"
  "max_cap1:1:max:1:8"
  "deep_cap2:2:deepest:1:8"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name bs sc cap mi <<< "$cfg"
  echo "########## $name (B_step=$bs sched=$sc per_layer_cap=$cap max_inflight=$mi) ##########"
  env SGLANG_AFD_FARM_B_STEP=$bs \
      SGLANG_AFD_FARM_SCHED=$sc \
      SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER=$cap \
      SGLANG_AFD_FARM_MAX_INFLIGHT=$mi \
      SGLANG_AFD_FARM_NATURAL_BATCH=0 \
      SGLANG_AFD_FARM_COALESCE_K=1 \
      OUT_DIR=/tmp/afd_pipe_req/$name BASE_PORT=$port \
      bash bench_farm_e2e.sh 2>&1 | tail -3
  echo "--- $name STAGE ---"
  rg -o "AFD farm STAGE.*" /tmp/afd_pipe_req/$name/sticky/attn.log 2>/dev/null | tail -1
  port=$((port + 10))
done
echo "PIPE_REQ_DONE"
