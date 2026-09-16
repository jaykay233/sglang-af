#!/usr/bin/env bash
# Does splitting the batch into windows + letting FFN gather produce real overlap?
# Measures true pipeline depth (STAGE) and e2e for each config.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1
export SGLANG_AFD_FARM_PHASE_TIMING=1

port=34800
# name : B_STEP : NATURAL_BATCH : MAX_INFLIGHT : COALESCE_K
CONFIGS=(
  "base:8:0:2:1"
  "win4:4:0:4:1"
  "win4nb:4:1:4:1"
  "win2nb:2:1:8:1"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name bs nb mi ck <<< "$cfg"
  echo "########## $name (B_step=$bs natural_batch=$nb max_inflight=$mi ck=$ck) ##########"
  env SGLANG_AFD_FARM_B_STEP=$bs \
      SGLANG_AFD_FARM_NATURAL_BATCH=$nb \
      SGLANG_AFD_FARM_MAX_INFLIGHT=$mi \
      SGLANG_AFD_FARM_COALESCE_K=$ck \
      OUT_DIR=/tmp/afd_pipe_sweep/$name BASE_PORT=$port \
      bash bench_farm_e2e.sh 2>&1 | tail -3
  echo "--- $name STAGE ---"
  rg -o "AFD farm STAGE.*" /tmp/afd_pipe_sweep/$name/sticky/attn.log 2>/dev/null | tail -1
  port=$((port + 10))
done
echo "PIPE_SWEEP_DONE"
