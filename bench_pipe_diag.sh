#!/usr/bin/env bash
# Measure the farm's TRUE pipeline depth and where host wall-clock goes.
# Enables stage stats (in-flight layers, hop latency, block time) + phase timing.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5 BASE_PORT=34700
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export OUT_DIR=/tmp/afd_pipe_diag

# True pipeline-depth instrumentation.
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1
export SGLANG_AFD_FARM_PHASE_TIMING=1

bash bench_farm_e2e.sh 2>&1 | tail -10

echo "===== STAGE (pipeline depth) ====="
rg -o "AFD farm STAGE.*" /tmp/afd_pipe_diag/sticky/attn.log 2>/dev/null | tail -6
echo "===== PHASE (host wall) ====="
rg -o "AFD farm PHASE.*" /tmp/afd_pipe_diag/sticky/attn.log 2>/dev/null | tail -6
echo "PIPE_DIAG_DONE"
