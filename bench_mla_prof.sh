#!/usr/bin/env bash
# Short farm run with AFD detail profiling on, to break down attn_core (MLA).
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5 BASE_PORT=33810
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export OUT_DIR=/tmp/afd_qproj_prof
export SGLANG_AFD_PROFILE_DETAIL=1
export SGLANG_AFD_PROFILE_DETAIL_EVERY=128

bash bench_farm_e2e.sh 2>&1 | tail -12
echo "MLA_PROF_DONE"
