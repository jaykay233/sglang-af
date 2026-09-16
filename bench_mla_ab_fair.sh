#!/usr/bin/env bash
# Fair A/B: triton vs flashinfer MLA backend, identical load, 2 reps each,
# with AFD detail profiling so we can see attn_core itself move.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=3 FFN_GPU=4
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75
export SGLANG_AFD_PROFILE_DETAIL=1
export SGLANG_AFD_PROFILE_DETAIL_EVERY=256

port=34000
for be in triton flashinfer; do
  for rep in 1 2; do
    echo "########## $be rep$rep ##########"
    skip=""
    [[ "$be" == "flashinfer" ]] && skip="SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1"
    env $skip OUT_DIR=/tmp/afd_mla_ab/$be$rep BASE_PORT=$port ATTN_BACKEND=$be \
      bash bench_farm_e2e.sh 2>&1 | tail -3
    port=$((port + 10))
  done
done
echo "MLA_AB_FAIR_DONE"
