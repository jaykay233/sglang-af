#!/usr/bin/env bash
# Hop decomposition: where does the ~2.8ms fixed FFN hop cost go?
# Emits AFD_DETAIL_PROFILE blocks (a2f_sync / post_poll / compute_wall / compute_cuda
# / ffn_* sections / respond) plus STAGE + PHASE for the top-level breakdown.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75

# Hop decomposition hooks
export SGLANG_AFD_PROFILE_DETAIL=1
export SGLANG_AFD_FARM_FFN_SECTION_TIME=1
export SGLANG_AFD_PROFILE_DETAIL_EVERY=1024
# Top-level breakdown
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1
export SGLANG_AFD_FARM_PHASE_TIMING=1

for rep in 1 2; do
  d=/tmp/afd_hop_prof/rep$rep
  rm -rf "$d"
  echo "########## hop_prof rep=$rep ##########"
  OUT_DIR="$d" BASE_PORT=$((35900 + rep * 20)) bash bench_farm_e2e.sh 2>&1 | tail -4
  echo "--- rep$rep STAGE ---"
  rg -o "AFD farm STAGE.*" "$d/sticky/attn.log" 2>/dev/null | tail -1
  echo "--- rep$rep PHASE ---"
  rg -o "AFD farm PHASE.*" "$d/sticky/attn.log" 2>/dev/null | tail -1
done
echo "HOP_PROF_DONE"
