#!/usr/bin/env bash
# §8 probe: is ffn_routed's ~1.6ms host cost CUDA launch overhead or Python dispatch?
# One-shot torch.profiler census inside the FFN worker.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75

# detail profiling + the new one-shot kernel census
export SGLANG_AFD_PROFILE_DETAIL=1
export SGLANG_AFD_FARM_FFN_SECTION_TIME=1
export SGLANG_AFD_FARM_FFN_PROBE=1
export SGLANG_AFD_PROFILE_DETAIL_EVERY=100000
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1

d=/tmp/afd_ffn_probe
rm -rf "$d"
OUT_DIR="$d" BASE_PORT=36400 bash bench_farm_e2e.sh 2>&1 | tail -4
echo "FFN_PROBE_DONE"
