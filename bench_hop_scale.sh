#!/usr/bin/env bash
# Discriminator: is ffn_routed host-bound (launch/dispatch) or GPU-bound?
# Vary tokens-per-hop ~4x via B_STEP. Host-bound => ffn_routed flat.
set -uo pipefail
cd /root/.cuda/sglang

export ATTN_GPU=4 FFN_GPU=5
export NUM_PROMPTS=16 MAX_CONCURRENCY=8
export RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=64
export MODES=sticky
export MEM_FRACTION=0.75

export SGLANG_AFD_PROFILE_DETAIL=1
export SGLANG_AFD_FARM_FFN_SECTION_TIME=1
export SGLANG_AFD_PROFILE_DETAIL_EVERY=1024
export SGLANG_AFD_FARM_STAGE_STATS_EVERY=1

# name : B_STEP : MAX_INFLIGHT
CONFIGS=(
  "bs2:2:8"
  "bs16:16:2"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r name bs mi <<< "$cfg"
  d=/tmp/afd_hop_scale/$name
  rm -rf "$d"
  echo "########## $name (B_STEP=$bs max_inflight=$mi) ##########"
  env SGLANG_AFD_FARM_B_STEP=$bs SGLANG_AFD_FARM_MAX_INFLIGHT=$mi \
      OUT_DIR="$d" BASE_PORT=$((36100 + bs * 10)) \
      bash bench_farm_e2e.sh 2>&1 | tail -3
  echo "--- $name B_win ---"
  rg -o "AFD farm B_win.*" "$d/sticky/attn.log" 2>/dev/null | tail -1
  echo "--- $name FFN routed/prep ---"
  rg -o "ffn_routed_us:.*|ffn_prep_us:.*|compute_wall_us:.*|compute_cuda_us:.*" "$d/sticky/ffn.log" 2>/dev/null | tail -4
  echo "--- $name STAGE ---"
  rg -o "AFD farm STAGE.*" "$d/sticky/attn.log" 2>/dev/null | tail -1
done
echo "HOP_SCALE_DONE"
