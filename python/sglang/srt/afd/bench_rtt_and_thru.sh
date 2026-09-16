#!/usr/bin/env bash
# 1) REMOTE_FROM_LAYER sweep (in-graph TPOT → 15–18ms band)
# 2) PD vs AFD throughput (high concurrency; AFD optional FFN gather)
set -euo pipefail
source /root/.cuda/afd_env.sh

BENCH=/root/.cuda/sglang/python/sglang/srt/afd/bench_pd_vs_afd.sh
ROOT_OUT="${ROOT_OUT:-/tmp/afd_rtt_thru}"
PREFILL_GPU="${PREFILL_GPU:-4}"
DECODE_GPU="${DECODE_GPU:-5}"
FFN_GPU="${FFN_GPU:-6}"
mkdir -p "$ROOT_OUT"
SUMMARY="$ROOT_OUT/SUMMARY.txt"
: >"$SUMMARY"

run_one() {
  local tag=$1
  shift
  local out="$ROOT_OUT/$tag"
  mkdir -p "$out"
  echo "===== $tag $* =====" | tee -a "$SUMMARY"
  # shellcheck disable=SC2086
  env OUT_DIR="$out" BASE_PORT="$BASE_PORT" \
    PREFILL_GPU="$PREFILL_GPU" DECODE_GPU="$DECODE_GPU" FFN_GPU="$FFN_GPU" \
    SGLANG_AFD_TRANSPORT=cuda_ipc \
    "$@" \
    bash "$BENCH" >"$out/run.log" 2>&1 || {
      echo "FAIL $tag" | tee -a "$SUMMARY"
      rg -n 'TimeoutError|Traceback|ERROR|AFD_PD' "$out/run.log" | tail -20 | tee -a "$SUMMARY" || true
      return 1
    }
  rg -n 'median_tpot_ms|output_throughput|request_throughput|AFD_PD_COMPARE_OK|remote_from=' \
    "$out/run.log" | tee -a "$SUMMARY" || true
  echo | tee -a "$SUMMARY"
}

echo "Comfy check: $(pgrep -af 'python main.py --listen' | wc -l) procs" | tee -a "$SUMMARY"

# ---------- Part A: from_layer TPOT sweep (in-graph, fair vs prior 18→20ms) ----------
BASE_PORT=34000
for K in 20 22 24; do
  run_one "tpot_from_${K}" \
    MODES=pd_afd \
    AFD_IN_GRAPH_WAIT=1 AFD_TRUE_OVERLAP=0 \
    SGLANG_AFD_REMOTE_FROM_LAYER="$K" \
    NUM_PROMPTS=16 RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=64 \
    MAX_CONCURRENCY=8 DECODE_MAX_BS=16 WARMUP_REQUESTS=2 \
    || true
  BASE_PORT=$((BASE_PORT + 100))
done

# ---------- Part B: throughput PD vs AFD ----------
# High concurrency; AFD keeps best RTT knob (from_layer=22 as mid) + in-graph.
# Second AFD variant: NUM_MB=4 + gather (breakable) for multi-mb FFN batching.
BASE_PORT=35000
run_one "thru_pd_vs_ingraph" \
  MODES=pd,pd_afd \
  AFD_IN_GRAPH_WAIT=1 AFD_TRUE_OVERLAP=0 \
  SGLANG_AFD_REMOTE_FROM_LAYER="${THRU_FROM_LAYER:-22}" \
  NUM_PROMPTS=64 RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=64 \
  MAX_CONCURRENCY=16 DECODE_MAX_BS=16 WARMUP_REQUESTS=4 \
  || true

BASE_PORT=35100
run_one "thru_afd_gather" \
  MODES=pd_afd \
  AFD_IN_GRAPH_WAIT=0 AFD_TRUE_OVERLAP=0 \
  AFD_LAYER_PIPELINE=0 SGLANG_AFD_PIPELINE=1 \
  SGLANG_AFD_NUM_MB=4 \
  SGLANG_AFD_REMOTE_FROM_LAYER="${THRU_FROM_LAYER:-22}" \
  SGLANG_AFD_FFN_GATHER_US=500 SGLANG_AFD_FFN_GATHER_MAX=4 \
  NUM_PROMPTS=64 RANDOM_INPUT_LEN=256 RANDOM_OUTPUT_LEN=64 \
  MAX_CONCURRENCY=16 DECODE_MAX_BS=16 WARMUP_REQUESTS=4 \
  || true

echo "DONE → $SUMMARY" | tee -a "$SUMMARY"
