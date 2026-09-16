#!/usr/bin/env bash
# Replica (no AFD, single GPU) vs 1A1F under identical model/client config.
#
# The AFD harness only ever benchmarks AF configurations, so the plain
# single-GPU server was never measured on the same workload. This runs both,
# interleaved, and reports the same metrics from the same client invocation.
#
#   bash bench_replica_vs_1a1f.sh
#
# Env: REPS (default 3), G (client config: small|std), REPLICA_GPU, plus the
# usual ATTN_GPU_1A1F / FFN_GPUS_1A1F.
set -uo pipefail

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT_ROOT="${OUT_ROOT:-/tmp/afd_rep_vs_1a1f}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
REPLICA_GPU="${REPLICA_GPU:-7}"
REPS="${REPS:-3}"
G="${G:-small}"

case "$G" in
  small) NUM_PROMPTS=16;  RANDOM_OUTPUT_LEN=32;  MAX_NUM_TOKEN=384  ;;
  std)   NUM_PROMPTS=64;  RANDOM_OUTPUT_LEN=192; MAX_NUM_TOKEN=1024 ;;
  *) echo "G must be small|std" >&2; exit 1 ;;
esac
RANDOM_INPUT_LEN=128
MAX_CONCURRENCY=16
WARMUP_REQUESTS=1
BASE_PORT="${BASE_PORT:-33910}"

mkdir -p "$OUT_ROOT"
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

declare -a CLEAN_PIDS=()

cleanup_all() {
  local p
  for p in "${CLEAN_PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    kill -TERM "$p" 2>/dev/null || true
    pkill -TERM -P "$p" 2>/dev/null || true
  done
  sleep 3
  for p in "${CLEAN_PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    kill -KILL "$p" 2>/dev/null || true
    pkill -KILL -P "$p" 2>/dev/null || true
  done
  CLEAN_PIDS=()
}
trap cleanup_all EXIT

wait_health() {
  local port="$1" i
  for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

run_client() {  # $1=port $2=outfile
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$1" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --output-file "$2" \
    --disable-tqdm >/dev/null 2>&1
}

report() {  # $1=json $2=label
  python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("%-9s out_tps=%7.2f  med_tpot=%8.2f  p99_tpot=%8.2f  med_ttft=%8.2f  med_e2e=%10.2f"
      % (sys.argv[2], d["output_throughput"], d["median_tpot_ms"],
         d["p99_tpot_ms"], d["median_ttft_ms"], d["median_e2e_latency_ms"]))
PY
}

run_replica() {  # $1=tag
  local tag="$1" port="$2"
  local d="$OUT_ROOT/$tag"; mkdir -p "$d"
  (
    unset SGLANG_AFD_MODE SGLANG_AFD_FARM SGLANG_AFD_POOL SGLANG_AFD_TRANSPORT \
          SGLANG_AFD_MODULE_STUBS SGLANG_AFD_POOL_NUM_ATTN SGLANG_AFD_POOL_NUM_FFN \
          SGLANG_AFD_POOL_LOCAL_RANK SGLANG_AFD_POOL_ENDPOINT_DIR
    export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
    export CUDA_VISIBLE_DEVICES="$REPLICA_GPU"
    python3 -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --host 127.0.0.1 --tp-size 1 \
      --dtype bfloat16 --mem-fraction-static "$MEM_FRACTION" \
      --max-running-requests "$MAX_CONCURRENCY" --context-length "$CONTEXT_LENGTH" \
      --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
      --disaggregation-mode null \
      --port "$port"
  ) >"$d/server.log" 2>&1 &
  CLEAN_PIDS+=("$!")
  if ! wait_health "$port"; then
    echo "replica $tag: server failed to become healthy" >&2
    tail -20 "$d/server.log" >&2
    return 1
  fi
  run_client "$port" "$d/bench.json"
  report "$d/bench.json" "$tag"
}

run_1a1f() {  # $1=tag
  local tag="$1"
  local d="$OUT_ROOT/$tag"
  rm -rf "$d"
  ATTN_GPU_1A1F="${ATTN_GPU_1A1F:-7}" FFN_GPUS_1A1F="${FFN_GPUS_1A1F:-4}" CASES=1a1f \
  BASE_PORT="$BASE_PORT" MEM_FRACTION="$MEM_FRACTION" CONTEXT_LENGTH="$CONTEXT_LENGTH" \
  SGLANG_AFD_FARM_MAX_INFLIGHT=32 \
  SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS=250 SGLANG_AFD_FFN_SLEEP_ON_IDLE=1 \
  SGLANG_AFD_MOE_SUM_REDUCE_COMPILE=0 \
  SGLANG_AFD_FARM_PERSISTENT=0 \
  NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  RANDOM_INPUT_LEN="$RANDOM_INPUT_LEN" RANDOM_OUTPUT_LEN="$RANDOM_OUTPUT_LEN" \
  WARMUP_REQUESTS="$WARMUP_REQUESTS" MAX_NUM_TOKEN="$MAX_NUM_TOKEN" \
  OUT_DIR="$d" \
  bash python/sglang/srt/afd/farm/bench_pool_e2e.sh >"$d.log" 2>&1
  report "$d/1a1f_bench.json" "$tag"
}

echo "=== replica(no-AFD,1GPU) vs 1A1F | config=$G n=$NUM_PROMPTS out=$RANDOM_OUTPUT_LEN reps=$REPS ==="
for r in $(seq 1 "$REPS"); do
  run_1a1f "1a1f_${G}_r${r}" "$((BASE_PORT))"
  cleanup_all; sleep 3
  run_replica "replica_${G}_r${r}" "$((BASE_PORT + 40))"
  cleanup_all; sleep 3
done
echo "=== done ==="
