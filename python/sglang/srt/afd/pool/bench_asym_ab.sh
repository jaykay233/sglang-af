#!/usr/bin/env bash
# Decisive test for the NA1F FFN head-of-line fix.
#
# The synthetic bench_af_pool drives both Attn clients with the same lockstep
# workload, so link0 is rarely idle while link1 has a ready hop — the HOL
# condition never arises, and legacy vs drain-all measured identically.
#
# Here the load is ASYMMETRIC: every request is sent straight to Attn1, so
# Attn0's FFN link is permanently idle. That is exactly when per-link blocking
# hurts: legacy _poll_ready blocks on idle link0 for the whole poll timeout
# before it ever looks at link1, which holds the ready hop.
#
#   ARM=drainall  SGLANG_AFD_FFN_POLL_DRAIN_ALL=1  (new)
#   ARM=legacy    SGLANG_AFD_FFN_POLL_DRAIN_ALL=0  (old)
#
# Self-prefill (PD=null, no Prefill process): Attn0 GPU5, Attn1 GPU6, FFN GPU7.
set -uo pipefail
source /root/.cuda/afd_env.sh

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
ARM="${ARM:?set ARM=drainall|legacy}"
OUT="${OUT_DIR:-/tmp/afd_asym_ab}/${ARM}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
IN_LEN="${RANDOM_INPUT_LEN:-1}"
OUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
MAX_RUNNING="${MAX_RUNNING_REQUESTS:-8}"

ATTN0_GPU="${ATTN0_GPU:-5}"
ATTN1_GPU="${ATTN1_GPU:-6}"
FFN_GPU="${FFN_GPU:-7}"
BASE="${PORT_BASE:-36000}"
a0=$((BASE+1)) a1=$((BASE+2)) ffn=$((BASE+0))

rm -rf "$OUT"; mkdir -p "$OUT/socks"

common_model=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.82
  --max-running-requests "$MAX_RUNNING" --context-length 4096 --max-total-tokens 50000
  --cuda-graph-backend-prefill disabled
)

case "$ARM" in
  drainall) DRAIN=1 ;;
  legacy)   DRAIN=0 ;;
  *) echo "ARM must be drainall|legacy" >&2; exit 2 ;;
esac

kill_ports() { local p; for p in "$@"; do fuser -k "${p}/tcp" >/dev/null 2>&1 || true; done; sleep 2; }

wait_http() {
  local url=$1 name=$2 pid=$3 timeout=${4:-360} i
  for i in $(seq 1 "$timeout"); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "$name DIED" >&2; return 1; fi
    curl -sf "$url" >/dev/null 2>&1 && { echo "$name healthy ${i}s"; return 0; }
    sleep 1
  done
  echo "$name TIMEOUT" >&2; return 1
}

stop_pids() {
  local f=$1
  [[ -f "$f" ]] || return 0
  while read -r p; do kill -TERM "$p" 2>/dev/null || true; pkill -TERM -P "$p" 2>/dev/null || true; done <"$f"
  sleep 4
  while read -r p; do kill -KILL "$p" 2>/dev/null || true; pkill -KILL -P "$p" 2>/dev/null || true; done <"$f"
}

cleanup() {
  stop_pids "$OUT/pids"
  kill_ports "$a0" "$a1" "$ffn"
}
trap cleanup EXIT

start_ffn() {
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$FFN_GPU"
    export SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN=2 SGLANG_AFD_POOL_NUM_FFN=1
    export SGLANG_AFD_POOL_LOCAL_RANK=0 SGLANG_AFD_POOL_ENDPOINT_DIR="$OUT/socks"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=8
    export SGLANG_AFD_MODULE_STUBS=1 SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN=256
    export SGLANG_AFD_FFN_CUDA_GRAPH=1
    # --- the experiment ---
    export SGLANG_AFD_FFN_POLL_DRAIN_ALL="$DRAIN"
    # new KPI: FFN-side self-reported serve occupancy
    export SGLANG_AFD_POOL_UTIL_FILE="$OUT/ffn_util.csv"
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --port "$ffn" --disaggregation-mode null --skip-server-warmup
  ) >"$OUT/ffn.log" 2>&1 &
  echo $!
}

start_attn() {
  local gpu=$1 port=$2 rank=$3 log=$4
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_AFD_MODE=attn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN=2 SGLANG_AFD_POOL_NUM_FFN=1
    export SGLANG_AFD_POOL_LOCAL_RANK="$rank" SGLANG_AFD_POOL_ENDPOINT_DIR="$OUT/socks"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=8
    export SGLANG_AFD_POOL_ROUTE=least_inflight
    export SGLANG_AFD_MODULE_STUBS=1
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN=256
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --cuda-graph-backend-decode breakable --cuda-graph-max-bs-decode "$MAX_RUNNING" \
      --port "$port" --disaggregation-mode null
  ) >"$log" 2>&1 &
  echo $!
}

run_bench() {
  local url=$1 outfile=$2
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat --base-url "$url" --model "$MODEL" \
    --dataset-name random --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
    --random-range-ratio 0.0 --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" --warmup-requests 2 \
    --output-file "$outfile" --disable-tqdm 2>&1 | tail -25
}

echo "=================== ARM=$ARM DRAIN_ALL=$DRAIN ==================="
echo "GPUs: attn0=$ATTN0_GPU attn1=$ATTN1_GPU ffn=$FFN_GPU  ports=$a0,$a1,$ffn"
echo "load: ALL requests -> Attn1 only (Attn0 stays IDLE)  in=$IN_LEN out=$OUT_LEN conc=$MAX_CONCURRENCY"
: >"$OUT/pids"
kill_ports "$ffn" "$a0" "$a1"

start_ffn >"$OUT/pids"
echo "ffn pid $(cat "$OUT/pids")"
for i in $(seq 1 90); do
  rg -q "AfPool FFN|Load weight end|waiting" "$OUT/ffn.log" 2>/dev/null && break
  sleep 2
done
echo "--- ffn log tail ---"; tail -3 "$OUT/ffn.log"

start_attn "$ATTN0_GPU" "$a0" 0 "$OUT/attn0.log" | tee -a "$OUT/pids" >/dev/null
p0=$(tail -1 "$OUT/pids")
wait_http "http://127.0.0.1:$a0/health" Attn0 "$p0" 360 || { echo "ATTN0 BRINGUP FAILED"; tail -30 "$OUT/attn0.log"; exit 1; }

start_attn "$ATTN1_GPU" "$a1" 1 "$OUT/attn1.log" >>"$OUT/pids"
p1=$(tail -1 "$OUT/pids")
wait_http "http://127.0.0.1:$a1/health" Attn1 "$p1" 360 || { echo "ATTN1 BRINGUP FAILED"; tail -30 "$OUT/attn1.log"; exit 1; }

# Both links up before traffic so FFN is not still waiting on a handshake.
sleep 3
echo "--- ffn link status ---"
rg -o "AfPool FFN rank=0 link attn=[01] endpoint=.* ready" "$OUT/ffn.log" | tail -4 || true

echo "=================== BENCH (asymmetric: Attn1 only) ==================="
run_bench "http://127.0.0.1:$a1" "$OUT/bench.json" | tail -25

echo "=================== FFN self-reported utilisation ==================="
if [[ -f "$OUT/ffn_util.csv" ]]; then
  echo "busy_s,elapsed_s,frac,tasks = $(cat "$OUT/ffn_util.csv")"
else
  echo "(no ffn_util.csv)"
fi

echo "=================== FFN poll log ==================="
rg -c "get_batch failed" "$OUT/ffn.log" 2>/dev/null || echo "get_batch errors: 0"

cleanup
trap - EXIT
echo "ASYM_ARM_${ARM}_DONE"
