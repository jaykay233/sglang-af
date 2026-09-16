#!/usr/bin/env bash
# AfPool model E2E: 1A1F baseline vs 1A2F/1A4F/2A4F on a single 8-GPU host.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT_DIR="${OUT_DIR:-/tmp/afd_pool_e2e}"
BASE_PORT="${BASE_PORT:-32910}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
B_STEP="${SGLANG_AFD_FARM_B_STEP:-16}"
B_WIN_K="${SGLANG_AFD_FARM_B_WIN_K:-8}"
COALESCE_K="${SGLANG_AFD_FARM_COALESCE_K:-4}"
LAYER_BURST="${SGLANG_AFD_FARM_LAYER_BURST:-0}"
FARM_SCHED="${SGLANG_AFD_FARM_SCHED:-max}"
MAX_INFLIGHT_PER_FFN="${SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN:-8}"
MAX_INFLIGHT_PER_LAYER="${SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER:-1}"
NUM_CONTEXTS="${SGLANG_AFD_FARM_NUM_CONTEXTS:-2}"
CONTEXT_STAGGER_LAYERS="${SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS:-1}"
CONTEXTS_PER_STAGE="${SGLANG_AFD_FARM_CONTEXTS_PER_STAGE:-1}"
STAGE_STATS_EVERY="${SGLANG_AFD_FARM_STAGE_STATS_EVERY:-0}"
BENCH_SEED="${BENCH_SEED:-}"
NUM_MB="${SGLANG_AFD_NUM_MB:-4}"
MAX_NUM_TOKEN="${SGLANG_AFD_MAX_NUM_TOKEN:-256}"

# Keep validation deterministic and within the idle-A800 memory envelope.
ATTN_GPU_1A1F="${ATTN_GPU_1A1F:-0}"
FFN_GPUS_1A1F="${FFN_GPUS_1A1F:-1}"
ATTN_GPU_1A2F="${ATTN_GPU_1A2F:-0}"
FFN_GPUS_1A2F="${FFN_GPUS_1A2F:-1,4}"
ATTN_GPU_1A4F="${ATTN_GPU_1A4F:-0}"
FFN_GPUS_1A4F="${FFN_GPUS_1A4F:-1,2,4,5}"
ATTN_GPUS_2A4F="${ATTN_GPUS_2A4F:-0,1}"
FFN_GPUS_2A4F="${FFN_GPUS_2A4F:-2,4,5,7}"
CASES="${CASES:-1a1f,1a2f,1a4f,2a4f}"

mkdir -p "$OUT_DIR"
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests "$MAX_CONCURRENCY"
  --context-length "$CONTEXT_LENGTH"
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode disabled
  --disaggregation-mode null
)

# The AFD FFN server never serves a real request, so its scheduler loop is pure
# idle spin. With --sleep-on-idle it parks on a zmq poll instead, freeing the GIL
# for the af-pool-ffn*-serve compute thread (progress.md §15.7).
declare -a FFN_IDLE_ARGS=()
if [[ "${SGLANG_AFD_FFN_SLEEP_ON_IDLE:-0}" == "1" ]]; then
  FFN_IDLE_ARGS+=(--sleep-on-idle)
fi

declare -a CLEANUP_PIDS=()
declare -a CLEANUP_PORTS=()
declare -a CLEANUP_DIRS=()

cleanup_case() {
  local pid
  for pid in "${CLEANUP_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${CLEANUP_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -KILL "$pid" 2>/dev/null || true
    pkill -KILL -P "$pid" 2>/dev/null || true
  done
  local port
  for port in "${CLEANUP_PORTS[@]:-}"; do
    [[ "$port" -gt 0 ]] || continue
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  done
  local dir
  for dir in "${CLEANUP_DIRS[@]:-}"; do
    [[ -n "$dir" ]] || continue
    rm -rf "$dir"
  done
  CLEANUP_PIDS=()
  CLEANUP_PORTS=()
  CLEANUP_DIRS=()
}

trap 'cleanup_case; echo "INTERRUPTED" >&2' INT TERM EXIT

apply_common_env() {
  export SGLANG_AFD_TRANSPORT=cuda_ipc
  export SGLANG_AFD_MODULE_STUBS=1
  export SGLANG_AFD_ROUTING_SCHEME=a
  export SGLANG_AFD_PIPELINE=0
  export SGLANG_AFD_USE_WAIT_FLAG=0
  # NOTE: must stay overridable — hardcoding this to 0 silently disabled the
  # FFN CUDA-graph path, which made an earlier "CG doesn't help" measurement
  # (progress.md §8) test nothing at all.
  export SGLANG_AFD_FFN_CUDA_GRAPH="${SGLANG_AFD_FFN_CUDA_GRAPH:-0}"
  export SGLANG_AFD_POOL_SERVE_SYNC=0
  export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
  # MoE combine: use the fused sgl_kernel moe_sum_reduce instead of the
  # torch.compile helper. Even with Dynamo disabled (above) the compiled stub
  # lowers to sum + mul_ (+ reshapes) instead of one kernel; measured faster and
  # it is the small-token path the farm always takes (progress.md §14).
  export SGLANG_AFD_MOE_SUM_REDUCE_COMPILE="${SGLANG_AFD_MOE_SUM_REDUCE_COMPILE:-0}"
  # Throttle the SGLang scheduler's idle housekeeping. The AFD FFN server never
  # serves real requests, so without this it spins on_idle (get_pool_stats +
  # invariant checks + publish_load_snapshot) at full rate, holding the GIL and
  # starving the FFN compute thread (progress.md §15).
  export SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS="${SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS:-250}"
  export SGLANG_AFD_FARM=1
  export SGLANG_AFD_FARM_B_STEP="$B_STEP"
  export SGLANG_AFD_FARM_B_WIN_K="$B_WIN_K"
  export SGLANG_AFD_FARM_MAX_INFLIGHT="$NUM_MB"
  export SGLANG_AFD_FARM_COALESCE_K="$COALESCE_K"
  export SGLANG_AFD_FARM_LAYER_BURST="$LAYER_BURST"
  export SGLANG_AFD_FARM_SCHED="$FARM_SCHED"
  export SGLANG_AFD_NUM_MB="$NUM_MB"
  export SGLANG_AFD_MAX_NUM_TOKEN="$MAX_NUM_TOKEN"
  export SGLANG_AFD_FARM_LOG_EVERY=32
  export SGLANG_AFD_FARM_NUM_CONTEXTS="$NUM_CONTEXTS"
  export SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS="$CONTEXT_STAGGER_LAYERS"
  export SGLANG_AFD_FARM_CONTEXTS_PER_STAGE="$CONTEXTS_PER_STAGE"
  export SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER="$MAX_INFLIGHT_PER_LAYER"
  export SGLANG_AFD_FARM_STAGE_STATS_EVERY="$STAGE_STATS_EVERY"
  export SGLANG_AFD_POOL=1
  export SGLANG_AFD_POOL_NUM_ATTN=1
  export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN="$MAX_INFLIGHT_PER_FFN"
  export SGLANG_AFD_POOL_ROUTE=least_inflight
  unset SGLANG_AFD_FARM_SPIN_WAIT || true
  unset SGLANG_AFD_FARM_PERSISTENT_LINEAR || true
  unset SGLANG_AFD_FARM_LAYER_CG || true
}

wait_health() {
  local pid=$1
  local port=$2
  local log_dir=$3
  local log_file="${4:-$log_dir/attn.log}"
  local i
  for i in $(seq 1 360); do
    kill -0 "$pid" 2>/dev/null || {
      tail -120 "$log_file" >&2
      return 1
    }
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -120 "$log_file" >&2
  return 1
}

run_case() {
  local tag=$1
  local num_ffn=$2
  local attn_gpu=$3
  local ffn_gpus_csv=$4
  local offset=$5

  local attn_port=$((BASE_PORT + offset))
  local endpoint_dir="$OUT_DIR/${tag}_endpoints"
  local log_dir="$OUT_DIR/$tag"
  mkdir -p "$log_dir"
  rm -rf "$endpoint_dir"
  mkdir -p "$endpoint_dir"
  : >"$log_dir/pids"

  local -a ffn_gpus
  IFS=',' read -r -a ffn_gpus <<<"$ffn_gpus_csv"
  if [[ "${#ffn_gpus[@]}" -ne "$num_ffn" ]]; then
    echo "ERROR: ${tag} needs ${num_ffn} FFN GPUs, got ${ffn_gpus_csv}" >&2
    return 1
  fi

  apply_common_env
  export SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
  export SGLANG_AFD_POOL_ENDPOINT_DIR="$endpoint_dir"
  CLEANUP_DIRS+=("$endpoint_dir")

  echo "=== ${tag}: 1A${num_ffn}F Attn@${attn_gpu} FFN@${ffn_gpus_csv} ==="
  local j
  for ((j=0; j<num_ffn; j++)); do
    local ffn_port=$((attn_port + 1 + j))
    CLEANUP_PORTS+=("$ffn_port")
    (
      export SGLANG_AFD_MODE=ffn
      export SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
      export SGLANG_AFD_POOL_LOCAL_RANK="$j"
      export CUDA_VISIBLE_DEVICES="${ffn_gpus[$j]}"
      python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
        "${FFN_IDLE_ARGS[@]}" \
        --port "$ffn_port" --skip-server-warmup
    ) >"$log_dir/ffn${j}.log" 2>&1 &
    local ffn_pid=$!
    echo "$ffn_pid" >>"$log_dir/pids"
    CLEANUP_PIDS+=("$ffn_pid")
  done

  echo "Waiting for ${num_ffn} FFN listeners..."
  local i
  for i in $(seq 1 300); do
    local ready=0
    for ((j=0; j<num_ffn; j++)); do
      if rg -q "AFD cuda_ipc FFN waiting for Attn|AfPool FFN rank=.*ready" \
        "$log_dir/ffn${j}.log" 2>/dev/null; then
        ready=$((ready + 1))
      fi
    done
    [[ "$ready" -eq "$num_ffn" ]] && break
    [[ $i -eq 300 ]] && {
      for ((j=0; j<num_ffn; j++)); do
        echo "--- ffn${j}.log ---" >&2
        tail -100 "$log_dir/ffn${j}.log" >&2 || true
      done
      return 1
    }
    sleep 2
  done

  (
    export SGLANG_AFD_MODE=attn
    export SGLANG_AFD_POOL_LOCAL_RANK=0
    export CUDA_VISIBLE_DEVICES="$attn_gpu"
    python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$attn_port"
  ) >"$log_dir/attn.log" 2>&1 &
  local attn_pid=$!
  echo "$attn_pid" >>"$log_dir/pids"
  CLEANUP_PIDS+=("$attn_pid")
  CLEANUP_PORTS+=("$attn_port")

  echo "Waiting for Attn health..."
  wait_health "$attn_pid" "$attn_port" "$log_dir"

  rg -n "AfPool Attn ready|AfPool FFN ready|AFD cuda_ipc FFN waiting" \
    "$log_dir/attn.log" "$log_dir"/ffn*.log 2>/dev/null | tail -30 || true

  echo "=== Bench ${tag}: n=${NUM_PROMPTS} conc=${MAX_CONCURRENCY} ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:${attn_port}" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests "$WARMUP_REQUESTS" \
    ${BENCH_SEED:+--seed "$BENCH_SEED"} \
    --output-file "$OUT_DIR/${tag}_bench.json" \
    --disable-tqdm \
    2>&1 | tee "$log_dir/bench.log"

  rg -n "AFD farm (on|occupancy|amortize)|coalesce|spin|Scheduler hit an exception|Traceback" \
    "$log_dir/attn.log" 2>/dev/null | tail -40 \
    | tee "$log_dir/farm_tail.txt" || true

  cleanup_case
  sleep 3
}

run_case_2a4f() {
  local tag="2a4f"
  local num_attn=2
  local num_ffn=4
  local attn_gpus_csv="$ATTN_GPUS_2A4F"
  local ffn_gpus_csv="$FFN_GPUS_2A4F"
  local offset=60

  local attn_base_port=$((BASE_PORT + offset))
  local endpoint_dir="$OUT_DIR/${tag}_endpoints"
  local log_dir="$OUT_DIR/$tag"
  mkdir -p "$log_dir"
  rm -rf "$endpoint_dir"
  mkdir -p "$endpoint_dir"
  : >"$log_dir/pids"

  local -a attn_gpus ffn_gpus
  IFS=',' read -r -a attn_gpus <<<"$attn_gpus_csv"
  IFS=',' read -r -a ffn_gpus <<<"$ffn_gpus_csv"
  if [[ "${#attn_gpus[@]}" -ne "$num_attn" ]]; then
    echo "ERROR: ${tag} needs ${num_attn} Attn GPUs, got ${attn_gpus_csv}" >&2
    return 1
  fi
  if [[ "${#ffn_gpus[@]}" -ne "$num_ffn" ]]; then
    echo "ERROR: ${tag} needs ${num_ffn} FFN GPUs, got ${ffn_gpus_csv}" >&2
    return 1
  fi

  local reqs_per_attn=$(( (NUM_PROMPTS + num_attn - 1) / num_attn ))
  local conc_per_attn=$(( (MAX_CONCURRENCY + num_attn - 1) / num_attn ))

  apply_common_env
  export SGLANG_AFD_POOL_NUM_ATTN="$num_attn"
  export SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
  export SGLANG_AFD_POOL_ENDPOINT_DIR="$endpoint_dir"
  CLEANUP_DIRS+=("$endpoint_dir")

  echo "=== ${tag}: ${num_attn}A${num_ffn}F Attn@${attn_gpus_csv} FFN@${ffn_gpus_csv} ==="
  local j
  for ((j=0; j<num_ffn; j++)); do
    local ffn_port=$((attn_base_port + 10 + j))
    CLEANUP_PORTS+=("$ffn_port")
    (
      export SGLANG_AFD_MODE=ffn
      export SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
      export SGLANG_AFD_POOL_LOCAL_RANK="$j"
      export CUDA_VISIBLE_DEVICES="${ffn_gpus[$j]}"
      python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
        "${FFN_IDLE_ARGS[@]}" \
        --port "$ffn_port" --skip-server-warmup
    ) >"$log_dir/ffn${j}.log" 2>&1 &
    local ffn_pid=$!
    echo "$ffn_pid" >>"$log_dir/pids"
    CLEANUP_PIDS+=("$ffn_pid")
  done

  echo "Waiting for ${num_ffn} FFN listeners..."
  local i
  for i in $(seq 1 300); do
    local ready=0
    for ((j=0; j<num_ffn; j++)); do
      if rg -q "AFD cuda_ipc FFN waiting for Attn|AfPool FFN rank=.*link0 ready" \
        "$log_dir/ffn${j}.log" 2>/dev/null; then
        ready=$((ready + 1))
      fi
    done
    [[ "$ready" -eq "$num_ffn" ]] && break
    [[ $i -eq 300 ]] && {
      for ((j=0; j<num_ffn; j++)); do
        echo "--- ffn${j}.log ---" >&2
        tail -100 "$log_dir/ffn${j}.log" >&2 || true
      done
      return 1
    }
    sleep 2
  done

  local attn_pid i
  local -a attn_pids=()
  for ((i=0; i<num_attn; i++)); do
    local attn_port=$((attn_base_port + i))
    CLEANUP_PORTS+=("$attn_port")
    (
      export SGLANG_AFD_MODE=attn
      export SGLANG_AFD_POOL_LOCAL_RANK="$i"
      export CUDA_VISIBLE_DEVICES="${attn_gpus[$i]}"
      python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$attn_port"
    ) >"$log_dir/attn${i}.log" 2>&1 &
    attn_pid=$!
    attn_pids+=("$attn_pid")
    echo "$attn_pid" >>"$log_dir/pids"
    CLEANUP_PIDS+=("$attn_pid")
  done

  echo "Waiting for ${num_attn} Attn health endpoints..."
  for ((i=0; i<num_attn; i++)); do
    wait_health \
      "${attn_pids[$i]}" \
      "$((attn_base_port + i))" \
      "$log_dir" \
      "$log_dir/attn${i}.log"
  done

  rg -n "AfPool Attn ready|AfPool FFN rank=.*link attn=|AfPool FFN rank=.*link0 ready" \
    "$log_dir"/attn*.log "$log_dir"/ffn*.log 2>/dev/null | tail -60 || true

  echo "=== Bench ${tag}: ${reqs_per_attn} requests and concurrency ${conc_per_attn} per Attn ==="
  local -a bench_pids=()
  for ((i=0; i<num_attn; i++)); do
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat \
      --base-url "http://127.0.0.1:$((attn_base_port + i))" \
      --model "$MODEL" \
      --dataset-name random \
      --num-prompts "$reqs_per_attn" \
      --random-input-len "$RANDOM_INPUT_LEN" \
      --random-output-len "$RANDOM_OUTPUT_LEN" \
      --random-range-ratio 0.0 \
      --request-rate "$REQUEST_RATE" \
      --max-concurrency "$conc_per_attn" \
      --warmup-requests "$WARMUP_REQUESTS" \
      ${BENCH_SEED:+--seed "$BENCH_SEED"} \
      --output-file "$OUT_DIR/${tag}_attn${i}_bench.json" \
      --disable-tqdm \
      >"$log_dir/bench_attn${i}.log" 2>&1 &
    bench_pids+=("$!")
  done
  for i in "${!bench_pids[@]}"; do
    wait "${bench_pids[$i]}"
    tail -35 "$log_dir/bench_attn${i}.log"
  done

  python3 - <<'PY' "$OUT_DIR" "$num_attn"
import json
import os
import sys

out_dir = sys.argv[1]
num_attn = int(sys.argv[2])
rows = []
for i in range(num_attn):
    path = os.path.join(out_dir, f"2a4f_attn{i}_bench.json")
    with open(path, encoding="utf-8") as f:
        rows.append(json.load(f))

wall = max(float(r["duration"]) for r in rows)
output_tokens = sum(float(r["total_output_tokens"]) for r in rows)
input_tokens = sum(float(r["total_input_tokens"]) for r in rows)

def weighted(key):
    denom = sum(float(r["total_output_tokens"]) for r in rows)
    return sum(float(r[key]) * float(r["total_output_tokens"]) for r in rows) / denom

aggregate = {
    "duration": wall,
    "completed": sum(int(r["completed"]) for r in rows),
    "total_input_tokens": input_tokens,
    "total_output_tokens": output_tokens,
    "request_throughput": sum(int(r["completed"]) for r in rows) / wall,
    "input_throughput": input_tokens / wall,
    "output_throughput": output_tokens / wall,
    "mean_e2e_latency_ms": weighted("mean_e2e_latency_ms"),
    "median_e2e_latency_ms": weighted("median_e2e_latency_ms"),
    "p99_e2e_latency_ms": max(float(r["p99_e2e_latency_ms"]) for r in rows),
    "mean_ttft_ms": weighted("mean_ttft_ms"),
    "median_ttft_ms": weighted("median_ttft_ms"),
    "p99_ttft_ms": max(float(r["p99_ttft_ms"]) for r in rows),
    "mean_tpot_ms": weighted("mean_tpot_ms"),
    "median_tpot_ms": weighted("median_tpot_ms"),
    "p99_tpot_ms": max(float(r["p99_tpot_ms"]) for r in rows),
    "aggregation": "cluster wall=max(attn durations); latency medians weighted by output tokens; p99=max(attn p99)",
}
path = os.path.join(out_dir, "2a4f_bench.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(aggregate, f, indent=2)
print(
    f"Aggregate 2A4F: ok={aggregate['completed']} "
    f"out_tps={aggregate['output_throughput']:.1f} "
    f"med_tpot={aggregate['median_tpot_ms']:.1f} "
    f"p99_tpot={aggregate['p99_tpot_ms']:.1f}"
)
PY

  rg -n "Scheduler hit an exception|Traceback|RuntimeError|exceeds the registered slot" \
    "$log_dir" 2>/dev/null | tail -40 || true

  cleanup_case
  sleep 3
}

summarize() {
  python3 - <<'PY' "$OUT_DIR"
import json
import os
import sys

out_dir = sys.argv[1]
rows = {}
for tag in ("1a1f", "1a2f", "1a4f", "2a4f"):
    path = os.path.join(out_dir, f"{tag}_bench.json")
    if os.path.isfile(path):
        rows[tag] = json.load(open(path, encoding="utf-8"))

keys = (
    "completed",
    "output_throughput",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_ttft_ms",
    "median_e2e_latency_ms",
)
print(f"{'case':<8} {'ok':>4} {'out_tps':>10} {'med_tpot':>10} {'p99_tpot':>10} {'med_ttft':>10} {'med_e2e':>10}")
for tag, data in rows.items():
    def fmt(key):
        value = data.get(key)
        return "-" if value is None else f"{value:.1f}"
    print(
        f"{tag:<8} {int(data.get('completed', 0)):>4} "
        f"{fmt('output_throughput'):>10} {fmt('median_tpot_ms'):>10} "
        f"{fmt('p99_tpot_ms'):>10} {fmt('median_ttft_ms'):>10} "
        f"{fmt('median_e2e_latency_ms'):>10}"
    )

summary = {"rows": {tag: {k: rows.get(tag, {}).get(k) for k in keys} for tag in rows}}
with open(os.path.join(out_dir, "SUMMARY.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

if "1a1f" in rows:
    a = float(rows["1a1f"].get("output_throughput") or 0)
    if a > 0:
        for tag in ("1a2f", "1a4f", "2a4f"):
            if tag not in rows:
                continue
            b = float(rows[tag].get("output_throughput") or 0)
            print(f"{tag.upper()}/1A1F output tok/s = {b / a:.3f}x")
PY
}

IFS=',' read -r -a selected_cases <<<"$CASES"
for case_tag in "${selected_cases[@]}"; do
  case "$case_tag" in
    1a1f) run_case "1a1f" 1 "$ATTN_GPU_1A1F" "$FFN_GPUS_1A1F" 0 ;;
    1a2f) run_case "1a2f" 2 "$ATTN_GPU_1A2F" "$FFN_GPUS_1A2F" 20 ;;
    1a4f) run_case "1a4f" 4 "$ATTN_GPU_1A4F" "$FFN_GPUS_1A4F" 40 ;;
    2a4f) run_case_2a4f ;;
    *) echo "ERROR: unknown case ${case_tag}" >&2; exit 1 ;;
  esac
done
cleanup_case
trap - INT TERM EXIT
summarize
echo "AFD_POOL_E2E_OK"
