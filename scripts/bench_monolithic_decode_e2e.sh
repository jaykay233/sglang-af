#!/usr/bin/env bash
# Monolithic decode baselines for AFD comparisons on a single A800 host.
#
# TP cases use one model instance with tensor parallelism. DP cases use one
# full-model replica per GPU because this model's 16 attention heads are not
# divisible by 3, 5, or 6.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT_DIR="${OUT_DIR:-/tmp/monolithic_decode_e2e}"
BASE_PORT="${BASE_PORT:-33110}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
GPUS_CSV="${GPUS_CSV:-0,1,2,4,5,7}"
CASES="${CASES:-dp1,dp2,tp2,dp3,dp4,tp4,dp5,dp6}"

mkdir -p "$OUT_DIR"
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --dtype bfloat16
  --mem-fraction-static "$MEM_FRACTION"
  --context-length "$CONTEXT_LENGTH"
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode disabled
  --disaggregation-mode null
)

declare -a CLEANUP_PIDS=()
declare -a CLEANUP_PORTS=()

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
  CLEANUP_PIDS=()
  CLEANUP_PORTS=()
}

trap 'cleanup_case; echo "INTERRUPTED" >&2' INT TERM EXIT

wait_health() {
  local pid=$1
  local port=$2
  local log_file=$3
  local i
  for i in $(seq 1 360); do
    kill -0 "$pid" 2>/dev/null || {
      tail -140 "$log_file" >&2
      return 1
    }
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  tail -140 "$log_file" >&2
  return 1
}

run_bench() {
  local port=$1
  local requests=$2
  local concurrency=$3
  local output_file=$4
  local log_file=$5

  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:${port}" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$requests" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$concurrency" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --output-file "$output_file" \
    --disable-tqdm \
    >"$log_file" 2>&1
}

run_parallel_replicas() {
  local tag=$1
  local replicas=$2
  local offset=$3

  (( replicas <= ${#GPUS[@]} )) || {
    echo "ERROR: ${tag} needs ${replicas} GPUs, only ${#GPUS[@]} configured" >&2
    return 1
  }

  local log_dir="$OUT_DIR/$tag"
  mkdir -p "$log_dir"
  : >"$log_dir/pids"

  local base_requests=$((NUM_PROMPTS / replicas))
  local extra_requests=$((NUM_PROMPTS % replicas))
  local base_concurrency=$((MAX_CONCURRENCY / replicas))
  local extra_concurrency=$((MAX_CONCURRENCY % replicas))

  local i
  local -a pids=()
  local -a ports=()
  local -a requests=()
  local -a concurrencies=()

  echo "=== ${tag}: ${replicas} full-model replicas ==="
  for ((i=0; i<replicas; i++)); do
    local gpu="${GPUS[$i]}"
    local port=$((BASE_PORT + offset + i))
    local requests_i=$((base_requests + (i < extra_requests ? 1 : 0)))
    local concurrency_i=$((base_concurrency + (i < extra_concurrency ? 1 : 0)))
    [[ "$concurrency_i" -gt 0 ]] || concurrency_i=1

    ports+=("$port")
    requests+=("$requests_i")
    concurrencies+=("$concurrency_i")
    CLEANUP_PORTS+=("$port")

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
        --tp-size 1 \
        --max-running-requests "$concurrency_i" \
        --port "$port"
    ) >"$log_dir/replica${i}.log" 2>&1 &
    pids+=("$!")
    echo "${pids[$i]}" >>"$log_dir/pids"
    CLEANUP_PIDS+=("${pids[$i]}")
  done

  echo "Waiting for ${replicas} health endpoints..."
  for ((i=0; i<replicas; i++)); do
    wait_health "${pids[$i]}" "${ports[$i]}" "$log_dir/replica${i}.log"
  done

  echo "=== Bench ${tag}: ${NUM_PROMPTS} requests, total concurrency ${MAX_CONCURRENCY} ==="
  local -a bench_pids=()
  for ((i=0; i<replicas; i++)); do
    run_bench \
      "${ports[$i]}" \
      "${requests[$i]}" \
      "${concurrencies[$i]}" \
      "$OUT_DIR/${tag}_replica${i}_bench.json" \
      "$log_dir/bench${i}.log" &
    bench_pids+=("$!")
  done

  for ((i=0; i<replicas; i++)); do
    wait "${bench_pids[$i]}"
    tail -35 "$log_dir/bench${i}.log"
  done

  python3 - <<'PY' "$OUT_DIR" "$tag" "$replicas" "$OUT_DIR/${tag}_bench.json"
import json
import os
import sys

out_dir, tag, replicas_s, output_path = sys.argv[1:]
replicas = int(replicas_s)
rows = []
for i in range(replicas):
    path = os.path.join(out_dir, f"{tag}_replica{i}_bench.json")
    with open(path, encoding="utf-8") as f:
        rows.append(json.load(f))

wall = max(float(row["duration"]) for row in rows)
output_tokens = sum(float(row["total_output_tokens"]) for row in rows)
input_tokens = sum(float(row["total_input_tokens"]) for row in rows)

def weighted(key):
    return sum(
        float(row[key]) * float(row["total_output_tokens"]) for row in rows
    ) / output_tokens

aggregate = {
    "tag": tag,
    "gpu_count": replicas,
    "mode": "dp",
    "duration": wall,
    "completed": sum(int(row["completed"]) for row in rows),
    "total_input_tokens": input_tokens,
    "total_output_tokens": output_tokens,
    "request_throughput": sum(int(row["completed"]) for row in rows) / wall,
    "input_throughput": input_tokens / wall,
    "output_throughput": output_tokens / wall,
    "mean_e2e_latency_ms": weighted("mean_e2e_latency_ms"),
    "median_e2e_latency_ms": weighted("median_e2e_latency_ms"),
    "p99_e2e_latency_ms": max(float(row["p99_e2e_latency_ms"]) for row in rows),
    "mean_ttft_ms": weighted("mean_ttft_ms"),
    "median_ttft_ms": weighted("median_ttft_ms"),
    "p99_ttft_ms": max(float(row["p99_ttft_ms"]) for row in rows),
    "mean_tpot_ms": weighted("mean_tpot_ms"),
    "median_tpot_ms": weighted("median_tpot_ms"),
    "p99_tpot_ms": max(float(row["p99_tpot_ms"]) for row in rows),
    "aggregation": (
        "wall=max(replica durations); throughput uses sum(tokens)/wall; "
        "latencies weighted by output tokens; p99=max(replica p99)"
    ),
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(aggregate, f, indent=2)

print(
    f"Aggregate {tag}: ok={aggregate['completed']} "
    f"out_tps={aggregate['output_throughput']:.2f} "
    f"med_tpot={aggregate['median_tpot_ms']:.2f} "
    f"p99_tpot={aggregate['p99_tpot_ms']:.2f}"
)
PY

  rg -n "Scheduler hit an exception|Traceback|RuntimeError" "$log_dir" 2>/dev/null \
    | tail -40 || true

  cleanup_case
  sleep 3
}

run_tp() {
  local tag=$1
  local tp_size=$2
  local offset=$3

  (( tp_size <= ${#GPUS[@]} )) || {
    echo "ERROR: ${tag} needs ${tp_size} GPUs, only ${#GPUS[@]} configured" >&2
    return 1
  }

  local log_dir="$OUT_DIR/$tag"
  local port=$((BASE_PORT + offset))
  local gpu_csv
  gpu_csv=$(IFS=,; echo "${GPUS[*]:0:tp_size}")
  mkdir -p "$log_dir"
  : >"$log_dir/pids"
  CLEANUP_PORTS+=("$port")

  echo "=== ${tag}: TP${tp_size} on GPUs ${gpu_csv} ==="
  (
    export CUDA_VISIBLE_DEVICES="$gpu_csv"
    python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --tp-size "$tp_size" \
      --max-running-requests "$MAX_CONCURRENCY" \
      --port "$port"
  ) >"$log_dir/server.log" 2>&1 &
  local server_pid=$!
  echo "$server_pid" >>"$log_dir/pids"
  CLEANUP_PIDS+=("$server_pid")

  wait_health "$server_pid" "$port" "$log_dir/server.log"

  echo "=== Bench ${tag}: ${NUM_PROMPTS} requests, concurrency ${MAX_CONCURRENCY} ==="
  run_bench \
    "$port" \
    "$NUM_PROMPTS" \
    "$MAX_CONCURRENCY" \
    "$OUT_DIR/${tag}_bench.json" \
    "$log_dir/bench.log"
  tail -35 "$log_dir/bench.log"

  python3 - <<'PY' "$OUT_DIR/${tag}_bench.json" "$tag" "$tp_size"
import json
import sys

path, tag, tp_size_s = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    row = json.load(f)
row["tag"] = tag
row["gpu_count"] = int(tp_size_s)
row["mode"] = "tp"
with open(path, "w", encoding="utf-8") as f:
    json.dump(row, f, indent=2)
print(
    f"Result {tag}: ok={row['completed']} "
    f"out_tps={row['output_throughput']:.2f} "
    f"med_tpot={row['median_tpot_ms']:.2f} "
    f"p99_tpot={row['p99_tpot_ms']:.2f}"
)
PY

  rg -n "Scheduler hit an exception|Traceback|RuntimeError" "$log_dir" 2>/dev/null \
    | tail -40 || true

  cleanup_case
  sleep 3
}

summarize() {
  python3 - <<'PY' "$OUT_DIR"
import json
import os
import sys

out_dir = sys.argv[1]
tags = ("dp1", "dp2", "tp2", "dp3", "dp4", "tp4", "dp5", "dp6")
keys = (
    "gpu_count",
    "mode",
    "completed",
    "output_throughput",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_ttft_ms",
    "median_e2e_latency_ms",
)
rows = {}
for tag in tags:
    path = os.path.join(out_dir, f"{tag}_bench.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            rows[tag] = json.load(f)

print(
    f"{'case':<6} {'gpu':>3} {'mode':>4} {'ok':>4} {'out_tps':>10} "
    f"{'med_tpot':>10} {'p99_tpot':>10} {'med_ttft':>10} {'med_e2e':>10}"
)
for tag, row in rows.items():
    def fmt(key):
        value = row.get(key)
        return "-" if value is None else f"{value:.2f}"
    print(
        f"{tag:<6} {int(row['gpu_count']):>3} {row['mode']:>4} "
        f"{int(row['completed']):>4} {fmt('output_throughput'):>10} "
        f"{fmt('median_tpot_ms'):>10} {fmt('p99_tpot_ms'):>10} "
        f"{fmt('median_ttft_ms'):>10} {fmt('median_e2e_latency_ms'):>10}"
    )

summary = {"rows": {tag: {key: row.get(key) for key in keys} for tag, row in rows.items()}}
with open(os.path.join(out_dir, "SUMMARY.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
PY
}

IFS=',' read -r -a SELECTED_CASES <<<"$CASES"
for case_tag in "${SELECTED_CASES[@]}"; do
  case "$case_tag" in
    dp1) run_parallel_replicas "dp1" 1 0 ;;
    dp2) run_parallel_replicas "dp2" 2 20 ;;
    tp2) run_tp "tp2" 2 40 ;;
    dp3) run_parallel_replicas "dp3" 3 60 ;;
    dp4) run_parallel_replicas "dp4" 4 80 ;;
    tp4) run_tp "tp4" 4 100 ;;
    dp5) run_parallel_replicas "dp5" 5 120 ;;
    dp6) run_parallel_replicas "dp6" 6 140 ;;
    *) echo "ERROR: unknown case ${case_tag}" >&2; exit 1 ;;
  esac
done

cleanup_case
trap - INT TERM EXIT
summarize
echo "MONOLITHIC_DECODE_E2E_OK"
