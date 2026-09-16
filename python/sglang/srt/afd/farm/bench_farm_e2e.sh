#!/usr/bin/env bash
# True AF (1A+1F cuda_ipc) + decode farm e2e tok/s · TPOT sweep.
# Configs: sticky (COALESCE_K=1) | coalesce (K>1) | spin_wait (K=1+SPIN_WAIT).
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT_DIR="${OUT_DIR:-/tmp/afd_farm_e2e}"
ATTN_GPU="${ATTN_GPU:-2}"
FFN_GPU="${FFN_GPU:-3}"
BASE_PORT="${BASE_PORT:-32710}"
# Safe defaults on A800 + ComfyUI (~40GB free): pad enough, FFN CG off, mid mem-fraction.
MEM_FRACTION="${MEM_FRACTION:-0.75}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
# comma list: sticky,coalesce,spin_wait
MODES="${MODES:-sticky,coalesce,spin_wait}"
B_STEP="${SGLANG_AFD_FARM_B_STEP:-8}"
B_WIN_K="${SGLANG_AFD_FARM_B_WIN_K:-4}"
COALESCE_K_THROUGHPUT="${COALESCE_K_THROUGHPUT:-4}"

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

kill_ports() {
  local attn_port=$1 ffn_port=$2
  local killed=0
  if [[ "$attn_port" -gt 0 ]]; then
    fuser -k "${attn_port}/tcp" >/dev/null 2>&1 || true
    killed=1
  fi
  if [[ "$ffn_port" -gt 0 ]]; then
    fuser -k "${ffn_port}/tcp" >/dev/null 2>&1 || true
    killed=1
  fi
  if [[ "$killed" -eq 1 ]]; then
    sleep 2
  fi
}

cleanup_mode() {
  local tag=$1
  if [[ -f "$OUT_DIR/$tag.pids" ]]; then
    while read -r p; do
      [[ -n "$p" ]] || continue
      kill -TERM "$p" 2>/dev/null || true
      pkill -TERM -P "$p" 2>/dev/null || true
    done <"$OUT_DIR/$tag.pids"
    sleep 2
    while read -r p; do
      [[ -n "$p" ]] || continue
      kill -KILL "$p" 2>/dev/null || true
      pkill -KILL -P "$p" 2>/dev/null || true
    done <"$OUT_DIR/$tag.pids"
    rm -f "$OUT_DIR/$tag.pids"
  fi
  kill_ports "${ATTN_PORT:-0}" "${FFN_PORT:-0}"
  rm -f "${IPC_ENDPOINT:-}" 2>/dev/null || true
}

apply_mode_env() {
  local mode=$1
  export SGLANG_AFD_TRANSPORT=cuda_ipc
  export SGLANG_AFD_IPC_ENDPOINT="$IPC_ENDPOINT"
  export SGLANG_AFD_MODULE_STUBS=1
  export SGLANG_AFD_ROUTING_SCHEME=a
  export SGLANG_AFD_PIPELINE=0
  export SGLANG_AFD_USE_WAIT_FLAG=0
  export SGLANG_AFD_FFN_CUDA_GRAPH="${SGLANG_AFD_FFN_CUDA_GRAPH:-0}"
  export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
  export SGLANG_AFD_FARM=1
  export SGLANG_AFD_FARM_B_STEP="$B_STEP"
  export SGLANG_AFD_FARM_B_WIN_K="$B_WIN_K"
  export SGLANG_AFD_FARM_MAX_INFLIGHT="${SGLANG_AFD_FARM_MAX_INFLIGHT:-2}"
  export SGLANG_AFD_NUM_MB="${SGLANG_AFD_NUM_MB:-2}"
  # Prefill A2F pad ≥ max hop tokens; with FFN_CG=0 Lite cost is MiB-scale.
  export SGLANG_AFD_MAX_NUM_TOKEN="${SGLANG_AFD_MAX_NUM_TOKEN:-256}"
  export SGLANG_AFD_FARM_LOG_EVERY=32
  export SGLANG_AFD_FARM_PHASE_TIMING="${SGLANG_AFD_FARM_PHASE_TIMING:-0}"
  export SGLANG_AFD_FARM_STAGE_STATS_EVERY="${SGLANG_AFD_FARM_STAGE_STATS_EVERY:-0}"
  echo "  mem_frac=$MEM_FRACTION max_tok=$SGLANG_AFD_MAX_NUM_TOKEN ctx=$CONTEXT_LENGTH FFN_CG=0"
  unset SGLANG_AFD_FARM_SPIN_WAIT || true
  unset SGLANG_AFD_FARM_PERSISTENT_LINEAR || true
  unset SGLANG_AFD_FARM_LAYER_CG || true

  case "$mode" in
    sticky)
      # Respect an explicit COALESCE_K so the P2 sweep can vary it; default 1.
      export SGLANG_AFD_FARM_COALESCE_K="${SGLANG_AFD_FARM_COALESCE_K:-1}"
      ;;
    coalesce)
      export SGLANG_AFD_FARM_COALESCE_K="$COALESCE_K_THROUGHPUT"
      ;;
    spin_wait)
      export SGLANG_AFD_FARM_COALESCE_K=1
      export SGLANG_AFD_FARM_SPIN_WAIT=1
      ;;
    *)
      echo "ERROR: unknown mode=$mode (sticky|coalesce|spin_wait)" >&2
      return 1
      ;;
  esac
}

start_af_farm() {
  local tag=$1
  local offset=$2
  ATTN_PORT=$((BASE_PORT + offset))
  FFN_PORT=$((BASE_PORT + offset + 1))
  IPC_ENDPOINT="$OUT_DIR/${tag}_cuda_ipc.sock"
  local log="$OUT_DIR/$tag"
  mkdir -p "$log"
  : >"$OUT_DIR/$tag.pids"
  rm -f "$IPC_ENDPOINT"
  echo "$ATTN_PORT" >"$OUT_DIR/$tag.attn_port"

  apply_mode_env "$tag"
  echo "=== Bring-up AF farm mode=$tag Attn@$ATTN_GPU:$ATTN_PORT FFN@$FFN_GPU:$FFN_PORT ==="
  echo "  COALESCE_K=$SGLANG_AFD_FARM_COALESCE_K SPIN_WAIT=${SGLANG_AFD_FARM_SPIN_WAIT:-0}"

  (
    export SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
    export CUDA_VISIBLE_DEVICES="$FFN_GPU"
    python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --port "$FFN_PORT" --skip-server-warmup
  ) >"$log/ffn.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local ffn_pid=$!

  echo "Waiting for FFN..."
  for i in $(seq 1 240); do
    kill -0 "$ffn_pid" 2>/dev/null || { tail -80 "$log/ffn.log" >&2; return 1; }
    if rg -q "AFD cuda_ipc FFN waiting|Load weight end" "$log/ffn.log" 2>/dev/null; then
      echo "FFN ready after ${i}s"
      break
    fi
    [[ $i -eq 240 ]] && { tail -100 "$log/ffn.log" >&2; return 1; }
    sleep 2
  done

  (
    export SGLANG_AFD_MODE=attn
    export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
    python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --port "$ATTN_PORT"
  ) >"$log/attn.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local attn_pid=$!

  echo "Waiting for Attn health..."
  for i in $(seq 1 360); do
    kill -0 "$attn_pid" 2>/dev/null || { tail -100 "$log/attn.log" >&2; return 1; }
    if curl -sf "http://127.0.0.1:$ATTN_PORT/health" >/dev/null 2>&1; then
      echo "Attn healthy after ${i}s"
      break
    fi
    [[ $i -eq 360 ]] && { tail -100 "$log/attn.log" >&2; return 1; }
    sleep 2
  done

  rg -n "AFD farm on|weight filter|MODE|spin_wait|coalesce" \
    "$log/attn.log" "$log/ffn.log" 2>/dev/null | tail -20 || true
}

run_bench() {
  local tag=$1
  local port
  port=$(cat "$OUT_DIR/$tag.attn_port")
  local out_json="$OUT_DIR/${tag}_bench.json"
  echo "=== Bench $tag → :$port n=$NUM_PROMPTS conc=$MAX_CONCURRENCY ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$port" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --output-file "$out_json" \
    --disable-tqdm \
    2>&1 | tee "$OUT_DIR/${tag}_bench.log"

  # Farm amortize / occupancy snippets after load
  rg -n "AFD farm (on|occupancy|amortize)|coalesce|spin" \
    "$OUT_DIR/$tag/attn.log" 2>/dev/null | tail -40 \
    | tee "$OUT_DIR/${tag}_farm_tail.txt" || true
}

summarize() {
  python3 - <<'PY' "$OUT_DIR" "$MODES"
import json, os, sys
out_dir, modes = sys.argv[1], [m.strip() for m in sys.argv[2].split(",") if m.strip()]
keys = [
    "completed", "output_throughput", "request_throughput",
    "median_tpot_ms", "mean_tpot_ms", "p99_tpot_ms",
    "median_itl_ms", "median_ttft_ms", "median_e2e_latency_ms",
]
print(f"{'mode':<12} {'ok':>4} {'out_tps':>10} {'med_tpot':>10} {'p99_tpot':>10} {'med_ttft':>10} {'med_e2e':>10}")
rows = {}
for tag in modes:
    path = os.path.join(out_dir, f"{tag}_bench.json")
    if not os.path.isfile(path):
        print(f"{tag:<12} {'—':>4} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
        continue
    d = json.load(open(path))
    rows[tag] = d
    ok = d.get("completed", 0)
    def f(k, fmt="{:.1f}"):
        v = d.get(k)
        return "—" if v is None else fmt.format(v)
    print(
        f"{tag:<12} {ok:>4} {f('output_throughput'):>10} "
        f"{f('median_tpot_ms'):>10} {f('p99_tpot_ms'):>10} "
        f"{f('median_ttft_ms'):>10} {f('median_e2e_latency_ms'):>10}"
    )
summary = {"modes": modes, "rows": {t: {k: rows[t].get(k) for k in keys} for t in rows}}
with open(os.path.join(out_dir, "SUMMARY.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote", os.path.join(out_dir, "SUMMARY.json"))
if "sticky" in rows and "coalesce" in rows:
    s, c = rows["sticky"], rows["coalesce"]
    if s.get("output_throughput") and c.get("output_throughput"):
        print(f"coalesce/sticky tok/s = {c['output_throughput']/s['output_throughput']:.3f}x")
    if s.get("median_tpot_ms") and c.get("median_tpot_ms"):
        print(f"coalesce/sticky med TPOT = {c['median_tpot_ms']/s['median_tpot_ms']:.3f}x")
if "sticky" in rows and "spin_wait" in rows:
    s, w = rows["sticky"], rows["spin_wait"]
    if s.get("median_tpot_ms") and w.get("median_tpot_ms"):
        print(f"spin_wait/sticky med TPOT = {w['median_tpot_ms']/s['median_tpot_ms']:.3f}x")
    if s.get("output_throughput") and w.get("output_throughput"):
        print(f"spin_wait/sticky tok/s = {w['output_throughput']/s['output_throughput']:.3f}x")
PY
}

trap 'cleanup_mode sticky; cleanup_mode coalesce; cleanup_mode spin_wait' EXIT

IFS=',' read -r -a MODE_ARR <<<"$MODES"
offset=0
for mode in "${MODE_ARR[@]}"; do
  mode=$(echo "$mode" | tr -d '[:space:]')
  [[ -n "$mode" ]] || continue
  cleanup_mode "$mode" || true
  start_af_farm "$mode" "$offset" || { echo "FAIL bring-up $mode" >&2; exit 1; }
  run_bench "$mode" || { echo "FAIL bench $mode" >&2; exit 1; }
  cleanup_mode "$mode"
  offset=$((offset + 10))
  sleep 3
done

summarize
echo "AFD_FARM_E2E_OK"
