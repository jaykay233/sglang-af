#!/usr/bin/env bash
# Fair decode-speed comparison: 2× full replica (DP=2) vs 1A1F decode farm.
#
# Same GPU count (2), same model, same server flags (CUDA graphs off,
# mem-fraction 0.75, ctx 2048, max-running 32), same client load
# (in=128 / out=128 / conc=32). Baseline = 2 independent full replicas,
# one per GPU; farm = 1 Attn + 1 FFN pipeline over the same 2 GPUs.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then source /root/.cuda/afd_env.sh; fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT="${OUT:-/tmp/afd_vs_monolithic}"
GPU0="${GPU0:-2}"
GPU1="${GPU1:-3}"
MEM="${MEM_FRACTION:-0.75}"
PROMPTS="${NUM_PROMPTS:-64}"
CONC="${MAX_CONCURRENCY:-32}"
IN_LEN="${RANDOM_INPUT_LEN:-128}"
OUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
WARM="${WARMUP_REQUESTS:-2}"

mkdir -p "$OUT"

COMMON=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --dtype bfloat16 --mem-fraction-static "$MEM"
  --max-running-requests "$CONC" --context-length 2048
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled
  --disaggregation-mode null
)

wait_health() {
  local port=$1 pid=$2 log=$3
  for _ in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { tail -40 "$log" >&2; return 1; }
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  tail -40 "$log" >&2
  return 1
}

run_bench() {
  local port=$1 tag=$2
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat --base-url "http://127.0.0.1:$port" \
    --model "$MODEL" --dataset-name random \
    --num-prompts "$PROMPTS" --random-input-len "$IN_LEN" \
    --random-output-len "$OUT_LEN" --random-range-ratio 0.0 \
    --request-rate inf --max-concurrency "$CONC" \
    --warmup-requests "$WARM" --output-file "$OUT/${tag}_bench.json" \
    --disable-tqdm >"$OUT/${tag}_bench.log" 2>&1 || true
}

cleanup() {
  for f in "$OUT"/*.pids; do
    [[ -f "$f" ]] || continue
    while read -r p; do [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true; done <"$f"
    rm -f "$f"
  done
  fuser -k 33200/tcp 33201/tcp 33202/tcp >/dev/null 2>&1 || true
  sleep 2
}
trap cleanup EXIT
cleanup

echo "### BASELINE: 2x full replica (DP=2) on GPU $GPU0,$GPU1"
: >"$OUT/mono.pids"
for i in 0 1; do
  port=$((33200 + i))
  gpu=$([[ $i -eq 0 ]] && echo "$GPU0" || echo "$GPU1")
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    unset SGLANG_AFD_MODE SGLANG_AFD_FARM || true
    python3 -m sglang.launch_server "${COMMON[@]}" --port "$port"
  ) >"$OUT/mono$i.log" 2>&1 &
  echo $! >>"$OUT/mono.pids"
  wait_health "$port" $! "$OUT/mono$i.log" || { echo "mono$i failed" >&2; exit 1; }
  run_bench "$port" "mono$i"
  echo "  mono$i done"
done
cleanup

echo "### FARM: 1A1F on GPU $GPU0,$GPU1"
export SGLANG_AFD_TRANSPORT=cuda_ipc SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME=a SGLANG_AFD_PIPELINE=0
export SGLANG_AFD_USE_WAIT_FLAG=0 SGLANG_AFD_FFN_CUDA_GRAPH=0
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export SGLANG_AFD_FARM=1 SGLANG_AFD_FARM_B_STEP=16 SGLANG_AFD_FARM_B_WIN_K=4
export SGLANG_AFD_FARM_COALESCE_K=4 SGLANG_AFD_FARM_MAX_INFLIGHT=2
export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN=256
export SGLANG_AFD_FARM_LOG_EVERY=32
export SGLANG_AFD_IPC_ENDPOINT="$OUT/farm_cuda_ipc.sock"
rm -f "$SGLANG_AFD_IPC_ENDPOINT"
: >"$OUT/farm.pids"

(
  export SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$GPU1"
  python3 -m sglang.launch_server "${COMMON[@]}" --port 33202 --skip-server-warmup
) >"$OUT/farm_ffn.log" 2>&1 &
echo $! >>"$OUT/farm.pids"
sleep 5
(
  export SGLANG_AFD_MODE=attn CUDA_VISIBLE_DEVICES="$GPU0"
  python3 -m sglang.launch_server "${COMMON[@]}" --port 33201
) >"$OUT/farm_attn.log" 2>&1 &
echo $! >>"$OUT/farm.pids"
wait_health 33201 "$(cat "$OUT/farm.pids" | tail -1)" "$OUT/farm_attn.log" \
  || { echo "farm failed" >&2; exit 1; }
run_bench 33201 "farm"
echo "  farm done"
cleanup

echo
echo "==================== DECODE SPEED: 2 GPUs, same load ===================="
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def load(tag):
    p = os.path.join(out, f"{tag}_bench.json")
    return json.load(open(p)) if os.path.isfile(p) else None
mono = [load("mono0"), load("mono1")]
mono = [m for m in mono if m]
farm = load("farm")
mt = sum(m.get("output_throughput", 0) for m in mono)
mtpot = (sum(m.get("median_tpot_ms", 0) for m in mono) / len(mono)) if mono else None
print(f"{'setup':<26} {'out_tps':>10} {'med_tpot':>10} {'med_ttft':>10} {'ok':>5}")
print(f"{'2x full replica (DP=2)':<26} {mt:>10.1f} "
      f"{(mtpot or 0):>10.1f} "
      f"{(mono[0].get('median_ttft_ms', 0) if mono else 0):>10.1f} "
      f"{sum(m.get('completed', 0) for m in mono):>5}")
if farm:
    print(f"{'1A1F farm':<26} {farm.get('output_throughput', 0):>10.1f} "
          f"{farm.get('median_tpot_ms', 0):>10.1f} "
          f"{farm.get('median_ttft_ms', 0):>10.1f} "
          f"{farm.get('completed', 0):>5}")
    if mt and farm.get("output_throughput"):
        print(f"\nfarm/mono tok/s = {farm['output_throughput'] / mt:.3f}x")
    if mtpot and farm.get("median_tpot_ms"):
        print(f"farm/mono med TPOT = {farm['median_tpot_ms'] / mtpot:.3f}x")
PY
echo "AFD_VS_MONOLITHIC_OK"
