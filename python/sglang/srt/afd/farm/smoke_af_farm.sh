#!/usr/bin/env bash
# True AF (1A+1F cuda_ipc) + decode farm smoke on DeepSeek-V2-Lite.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_farm_smoke}"
ATTN_PORT="${ATTN_PORT:-32610}"
FFN_PORT="${FFN_PORT:-32611}"
ATTN_GPU="${ATTN_GPU:-2}"
FFN_GPU="${FFN_GPU:-3}"
IPC_ENDPOINT="${SGLANG_AFD_IPC_ENDPOINT:-$LOG_DIR/afd_farm_cuda_ipc.sock}"
mkdir -p "$LOG_DIR"
rm -f "$IPC_ENDPOINT"

[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

export SGLANG_AFD_TRANSPORT=cuda_ipc
export SGLANG_AFD_IPC_ENDPOINT="$IPC_ENDPOINT"
export SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME=a
export SGLANG_AFD_PIPELINE=0
export SGLANG_AFD_USE_WAIT_FLAG=0
export SGLANG_AFD_FFN_CUDA_GRAPH=0
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

# Decode farm
export SGLANG_AFD_FARM=1
export SGLANG_AFD_FARM_B_STEP="${SGLANG_AFD_FARM_B_STEP:-8}"
export SGLANG_AFD_FARM_B_WIN_K="${SGLANG_AFD_FARM_B_WIN_K:-4}"
export SGLANG_AFD_FARM_COALESCE_K="${SGLANG_AFD_FARM_COALESCE_K:-2}"
export SGLANG_AFD_FARM_MAX_INFLIGHT="${SGLANG_AFD_FARM_MAX_INFLIGHT:-2}"
export SGLANG_AFD_FARM_NUM_CONTEXTS="${SGLANG_AFD_FARM_NUM_CONTEXTS:-2}"
export SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS="${SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS:-1}"
export SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER="${SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER:-1}"
export SGLANG_AFD_FARM_STAGE_STATS_EVERY="${SGLANG_AFD_FARM_STAGE_STATS_EVERY:-1}"
export SGLANG_AFD_FARM_LOG_EVERY=1
export SGLANG_AFD_NUM_MB=2
# A2F pad: cover short chat + light serving; keep FFN CG off (below) so pad stays cheap.
export SGLANG_AFD_MAX_NUM_TOKEN="${SGLANG_AFD_MAX_NUM_TOKEN:-256}"
export SGLANG_AFD_FFN_CUDA_GRAPH=0

COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  # ~0.72–0.78 on A800 with ~40GB free after ComfyUI; 0.48 starves KV.
  --mem-fraction-static "${MEM_FRACTION:-0.75}"
  --max-running-requests 8
  --context-length 2048
  --cuda-graph-backend-prefill disabled
  --cuda-graph-backend-decode disabled
  --disaggregation-mode null
)

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
    pkill -TERM -P "$p" 2>/dev/null || true
  done
  sleep 1
  for p in "${PIDS[@]:-}"; do
    kill -KILL "$p" 2>/dev/null || true
    pkill -KILL -P "$p" 2>/dev/null || true
  done
  rm -f "$IPC_ENDPOINT" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "AFD farm smoke ATTN_GPU=$ATTN_GPU FFN_GPU=$FFN_GPU"
echo "  FARM=1 B_STEP=$SGLANG_AFD_FARM_B_STEP COALESCE_K=$SGLANG_AFD_FARM_COALESCE_K"

(
  export SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$FFN_GPU"
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$FFN_PORT" --skip-server-warmup
) >"$LOG_DIR/ffn.log" 2>&1 &
PIDS+=($!)
FFN_PID=${PIDS[-1]}

echo "Waiting for FFN..."
for i in $(seq 1 240); do
  kill -0 "$FFN_PID" 2>/dev/null || { tail -100 "$LOG_DIR/ffn.log" >&2; exit 1; }
  if rg -q "AFD cuda_ipc FFN waiting|Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null; then
    echo "FFN ready after ${i}s"
    break
  fi
  [[ $i -eq 240 ]] && { tail -120 "$LOG_DIR/ffn.log" >&2; exit 1; }
  sleep 2
done

(
  export SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$ATTN_PORT"
) >"$LOG_DIR/attn.log" 2>&1 &
PIDS+=($!)
ATTN_PID=${PIDS[-1]}

echo "Waiting for Attn health..."
for i in $(seq 1 360); do
  kill -0 "$ATTN_PID" 2>/dev/null || { tail -120 "$LOG_DIR/attn.log" >&2; exit 1; }
  if curl -sf "http://127.0.0.1:$ATTN_PORT/health" >/dev/null 2>&1; then
    echo "Attn healthy after ${i}s"
    break
  fi
  [[ $i -eq 360 ]] && { tail -120 "$LOG_DIR/attn.log" >&2; exit 1; }
  sleep 2
done

# Confirm farm / AF wiring
echo "--- AFD / farm log snippets ---"
rg -n "AFD farm on|AFD (auto-init|runtime ready)|cuda_ipc|weight filter|MODE" \
  "$LOG_DIR/attn.log" "$LOG_DIR/ffn.log" 2>/dev/null | tail -40 || true

echo "Sending decode request..."
RESP=$(curl -sf "http://127.0.0.1:$ATTN_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 48,
    "temperature": 0
  }')
echo "$RESP" | tee "$LOG_DIR/response.json"
python3 - <<'PY' "$LOG_DIR/response.json"
import json,sys
o=json.load(open(sys.argv[1]))
text=o["choices"][0]["message"]["content"]
print("GENERATED:", text[:400])
assert text.strip(), "empty generation"
print("AFD_FARM_SMOKE_OK")
PY

echo "--- post-request farm occupancy ---"
rg -n "AFD farm (on|occupancy|B_win)|farm failed|sequential fallback" \
  "$LOG_DIR/attn.log" 2>/dev/null | tail -30 || true
