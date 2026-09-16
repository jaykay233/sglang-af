#!/usr/bin/env bash
# Same-host AFD smoke via CUDA IPC (no StepMesh / RDMA).
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_cuda_ipc_smoke}"
ATTN_PORT="${ATTN_PORT:-32500}"
FFN_PORT="${FFN_PORT:-32501}"
ATTN_GPU="${ATTN_GPU:-1}"
FFN_GPU="${FFN_GPU:-2}"
IPC_ENDPOINT="${SGLANG_AFD_IPC_ENDPOINT:-$LOG_DIR/afd_cuda_ipc.sock}"
mkdir -p "$LOG_DIR"
rm -f "$IPC_ENDPOINT"

[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

export SGLANG_AFD_TRANSPORT=cuda_ipc
export SGLANG_AFD_IPC_ENDPOINT="$IPC_ENDPOINT"
export SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME=a
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_MAX_NUM_TOKEN=8
export SGLANG_AFD_PIPELINE=0
export SGLANG_AFD_USE_WAIT_FLAG=0
export SGLANG_AFD_FFN_CUDA_GRAPH=1
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1

COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static 0.5
  --max-running-requests 2
  --context-length 2048
  --cuda-graph-backend-prefill disabled
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

echo "cuda_ipc smoke ATTN_GPU=$ATTN_GPU FFN_GPU=$FFN_GPU endpoint=$IPC_ENDPOINT"

(
  export SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$FFN_GPU"
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$FFN_PORT" --skip-server-warmup \
    --cuda-graph-backend-decode disabled
) >"$LOG_DIR/ffn.log" 2>&1 &
PIDS+=($!)
FFN_PID=${PIDS[-1]}

echo "Waiting for FFN listen..."
for i in $(seq 1 180); do
  kill -0 "$FFN_PID" 2>/dev/null || { tail -80 "$LOG_DIR/ffn.log" >&2; exit 1; }
  if rg -q "AFD cuda_ipc FFN waiting|Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null; then
    echo "FFN listening after ${i}s"
    break
  fi
  [[ $i -eq 180 ]] && { tail -100 "$LOG_DIR/ffn.log" >&2; exit 1; }
  sleep 2
done

(
  export SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$ATTN_PORT" \
    --cuda-graph-backend-decode breakable \
    --cuda-graph-max-bs-decode 4
) >"$LOG_DIR/attn.log" 2>&1 &
PIDS+=($!)
ATTN_PID=${PIDS[-1]}

echo "Waiting for Attn health..."
for i in $(seq 1 300); do
  kill -0 "$ATTN_PID" 2>/dev/null || { tail -100 "$LOG_DIR/attn.log" >&2; exit 1; }
  if curl -sf "http://127.0.0.1:$ATTN_PORT/health" >/dev/null 2>&1; then
    echo "Attn healthy after ${i}s"
    break
  fi
  [[ $i -eq 300 ]] && { tail -100 "$LOG_DIR/attn.log" >&2; exit 1; }
  sleep 2
done

rg -q "transport=cuda_ipc|cuda_ipc buffers ready" "$LOG_DIR/ffn.log" || {
  echo "FFN missing cuda_ipc ready marker" >&2
  tail -40 "$LOG_DIR/ffn.log" >&2
  exit 1
}
rg -q "transport=cuda_ipc|cuda_ipc buffers ready" "$LOG_DIR/attn.log" || {
  echo "Attn missing cuda_ipc ready marker" >&2
  tail -40 "$LOG_DIR/attn.log" >&2
  exit 1
}

curl -sf "http://127.0.0.1:$ATTN_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"用一句话介绍北京"}],"max_tokens":32,"temperature":0}' \
  | tee "$LOG_DIR/response.json"

python3 - <<'PY' "$LOG_DIR/response.json"
import json, sys, re
path = sys.argv[1]
obj = json.load(open(path))
text = obj["choices"][0]["message"]["content"]
print("GEN:", text)
# crude sanity: chinese chars or beijing
ok = bool(re.search(r"[\u4e00-\u9fff]|Beijing|beijing|中国|首都", text))
assert ok, f"unexpected generation: {text!r}"
print("AFD_CUDA_IPC_SMOKE_OK")
PY
