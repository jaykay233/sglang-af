#!/usr/bin/env bash
# AFD Fake smoke on DeepSeek-V2-Lite-Chat (single GPU, in-process FFN).
set -euo pipefail

# CUDA 12.x + tvm_ffi for deep_gemm / native DeepseekV2 (not Transformers fallback).
if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-31000}"
LOG_DIR="${LOG_DIR:-/tmp/afd_lite_smoke}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi
n_shards=$(find "$MODEL" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
if [[ "$n_shards" -lt 4 ]]; then
  echo "ERROR: expected 4 weight shards, found $n_shards under $MODEL" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export SGLANG_AFD_MODE=attn
export SGLANG_AFD_TRANSPORT=fake
# Fake keeps FFN in-process — need real MLP/experts modules + weights.
export SGLANG_AFD_MODULE_STUBS=0
export SGLANG_AFD_ROUTING_SCHEME=a
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_MAX_NUM_TOKEN=64

echo "Launching AFD Fake server on GPU=$GPU port=$PORT model=$MODEL"
# mem-fraction: ComfyUI leaves ~40GB free; Lite ~32GB bf16 needs headroom for KV.
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --max-running-requests 4 \
  --context-length 2048 \
  --disable-cuda-graph \
  >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"
echo "$SERVER_PID" >"$LOG_DIR/server.pid"

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Waiting for server..."
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server died; tail log:" >&2
    tail -80 "$LOG_DIR/server.log" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
    echo "Server healthy after ${i}s"
    break
  fi
  if rg -q "The server is fired up|Uvicorn running|Application startup complete" "$LOG_DIR/server.log" 2>/dev/null; then
    echo "Server log reports ready after ${i}s"
    break
  fi
  if [[ "$i" -eq 180 ]]; then
    echo "Timeout waiting for server; tail log:" >&2
    tail -100 "$LOG_DIR/server.log" >&2
    exit 1
  fi
  sleep 2
done

echo "Sending generation request..."
RESP=$(curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 64,
    "temperature": 0
  }')
echo "$RESP" | tee "$LOG_DIR/response.json"
python3 - <<'PY' "$LOG_DIR/response.json"
import json,sys
p=sys.argv[1]
o=json.load(open(p))
text=o["choices"][0]["message"]["content"]
print("GENERATED:", text[:500])
assert text.strip(), "empty generation"
print("AFD_LITE_SMOKE_OK")
PY

# Confirm AFD was initialized
if rg -q "AFD (auto-init|runtime ready)" "$LOG_DIR/server.log"; then
  echo "AFD init lines:"
  rg "AFD (auto-init|runtime ready|weight filter)" "$LOG_DIR/server.log" | tail -10
else
  echo "WARNING: no AFD init log found" >&2
fi
