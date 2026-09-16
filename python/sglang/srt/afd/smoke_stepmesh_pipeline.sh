#!/usr/bin/env bash
# StepMesh AFD smoke with 3-stage microbatch pipeline (NUM_MB=3).
# Same topology as smoke_stepmesh_lite.sh; enables SGLANG_AFD_PIPELINE.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_pipeline}"
ATTN_PORT="${ATTN_PORT:-32100}"
FFN_PORT="${FFN_PORT:-32101}"
PS_PORT="${PS_PORT:-18224}"
RNIC="${RNIC:-eth1}"
ATTN_GPU="${ATTN_GPU:-1}"
FFN_GPU="${FFN_GPU:-0}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

SCHEDULER_IP=$(ip -o -4 addr show "$RNIC" | awk '{print $4}' | cut -d/ -f1 | head -1)
if [[ -z "$SCHEDULER_IP" ]]; then
  echo "ERROR: no IPv4 on RNIC=$RNIC" >&2
  exit 1
fi

export SGLANG_AFD_TRANSPORT=stepmesh
export SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME="${SGLANG_AFD_ROUTING_SCHEME:-a}"
export SGLANG_AFD_A2F_DTYPE="${SGLANG_AFD_A2F_DTYPE:-auto}"
export SGLANG_AFD_PIPELINE=1
export SGLANG_AFD_NUM_MB=3
export SGLANG_AFD_MAX_NUM_TOKEN=128

export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI="$SCHEDULER_IP"
export DMLC_PS_ROOT_PORT="$PS_PORT"
export DMLC_ENABLE_RDMA=ibverbs
export BYTEPS_ENABLE_IPC="${BYTEPS_ENABLE_IPC:-0}"
export DMLC_INTERFACE=auto
export DMLC_NODE_HOST="$SCHEDULER_IP"
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_BIND_CPU_CORE=0
export PS_VERBOSE="${PS_VERBOSE:-0}"

COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static 0.85
  --max-running-requests 2
  --context-length 2048
  --disable-cuda-graph
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
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "PIPELINE NUM_MB=3 RNIC=$RNIC IP=$SCHEDULER_IP PS_PORT=$PS_PORT"

python3 - <<'PY' >"$LOG_DIR/scheduler.log" 2>&1 &
import os, time
os.environ["DMLC_ROLE"] = "scheduler"
import fserver_lib as f
f.init()
while True:
    time.sleep(3600)
PY
PIDS+=($!)
sleep 2

(
  export DMLC_ROLE=server
  export SGLANG_AFD_MODE=ffn
  export SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$FFN_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$FFN_PORT" \
    --skip-server-warmup
) >"$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!
PIDS+=($FFN_PID)

echo "Waiting for FFN weight load..."
for i in $(seq 1 240); do
  if ! kill -0 "$FFN_PID" 2>/dev/null; then
    echo "FFN died; tail:" >&2
    tail -80 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  if rg -q "Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null; then
    echo "FFN weights loaded after ${i}s"
    break
  fi
  if [[ "$i" -eq 240 ]]; then
    echo "FFN weight-load timeout" >&2
    tail -100 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  sleep 2
done

(
  export DMLC_ROLE=worker
  export SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN_PORT"
) >"$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!
PIDS+=($ATTN_PID)

echo "Waiting for Attn HTTP..."
for i in $(seq 1 240); do
  if ! kill -0 "$ATTN_PID" 2>/dev/null; then
    echo "Attn died; tail:" >&2
    tail -80 "$LOG_DIR/attn.log" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$ATTN_PORT/get_model_info" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:$ATTN_PORT/model_info" >/dev/null 2>&1; then
    echo "Attn healthy after ${i}s"
    break
  fi
  if [[ "$i" -eq 240 ]]; then
    echo "Attn timeout" >&2
    tail -100 "$LOG_DIR/attn.log" >&2
    exit 1
  fi
  sleep 2
done

RESP=$(curl -sf "http://127.0.0.1:$ATTN_PORT/v1/chat/completions" \
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
o=json.load(open(sys.argv[1]))
text=o["choices"][0]["message"]["content"]
print("GENERATED:", text[:500])
assert text.strip(), "empty generation"
print("AFD_STEPMESH_PIPELINE_SMOKE_OK")
PY

rg "AFD (auto-init|runtime ready)|pipeline|NUM_MB|mb=" "$LOG_DIR/attn.log" | tail -15 || true
