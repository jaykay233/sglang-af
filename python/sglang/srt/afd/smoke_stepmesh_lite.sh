#!/usr/bin/env bash
# AFD StepMesh smoke: 1 Attn (worker) + 1 FFN (server) + scheduler on one host.
# Requires RDMA NIC (RoCE/IB) and source /root/.cuda/afd_env.sh (or equiv).
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_smoke}"
ATTN_PORT="${ATTN_PORT:-32000}"
FFN_PORT="${FFN_PORT:-32001}"
# Prefer explicit PS_PORT; do not inherit stale DMLC_PS_ROOT_PORT from prior tests.
PS_PORT="${PS_PORT:-18223}"
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
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_MAX_NUM_TOKEN=64

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
    kill "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

echo "RNIC=$RNIC SCHEDULER_IP=$SCHEDULER_IP PS_PORT=$PS_PORT"
echo "FFN_GPU=$FFN_GPU ATTN_GPU=$ATTN_GPU model=$MODEL"

# --- scheduler (blocks inside f.init / event loop) ---
python3 - <<'PY' >"$LOG_DIR/scheduler.log" 2>&1 &
import os, time
os.environ["DMLC_ROLE"] = "scheduler"
import fserver_lib as f
f.init()
while True:
    time.sleep(3600)
PY
PIDS+=($!)
echo "SCHED_PID=${PIDS[-1]}"
sleep 2

# --- FFN (StepMesh server) ---
# FFN stubs self_attn / skips Attn weights — must not run normal forward/warmup.
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
echo "FFN_PID=$FFN_PID"

# Wait until FFN finishes weight load (f.init then blocks until Attn joins).
echo "Waiting for FFN weight load..."
for i in $(seq 1 240); do
  if ! kill -0 "$FFN_PID" 2>/dev/null; then
    echo "FFN died; tail:" >&2
    tail -80 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  if rg -q "Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null; then
    echo "FFN weights loaded after ${i}s (f.init may wait for Attn)"
    break
  fi
  if [[ "$i" -eq 240 ]]; then
    echo "FFN weight-load timeout; tail:" >&2
    tail -100 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  sleep 2
done

# --- Attn (StepMesh worker) — must start so FFN f.init can finish ---
(
  export DMLC_ROLE=worker
  export SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN_PORT"
) >"$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!
PIDS+=($ATTN_PID)
echo "ATTN_PID=$ATTN_PID"

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
    echo "Attn timeout; tail:" >&2
    tail -100 "$LOG_DIR/attn.log" >&2
    echo "--- ffn tail ---" >&2
    tail -40 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  sleep 2
done

# Both sides should have completed StepMesh init by now.
if ! rg -q "AFD (auto-init|runtime ready)" "$LOG_DIR/ffn.log" 2>/dev/null; then
  echo "WARNING: FFN AFD runtime not logged yet" >&2
fi
if ! rg -q "AFD (auto-init|runtime ready)" "$LOG_DIR/attn.log" 2>/dev/null; then
  echo "WARNING: Attn AFD runtime not logged yet" >&2
fi

echo "Sending generation request to Attn..."
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
print("AFD_STEPMESH_SMOKE_OK")
PY

echo "AFD init lines (attn):"
rg "AFD (auto-init|runtime ready)|StepMesh AFD" "$LOG_DIR/attn.log" | tail -10 || true
echo "AFD init lines (ffn):"
rg "AFD (auto-init|runtime ready)|StepMesh AFD|FFN poll" "$LOG_DIR/ffn.log" | tail -10 || true
