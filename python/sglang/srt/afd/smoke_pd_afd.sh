#!/usr/bin/env bash
# Prefill∥Decode-AFD: PD mooncake + StepMesh AFD on decode Attn + FFN.
# Topology (3 GPUs):
#   Prefill GPU0  — AFD null, PD prefill (full model)
#   Decode  GPU1  — AFD attn + PD decode
#   FFN     GPU2  — AFD ffn, PD null
# Router on LB_PORT; StepMesh scheduler on eth1.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

# Router package (sglang_router)
python3 -c "import sglang_router" 2>/dev/null || \
  pip install -q "sglang-router==0.3.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_pd_smoke}"
PREFILL_PORT="${PREFILL_PORT:-32300}"
DECODE_PORT="${DECODE_PORT:-32301}"
FFN_PORT="${FFN_PORT:-32302}"
LB_PORT="${LB_PORT:-32310}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-32350}"
PS_PORT="${PS_PORT:-18226}"
RNIC="${RNIC:-eth1}"
IB_DEVICE="${IB_DEVICE:-mlx5_1}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
FFN_GPU="${FFN_GPU:-2}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

SCHEDULER_IP=$(ip -o -4 addr show "$RNIC" | awk '{print $4}' | cut -d/ -f1 | head -1)
[[ -n "$SCHEDULER_IP" ]] || { echo "ERROR: no IPv4 on RNIC=$RNIC" >&2; exit 1; }

export SGLANG_AFD_TRANSPORT=stepmesh
export SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME="${SGLANG_AFD_ROUTING_SCHEME:-a}"
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_MAX_NUM_TOKEN=64
export SGLANG_AFD_PIPELINE=0
export SGLANG_AFD_USE_WAIT_FLAG=0

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

MODEL_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static 0.82
  --max-running-requests 2
  --context-length 2048
  --disable-cuda-graph
)

PD_ARGS=(
  --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device "$IB_DEVICE"
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

echo "PD+AFD prefill=$PREFILL_GPU decode=$DECODE_GPU ffn=$FFN_GPU ib=$IB_DEVICE"

# Ensure PS port is free
if fuser "$PS_PORT/tcp" >/dev/null 2>&1; then
  echo "Killing stale listeners on PS_PORT=$PS_PORT"
  fuser -k "$PS_PORT/tcp" 2>/dev/null || true
  sleep 1
fi

# StepMesh scheduler
python3 - <<'PY' >"$LOG_DIR/scheduler.log" 2>&1 &
import os, time
os.environ["DMLC_ROLE"] = "scheduler"
import fserver_lib as f
f.init()
print("SCHEDULER_READY", flush=True)
while True:
    time.sleep(3600)
PY
SCHED_PID=$!
PIDS+=($SCHED_PID)
for i in $(seq 1 30); do
  if ! kill -0 "$SCHED_PID" 2>/dev/null; then
    echo "Scheduler died:" >&2; cat "$LOG_DIR/scheduler.log" >&2; exit 1
  fi
  if rg -q "SCHEDULER_READY|Bind to" "$LOG_DIR/scheduler.log" 2>/dev/null; then
    if rg -q "bind failed|Check failed" "$LOG_DIR/scheduler.log" 2>/dev/null; then
      echo "Scheduler bind failed:" >&2; cat "$LOG_DIR/scheduler.log" >&2; exit 1
    fi
    echo "Scheduler up after ${i}s"
    break
  fi
  [[ "$i" -eq 30 ]] && { echo "Scheduler timeout"; cat "$LOG_DIR/scheduler.log"; exit 1; }
  sleep 0.5
done

# Prefill: no AFD
(
  unset SGLANG_AFD_MODE || true
  export SGLANG_AFD_MODE=null
  export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"
  python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
    --port "$PREFILL_PORT" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    "${PD_ARGS[@]}"
) >"$LOG_DIR/prefill.log" 2>&1 &
PIDS+=($!)
PREFILL_PID=${PIDS[-1]}

# FFN: AFD ffn, start early so weights load before decode joins StepMesh
(
  export DMLC_ROLE=server
  export SGLANG_AFD_MODE=ffn
  export SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$FFN_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
    --port "$FFN_PORT" \
    --disaggregation-mode null \
    --skip-server-warmup
) >"$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!
PIDS+=($FFN_PID)

echo "Waiting for Prefill health + FFN weights..."
for i in $(seq 1 300); do
  pref_ok=0
  ffn_ok=0
  curl -sf "http://127.0.0.1:$PREFILL_PORT/health" >/dev/null 2>&1 && pref_ok=1
  rg -q "Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null && ffn_ok=1
  if [[ "$pref_ok" -eq 1 && "$ffn_ok" -eq 1 ]]; then
    echo "Prefill+FFN ready after ${i}s"
    break
  fi
  if ! kill -0 "$PREFILL_PID" 2>/dev/null; then
    echo "Prefill died" >&2; tail -80 "$LOG_DIR/prefill.log" >&2; exit 1
  fi
  if ! kill -0 "$FFN_PID" 2>/dev/null; then
    echo "FFN died" >&2; tail -80 "$LOG_DIR/ffn.log" >&2; exit 1
  fi
  [[ "$i" -eq 300 ]] && {
    echo "timeout pref=$pref_ok ffn=$ffn_ok" >&2
    tail -60 "$LOG_DIR/prefill.log" >&2
    tail -60 "$LOG_DIR/ffn.log" >&2
    exit 1
  }
  sleep 2
done

# Decode Attn: AFD attn + PD decode
(
  export DMLC_ROLE=worker
  export SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$DECODE_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
    --port "$DECODE_PORT" \
    --disaggregation-mode decode \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    "${PD_ARGS[@]}"
) >"$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
PIDS+=($DECODE_PID)

echo "Waiting for Decode health..."
for i in $(seq 1 300); do
  if ! kill -0 "$DECODE_PID" 2>/dev/null; then
    echo "Decode died" >&2; tail -100 "$LOG_DIR/decode.log" >&2; exit 1
  fi
  if curl -sf "http://127.0.0.1:$DECODE_PORT/health" >/dev/null 2>&1; then
    echo "Decode healthy after ${i}s"
    break
  fi
  [[ "$i" -eq 300 ]] && { tail -120 "$LOG_DIR/decode.log" >&2; exit 1; }
  sleep 2
done

# Router
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill "http://127.0.0.1:$PREFILL_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --host 127.0.0.1 \
  --port "$LB_PORT" \
  >"$LOG_DIR/router.log" 2>&1 &
PIDS+=($!)

for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$LB_PORT/health" >/dev/null 2>&1; then
    echo "Router healthy after ${i}s"
    break
  fi
  [[ "$i" -eq 60 ]] && { tail -40 "$LOG_DIR/router.log" >&2; exit 1; }
  sleep 1
done

# Policy log checks
rg -q "AFD×PD: decode Attn|AFD.*decode" "$LOG_DIR/decode.log" \
  || rg -q "AFD auto-init|runtime ready" "$LOG_DIR/decode.log" \
  || echo "WARNING: AFD×PD log not found on decode" >&2

RESP=$(curl -sf "http://127.0.0.1:$LB_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 48,
    "temperature": 0
  }')
echo "$RESP" | tee "$LOG_DIR/response.json"
python3 - <<'PY' "$LOG_DIR/response.json" "$LOG_DIR/decode.log" "$LOG_DIR/ffn.log"
import json,sys
o=json.load(open(sys.argv[1]))
text=o["choices"][0]["message"]["content"]
print("GENERATED:", text[:500])
assert text.strip(), "empty generation"
d=open(sys.argv[2]).read()
f=open(sys.argv[3]).read()
assert "AFD" in d and ("attn" in d.lower() or "auto-init" in d), "decode missing AFD"
assert "AFD" in f and ("ffn" in f.lower() or "auto-init" in f), "ffn missing AFD"
print("AFD_PD_STEPMESH_SMOKE_OK")
PY

echo "--- decode AFD/PD lines ---"
rg "AFD|disaggregation|PD" "$LOG_DIR/decode.log" | tail -20 || true
