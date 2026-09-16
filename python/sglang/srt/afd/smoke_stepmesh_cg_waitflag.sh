#!/usr/bin/env bash
# StepMesh AFD: breakable CUDA Graph + wait_flag (no --disable-cuda-graph).
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_cg}"
ATTN_PORT="${ATTN_PORT:-32200}"
FFN_PORT="${FFN_PORT:-32201}"
PS_PORT="${PS_PORT:-18225}"
RNIC="${RNIC:-eth1}"
ATTN_GPU="${ATTN_GPU:-1}"
FFN_GPU="${FFN_GPU:-0}"
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
export SGLANG_AFD_A2F_DTYPE="${SGLANG_AFD_A2F_DTYPE:-auto}"
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_MAX_NUM_TOKEN=64
export SGLANG_AFD_USE_WAIT_FLAG=1
export SGLANG_AFD_PIPELINE=0
# Breakable CG capture otherwise stalls inside torch._dynamo compiling MoE topk.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

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

# Decode breakable CG + wait_flag; disable prefill CG (capture OOM/hangs with stubs).
COMMON_ARGS=(
  --model-path "$MODEL"
  --trust-remote-code
  --host 127.0.0.1
  --tp-size 1
  --dtype bfloat16
  --mem-fraction-static 0.82
  --max-running-requests 2
  --context-length 2048
  --cuda-graph-backend-decode breakable
  --cuda-graph-backend-prefill disabled
  --cuda-graph-max-bs-decode 4
  --disaggregation-mode null
)

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
    # Kill process group / children (scheduler + detokenizer)
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

echo "CG+wait_flag RNIC=$RNIC IP=$SCHEDULER_IP PS_PORT=$PS_PORT"

# Ensure PS port is free (stale scheduler causes silent mesh hang).
if fuser "$PS_PORT/tcp" >/dev/null 2>&1; then
  echo "Killing stale listeners on PS_PORT=$PS_PORT"
  fuser -k "$PS_PORT/tcp" 2>/dev/null || true
  sleep 1
fi

python3 - <<'PY' >"$LOG_DIR/scheduler.log" 2>&1 &
import os, time, sys
os.environ["DMLC_ROLE"] = "scheduler"
import fserver_lib as f
f.init()
print("SCHEDULER_READY", flush=True)
while True:
    time.sleep(3600)
PY
SCHED_PID=$!
PIDS+=($SCHED_PID)
echo "SCHED_PID=$SCHED_PID"
for i in $(seq 1 30); do
  if ! kill -0 "$SCHED_PID" 2>/dev/null; then
    echo "Scheduler died during bind; log:" >&2
    cat "$LOG_DIR/scheduler.log" >&2
    exit 1
  fi
  if rg -q "SCHEDULER_READY|Bind to" "$LOG_DIR/scheduler.log" 2>/dev/null; then
    if rg -q "bind failed|Check failed" "$LOG_DIR/scheduler.log" 2>/dev/null; then
      echo "Scheduler bind failed:" >&2
      cat "$LOG_DIR/scheduler.log" >&2
      exit 1
    fi
    echo "Scheduler up after ${i}s"
    break
  fi
  [[ "$i" -eq 30 ]] && { echo "Scheduler timeout"; cat "$LOG_DIR/scheduler.log"; exit 1; }
  sleep 0.5
done

(
  export DMLC_ROLE=server
  export SGLANG_AFD_MODE=ffn
  export SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export SGLANG_AFD_FFN_CUDA_GRAPH=1
  # wait_flag is Attn-side; FFN still uses normal poll
  export SGLANG_AFD_USE_WAIT_FLAG=0
  export CUDA_VISIBLE_DEVICES="$FFN_GPU"
  export STEPMESH_GPU=0
  # Model decode CG auto-disabled on FFN; MLP/MoE graphs via SGLANG_AFD_FFN_CUDA_GRAPH.
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
    --port "$FFN_PORT" \
    --skip-server-warmup \
    --cuda-graph-backend-prefill disabled
) >"$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!
PIDS+=($FFN_PID)

echo "Waiting for FFN weight load + FFN CUDA graph..."
for i in $(seq 1 240); do
  if ! kill -0 "$FFN_PID" 2>/dev/null; then
    echo "FFN died; tail:" >&2
    tail -80 "$LOG_DIR/ffn.log" >&2
    exit 1
  fi
  if rg -q "AFD FFN CUDA graph|AFD auto-init done|Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null \
    && rg -q "Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null; then
    echo "FFN weights loaded after ${i}s"
    break
  fi
  [[ "$i" -eq 240 ]] && { tail -100 "$LOG_DIR/ffn.log" >&2; exit 1; }
  sleep 2
done

(
  export DMLC_ROLE=worker
  export SGLANG_AFD_MODE=attn
  export SGLANG_AFD_USE_WAIT_FLAG=1
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU"
  export STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN_PORT"
) >"$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!
PIDS+=($ATTN_PID)

echo "Waiting for Attn HTTP (decode CG capture can take a few minutes)..."
for i in $(seq 1 300); do
  if ! kill -0 "$ATTN_PID" 2>/dev/null; then
    echo "Attn died; tail:" >&2
    tail -100 "$LOG_DIR/attn.log" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$ATTN_PORT/health" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:$ATTN_PORT/get_model_info" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:$ATTN_PORT/model_info" >/dev/null 2>&1; then
    echo "Attn healthy after ${i}s"
    break
  fi
  if (( i % 15 == 0 )); then
    echo "  still waiting ${i}s... last attn log:"
    tail -3 "$LOG_DIR/attn.log" 2>/dev/null || true
  fi
  [[ "$i" -eq 300 ]] && { echo "Attn timeout"; tail -120 "$LOG_DIR/attn.log" >&2; exit 1; }
  sleep 2
done

# Require wait_flag + breakable CG logged
if ! rg -q "wait_flag sync enabled|AFD wait_flag" "$LOG_DIR/attn.log"; then
  echo "WARNING: wait_flag enable line not found — check fallback" >&2
  rg "wait_flag|CUDA graph|breakable" "$LOG_DIR/attn.log" | tail -20 || true
fi
if ! rg -qi "breakable|cuda.graph" "$LOG_DIR/attn.log"; then
  echo "WARNING: no CUDA graph / breakable log lines" >&2
fi

RESP=$(curl -sf "http://127.0.0.1:$ATTN_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 48,
    "temperature": 0
  }')
echo "$RESP" | tee "$LOG_DIR/response.json"
python3 - <<'PY' "$LOG_DIR/response.json" "$LOG_DIR/attn.log"
import json,sys,re
o=json.load(open(sys.argv[1]))
text=o["choices"][0]["message"]["content"]
print("GENERATED:", text[:500])
assert text.strip(), "empty generation"
log=open(sys.argv[2]).read()
assert "wait_flag sync enabled" in log or "AFD wait_flag" in log, "wait_flag not enabled"
ok_cg = (
    "breakable" in log.lower()
    or "CudaGraph" in log
    or "cuda graph" in log.lower()
    or "Capture target decode CUDA graph" in log
)
assert ok_cg, "no evidence of CUDA graph / breakable path"
cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
latin = len(re.findall(r"[A-Za-z]", text))
weird = len(re.findall(r"[*~=\\[\]{}|]", text))
assert cjk >= 4 or latin >= 8, f"nonsensical generation cjk={cjk} latin={latin}"
assert weird < max(8, len(text) // 3), f"too much noise in generation"
print("AFD_STEPMESH_CG_WAITFLAG_SMOKE_OK")
PY

rg "wait_flag|breakable|AFD (auto-init|runtime ready)" "$LOG_DIR/attn.log" | tail -20 || true
