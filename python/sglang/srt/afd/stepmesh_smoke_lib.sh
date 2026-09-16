#!/usr/bin/env bash
# Shared StepMesh 1A1F launcher helpers for AFD variant smokes.
# shellcheck disable=SC2034

afd_stepmesh_prep() {
  if [[ -f /root/.cuda/afd_env.sh ]]; then
    # shellcheck disable=SC1091
    source /root/.cuda/afd_env.sh
  fi
  MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
  LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_var}"
  ATTN_PORT="${ATTN_PORT:-32600}"
  FFN_PORT="${FFN_PORT:-32601}"
  PS_PORT="${PS_PORT:-18350}"
  RNIC="${RNIC:-eth1}"
  ATTN_GPU="${ATTN_GPU:-1}"
  FFN_GPU="${FFN_GPU:-0}"
  mkdir -p "$LOG_DIR"
  [[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; return 1; }
  SCHEDULER_IP=$(ip -o -4 addr show "$RNIC" | awk '{print $4}' | cut -d/ -f1 | head -1)
  [[ -n "$SCHEDULER_IP" ]] || { echo "ERROR: no IPv4 on $RNIC" >&2; return 1; }

  export SGLANG_AFD_TRANSPORT=stepmesh
  export SGLANG_AFD_MODULE_STUBS=1
  export SGLANG_AFD_ROUTING_SCHEME="${SGLANG_AFD_ROUTING_SCHEME:-a}"
  export SGLANG_AFD_A2F_DTYPE="${SGLANG_AFD_A2F_DTYPE:-auto}"
  export SGLANG_AFD_NUM_MB="${SGLANG_AFD_NUM_MB:-1}"
  export SGLANG_AFD_MAX_NUM_TOKEN="${SGLANG_AFD_MAX_NUM_TOKEN:-64}"
  export SGLANG_AFD_PIPELINE="${SGLANG_AFD_PIPELINE:-0}"
  export SGLANG_AFD_USE_WAIT_FLAG="${SGLANG_AFD_USE_WAIT_FLAG:-0}"
  export SGLANG_AFD_TIMELINE="${SGLANG_AFD_TIMELINE:-0}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

  export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
  export DMLC_NUM_SERVER="${DMLC_NUM_SERVER:-1}"
  export DMLC_PS_ROOT_URI="$SCHEDULER_IP"
  export DMLC_PS_ROOT_PORT="$PS_PORT"
  export DMLC_ENABLE_RDMA=ibverbs
  # Opt-in same-host shm IPC (BYTEPS_ENABLE_IPC=1). Default off — GPU A2F was
  # slower than GDR RDMA in PD+AFD bench.
  export BYTEPS_ENABLE_IPC="${BYTEPS_ENABLE_IPC:-0}"
  export DMLC_INTERFACE=auto
  export DMLC_NODE_HOST="$SCHEDULER_IP"
  export STEPMESH_SPLIT_QP_LAG=0
  export STEPMESH_BIND_CPU_CORE=0
  export PS_VERBOSE="${PS_VERBOSE:-0}"

  if fuser "$PS_PORT/tcp" >/dev/null 2>&1; then
    fuser -k "$PS_PORT/tcp" 2>/dev/null || true
    sleep 1
  fi
}

afd_stepmesh_start_scheduler() {
  python3 - <<'PY' >"$LOG_DIR/scheduler.log" 2>&1 &
import os, time
os.environ["DMLC_ROLE"] = "scheduler"
import fserver_lib as f
f.init()
print("SCHEDULER_READY", flush=True)
while True:
    time.sleep(3600)
PY
  local pid=$!
  PIDS+=("$pid")
  for i in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Scheduler died:" >&2; cat "$LOG_DIR/scheduler.log" >&2; return 1
    fi
    if rg -q "SCHEDULER_READY|Bind to" "$LOG_DIR/scheduler.log" 2>/dev/null; then
      if rg -q "bind failed|Check failed" "$LOG_DIR/scheduler.log" 2>/dev/null; then
        echo "Scheduler bind failed:" >&2; cat "$LOG_DIR/scheduler.log" >&2; return 1
      fi
      echo "Scheduler up"
      return 0
    fi
    sleep 0.5
  done
  echo "Scheduler timeout" >&2; return 1
}

afd_stepmesh_cleanup_trap() {
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
}

afd_assert_sane_generation() {
  python3 - <<'PY' "$1" "$2"
import json, re, sys
path, tag = sys.argv[1], sys.argv[2]
o = json.load(open(path))
text = o["choices"][0]["message"]["content"]
print("GENERATED:", text[:500])
assert text.strip(), "empty generation"
# Reject obvious garbage (CG+wait_flag regression): mostly non-CJK/latin noise.
cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
latin = len(re.findall(r"[A-Za-z]", text))
weird = len(re.findall(r"[*~=\\[\]{}|]", text))
assert cjk >= 4 or latin >= 8, f"nonsensical generation cjk={cjk} latin={latin}: {text[:120]!r}"
assert weird < max(8, len(text) // 3), f"too much noise chars: {text[:120]!r}"
print(tag)
PY
}
