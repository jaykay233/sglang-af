#!/usr/bin/env bash
# MxN bring-up: 2 Attn workers × 1 FFN server (DMLC_NUM_WORKER=2).
# Each Attn has its own HTTP port; FFN serves both via StepMesh get_batch.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/stepmesh_smoke_lib.sh"

export DMLC_NUM_WORKER=2
export DMLC_NUM_SERVER=1
LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_mxn}"
ATTN0_PORT="${ATTN0_PORT:-32640}"
ATTN1_PORT="${ATTN1_PORT:-32641}"
FFN_PORT="${FFN_PORT:-32642}"
PS_PORT="${PS_PORT:-18354}"
ATTN0_GPU="${ATTN0_GPU:-1}"
ATTN1_GPU="${ATTN1_GPU:-2}"
FFN_GPU="${FFN_GPU:-0}"
PIDS=()
afd_stepmesh_prep
# override ports/gpus from prep defaults
ATTN_PORT=$ATTN0_PORT
afd_stepmesh_cleanup_trap
afd_stepmesh_start_scheduler

COMMON_ARGS=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.82
  --max-running-requests 2 --context-length 2048
  --disable-cuda-graph --disaggregation-mode null
)

echo "MxN 2A1F FFN=$FFN_GPU ATTN0=$ATTN0_GPU ATTN1=$ATTN1_GPU"

(
  export DMLC_ROLE=server SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
  export CUDA_VISIBLE_DEVICES="$FFN_GPU" STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$FFN_PORT" --skip-server-warmup
) >"$LOG_DIR/ffn.log" 2>&1 &
FFN_PID=$!; PIDS+=($FFN_PID)

for i in $(seq 1 240); do
  kill -0 "$FFN_PID" 2>/dev/null || { tail -80 "$LOG_DIR/ffn.log" >&2; exit 1; }
  rg -q "Load weight end" "$LOG_DIR/ffn.log" 2>/dev/null && break
  [[ $i -eq 240 ]] && exit 1
  sleep 2
done
echo "FFN ready — starting both Attn workers"

# StepMesh preferred rank = DMLC_GROUP_SIZE * DMLC_NODE_RANK + STEPMESH_GPU.
# With CUDA_VISIBLE_DEVICES remapping STEPMESH_GPU=0, ranks must differ via DMLC_NODE_RANK.
(
  export DMLC_ROLE=worker SGLANG_AFD_MODE=attn
  export DMLC_NODE_RANK=0 SGLANG_AFD_WORKER_RANK=0
  export CUDA_VISIBLE_DEVICES="$ATTN0_GPU" STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN0_PORT"
) >"$LOG_DIR/attn0.log" 2>&1 &
ATTN0_PID=$!; PIDS+=($ATTN0_PID)

(
  export DMLC_ROLE=worker SGLANG_AFD_MODE=attn
  export DMLC_NODE_RANK=1 SGLANG_AFD_WORKER_RANK=1
  export CUDA_VISIBLE_DEVICES="$ATTN1_GPU" STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN1_PORT"
) >"$LOG_DIR/attn1.log" 2>&1 &
ATTN1_PID=$!; PIDS+=($ATTN1_PID)

wait_http() {
  local port=$1 pid=$2 name=$3
  for i in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name died" >&2; tail -80 "$LOG_DIR/${name}.log" >&2; exit 1
    fi
    curl -sf "http://127.0.0.1:$port/get_model_info" >/dev/null 2>&1 && {
      echo "$name healthy after ${i}s"; return 0
    }
    (( i % 20 == 0 )) && echo "  waiting $name ${i}s..."
    sleep 2
  done
  echo "$name timeout" >&2; tail -100 "$LOG_DIR/${name}.log" >&2; exit 1
}

wait_http "$ATTN0_PORT" "$ATTN0_PID" attn0
wait_http "$ATTN1_PORT" "$ATTN1_PID" attn1

rg -q "worker_rank=0" "$LOG_DIR/attn0.log" || echo "WARNING: attn0 rank log missing" >&2
rg -q "worker_rank=1" "$LOG_DIR/attn1.log" || echo "WARNING: attn1 rank log missing" >&2
rg -q "dmlc_node_rank=0" "$LOG_DIR/attn0.log" || echo "WARNING: attn0 DMLC_NODE_RANK missing" >&2
rg -q "dmlc_node_rank=1" "$LOG_DIR/attn1.log" || echo "WARNING: attn1 DMLC_NODE_RANK missing" >&2

# StepMesh get_batch waits for ALL workers — issue concurrent requests.
BODY='{"model":"DeepSeek-V2-Lite-Chat","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":48,"temperature":0}'
curl -sf "http://127.0.0.1:$ATTN0_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" >"$LOG_DIR/response_${ATTN0_PORT}.json" &
PID_R0=$!
curl -sf "http://127.0.0.1:$ATTN1_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" >"$LOG_DIR/response_${ATTN1_PORT}.json" &
PID_R1=$!
wait "$PID_R0" "$PID_R1"

for port in "$ATTN0_PORT" "$ATTN1_PORT"; do
  cat "$LOG_DIR/response_${port}.json"
  echo
  afd_assert_sane_generation "$LOG_DIR/response_${port}.json" "AFD_MXN_PORT_${port}_OK"
done

echo "AFD_STEPMESH_MXN_SMOKE_OK"
