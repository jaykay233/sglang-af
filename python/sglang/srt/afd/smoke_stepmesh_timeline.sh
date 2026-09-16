#!/usr/bin/env bash
# StepMesh pipeline timeline: NUM_MB=3 + SGLANG_AFD_TIMELINE logs + Fake overlap proxy.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/stepmesh_smoke_lib.sh"

export SGLANG_AFD_PIPELINE=1
export SGLANG_AFD_NUM_MB=3
export SGLANG_AFD_TIMELINE=1
export SGLANG_AFD_MAX_NUM_TOKEN=128
LOG_DIR="${LOG_DIR:-/tmp/afd_stepmesh_timeline}"
ATTN_PORT="${ATTN_PORT:-32630}"
FFN_PORT="${FFN_PORT:-32631}"
PS_PORT="${PS_PORT:-18353}"
PIDS=()
afd_stepmesh_prep
afd_stepmesh_cleanup_trap

echo "=== Fake overlap proxy (same schedule as StepMesh pipeline) ==="
python3 -m sglang.srt.afd.bench_stepmesh_timeline --fake --min-speedup 1.25

afd_stepmesh_start_scheduler

COMMON_ARGS=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85
  --max-running-requests 2 --context-length 2048
  --disable-cuda-graph --disaggregation-mode null
)

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

(
  export DMLC_ROLE=worker SGLANG_AFD_MODE=attn
  export CUDA_VISIBLE_DEVICES="$ATTN_GPU" STEPMESH_GPU=0
  python3 -m sglang.launch_server "${COMMON_ARGS[@]}" --port "$ATTN_PORT"
) >"$LOG_DIR/attn.log" 2>&1 &
ATTN_PID=$!; PIDS+=($ATTN_PID)

for i in $(seq 1 240); do
  kill -0 "$ATTN_PID" 2>/dev/null || { tail -80 "$LOG_DIR/attn.log" >&2; exit 1; }
  curl -sf "http://127.0.0.1:$ATTN_PORT/get_model_info" >/dev/null 2>&1 && break
  [[ $i -eq 240 ]] && exit 1
  sleep 2
done

# Longer prompt → more tokens so NUM_MB=3 actually splits.
RESP=$(curl -sf "http://127.0.0.1:$ATTN_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V2-Lite-Chat","messages":[{"role":"user","content":"请用三句话介绍深度学习。"}],"max_tokens":96,"temperature":0}')
echo "$RESP" | tee "$LOG_DIR/response.json"
afd_assert_sane_generation "$LOG_DIR/response.json" "AFD_STEPMESH_TIMELINE_GEN_OK"

echo "=== StepMesh AFD_TIMELINE samples ==="
rg "AFD_TIMELINE" "$LOG_DIR/attn.log" | tail -20 || true
python3 -m sglang.srt.afd.bench_stepmesh_timeline --attn-log "$LOG_DIR/attn.log"
echo "AFD_STEPMESH_TIMELINE_SMOKE_OK"
