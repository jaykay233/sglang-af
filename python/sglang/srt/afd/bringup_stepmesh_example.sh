#!/usr/bin/env bash
# Example Attn+FFN bring-up with StepMesh (edit hosts / model path first).
# Fake CI path (no RDMA):
#   python -m sglang.srt.afd.smoke
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
HOST="${DMLC_PS_ROOT_URI:-127.0.0.1}"
PORT="${DMLC_PS_ROOT_PORT:-8000}"

export SGLANG_AFD_TRANSPORT=stepmesh
export SGLANG_AFD_MODULE_STUBS=1
export SGLANG_AFD_ROUTING_SCHEME="${SGLANG_AFD_ROUTING_SCHEME:-a}"
export SGLANG_AFD_A2F_DTYPE="${SGLANG_AFD_A2F_DTYPE:-auto}"
export DMLC_PS_ROOT_URI="$HOST"
export DMLC_PS_ROOT_PORT="$PORT"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export DMLC_NUM_SERVER="${DMLC_NUM_SERVER:-1}"

echo "Starting FFN worker (GPU 0)..."
SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m sglang.launch_server --model-path "$MODEL_PATH" \
  --disaggregation-mode null \
  --port 30001 &
FFN_PID=$!

cleanup() { kill "$FFN_PID" 2>/dev/null || true; }
trap cleanup EXIT

sleep 5
echo "Starting Attn/decode worker (GPU 1)..."
SGLANG_AFD_MODE=attn CUDA_VISIBLE_DEVICES=1 \
  python -m sglang.launch_server --model-path "$MODEL_PATH" \
  --disaggregation-mode null \
  --port 30000

wait
