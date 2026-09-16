#!/usr/bin/env bash
# Compare DeepSeek-V2-Lite under PD disaggregation:
#   A) PD only        — Prefill + Decode (full MLP on Decode)
#   B) PD + AFD       — Prefill + Decode-Attn + FFN (StepMesh)
# Same prompts via sglang.benchmark.serving; report TTFT / ITL / throughput.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

# Keep CUDA env stable across every launch subshell (avoid Error 802 / "no accelerator").
AFD_ENV_SH=/root/.cuda/afd_env.sh
cuda_preflight() {
  local tries=${1:-30}
  for i in $(seq 1 "$tries"); do
    if nvidia-smi -L >/dev/null 2>&1 \
      && python3 -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count()>0" 2>/dev/null; then
      echo "CUDA preflight ok (try $i)"
      return 0
    fi
    echo "CUDA not ready (try $i/$tries), cooling..."
    sleep 2
  done
  echo "ERROR: CUDA still unavailable after ${tries} tries" >&2
  nvidia-smi 2>&1 | tail -20 >&2 || true
  python3 -c "import torch; print(torch.cuda.is_available())" 2>&1 >&2 || true
  return 1
}

python3 -c "import sglang_router" 2>/dev/null || \
  pip install -q "sglang-router==0.3.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT_DIR="${OUT_DIR:-/tmp/afd_pd_compare}"
RNIC="${RNIC:-eth1}"
IB_DEVICE="${IB_DEVICE:-mlx5_1}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
FFN_GPU="${FFN_GPU:-2}"

# Base ports (shifted per config to avoid stale binds)
BASE_PORT="${BASE_PORT:-32800}"
# Decode batching: raise concurrency so PD decode actually batches (>1).
NUM_PROMPTS="${NUM_PROMPTS:-32}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-256}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
DECODE_MAX_BS="${DECODE_MAX_BS:-16}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
MODES="${MODES:-pd,pd_afd}"  # comma list

mkdir -p "$OUT_DIR"
[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model missing $MODEL" >&2; exit 1; }

SCHEDULER_IP=$(ip -o -4 addr show "$RNIC" | awk '{print $4}' | cut -d/ -f1 | head -1)
[[ -n "$SCHEDULER_IP" ]] || { echo "ERROR: no IPv4 on $RNIC" >&2; exit 1; }

MODEL_ARGS=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.82
  --max-running-requests "$DECODE_MAX_BS" --context-length 4096
  --max-total-tokens "${MAX_TOTAL_TOKENS:-100000}"
  --cuda-graph-backend-prefill disabled
)
# PD-only decode: full CUDA graph (local MLP).
PD_DECODE_CG_ARGS=(
  --cuda-graph-backend-decode full
  --cuda-graph-max-bs-decode "$DECODE_MAX_BS"
)
# AFD decode Attn: breakable CG (remote FFN is a graph break).
# Override with AFD_DECODE_CG_BACKEND=disabled for correctness probes.
AFD_DECODE_CG_ARGS=(
  --cuda-graph-backend-decode "${AFD_DECODE_CG_BACKEND:-breakable}"
  --cuda-graph-max-bs-decode "$DECODE_MAX_BS"
)
PD_ARGS=(
  --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device "$IB_DEVICE"
)

kill_cluster() {
  local tag=$1
  if [[ -f "$OUT_DIR/$tag.pids" ]]; then
    while read -r p; do
      [[ -n "$p" ]] || continue
      kill -TERM "$p" 2>/dev/null || true
      pkill -TERM -P "$p" 2>/dev/null || true
    done <"$OUT_DIR/$tag.pids"
    sleep 2
    while read -r p; do
      [[ -n "$p" ]] || continue
      kill -KILL "$p" 2>/dev/null || true
      pkill -KILL -P "$p" 2>/dev/null || true
    done <"$OUT_DIR/$tag.pids"
    rm -f "$OUT_DIR/$tag.pids"
  fi
  # Quiet port cleanup — never dump PIDs to stdout (breaks logs / confuses Cursor).
  for port in "${PREFILL_PORT:-}" "${DECODE_PORT:-}" "${FFN_PORT:-}" "${LB_PORT:-}" "${PS_PORT:-}" "${BOOTSTRAP_PORT:-}"; do
    [[ -n "$port" && "$port" != "0" ]] || continue
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
  done
  # Hard GPU kills can leave driver Error 802 briefly — wait for CUDA again.
  sleep 3
  cuda_preflight 20 || true
}

wait_http() {
  local url=$1 name=$2 pid=${3:-} timeout=${4:-300}
  for i in $(seq 1 "$timeout"); do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$name died" >&2
      return 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$name healthy after ${i}s"
      return 0
    fi
    sleep 2
  done
  echo "$name timeout" >&2
  return 1
}

start_pd_only() {
  local tag=pd
  PREFILL_PORT=$((BASE_PORT + 0))
  DECODE_PORT=$((BASE_PORT + 1))
  LB_PORT=$((BASE_PORT + 10))
  BOOTSTRAP_PORT=$((BASE_PORT + 50))
  FFN_PORT=0
  PS_PORT=0
  local log="$OUT_DIR/$tag"
  mkdir -p "$log"
  : >"$OUT_DIR/$tag.pids"

  echo "=== Bring-up PD-only Prefill=$PREFILL_GPU Decode=$DECODE_GPU ==="
  cuda_preflight 30 || return 1
  (
    # shellcheck disable=SC1090
    source "$AFD_ENV_SH"
    export SGLANG_AFD_MODE=null
    unset DMLC_ROLE || true
    export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"
    python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
      --port "$PREFILL_PORT" \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
      "${PD_ARGS[@]}"
  ) >"$log/prefill.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local pref_pid=$!

  (
    # shellcheck disable=SC1090
    source "$AFD_ENV_SH"
    export SGLANG_AFD_MODE=null
    unset DMLC_ROLE || true
    export CUDA_VISIBLE_DEVICES="$DECODE_GPU"
    python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
      "${PD_DECODE_CG_ARGS[@]}" \
      --port "$DECODE_PORT" \
      --disaggregation-mode decode \
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
      "${PD_ARGS[@]}"
  ) >"$log/decode.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local dec_pid=$!

  wait_http "http://127.0.0.1:$PREFILL_PORT/health" Prefill "$pref_pid" || {
    tail -80 "$log/prefill.log" >&2; return 1; }
  wait_http "http://127.0.0.1:$DECODE_PORT/health" Decode "$dec_pid" 400 || {
    tail -80 "$log/decode.log" >&2; return 1; }

  python3 -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --host 127.0.0.1 --port "$LB_PORT" \
    >"$log/router.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  wait_http "http://127.0.0.1:$LB_PORT/health" Router || {
    tail -40 "$log/router.log" >&2; return 1; }
  echo "$LB_PORT" >"$OUT_DIR/$tag.lb_port"
}

start_pd_afd() {
  local tag=pd_afd
  PREFILL_PORT=$((BASE_PORT + 100))
  DECODE_PORT=$((BASE_PORT + 101))
  FFN_PORT=$((BASE_PORT + 102))
  LB_PORT=$((BASE_PORT + 110))
  BOOTSTRAP_PORT=$((BASE_PORT + 150))
  PS_PORT=$((BASE_PORT + 180))
  local log="$OUT_DIR/$tag"
  mkdir -p "$log"
  : >"$OUT_DIR/$tag.pids"

  # stepmesh (default) | cuda_ipc | nvlink
  export SGLANG_AFD_TRANSPORT="${SGLANG_AFD_TRANSPORT:-stepmesh}"
  if [[ "$SGLANG_AFD_TRANSPORT" == "nvlink" || "$SGLANG_AFD_TRANSPORT" == "cuda-ipc" ]]; then
    export SGLANG_AFD_TRANSPORT=cuda_ipc
  fi
  export SGLANG_AFD_MODULE_STUBS=1
  export SGLANG_AFD_ROUTING_SCHEME=a
  export SGLANG_AFD_NUM_MB="${SGLANG_AFD_NUM_MB:-3}"
  export SGLANG_AFD_MAX_NUM_TOKEN="${SGLANG_AFD_MAX_NUM_TOKEN:-$DECODE_MAX_BS}"
  if (( SGLANG_AFD_MAX_NUM_TOKEN < DECODE_MAX_BS )); then
    export SGLANG_AFD_MAX_NUM_TOKEN="$DECODE_MAX_BS"
  fi
  export SGLANG_AFD_PIPELINE="${SGLANG_AFD_PIPELINE:-1}"
  # P7 / StepMesh stages: staggered multi-mb layer pipeline.
  export SGLANG_AFD_STEPMESH_STAGES="${AFD_STEPMESH_STAGES:-${SGLANG_AFD_STEPMESH_STAGES:-0}}"
  export SGLANG_AFD_LAYER_PIPELINE="${AFD_LAYER_PIPELINE:-${SGLANG_AFD_LAYER_PIPELINE:-0}}"
  # P8: in-graph wait_flag inside full decode CG (mutually exclusive w/ layer pipe).
  export SGLANG_AFD_IN_GRAPH_WAIT="${AFD_IN_GRAPH_WAIT:-${SGLANG_AFD_IN_GRAPH_WAIT:-0}}"
  # True Attn∥FFN overlap (breakable CG + dual-mb + deferred wait_flag).
  export SGLANG_AFD_TRUE_OVERLAP="${AFD_TRUE_OVERLAP:-${SGLANG_AFD_TRUE_OVERLAP:-0}}"
  if [[ "${SGLANG_AFD_TRUE_OVERLAP}" == "1" || "${SGLANG_AFD_TRUE_OVERLAP}" == "true" ]]; then
    export SGLANG_AFD_LAYER_PIPELINE=1
    export SGLANG_AFD_PIPELINE=0
    export SGLANG_AFD_IN_GRAPH_WAIT=0
    # Dual-mb is the Lite sweet spot (do not inherit default NUM_MB=3).
    if [[ -z "${AFD_NUM_MB:-}" && -z "${SGLANG_AFD_NUM_MB_FORCE:-}" ]]; then
      export SGLANG_AFD_NUM_MB=2
    else
      export SGLANG_AFD_NUM_MB="${AFD_NUM_MB:-${SGLANG_AFD_NUM_MB_FORCE:-2}}"
    fi
    if [[ "${SGLANG_AFD_STEPMESH_STAGES}" == "0" || -z "${SGLANG_AFD_STEPMESH_STAGES}" ]]; then
      export SGLANG_AFD_STEPMESH_STAGES="${SGLANG_AFD_NUM_MB}"
    fi
  fi
  export SGLANG_AFD_REMOTE_FROM_LAYER="${SGLANG_AFD_REMOTE_FROM_LAYER:-0}"
  export SGLANG_AFD_REMOTE_MOE_ONLY="${SGLANG_AFD_REMOTE_MOE_ONLY:-0}"
  export SGLANG_AFD_LAYER_MERGE_K="${SGLANG_AFD_LAYER_MERGE_K:-1}"
  export SGLANG_AFD_FFN_GATHER_US="${SGLANG_AFD_FFN_GATHER_US:-0}"
  export SGLANG_AFD_FFN_GATHER_MAX="${SGLANG_AFD_FFN_GATHER_MAX:-4}"
  # Layer-merge forces from=0, breakable; multi-mb gather; FFN-local interior KV.
  if [[ "${SGLANG_AFD_LAYER_MERGE_K}" != "1" && "${SGLANG_AFD_LAYER_MERGE_K}" != "0" ]]; then
    export SGLANG_AFD_REMOTE_FROM_LAYER=0
    export SGLANG_AFD_IN_GRAPH_WAIT=0
    export SGLANG_AFD_LAYER_PIPELINE=0
    export SGLANG_AFD_TRUE_OVERLAP=0
    export SGLANG_AFD_PIPELINE=0
    export SGLANG_AFD_NUM_MB="${SGLANG_AFD_NUM_MB:-2}"
    export SGLANG_AFD_STEPMESH_STAGES=0
    # Merge interiors are eager; FCG works for group-start MLP (sshapes match).
    export SGLANG_AFD_FFN_CUDA_GRAPH="${SGLANG_AFD_FFN_CUDA_GRAPH:-1}"
  fi
  if [[ "${SGLANG_AFD_STEPMESH_STAGES}" != "0" && "${SGLANG_AFD_STEPMESH_STAGES}" != "" ]]; then
    export SGLANG_AFD_LAYER_PIPELINE=1
    export SGLANG_AFD_PIPELINE=0
    export SGLANG_AFD_IN_GRAPH_WAIT=0
    export SGLANG_AFD_NUM_MB="${SGLANG_AFD_STEPMESH_STAGES}"
  fi
  if [[ "${SGLANG_AFD_IN_GRAPH_WAIT}" == "1" || "${SGLANG_AFD_IN_GRAPH_WAIT}" == "true" ]]; then
    export SGLANG_AFD_LAYER_PIPELINE=0
    export SGLANG_AFD_PIPELINE=0
    export SGLANG_AFD_NUM_MB=1
    export SGLANG_AFD_STEPMESH_STAGES=0
    export SGLANG_AFD_TRUE_OVERLAP=0
  fi
  if [[ "${SGLANG_AFD_LAYER_PIPELINE}" == "1" || "${SGLANG_AFD_LAYER_PIPELINE}" == "true" ]]; then
    export SGLANG_AFD_NUM_MB="${SGLANG_AFD_NUM_MB:-2}"
    export SGLANG_AFD_PIPELINE=0
    export SGLANG_AFD_IN_GRAPH_WAIT=0
    if [[ "${SGLANG_AFD_STEPMESH_STAGES}" == "0" || -z "${SGLANG_AFD_STEPMESH_STAGES}" ]]; then
      export SGLANG_AFD_STEPMESH_STAGES="${SGLANG_AFD_NUM_MB}"
    fi
  fi
  export TORCHDYNAMO_DISABLE=1
  export TORCH_COMPILE_DISABLE=1
  export SGLANG_AFD_FFN_CUDA_GRAPH="${SGLANG_AFD_FFN_CUDA_GRAPH:-1}"

  local use_stepmesh=0
  if [[ "$SGLANG_AFD_TRANSPORT" == "stepmesh" ]]; then
    use_stepmesh=1
    export SGLANG_AFD_USE_WAIT_FLAG="${SGLANG_AFD_USE_WAIT_FLAG:-1}"
    export DMLC_NUM_WORKER=1 DMLC_NUM_SERVER=1
    export DMLC_PS_ROOT_URI="$SCHEDULER_IP"
    export DMLC_PS_ROOT_PORT="$PS_PORT"
    export DMLC_ENABLE_RDMA=ibverbs
    export BYTEPS_ENABLE_IPC="${BYTEPS_ENABLE_IPC:-0}"
    export DMLC_INTERFACE=auto
    export DMLC_NODE_HOST="$SCHEDULER_IP"
    export STEPMESH_SPLIT_QP_LAG=0 STEPMESH_BIND_CPU_CORE=0
    export PS_VERBOSE=0
    fuser -k "$PS_PORT/tcp" >/dev/null 2>&1 || true
    sleep 1
  else
    # cuda_ipc: mailbox async is the overlap path; wait_flag optional.
    if [[ "${SGLANG_AFD_TRUE_OVERLAP}" == "1" || "${SGLANG_AFD_TRUE_OVERLAP}" == "true" ]]; then
      export SGLANG_AFD_USE_WAIT_FLAG="${SGLANG_AFD_USE_WAIT_FLAG:-0}"
    else
      export SGLANG_AFD_USE_WAIT_FLAG="${SGLANG_AFD_USE_WAIT_FLAG:-0}"
    fi
    export SGLANG_AFD_IPC_ENDPOINT="${SGLANG_AFD_IPC_ENDPOINT:-$OUT_DIR/afd_cuda_ipc.sock}"
    rm -f "$SGLANG_AFD_IPC_ENDPOINT" "$SGLANG_AFD_IPC_ENDPOINT.merge_kv" 2>/dev/null || true
  fi

  echo "AFD transport=$SGLANG_AFD_TRANSPORT max_token=$SGLANG_AFD_MAX_NUM_TOKEN mb=$SGLANG_AFD_NUM_MB stages=$SGLANG_AFD_STEPMESH_STAGES pipeline=$SGLANG_AFD_PIPELINE layer_pipeline=$SGLANG_AFD_LAYER_PIPELINE true_overlap=$SGLANG_AFD_TRUE_OVERLAP in_graph_wait=$SGLANG_AFD_IN_GRAPH_WAIT wait_flag=$SGLANG_AFD_USE_WAIT_FLAG remote_from=$SGLANG_AFD_REMOTE_FROM_LAYER merge_k=$SGLANG_AFD_LAYER_MERGE_K decode_bs=$DECODE_MAX_BS conc=$MAX_CONCURRENCY endpoint=${SGLANG_AFD_IPC_ENDPOINT:-n/a}"
  echo "=== Bring-up PD+AFD Prefill=$PREFILL_GPU Decode=$DECODE_GPU FFN=$FFN_GPU (transport=$SGLANG_AFD_TRANSPORT) ==="
  cuda_preflight 30 || return 1

  if (( use_stepmesh )); then
    (
      # shellcheck disable=SC1090
      source "$AFD_ENV_SH"
      export DMLC_ROLE=scheduler
      python3 - <<'PY'
import os, time
import fserver_lib as f
f.init()
print("SCHEDULER_READY", flush=True)
while True:
    time.sleep(3600)
PY
    ) >"$log/scheduler.log" 2>&1 &
    echo $! >>"$OUT_DIR/$tag.pids"
    local sched_pid=$!
    for i in $(seq 1 30); do
      kill -0 "$sched_pid" 2>/dev/null || { cat "$log/scheduler.log" >&2; return 1; }
      rg -q "SCHEDULER_READY|Bind to" "$log/scheduler.log" 2>/dev/null && break
      sleep 0.5
    done
    echo "Scheduler up"
  fi

  (
    # shellcheck disable=SC1090
    source "$AFD_ENV_SH"
    export SGLANG_AFD_MODE=null
    unset DMLC_ROLE || true
    export CUDA_VISIBLE_DEVICES="$PREFILL_GPU"
    python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
      --port "$PREFILL_PORT" \
      --disaggregation-mode prefill \
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
      "${PD_ARGS[@]}"
  ) >"$log/prefill.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local pref_pid=$!

  (
    # shellcheck disable=SC1090
    source "$AFD_ENV_SH"
    if (( use_stepmesh )); then
      export DMLC_ROLE=server
      export STEPMESH_GPU=0
    else
      unset DMLC_ROLE || true
    fi
    export SGLANG_AFD_MODE=ffn SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
    export SGLANG_AFD_TRANSPORT
    export SGLANG_AFD_FFN_CUDA_GRAPH
    export SGLANG_AFD_IPC_ENDPOINT
    export SGLANG_AFD_LAYER_MERGE_K
    export SGLANG_AFD_REMOTE_FROM_LAYER
    export CUDA_VISIBLE_DEVICES="$FFN_GPU"
    python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
      --port "$FFN_PORT" --disaggregation-mode null --skip-server-warmup
  ) >"$log/ffn.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local ffn_pid=$!

  wait_http "http://127.0.0.1:$PREFILL_PORT/health" Prefill "$pref_pid" || {
    tail -80 "$log/prefill.log" >&2; return 1; }
  for i in $(seq 1 240); do
    kill -0 "$ffn_pid" 2>/dev/null || { tail -80 "$log/ffn.log" >&2; return 1; }
    # cuda_ipc FFN blocks in register_buffers until Attn connects — wait for weight load.
    if [[ "$SGLANG_AFD_TRANSPORT" == "cuda_ipc" ]]; then
      rg -q "Load weight end|AFD cuda_ipc FFN waiting" "$log/ffn.log" 2>/dev/null && break
    else
      rg -q "AFD auto-init done|AFD FFN CUDA graph warm-up done" "$log/ffn.log" 2>/dev/null && break
    fi
    [[ $i -eq 240 ]] && { tail -80 "$log/ffn.log" >&2; return 1; }
    sleep 2
  done
  echo "FFN ready (weights + AFD init progressing)"

  (
    # shellcheck disable=SC1090
    source "$AFD_ENV_SH"
    if (( use_stepmesh )); then
      export DMLC_ROLE=worker DMLC_NODE_RANK=0
      export STEPMESH_GPU=0
    else
      unset DMLC_ROLE || true
    fi
    export SGLANG_AFD_MODE=attn
    export SGLANG_AFD_TRANSPORT
    export SGLANG_AFD_IPC_ENDPOINT
    export SGLANG_AFD_LAYER_MERGE_K
    export SGLANG_AFD_REMOTE_FROM_LAYER
    export SGLANG_AFD_IN_GRAPH_WAIT
    export CUDA_VISIBLE_DEVICES="$DECODE_GPU"
    python3 -m sglang.launch_server "${MODEL_ARGS[@]}" \
      "${AFD_DECODE_CG_ARGS[@]}" \
      --port "$DECODE_PORT" \
      --disaggregation-mode decode \
      --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
      "${PD_ARGS[@]}"
  ) >"$log/decode.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  local dec_pid=$!

  # CG capture can take several minutes; cuda_ipc handshake completes during Attn init.
  wait_http "http://127.0.0.1:$DECODE_PORT/health" Decode "$dec_pid" 400 || {
    tail -100 "$log/decode.log" >&2; return 1; }
  if [[ "$SGLANG_AFD_TRANSPORT" == "cuda_ipc" ]]; then
    rg -q "cuda_ipc buffers ready|AFD auto-init done" "$log/ffn.log" 2>/dev/null || \
      echo "WARN: FFN cuda_ipc ready marker missing" >&2
  fi

  python3 -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:$PREFILL_PORT" \
    --decode "http://127.0.0.1:$DECODE_PORT" \
    --host 127.0.0.1 --port "$LB_PORT" \
    >"$log/router.log" 2>&1 &
  echo $! >>"$OUT_DIR/$tag.pids"
  wait_http "http://127.0.0.1:$LB_PORT/health" Router || {
    tail -40 "$log/router.log" >&2; return 1; }
  echo "$LB_PORT" >"$OUT_DIR/$tag.lb_port"
}

run_bench() {
  local tag=$1
  local lb_port
  lb_port=$(cat "$OUT_DIR/$tag.lb_port")
  local out_json="$OUT_DIR/${tag}_bench.json"
  echo "=== Bench $tag → http://127.0.0.1:$lb_port (n=$NUM_PROMPTS) ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$lb_port" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --pd-separated \
    --output-file "$out_json" \
    --disable-tqdm \
    2>&1 | tee "$OUT_DIR/${tag}_bench.log"
}

summarize() {
  python3 - <<'PY' "$OUT_DIR" "$MODES"
import json, glob, os, sys
out_dir, modes = sys.argv[1], sys.argv[2].split(",")
rows = []
keys = [
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "completed",
]

def load(tag):
    path = os.path.join(out_dir, f"{tag}_bench.json")
    if not os.path.isfile(path):
        cands = sorted(glob.glob(os.path.join(out_dir, f"{tag}_bench*.json")))
        if not cands:
            return None
        path = cands[-1]
    text = open(path).read().strip()
    if not text:
        return None
    # serving may append multiple JSON objects into one file
    dec = json.JSONDecoder()
    objs = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, end = dec.raw_decode(text, i)
        objs.append(obj)
        i = end
    return objs[-1] if objs else None

print("\n========== PD vs PD+AFD (DeepSeek-V2-Lite) ==========")
print(f"{'metric':<28}" + "".join(f"{m:>14}" for m in modes))
data = {m: load(m.strip()) for m in modes}
for k in keys:
    vals = []
    for m in modes:
        d = data.get(m.strip())
        if not d or k not in d:
            vals.append("—")
        else:
            v = d[k]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
    print(f"{k:<28}" + "".join(f"{v:>14}" for v in vals))

# Relative deltas if both present
a, b = data.get("pd"), data.get("pd_afd")
if a and b:
    print("\n--- pd_afd relative to pd (positive = slower/higher) ---")
    for k in ("mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
              "mean_e2e_latency_ms", "output_throughput"):
        if k in a and k in b and a[k]:
            rel = (b[k] - a[k]) / a[k] * 100.0
            print(f"  {k}: {rel:+.1f}%  (pd={a[k]:.2f} → pd_afd={b[k]:.2f})")
print("AFD_PD_COMPARE_OK")
PY
}

# --- main ---
trap 'kill_cluster pd; kill_cluster pd_afd' EXIT

cuda_preflight 30 || exit 1

# free leftover servers (quiet)
ps -eo pid,cmd | awk '/launch_server|sglang_router|sglang::/ && !/awk/ {print $1}' \
  | xargs -r kill -9 2>/dev/null || true
sleep 3
cuda_preflight 30 || exit 1

IFS=',' read -ra MODE_ARR <<<"$MODES"
for mode in "${MODE_ARR[@]}"; do
  mode=$(echo "$mode" | tr -d ' ')
  case "$mode" in
    pd)
      kill_cluster pd
      start_pd_only
      run_bench pd
      kill_cluster pd
      ;;
    pd_afd)
      kill_cluster pd_afd
      start_pd_afd
      run_bench pd_afd
      kill_cluster pd_afd
      ;;
    *)
      echo "Unknown mode $mode" >&2; exit 1
      ;;
  esac
done

summarize
