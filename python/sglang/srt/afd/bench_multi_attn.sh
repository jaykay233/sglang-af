#!/usr/bin/env bash
# A/B: 1A1F vs 2A1F (2 GPUs) vs 2A1F (same GPU dual Attn processes)
set -euo pipefail
source /root/.cuda/afd_env.sh

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT="${OUT_DIR:-/tmp/afd_multi_attn_exp}"
IB="${IB_DEVICE:-mlx5_1}"
rm -rf "$OUT"
mkdir -p "$OUT"

NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
IN_LEN="${RANDOM_INPUT_LEN:-128}"
OUT_LEN="${RANDOM_OUTPUT_LEN:-32}"

common_model=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.82
  --max-running-requests 8 --context-length 4096 --max-total-tokens 50000
  --cuda-graph-backend-prefill disabled
)

kill_ports() {
  local p
  for p in "$@"; do fuser -k "${p}/tcp" >/dev/null 2>&1 || true; done
  sleep 2
}

wait_http() {
  local url=$1 name=$2 pid=$3 timeout=${4:-180}
  local i
  for i in $(seq 1 "$timeout"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name died" >&2
      return 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "$name healthy ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "$name timeout" >&2
  return 1
}

run_bench() {
  local tag=$1 lb=$2
  echo "=== Bench $tag lb=$lb ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$lb" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$IN_LEN" \
    --random-output-len "$OUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests 2 \
    --pd-separated \
    --output-file "$OUT/${tag}_bench.json" \
    --disable-tqdm \
    2>&1 | tee "$OUT/${tag}_bench.log" | tail -35
}

summarize() {
  python3 - <<'PY' "$OUT"
import json, glob, os, sys
out = sys.argv[1]
keys = ["median_tpot_ms","mean_tpot_ms","p99_tpot_ms","median_ttft_ms",
        "output_throughput","request_throughput","median_e2e_latency_ms","completed"]
tags = ["a1f1", "a2f1_2gpu", "a2f1_1gpu"]
print("\n========== Multi-Attn experiment (DeepSeek-V2-Lite) ==========")
print(f"{'metric':<28}" + "".join(f"{t:>14}" for t in tags))
data = {}
for t in tags:
    path = os.path.join(out, f"{t}_bench.json")
    if not os.path.isfile(path):
        data[t] = None
        continue
    text = open(path).read().strip()
    dec = json.JSONDecoder(); objs=[]; i=0
    while i < len(text):
        while i < len(text) and text[i].isspace(): i += 1
        if i >= len(text): break
        o, e = dec.raw_decode(text, i); objs.append(o); i = e
    data[t] = objs[-1] if objs else None
for k in keys:
    row = f"{k:<28}"
    for t in tags:
        d = data.get(t)
        if not d or k not in d:
            row += f"{'—':>14}"
        else:
            v = d[k]
            row += f"{v:>14.2f}" if isinstance(v, float) else f"{v:>14}"
    print(row)
print("MULTI_ATTN_EXP_OK")
PY
}

start_prefill() {
  local gpu=$1 port=$2 boot=$3 log=$4
  (
    source /root/.cuda/afd_env.sh
    export SGLANG_AFD_MODE=null CUDA_VISIBLE_DEVICES="$gpu"
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --port "$port" --disaggregation-mode prefill \
      --disaggregation-bootstrap-port "$boot" \
      --disaggregation-transfer-backend mooncake --disaggregation-ib-device "$IB"
  ) >"$log" 2>&1 &
  echo $!
}

start_ffn_pool() {
  local gpu=$1 port=$2 rank=$3 num_attn=$4 num_ffn=$5 ep=$6 log=$7
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN="$num_attn" SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
    export SGLANG_AFD_POOL_LOCAL_RANK="$rank" SGLANG_AFD_POOL_ENDPOINT_DIR="$ep"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=8
    export SGLANG_AFD_MODULE_STUBS=1 SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN=8
    export SGLANG_AFD_TRUE_OVERLAP=0 SGLANG_AFD_LAYER_PIPELINE=0
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --port "$port" --disaggregation-mode null --skip-server-warmup
  ) >"$log" 2>&1 &
  echo $!
}

start_attn_pool() {
  local gpu=$1 port=$2 rank=$3 num_attn=$4 num_ffn=$5 ep=$6 boot=$7 log=$8
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_AFD_MODE=attn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN="$num_attn" SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
    export SGLANG_AFD_POOL_LOCAL_RANK="$rank" SGLANG_AFD_POOL_ENDPOINT_DIR="$ep"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=8
    export SGLANG_AFD_POOL_ROUTE=least_inflight
    export SGLANG_AFD_MODULE_STUBS=1
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN=8
    export SGLANG_AFD_TRUE_OVERLAP=0 SGLANG_AFD_LAYER_PIPELINE=0 SGLANG_AFD_IN_GRAPH_WAIT=0
    export SGLANG_AFD_REMOTE_FROM_LAYER=0 SGLANG_AFD_LAYER_MERGE_K=1
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --cuda-graph-backend-decode breakable --cuda-graph-max-bs-decode 8 \
      --port "$port" --disaggregation-mode decode \
      --disaggregation-bootstrap-port "$boot" \
      --disaggregation-transfer-backend mooncake --disaggregation-ib-device "$IB"
  ) >"$log" 2>&1 &
  echo $!
}

# ---------- config 1: 1A1F ----------
run_a1f1() {
  local tag=a1f1
  local base=34000
  local pref=$((base+0)) dec=$((base+1)) ffn=$((base+2)) lb=$((base+10)) boot=$((base+50))
  local dir="$OUT/$tag"
  local ep="$dir/socks"
  mkdir -p "$dir" "$ep"
  kill_ports "$pref" "$dec" "$ffn" "$lb" "$boot"
  : >"$dir/pids"
  echo "=== Bring-up $tag: P4 A5 F6 ==="
  local p_pid f_pid a_pid
  p_pid=$(start_prefill 4 "$pref" "$boot" "$dir/prefill.log"); echo "$p_pid" >>"$dir/pids"
  f_pid=$(start_ffn_pool 6 "$ffn" 0 1 1 "$ep" "$dir/ffn.log"); echo "$f_pid" >>"$dir/pids"
  wait_http "http://127.0.0.1:$pref/health" Prefill "$p_pid" 180
  for i in $(seq 1 90); do
    rg -q "Load weight end|AFD cuda_ipc FFN waiting|AfPool FFN" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  a_pid=$(start_attn_pool 5 "$dec" 0 1 1 "$ep" "$boot" "$dir/attn.log"); echo "$a_pid" >>"$dir/pids"
  wait_http "http://127.0.0.1:$dec/health" Decode "$a_pid" 240
  python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:$pref" --decode "http://127.0.0.1:$dec" \
    --host 127.0.0.1 --port "$lb" >"$dir/router.log" 2>&1 &
  echo $! >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router $! 60
  run_bench "$tag" "$lb"
  while read -r p; do kill -TERM "$p" 2>/dev/null || true; pkill -TERM -P "$p" 2>/dev/null || true; done <"$dir/pids"
  sleep 3
  while read -r p; do kill -KILL "$p" 2>/dev/null || true; pkill -KILL -P "$p" 2>/dev/null || true; done <"$dir/pids"
  kill_ports "$pref" "$dec" "$ffn" "$lb" "$boot"
}

# ---------- config 2: 2A1F on 2 GPUs ----------
run_a2f1_2gpu() {
  local tag=a2f1_2gpu
  local base=34100
  local pref=$((base+0)) d0=$((base+1)) d1=$((base+2)) ffn=$((base+3)) lb=$((base+10)) boot=$((base+50))
  local dir="$OUT/$tag"
  local ep="$dir/socks"
  mkdir -p "$dir" "$ep"
  kill_ports "$pref" "$d0" "$d1" "$ffn" "$lb" "$boot"
  : >"$dir/pids"
  echo "=== Bring-up $tag: P4 A5 A6 F7 ==="
  local p_pid f_pid a0 a1
  p_pid=$(start_prefill 4 "$pref" "$boot" "$dir/prefill.log"); echo "$p_pid" >>"$dir/pids"
  f_pid=$(start_ffn_pool 7 "$ffn" 0 2 1 "$ep" "$dir/ffn.log"); echo "$f_pid" >>"$dir/pids"
  wait_http "http://127.0.0.1:$pref/health" Prefill "$p_pid" 180
  for i in $(seq 1 90); do
    rg -q "Load weight end|AFD cuda_ipc FFN waiting|AfPool" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  a0=$(start_attn_pool 5 "$d0" 0 2 1 "$ep" "$boot" "$dir/attn0.log"); echo "$a0" >>"$dir/pids"
  a1=$(start_attn_pool 6 "$d1" 1 2 1 "$ep" "$boot" "$dir/attn1.log"); echo "$a1" >>"$dir/pids"
  wait_http "http://127.0.0.1:$d0/health" Decode0 "$a0" 300
  wait_http "http://127.0.0.1:$d1/health" Decode1 "$a1" 300
  python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:$pref" \
    --decode "http://127.0.0.1:$d0" --decode "http://127.0.0.1:$d1" \
    --decode-policy round_robin \
    --host 127.0.0.1 --port "$lb" >"$dir/router.log" 2>&1 &
  echo $! >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router $! 60
  run_bench "$tag" "$lb"
  while read -r p; do kill -TERM "$p" 2>/dev/null || true; pkill -TERM -P "$p" 2>/dev/null || true; done <"$dir/pids"
  sleep 3
  while read -r p; do kill -KILL "$p" 2>/dev/null || true; pkill -KILL -P "$p" 2>/dev/null || true; done <"$dir/pids"
  kill_ports "$pref" "$d0" "$d1" "$ffn" "$lb" "$boot"
}

# ---------- config 3: 2A on SAME GPU ----------
run_a2f1_1gpu() {
  local tag=a2f1_1gpu
  local base=34200
  local pref=$((base+0)) d0=$((base+1)) d1=$((base+2)) ffn=$((base+3)) lb=$((base+10)) boot=$((base+50))
  local dir="$OUT/$tag"
  local ep="$dir/socks"
  mkdir -p "$dir" "$ep"
  kill_ports "$pref" "$d0" "$d1" "$ffn" "$lb" "$boot"
  : >"$dir/pids"
  echo "=== Bring-up $tag: P4  A5+A5(same)  F6 ==="
  local p_pid f_pid a0 a1
  p_pid=$(start_prefill 4 "$pref" "$boot" "$dir/prefill.log"); echo "$p_pid" >>"$dir/pids"
  f_pid=$(start_ffn_pool 6 "$ffn" 0 2 1 "$ep" "$dir/ffn.log"); echo "$f_pid" >>"$dir/pids"
  wait_http "http://127.0.0.1:$pref/health" Prefill "$p_pid" 180
  for i in $(seq 1 90); do
    rg -q "Load weight end|AFD cuda_ipc FFN waiting|AfPool" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  # both Attn on GPU 5
  a0=$(start_attn_pool 5 "$d0" 0 2 1 "$ep" "$boot" "$dir/attn0.log"); echo "$a0" >>"$dir/pids"
  a1=$(start_attn_pool 5 "$d1" 1 2 1 "$ep" "$boot" "$dir/attn1.log"); echo "$a1" >>"$dir/pids"
  wait_http "http://127.0.0.1:$d0/health" Decode0 "$a0" 300
  wait_http "http://127.0.0.1:$d1/health" Decode1 "$a1" 300
  python3 -m sglang_router.launch_router --pd-disaggregation --mini-lb \
    --prefill "http://127.0.0.1:$pref" \
    --decode "http://127.0.0.1:$d0" --decode "http://127.0.0.1:$d1" \
    --decode-policy round_robin \
    --host 127.0.0.1 --port "$lb" >"$dir/router.log" 2>&1 &
  echo $! >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router $! 60
  run_bench "$tag" "$lb"
  while read -r p; do kill -TERM "$p" 2>/dev/null || true; pkill -TERM -P "$p" 2>/dev/null || true; done <"$dir/pids"
  sleep 3
  while read -r p; do kill -KILL "$p" 2>/dev/null || true; pkill -KILL -P "$p" 2>/dev/null || true; done <"$dir/pids"
  kill_ports "$pref" "$d0" "$d1" "$ffn" "$lb" "$boot"
}

trap 'pkill -f "sglang.launch_server.*342|sglang.launch_server.*341|sglang.launch_server.*340" 2>/dev/null || true' EXIT

run_a1f1
run_a2f1_2gpu
run_a2f1_1gpu
summarize
