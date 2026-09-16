#!/usr/bin/env bash
# Attn self-prefill (no Prefill process): PD=null, in=1 decode-dominated.
# Compare 1A1F vs 2A1F (2 GPU) vs 2A same-GPU.
set -euo pipefail
source /root/.cuda/afd_env.sh

MODEL="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
OUT="${OUT_DIR:-/tmp/afd_attn_prefill_exp}"
mkdir -p "$OUT"

NUM_PROMPTS="${NUM_PROMPTS:-32}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
IN_LEN="${RANDOM_INPUT_LEN:-1}"
OUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
MAX_RUNNING="${MAX_RUNNING_REQUESTS:-8}"
POOL_INFLIGHT="${SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN:-8}"

common_model=(
  --model-path "$MODEL" --trust-remote-code --host 127.0.0.1
  --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.82
  --max-running-requests "$MAX_RUNNING" --context-length 4096 --max-total-tokens 50000
  --cuda-graph-backend-prefill disabled
)

kill_ports() {
  local p
  for p in "$@"; do fuser -k "${p}/tcp" >/dev/null 2>&1 || true; done
  sleep 2
}

wait_http() {
  local url=$1 name=$2 pid=$3 timeout=${4:-240}
  local i
  for i in $(seq 1 "$timeout"); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "$name died" >&2; return 1; fi
    if curl -sf "$url" >/dev/null 2>&1; then echo "$name healthy ${i}s"; return 0; fi
    sleep 1
  done
  echo "$name timeout" >&2
  return 1
}

run_bench() {
  local tag=$1 base_url=$2
  echo "=== Bench $tag → $base_url (in=$IN_LEN out=$OUT_LEN conc=$MAX_CONCURRENCY) ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "$base_url" \
    --model "$MODEL" \
    --dataset-name random \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$IN_LEN" \
    --random-output-len "$OUT_LEN" \
    --random-range-ratio 0.0 \
    --request-rate inf \
    --max-concurrency "$MAX_CONCURRENCY" \
    --warmup-requests 2 \
    --output-file "$OUT/${tag}_bench.json" \
    --disable-tqdm \
    2>&1 | tee "$OUT/${tag}_bench.log"
}

stop_pids() {
  local f=$1
  [[ -f "$f" ]] || return 0
  while read -r p; do kill -TERM "$p" 2>/dev/null || true; pkill -TERM -P "$p" 2>/dev/null || true; done <"$f"
  sleep 3
  while read -r p; do kill -KILL "$p" 2>/dev/null || true; pkill -KILL -P "$p" 2>/dev/null || true; done <"$f"
}

# MiniLB requires PD mode; for Attn-self-prefill (PD=null) use a tiny RR reverse proxy.
start_rr_lb() {
  local port=$1
  shift
  local log="${RR_LB_LOG:-/tmp/afd_rr_lb_${port}.log}"
  python3 - "$port" "$@" >"$log" 2>&1 <<'PY' &
import itertools, sys
from aiohttp import ClientSession, TCPConnector, web

port = int(sys.argv[1])
backends = list(sys.argv[2:])
cycle = itertools.cycle(backends)
skip_hop = {"host", "content-length", "transfer-encoding", "connection"}
session: ClientSession | None = None

async def on_start(app: web.Application) -> None:
    global session
    session = ClientSession(connector=TCPConnector(limit=0, force_close=False))

async def on_stop(app: web.Application) -> None:
    global session
    if session is not None:
        await session.close()

async def health(_request: web.Request) -> web.Response:
    return web.Response(text="ok")

async def proxy(request: web.Request) -> web.StreamResponse:
    assert session is not None
    backend = next(cycle)
    url = f"{backend}{request.rel_url}"
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_hop}
    async with session.request(
        request.method, url, headers=headers, data=body
    ) as resp:
        out_headers = {
            k: v for k, v in resp.headers.items() if k.lower() not in skip_hop
        }
        out = web.StreamResponse(status=resp.status, headers=out_headers)
        await out.prepare(request)
        async for chunk in resp.content.iter_chunked(65536):
            await out.write(chunk)
        await out.write_eof()
        return out

app = web.Application()
app.on_startup.append(on_start)
app.on_cleanup.append(on_stop)
app.router.add_get("/health", health)
app.router.add_route("*", "/{path:.*}", proxy)
web.run_app(app, host="127.0.0.1", port=port, print=lambda *_: None)
PY
  echo $!
}

start_ffn() {
  local gpu=$1 port=$2 rank=$3 num_attn=$4 num_ffn=$5 ep=$6 log=$7
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN="$num_attn" SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
    export SGLANG_AFD_POOL_LOCAL_RANK="$rank" SGLANG_AFD_POOL_ENDPOINT_DIR="$ep"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN="$POOL_INFLIGHT"
    export SGLANG_AFD_MODULE_STUBS=1 SGLANG_AFD_RELEASE_UNUSED_PARAMS=1
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN="${AFD_MAX_TOKEN:-256}"
    # Single-thread FFN serve is CUDA-safe; keep CG on unless explicitly disabled.
    export SGLANG_AFD_FFN_CUDA_GRAPH="${SGLANG_AFD_FFN_CUDA_GRAPH:-1}"
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --port "$port" --disaggregation-mode null --skip-server-warmup
  ) >"$log" 2>&1 &
  echo $!
}

start_attn() {
  local gpu=$1 port=$2 rank=$3 num_attn=$4 num_ffn=$5 ep=$6 log=$7
  # Optional 8th arg: force decode CG backend (breakable|disabled). Default: breakable.
  local cg_decode="${8:-breakable}"
  (
    source /root/.cuda/afd_env.sh
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_AFD_MODE=attn SGLANG_AFD_TRANSPORT=cuda_ipc
    export SGLANG_AFD_POOL=1
    export SGLANG_AFD_POOL_NUM_ATTN="$num_attn" SGLANG_AFD_POOL_NUM_FFN="$num_ffn"
    export SGLANG_AFD_POOL_LOCAL_RANK="$rank" SGLANG_AFD_POOL_ENDPOINT_DIR="$ep"
    export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN="$POOL_INFLIGHT"
    export SGLANG_AFD_POOL_ROUTE=least_inflight
    export SGLANG_AFD_MODULE_STUBS=1
    # Prefill on Attn can batch many tokens; keep A2F buffers large enough.
    export SGLANG_AFD_NUM_MB=2 SGLANG_AFD_MAX_NUM_TOKEN="${AFD_MAX_TOKEN:-256}"
    # PD=null → Attn does local prefill+decode; remote FFN via AfPool
    unset DMLC_ROLE || true
    python3 -m sglang.launch_server "${common_model[@]}" \
      --cuda-graph-backend-decode "$cg_decode" --cuda-graph-max-bs-decode "$MAX_RUNNING" \
      --port "$port" --disaggregation-mode null
  ) >"$log" 2>&1 &
  echo $!
}

summarize() {
  python3 - <<'PY' "$OUT"
import json, os, sys
out = sys.argv[1]
tags = ["a1f1", "a2f2_2gpu", "a2f1_2gpu", "a2f1_1gpu"]
keys = ["median_tpot_ms","mean_tpot_ms","p99_tpot_ms","median_ttft_ms",
        "median_itl_ms","output_throughput","request_throughput","completed"]
print("\n========== Attn-self-prefill multi-A (in=1) ==========")
print(f"{'metric':<28}" + "".join(f"{t:>14}" for t in tags))
data={}
for t in tags:
    path=os.path.join(out, f"{t}_bench.json")
    if not os.path.isfile(path):
        data[t]=None; continue
    text=open(path).read().strip(); dec=json.JSONDecoder(); objs=[]; i=0
    while i<len(text):
        while i<len(text) and text[i].isspace(): i+=1
        if i>=len(text): break
        o,e=dec.raw_decode(text,i); objs.append(o); i=e
    data[t]=objs[-1] if objs else None
for k in keys:
    row=f"{k:<28}"
    for t in tags:
        d=data.get(t)
        if not d or k not in d: row += f"{'—':>14}"
        else:
            v=d[k]
            row += f"{v:>14.2f}" if isinstance(v,float) else f"{v:>14}"
    print(row)
print("ATTN_PREFILL_MULTI_A_OK")
PY
}

run_a1f1() {
  local tag=a1f1 base=35000
  local ffn=$((base+0)) a0=$((base+1)) lb=$((base+10))
  local dir="$OUT/$tag"; local ep="$dir/socks"
  mkdir -p "$dir" "$ep"; : >"$dir/pids"
  kill_ports "$ffn" "$a0" "$lb"
  echo "=== $tag: A5 F6 (Attn self-prefill, no P) ==="
  local fp ap
  fp=$(start_ffn 6 "$ffn" 0 1 1 "$ep" "$dir/ffn.log"); echo "$fp" >>"$dir/pids"
  for i in $(seq 1 60); do
    rg -q "Load weight end|AfPool FFN|waiting" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  ap=$(start_attn 5 "$a0" 0 1 1 "$ep" "$dir/attn.log"); echo "$ap" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a0/health" Attn "$ap" 300
  # single worker — hit Attn directly (no router required)
  run_bench "$tag" "http://127.0.0.1:$a0"
  stop_pids "$dir/pids"
  kill_ports "$ffn" "$a0" "$lb"
}

run_a2_2gpu() {
  local tag=a2f1_2gpu base=35100
  local ffn=$((base+0)) a0=$((base+1)) a1=$((base+2)) lb=$((base+10))
  local dir="$OUT/$tag"; local ep="$dir/socks"
  mkdir -p "$dir" "$ep"; : >"$dir/pids"
  kill_ports "$ffn" "$a0" "$a1" "$lb"
  echo "=== $tag: A5 A6 F7 (2 GPU Attn) ==="
  local fp p0 p1
  fp=$(start_ffn 7 "$ffn" 0 2 1 "$ep" "$dir/ffn.log"); echo "$fp" >>"$dir/pids"
  for i in $(seq 1 60); do
    rg -q "Load weight end|AfPool FFN|waiting" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  # Stagger Attn bring-up so CG capture does not hammer FFN concurrently.
  p0=$(start_attn 5 "$a0" 0 2 1 "$ep" "$dir/attn0.log"); echo "$p0" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a0/health" Attn0 "$p0" 360
  p1=$(start_attn 6 "$a1" 1 2 1 "$ep" "$dir/attn1.log"); echo "$p1" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a1/health" Attn1 "$p1" 360
  local rp
  rp=$(start_rr_lb "$lb" "http://127.0.0.1:$a0" "http://127.0.0.1:$a1")
  echo "$rp" >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router "$rp" 60
  run_bench "$tag" "http://127.0.0.1:$lb"
  stop_pids "$dir/pids"
  kill_ports "$ffn" "$a0" "$a1" "$lb"
}

run_a2_1gpu() {
  local tag=a2f1_1gpu base=35200
  local ffn=$((base+0)) a0=$((base+1)) a1=$((base+2)) lb=$((base+10))
  local dir="$OUT/$tag"; local ep="$dir/socks"
  mkdir -p "$dir" "$ep"; : >"$dir/pids"
  kill_ports "$ffn" "$a0" "$a1" "$lb"
  echo "=== $tag: A5+A5(same GPU) F6 ==="
  local fp p0 p1
  fp=$(start_ffn 6 "$ffn" 0 2 1 "$ep" "$dir/ffn.log"); echo "$fp" >>"$dir/pids"
  for i in $(seq 1 60); do
    rg -q "Load weight end|AfPool FFN|waiting" "$dir/ffn.log" 2>/dev/null && break
    sleep 2
  done
  # Same-GPU: still try breakable CG (user request); stagger bring-up to reduce capture clash.
  p0=$(start_attn 5 "$a0" 0 2 1 "$ep" "$dir/attn0.log" breakable); echo "$p0" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a0/health" Attn0 "$p0" 360
  p1=$(start_attn 5 "$a1" 1 2 1 "$ep" "$dir/attn1.log" breakable); echo "$p1" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a1/health" Attn1 "$p1" 360
  local rp
  rp=$(start_rr_lb "$lb" "http://127.0.0.1:$a0" "http://127.0.0.1:$a1")
  echo "$rp" >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router "$rp" 60
  run_bench "$tag" "http://127.0.0.1:$lb"
  stop_pids "$dir/pids"
  kill_ports "$ffn" "$a0" "$a1" "$lb"
}

run_a2_2f() {
  # 2A2F full-mesh AfPool — proves multi-Attn + multi-FFN.
  local tag=a2f2_2gpu base=35300
  local f0=$((base+0)) f1=$((base+1)) a0=$((base+2)) a1=$((base+3)) lb=$((base+10))
  local dir="$OUT/$tag"; local ep="$dir/socks"
  mkdir -p "$dir" "$ep"; : >"$dir/pids"
  kill_ports "$f0" "$f1" "$a0" "$a1" "$lb"
  echo "=== $tag: A5 A6 F4 F7 (2A2F) ==="
  local fp0 fp1 p0 p1
  fp0=$(start_ffn 4 "$f0" 0 2 2 "$ep" "$dir/ffn0.log"); echo "$fp0" >>"$dir/pids"
  fp1=$(start_ffn 7 "$f1" 1 2 2 "$ep" "$dir/ffn1.log"); echo "$fp1" >>"$dir/pids"
  for i in $(seq 1 60); do
    rg -q "AfPool FFN|Load weight end" "$dir/ffn0.log" 2>/dev/null \
      && rg -q "AfPool FFN|Load weight end" "$dir/ffn1.log" 2>/dev/null && break
    sleep 2
  done
  p0=$(start_attn 5 "$a0" 0 2 2 "$ep" "$dir/attn0.log"); echo "$p0" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a0/health" Attn0 "$p0" 360
  p1=$(start_attn 6 "$a1" 1 2 2 "$ep" "$dir/attn1.log"); echo "$p1" >>"$dir/pids"
  wait_http "http://127.0.0.1:$a1/health" Attn1 "$p1" 360
  local rp
  rp=$(start_rr_lb "$lb" "http://127.0.0.1:$a0" "http://127.0.0.1:$a1")
  echo "$rp" >>"$dir/pids"
  wait_http "http://127.0.0.1:$lb/health" Router "$rp" 60
  run_bench "$tag" "http://127.0.0.1:$lb"
  stop_pids "$dir/pids"
  kill_ports "$f0" "$f1" "$a0" "$a1" "$lb"
}

# ONLY_HIGH_CONC=1 → a1f1 + a2f1_2gpu + a2f2 at current MAX_CONCURRENCY (no same-GPU).
if [[ "${ONLY_HIGH_CONC:-0}" == "1" ]]; then
  run_a1f1
  run_a2_2gpu
  run_a2_2f
  summarize
  exit 0
fi
# ONLY_SAME_GPU=1 → only a2f1_1gpu. RUN_SAME_GPU=1 includes it in full suite.
if [[ "${ONLY_SAME_GPU:-0}" == "1" ]]; then
  run_a2_1gpu
  summarize
  exit 0
fi
if [[ -f "$OUT/a1f1_bench.json" ]]; then
  echo "Reusing existing a1f1_bench.json"
else
  run_a1f1
fi
run_a2_2f
run_a2_2gpu
if [[ "${RUN_SAME_GPU:-0}" == "1" ]]; then
  run_a2_1gpu
fi
summarize
