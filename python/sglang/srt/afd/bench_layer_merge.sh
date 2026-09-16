#!/usr/bin/env bash
# Bench PD vs AFD from=0 merge_k=1 vs merge_k=2 (same-host cuda_ipc).
# Keeps ComfyUI alive — only stops sglang launch_server / router.
set -euo pipefail

if [[ -f /root/.cuda/afd_env.sh ]]; then
  # shellcheck disable=SC1091
  source /root/.cuda/afd_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-/data/share/models/DeepSeek-V2-Lite-Chat}"
export OUT_DIR="${OUT_DIR:-/tmp/afd_layer_merge_bench}"
export PREFILL_GPU="${PREFILL_GPU:-4}"
export DECODE_GPU="${DECODE_GPU:-5}"
export FFN_GPU="${FFN_GPU:-6}"
export BASE_PORT="${BASE_PORT:-33800}"
export NUM_PROMPTS="${NUM_PROMPTS:-16}"
export RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
export RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-64}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
export DECODE_MAX_BS="${DECODE_MAX_BS:-16}"
export SGLANG_AFD_TRANSPORT="${SGLANG_AFD_TRANSPORT:-cuda_ipc}"
export SGLANG_AFD_REMOTE_FROM_LAYER=0
export SGLANG_AFD_LAYER_PIPELINE=0
export SGLANG_AFD_TRUE_OVERLAP=0
export SGLANG_AFD_NUM_MB=1
export SGLANG_AFD_STEPMESH_STAGES=0

mkdir -p "$OUT_DIR"

ps -eo pid,cmd | awk '/sglang.launch_server|sglang_router|sglang::/ && !/awk|Comfy|main.py/ {print $1}' \
  | xargs -r kill -9 2>/dev/null || true
sleep 2

run_mode() {
  local modes=$1
  local merge_k=$2
  local in_graph=$3
  local tag_out=$4
  export MODES="$modes"
  export SGLANG_AFD_LAYER_MERGE_K="$merge_k"
  export SGLANG_AFD_REMOTE_FROM_LAYER=0
  if [[ "$merge_k" -gt 1 ]]; then
    export SGLANG_AFD_IN_GRAPH_WAIT=0
    export SGLANG_AFD_FFN_CUDA_GRAPH=0
  else
    export SGLANG_AFD_IN_GRAPH_WAIT="$in_graph"
    export SGLANG_AFD_FFN_CUDA_GRAPH=1
  fi
  export SGLANG_AFD_IPC_ENDPOINT="$OUT_DIR/afd_${tag_out}.sock"
  rm -f "$SGLANG_AFD_IPC_ENDPOINT" "$SGLANG_AFD_IPC_ENDPOINT.merge_kv" 2>/dev/null || true
  echo "===== $tag_out modes=$modes merge_k=$merge_k in_graph=$SGLANG_AFD_IN_GRAPH_WAIT ====="
  bash "$SCRIPT_DIR/bench_pd_vs_afd.sh" || true
  if [[ "$modes" == "pd" ]]; then
    cp -a "$OUT_DIR/pd_bench.json" "$OUT_DIR/${tag_out}_bench.json" 2>/dev/null || true
    cp -a "$OUT_DIR/pd_bench.log" "$OUT_DIR/${tag_out}_bench.log" 2>/dev/null || true
  else
    cp -a "$OUT_DIR/pd_afd_bench.json" "$OUT_DIR/${tag_out}_bench.json" 2>/dev/null || true
    cp -a "$OUT_DIR/pd_afd_bench.log" "$OUT_DIR/${tag_out}_bench.log" 2>/dev/null || true
  fi
}

run_mode pd 1 0 pd_only
run_mode pd_afd 1 1 afd_from0_k1
run_mode pd_afd 2 0 afd_from0_k2

python3 - <<'PY' "$OUT_DIR"
import json, os, sys
out = sys.argv[1]
tags = ["pd_only", "afd_from0_k1", "afd_from0_k2"]

def load(tag):
    path = os.path.join(out, f"{tag}_bench.json")
    if not os.path.isfile(path):
        return None
    text = open(path).read().strip()
    if not text:
        return None
    dec = json.JSONDecoder()
    objs, i = [], 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        obj, end = dec.raw_decode(text, i)
        objs.append(obj)
        i = end
    return objs[-1] if objs else None

print("\n========== Layer-merge AFD (REMOTE_FROM_LAYER=0) ==========")
print(f"{'metric':<24}" + "".join(f"{t:>16}" for t in tags))
data = {t: load(t) for t in tags}
keys = ["median_tpot_ms", "mean_tpot_ms", "p99_tpot_ms",
        "output_throughput", "median_ttft_ms", "completed"]
for k in keys:
    row = f"{k:<24}"
    for t in tags:
        d = data.get(t)
        if not d or k not in d:
            row += f"{'—':>16}"
        else:
            v = d[k]
            row += f"{v:>16.2f}" if isinstance(v, float) else f"{str(v):>16}"
    print(row)
print("LAYER_MERGE_BENCH_OK")
PY
