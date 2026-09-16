#!/usr/bin/env bash
# A/B the FFN worker's outer gather while transport-internal gather stays on.
set -euo pipefail

cd /root/.cuda/sglang/python/sglang/srt/afd

ROOT="${ROOT:-/tmp/afd_phase4_20260915}"
mkdir -p "$ROOT"

COMMON_ENV=(
  CUDA_VISIBLE_DEVICES=""
  SGLANG_AFD_FARM_PHASE_TIMING=1
  SGLANG_AFD_FARM_B_STEP=16
  SGLANG_AFD_FARM_B_WIN_K=1
  SGLANG_AFD_FARM_COALESCE_K=2
  SGLANG_AFD_FARM_LAYER_BURST=0
  SGLANG_AFD_FARM_SCHED=max
  SGLANG_AFD_FARM_LPU_GATHER_US=50
  SGLANG_AFD_FARM_LPU_STATS_EVERY=256
  SGLANG_AFD_NUM_MB=16
  SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=16
  CASES=1a1f
  ATTN_GPU_1A1F=4
  FFN_GPUS_1A1F=5
  NUM_PROMPTS=64
  MAX_CONCURRENCY=32
  RANDOM_INPUT_LEN=128
  RANDOM_OUTPUT_LEN=128
)

run_case() {
  local tag=$1
  shift
  echo "############################ $tag $(date)"
  env "${COMMON_ENV[@]}" "$@" \
    OUT_DIR="$ROOT/$tag" \
    BASE_PORT=41910 \
    bash farm/bench_pool_e2e.sh
  echo "############################ $tag done rc=$? $(date)"
  sleep 5
}

run_case double_gather env -u SGLANG_AFD_FARM_SKIP_EXTRA_GATHER
run_case single_gather env SGLANG_AFD_FARM_SKIP_EXTRA_GATHER=1

python3 - <<'PY' "$ROOT"
import json
import os
import re
import sys

root = sys.argv[1]
rows = {}
stats = {}
for tag in ("double_gather", "single_gather"):
    summary_path = os.path.join(root, tag, "SUMMARY.json")
    with open(summary_path, encoding="utf-8") as f:
        rows[tag] = json.load(f)["rows"]["1a1f"]
    ffn_path = os.path.join(root, tag, "1a1f", "ffn0.log")
    last = ""
    with open(ffn_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "LPU stats" in line:
                last = line.strip()
    stats[tag] = last

out = {"rows": rows, "lpu_stats_tail": stats}
with open(os.path.join(root, "SUMMARY.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

base = rows["double_gather"]
single = rows["single_gather"]
print(
    f"single/double output tok/s = "
    f"{single['output_throughput'] / base['output_throughput']:.3f}x"
)
print(
    f"single/double median TPOT = "
    f"{single['median_tpot_ms'] / base['median_tpot_ms']:.3f}x"
)
for tag in ("double_gather", "single_gather"):
    print(tag, stats[tag])
PY

echo "AFD_GATHER_AB_DONE $(date)"
