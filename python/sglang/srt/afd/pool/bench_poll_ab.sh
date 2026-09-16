#!/usr/bin/env bash
# A/B: NA1F FFN poll "drain all links non-blocking" vs legacy per-link blocking.
#
# SGLANG_AFD_FFN_POLL_DRAIN_ALL=1 -> new (non-blocking scan of every link)
# SGLANG_AFD_FFN_POLL_DRAIN_ALL=0 -> legacy (blocking get_batch per link)
#
# Each invocation also runs the 1A1F baseline (--compare-1a1f), so every rep
# yields both a 1A1F and a 2A1F number measured back-to-back on the same host.
set -u
cd /root/.cuda/sglang

GPUS=${GPUS:-0,1,3}
REPS=${REPS:-4}
SKIP_GATHER=${SKIP_GATHER:-1}
OUT=${OUT:-/tmp/afd_poll_ab2}
# Real per-layer ratio from progress.md §16.4: attn 1114us vs FFN 520us.
# attn-bound => the FFN idles in 1A1F, which is the regime 2A1F is meant to fix.
ATTN_US=${ATTN_US:-1100}
FFN_US=${FFN_US:-520}
REQS=${REQS:-16}
LAYERS=${LAYERS:-26}
TOKENS=${TOKENS:-8}
mkdir -p "$OUT"
: > "$OUT/results.txt"

run_arm() {
  local arm=$1 drain=$2 rep=$3
  local d="$OUT/${arm}_r${rep}"
  rm -rf "$d"; mkdir -p "$d"
  echo "=== arm=$arm POLL_DRAIN_ALL=$drain rep=$rep skip_gather=$SKIP_GATHER attn_us=$ATTN_US ffn_us=$FFN_US ==="
  SGLANG_AFD_FFN_POLL_DRAIN_ALL="$drain" \
  SGLANG_AFD_FARM_SKIP_EXTRA_GATHER="$SKIP_GATHER" \
    python3 -m sglang.srt.afd.bench_af_pool \
      --num-attn 2 --num-ffn 1 --gpus "$GPUS" \
      --reqs "$REQS" --layers "$LAYERS" --tokens "$TOKENS" \
      --attn-us "$ATTN_US" --ffn-us "$FFN_US" \
      --compare-1a1f \
      --endpoint-dir "$d" >"$d/log.txt" 2>&1
  local rc=$?
  grep -E "^RESULT |^SPEEDUP " "$d/log.txt" | sed "s/^/${arm},${rep},/" >>"$OUT/results.txt"
  echo "  rc=$rc"
  grep -E "^RESULT |^SPEEDUP " "$d/log.txt" | sed 's/^/  /'
}

for rep in $(seq 1 "$REPS"); do
  # Alternate order per rep to cancel out machine drift.
  if [ $((rep % 2)) -eq 1 ]; then
    run_arm legacy 0 "$rep"
    run_arm drainall 1 "$rep"
  else
    run_arm drainall 1 "$rep"
    run_arm legacy 0 "$rep"
  fi
done

echo
echo "############ raw results ############"
cat "$OUT/results.txt"
