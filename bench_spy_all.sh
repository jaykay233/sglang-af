#!/usr/bin/env bash
# Dump stacks of every sglang process during a live decode so we can find which
# thread competes with FFN compute for the GIL (progress.md §14.3).
set -uo pipefail
OUT=/tmp/afd_spy
RAW="$OUT/raw"
mkdir -p "$RAW"
: > "$OUT/all_stacks.txt"

for i in $(seq 1 200); do
  grep -q "=== Bench" "$OUT/bench.log" 2>/dev/null && break
  sleep 2
done
echo "BENCH_UP"

# Sample for a fixed window while decode is running.
deadline=$((SECONDS + 100))
round=0
while [ $SECONDS -lt $deadline ]; do
  round=$((round + 1))
  for p in $(pgrep -f "launch_server|sglang::" 2>/dev/null); do
    f="$RAW/${p}.txt"
    if [ ! -f "$f" ]; then
      # Identify this pid once.
      echo "=== PID $p: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-100)" >> "$OUT/all_stacks.txt"
    fi
    timeout 30 py-spy dump --pid "$p" > "$f.$$" 2>&1
    cat "$f.$$" >> "$OUT/all_stacks.txt" 2>/dev/null
    rm -f "$f.$$"
  done
  sleep 0.5
done
echo "ALLSAMPLED rounds=$round"
