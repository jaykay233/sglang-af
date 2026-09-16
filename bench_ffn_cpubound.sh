#!/usr/bin/env bash
# Is the FFN process CPU-bound or waiting? Sample /proc/<pid>/stat during decode.
# progress.md §13: decides whether CPU-side MoE optimizations can help at all.
#
# Usage: ATTN_GPU_1A1F=7 FFN_GPUS_1A1F=4 bash bench_ffn_cpubound.sh
set -uo pipefail
cd /root/.cuda/sglang

d=/tmp/afd_cpu
rm -rf "$d"
OUT_DIR="$d" ATTN_GPU_1A1F=7 FFN_GPUS_1A1F=4 CASES=1a1f \
  SGLANG_AFD_FARM_PERSISTENT=0 SGLANG_AFD_FARM_MAX_INFLIGHT=32 \
  NUM_PROMPTS=16 MAX_CONCURRENCY=16 RANDOM_INPUT_LEN=128 RANDOM_OUTPUT_LEN=32 \
  WARMUP_REQUESTS=1 CONTEXT_LENGTH=2048 MAX_NUM_TOKEN=384 \
  bash python/sglang/srt/afd/farm/bench_pool_e2e.sh >"$d/bench.out" 2>&1 &
bench_pid=$!

nproc_val=$(nproc)
echo "nproc=$nproc_val"

# Wait for the FFN pid file, then sample while the server is live.
for _ in $(seq 1 120); do
  [ -f "$d/1a1f/pids" ] && break
  sleep 2
done
if [ ! -f "$d/1a1f/pids" ]; then echo "NO_PIDS"; exit 1; fi
cat "$d/1a1f/pids"
ffn_pid=$(awk '/^ffn/{print $NF}' "$d/1a1f/pids" | head -1)
attn_pid=$(awk '/^attn/{print $NF}' "$d/1a1f/pids" | head -1)
echo "ffn_pid=$ffn_pid attn_pid=$attn_pid"

sample() { # pid -> "utime+stime" in clock ticks, plus nthreads
  awk '{print $14+$15, $20}' /proc/$1/stat 2>/dev/null
}

# Let the run reach steady-state decode.
sleep 25
hz=$(getconf CLK_TCK)
echo "CLK_TCK=$hz"
for tag in "$ffn_pid:FFN" "$attn_pid:ATTN"; do
  pid=${tag%%:*}; name=${tag##*:}
  read -r a _ < <(sample "$pid")
  t0=$(date +%s.%N)
  sleep 6
  read -r b nthreads < <(sample "$pid")
  t1=$(date +%s.%N)
  if [ -z "${a:-}" ] || [ -z "${b:-}" ]; then echo "$name: gone"; continue; fi
  cpu=$(awk -v d=$((b-a)) -v hz="$hz" -v dt="$(awk -v x=$t0 -v y=$t1 'BEGIN{print y-x}')" \
        'BEGIN{printf "%.2f", d/hz/dt}')
  echo "$name pid=$pid threads=$nthreads cpu_cores_used=$cpu"
done
echo "CPU_SAMPLED"
wait $bench_pid 2>/dev/null
echo "DONE"
