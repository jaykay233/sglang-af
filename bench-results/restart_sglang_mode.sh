#!/usr/bin/env bash
# Restart SGLang with a given scheduler mode for ablation.
set -euo pipefail

MODE="${1:?mode: baseline|mixed|budget}"
LOG="/root/.cuda/sglang/sglang-ablation-${MODE}.log"
MODEL="/data/share/tmp-1/Qwen2.5-0.5B"

export CUDA_HOME=/opt/cuda-13.0
export PATH=/opt/cuda-13.0/bin:/root/.cargo/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda-compat-13:/opt/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Kill existing launch_server carefully
pids=$(pgrep -f '/usr/bin/python3 -m sglang.launch_server' || true)
if [[ -n "${pids}" ]]; then
  kill ${pids} || true
  sleep 3
  pids=$(pgrep -f '/usr/bin/python3 -m sglang.launch_server' || true)
  if [[ -n "${pids}" ]]; then
    kill -9 ${pids} || true
    sleep 1
  fi
fi

EXTRA=()
case "${MODE}" in
  baseline) EXTRA=() ;;
  mixed) EXTRA=(--enable-mixed-chunk) ;;
  budget) EXTRA=(--enable-decode-token-budget --decode-token-budget-stall-limit 2) ;;
  *) echo "unknown mode"; exit 1 ;;
esac

cd /root/.cuda/sglang
rm -f "${LOG}"
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m sglang.launch_server \
  --model-path "${MODEL}" \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --attention-backend triton \
  "${EXTRA[@]}" \
  > "${LOG}" 2>&1 &

for i in $(seq 1 120); do
  if grep -qE 'The server is fired up|Uvicorn running on' "${LOG}" 2>/dev/null; then
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/v1/models || true)
    if [[ "${code}" == "200" ]]; then
      echo "READY mode=${MODE}"
      exit 0
    fi
  fi
  if ! pgrep -f '/usr/bin/python3 -m sglang.launch_server' >/dev/null; then
    echo "DEAD mode=${MODE}"
    tail -40 "${LOG}"
    exit 1
  fi
  sleep 2
done
echo "TIMEOUT mode=${MODE}"
tail -40 "${LOG}"
exit 1
