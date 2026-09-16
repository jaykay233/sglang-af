#!/usr/bin/env bash
# P2 parameter sweep for the decode farm.
#
# Runs bench_farm_e2e.sh once per config (isolated bring-up + teardown), then
# aggregates tok/s and TPOT into a single table.
#
# Each config is "tag|ENV=VAL ENV=VAL ...". Env vars are exported for that run
# only and are picked up by bench_farm_e2e.sh / the farm env layer.
#
# Usage:
#   bash bench_farm_sweep.sh                       # default P2 matrix
#   CONFIGS="base| ; bs8|SGLANG_AFD_FARM_B_STEP=8" bash bench_farm_sweep.sh
#
# Tunables honored per config:
#   SGLANG_AFD_FARM_B_STEP, SGLANG_AFD_FARM_B_WIN_K, SGLANG_AFD_FARM_COALESCE_K,
#   SGLANG_AFD_FARM_MAX_INFLIGHT, SGLANG_AFD_NUM_MB,
#   SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER, SGLANG_AFD_FARM_GLOBAL_TOKEN_BUDGET,
#   SGLANG_AFD_FARM_MAX_AGE_STEPS, SGLANG_AFD_FARM_NATURAL_BATCH,
#   SGLANG_AFD_FARM_PERSISTENT, SGLANG_AFD_ATTN_QUEUE_TARGET_TOKENS,
#   SGLANG_AFD_FFN_GATHER_US, SGLANG_AFD_FFN_GATHER_MAX
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OUT_DIR:-/tmp/afd_farm_p2}"
mkdir -p "$ROOT"

# P2 matrix. Comment rows out to shorten the run (~2-3 min each).
# Direction from the P2 sweep: B_step is monotone in tok/s and TPOT, so push
# B_step / coalesce UP to maximise tokens-per-launch; do NOT shrink B_step to
# chase cross-layer overlap (overlap rises but fixed per-hop cost dominates).
DEFAULT_CONFIGS=(
  # --- concurrency-16 baseline, coalesce=1 (matches best-throughput point) ---
  "b16c16|SGLANG_AFD_FARM_B_STEP=16 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_COALESCE_K=1"
  # --- Attn-side pre-pack: fewer, fatter A2F hops ---
  "b16c16k2|SGLANG_AFD_FARM_B_STEP=16 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_COALESCE_K=2"
  "b16c16k4|SGLANG_AFD_FARM_B_STEP=16 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_COALESCE_K=4"
  # --- P1.5 natural batching: Attn issues immediately, FFN gathers instead ---
  "b16c16nat|SGLANG_AFD_FARM_B_STEP=16 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_NATURAL_BATCH=1"
  # --- bigger windows ---
  "b32c16|SGLANG_AFD_FARM_B_STEP=32 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_COALESCE_K=1"
  "b32c16k2|SGLANG_AFD_FARM_B_STEP=32 MAX_CONCURRENCY=16 SGLANG_AFD_FARM_COALESCE_K=2"
)

CONFIGS_STR="${CONFIGS:-}"
if [[ -n "$CONFIGS_STR" ]]; then
  IFS=';' read -r -a CONFIG_ARR <<<"$CONFIGS_STR"
else
  CONFIG_ARR=("${DEFAULT_CONFIGS[@]}")
fi

echo "P2 sweep: ${#CONFIG_ARR[@]} configs -> $ROOT"
FAILED=()

for entry in "${CONFIG_ARR[@]}"; do
  tag="$(echo "${entry%%|*}" | tr -d '[:space:]')"
  envs="${entry#*|}"
  [[ -n "$tag" ]] || continue
  echo "=============================================================="
  echo ">>> P2 config tag=$tag envs=[$envs]"
  echo "=============================================================="
  # shellcheck disable=SC2086
  if env $envs OUT_DIR="$ROOT/$tag" MODES=sticky NUM_PROMPTS="${NUM_PROMPTS:-32}" \
      bash "$HERE/bench_farm_e2e.sh"; then
    echo ">>> config $tag OK"
  else
    echo ">>> config $tag FAILED" >&2
    FAILED+=("$tag")
  fi
  sleep 3
done

echo
echo "==================== P2 SWEEP SUMMARY ===================="
python3 - "$ROOT" "${CONFIG_ARR[@]}" <<'PY'
import json, os, sys
root = sys.argv[1]
entries = sys.argv[2:]
rows = []
for entry in entries:
    tag = entry.split("|", 1)[0].strip()
    if not tag:
        continue
    path = os.path.join(root, tag, "SUMMARY.json")
    if not os.path.isfile(path):
        rows.append((tag, None, None, None, None, None))
        continue
    d = json.load(open(path))
    r = (d.get("rows") or {}).get("sticky") or {}
    rows.append((
        tag,
        r.get("completed"),
        r.get("output_throughput"),
        r.get("median_tpot_ms"),
        r.get("median_ttft_ms"),
        r.get("median_e2e_latency_ms"),
    ))

print(f"{'config':<12} {'ok':>4} {'out_tps':>9} {'med_tpot':>9} "
      f"{'med_ttft':>9} {'med_e2e':>10}")
base = next((r for r in rows if r[0].startswith("b16c16|") or r[0] == "base"), None)
for tag, ok, tps, tpot, ttft, e2e in rows:
    def f(v, fmt="{:.1f}"):
        return "—" if v is None else fmt.format(v)
    extra = ""
    if base and tag != base[0] and base[2] and tps and base[3] and tpot:
        extra = f"   tps {tps / base[2]:.2f}x  tpot {tpot / base[3]:.2f}x"
    print(f"{tag:<12} {f(ok, '{:>4.0f}'):>4} {f(tps):>9} {f(tpot):>9} "
          f"{f(ttft):>9} {f(e2e):>10}{extra}")
PY

if ((${#FAILED[@]})); then
  echo "FAILED configs: ${FAILED[*]}" >&2
  exit 1
fi
echo "AFD_FARM_P2_SWEEP_OK"
