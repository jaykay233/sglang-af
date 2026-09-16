# SPDX-License-Identifier: Apache-2.0
"""CPU occupancy / B_win KPI sim for decode farm (no GPU).

Example::

    python -m sglang.srt.afd.farm.bench_occupancy --tokens 32 --layers 26 --b-step 16
    python -m sglang.srt.afd.farm.bench_occupancy --sweep-k 1,8,32 --tokens 32 --layers 26
"""

from __future__ import annotations

import argparse
import json
from typing import List

from sglang.srt.afd.farm.token_queue import (
    simulate_bwin_kpi,
    simulate_farm_occupancy,
    simulate_ffn_gather,
    sweep_farm_params,
)


def _parse_k_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(1, int(part)))
    return out or [1, 8, 32]


def main() -> int:
    p = argparse.ArgumentParser(description="AFD farm occupancy / B_win discrete-event sim")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--layers", type=int, default=26)
    p.add_argument("--b-step", type=int, default=16)
    p.add_argument("--b-win-k", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=4)
    p.add_argument("--attn-ticks", type=int, default=1)
    p.add_argument("--ffn-ticks", type=int, default=3)
    p.add_argument(
        "--sweep-k",
        type=str,
        default="",
        help="Comma list of B_win_k values (P2 KPI). Example: 1,8,32",
    )
    p.add_argument(
        "--sweep-coalesce",
        type=str,
        default="",
        help="Comma list of COALESCE_K values. Example: 1,2,4",
    )
    p.add_argument("--coalesce-k", type=int, default=1)
    p.add_argument("--per-layer-cap", type=int, default=0)
    p.add_argument("--token-budget", type=int, default=0)
    p.add_argument("--max-age", type=int, default=0)
    p.add_argument(
        "--sweep-params",
        action="store_true",
        help="P2 grid over B_step x per-layer cap x token budget x coalesce",
    )
    p.add_argument("--b-steps", type=str, default="8,16,32")
    p.add_argument("--per-layer-caps", type=str, default="0,1,2")
    p.add_argument("--token-budgets", type=str, default="0")
    p.add_argument("--max-ages", type=str, default="0")
    p.add_argument("--coalesce-ks", type=str, default="1")
    p.add_argument(
        "--sweep-ffn-gather",
        action="store_true",
        help="P2 FFN natural-batch size vs hop latency / gather window",
    )
    args = p.parse_args()

    if args.sweep_params:
        rows = sweep_farm_params(
            n_tokens=args.tokens,
            n_layers=args.layers,
            b_steps=_parse_k_list(args.b_steps),
            b_win_ks=[args.b_win_k],
            coalesce_ks=_parse_k_list(args.coalesce_ks),
            per_layer_caps=[int(x) for x in args.per_layer_caps.split(",") if x.strip()],
            global_token_budgets=[
                int(x) for x in args.token_budgets.split(",") if x.strip()
            ],
            max_age_steps_list=[int(x) for x in args.max_ages.split(",") if x.strip()],
            max_inflight=args.max_inflight,
            attn_ticks=args.attn_ticks,
            ffn_ticks=args.ffn_ticks,
        )
        print(
            f"{'rank':>4} {'B_step':>6} {'cap':>4} {'budget':>7} {'age':>4} "
            f"{'ticks':>7} {'busy':>5} {'peak':>4} {'tok_blk':>7} {'cap_blk':>7} "
            f"{'aged':>5} {'amort':>6} {'pk_tok':>6}"
        )
        for r in rows[:24]:
            print(
                f"{r['rank']:>4} {r['b_step']:>6} {r['per_layer_cap']:>4} "
                f"{r['global_token_budget']:>7} {r['max_age_steps']:>4} "
                f"{r['ticks']:>7} {float(r['mean_layers_busy']):>5.2f} "
                f"{r['peak_layers_busy']:>4} {r['token_blocks']:>7} "
                f"{r['layer_cap_blocks']:>7} {r['aged_picks']:>5} "
                f"{float(r['amortize_factor']):>6.2f} "
                f"{r['peak_inflight_tokens']:>6}"
            )
        print(json.dumps({"sweep_params": rows}, indent=2))
        return 0

    if args.sweep_ffn_gather:
        rows = []
        for lat in (1, 2, 4):
            for gather in (0, 1, 2, 4):
                d = simulate_ffn_gather(
                    n_layers=args.layers,
                    hops_per_layer=8,
                    hop_tokens=args.b_step,
                    hop_latency_ticks=lat,
                    gather_ticks=max(1, gather),
                    max_batch_tokens=64,
                )
                d["hop_latency_ticks"] = lat
                d["gather_ticks"] = max(1, gather)
                rows.append(d)
                print(
                    f"hop_lat={lat} gather={gather:>2}: "
                    f"mean_batch={float(d['mean_batch_tokens']):>5.1f}tok "
                    f"({float(d['mean_hops_per_launch']):.2f} hops) "
                    f"singletons={d['singleton_launches']}/{d['launches']} "
                    f"max={d['max_batch_tokens']}"
                )
        print(json.dumps({"sweep_ffn_gather": rows}, indent=2))
        return 0

    if args.sweep_k.strip():
        rows = []
        for k in _parse_k_list(args.sweep_k):
            d = simulate_bwin_kpi(
                n_layers=min(8, args.layers),
                n_per_layer=64,
                b_step=args.b_step,
                b_win_k=k,
                coalesce_k=args.coalesce_k,
                max_picks=64,
            )
            rows.append(d)
            print(
                f"K={k:3d} switches={d['layer_switches']} "
                f"mean_win_len={float(d['mean_win_len']):.2f} "
                f"amortize={float(d['amortize_factor']):.2f} picks={d['picks']}"
            )
        print(json.dumps({"sweep_k": rows}, indent=2))
        print(
            "NOTE: sticky B_win is scheduling; COALESCE_K>1 is true stock-kernel amortize."
        )
        return 0

    if args.sweep_coalesce.strip():
        rows = []
        for ck in _parse_k_list(args.sweep_coalesce):
            d = simulate_bwin_kpi(
                n_layers=min(8, args.layers),
                n_per_layer=64,
                b_step=args.b_step,
                b_win_k=args.b_win_k,
                coalesce_k=ck,
                max_picks=64,
            )
            rows.append(d)
            print(
                f"coalesce_k={ck:3d} picks={d['picks']} "
                f"mean_tok/launch={float(d['mean_tokens_per_launch']):.1f} "
                f"amortize_factor={float(d['amortize_factor']):.2f} "
                f"switches={d['layer_switches']}"
            )
        print(json.dumps({"sweep_coalesce": rows}, indent=2))
        return 0

    lock = simulate_farm_occupancy(
        n_tokens=args.tokens,
        n_layers=args.layers,
        b_step=args.tokens,
        b_win_k=1,
        coalesce_k=1,
        max_inflight=1,
        attn_ticks=args.attn_ticks,
        ffn_ticks=args.ffn_ticks,
    )
    farm = simulate_farm_occupancy(
        n_tokens=args.tokens,
        n_layers=args.layers,
        b_step=args.b_step,
        b_win_k=args.b_win_k,
        coalesce_k=args.coalesce_k,
        max_inflight=args.max_inflight,
        attn_ticks=args.attn_ticks,
        ffn_ticks=args.ffn_ticks,
        per_layer_cap=args.per_layer_cap,
        global_token_budget=args.token_budget,
        max_age_steps=args.max_age,
    )
    out = {
        "lockstep": lock.as_dict(),
        "farm": farm.as_dict(),
        "speedup_ticks": (lock.ticks / farm.ticks if farm.ticks else None),
        "occupancy_gain": farm.mean_layers_busy - lock.mean_layers_busy,
    }
    print(json.dumps(out, indent=2))
    print(
        f"lockstep ticks={lock.ticks} mean_layers={lock.mean_layers_busy:.2f} "
        f"peak={lock.peak_layers_busy} switches={lock.layer_switches}"
    )
    print(
        f"farm     ticks={farm.ticks} mean_layers={farm.mean_layers_busy:.2f} "
        f"peak={farm.peak_layers_busy} B_step={args.b_step} K={args.b_win_k} "
        f"coalesce={args.coalesce_k} amortize={farm.amortize_factor:.2f} "
        f"switches={farm.layer_switches} mean_win_len={farm.mean_win_len:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
