# SPDX-License-Identifier: Apache-2.0
"""Farm amortize KPI sweep (plan: measure COALESCE + PERSISTENT_LINEAR first).

Runs without a full model serve:

1. CPU discrete-event: coalesce_k → amortize_factor / picks / switches
2. GPU microbench: separate tiny GEMMs vs fused cuBLAS vs weight-outer Triton
   across (B_step, num_mb) — proxy for sticky windows with/without coalesce

Example::

    CUDA_VISIBLE_DEVICES=7 python -m sglang.srt.afd.farm.bench_farm_amortize
    CUDA_VISIBLE_DEVICES=7 python -m sglang.srt.afd.farm.bench_farm_amortize \\
        --b-steps 8,16 --mbs 1,2,4,8 --k 2048 --n 2112
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import torch

from sglang.srt.afd.farm.persistent_linear import (
    reset_persistent_linear_stats,
    weight_outer_linear,
)
from sglang.srt.afd.farm.token_queue import simulate_bwin_kpi, simulate_farm_occupancy
from sglang.srt.environ import envs


def _parse_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(max(1, int(part)))
    return out


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_ms(fn, *, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3 / float(iters)


def sweep_cpu_coalesce(
    *,
    b_step: int,
    coalesce_ks: List[int],
    b_win_k: int = 8,
    n_layers: int = 8,
    n_per_layer: int = 64,
) -> List[Dict[str, Any]]:
    rows = []
    for ck in coalesce_ks:
        d = simulate_bwin_kpi(
            n_layers=n_layers,
            n_per_layer=n_per_layer,
            b_step=b_step,
            b_win_k=b_win_k,
            coalesce_k=ck,
            max_picks=64,
        )
        occ = simulate_farm_occupancy(
            n_tokens=32,
            n_layers=n_layers,
            b_step=b_step,
            b_win_k=b_win_k,
            coalesce_k=ck,
            max_inflight=4,
            attn_ticks=1,
            ffn_ticks=3,
        )
        rows.append(
            {
                "coalesce_k": ck,
                "amortize_factor": float(d["amortize_factor"]),
                "mean_tokens_per_launch": float(d["mean_tokens_per_launch"]),
                "picks": int(d["picks"]),
                "layer_switches": int(d["layer_switches"]),
                "occ_mean_layers_busy": occ.mean_layers_busy,
                "occ_peak_layers_busy": occ.peak_layers_busy,
                "occ_ticks": occ.ticks,
            }
        )
    return rows


def sweep_gpu_linear(
    *,
    b_steps: List[int],
    mbs: List[int],
    k: int,
    n: int,
    dtype: torch.dtype,
    device: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return rows

    envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(True)
    w = torch.randn(n, k, device=device, dtype=dtype)

    for b_step in b_steps:
        for num_mb in mbs:
            m = b_step * num_mb
            x = torch.randn(m, k, device=device, dtype=dtype)

            def separate():
                outs = []
                for i in range(num_mb):
                    sl = x[i * b_step : (i + 1) * b_step]
                    outs.append(sl @ w.T)
                return torch.cat(outs, dim=0)

            def fused():
                return x @ w.T

            def outer():
                return weight_outer_linear(x, w, b_step=b_step)

            reset_persistent_linear_stats()
            y_s = separate()
            y_f = fused()
            y_o = outer()
            err_f = (y_s.float() - y_f.float()).abs().max().item()
            err_o = (y_s.float() - y_o.float()).abs().max().item()

            ms_s = _timed_ms(separate)
            ms_f = _timed_ms(fused)
            ms_o = _timed_ms(outer)

            # Bytes of W read if each mb reload vs once: proxy model.
            w_bytes = float(n * k * x.element_size())
            # separate ≈ num_mb full W reads; fused/outer ≈ 1 (ideal).
            rows.append(
                {
                    "b_step": b_step,
                    "num_mb": num_mb,
                    "M": m,
                    "K": k,
                    "N": n,
                    "ms_separate": round(ms_s, 4),
                    "ms_fused": round(ms_f, 4),
                    "ms_weight_outer": round(ms_o, 4),
                    "speedup_outer_vs_separate": round(ms_s / ms_o, 3) if ms_o else None,
                    "speedup_fused_vs_separate": round(ms_s / ms_f, 3) if ms_f else None,
                    "err_fused": err_f,
                    "err_outer": err_o,
                    "w_bytes": w_bytes,
                    "ideal_w_traffic_ratio_vs_separate": round(1.0 / num_mb, 4),
                    # coalesce_k=num_mb means one launch of M tokens
                    "equiv_coalesce_k": num_mb,
                }
            )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Farm COALESCE × PERSISTENT_LINEAR KPI")
    p.add_argument("--b-step", type=int, default=16, help="CPU sweep B_step")
    p.add_argument("--b-steps", type=str, default="8,16", help="GPU sweep B_steps")
    p.add_argument("--mbs", type=str, default="1,2,4,8", help="GPU num microbatches")
    p.add_argument("--coalesce", type=str, default="1,2,4,8")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=2112, help="DeepSeek-V2-Lite fused qkv_a out")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    coalesce_ks = _parse_ints(args.coalesce)
    cpu_rows = sweep_cpu_coalesce(b_step=args.b_step, coalesce_ks=coalesce_ks)

    print("=== CPU coalesce / occupancy ===")
    print(
        f"{'ck':>4} {'amortize':>8} {'tok/launch':>10} {'picks':>6} "
        f"{'switches':>8} {'occ_mean':>8} {'ticks':>6}"
    )
    for r in cpu_rows:
        print(
            f"{r['coalesce_k']:4d} {r['amortize_factor']:8.2f} "
            f"{r['mean_tokens_per_launch']:10.1f} {r['picks']:6d} "
            f"{r['layer_switches']:8d} {r['occ_mean_layers_busy']:8.2f} "
            f"{r['occ_ticks']:6d}"
        )

    gpu_rows: List[Dict[str, Any]] = []
    if not args.cpu_only and torch.cuda.is_available():
        dtype = getattr(torch, args.dtype)
        print("\n=== GPU linear (separate vs fused vs weight-outer) ===")
        print(
            f"{'B':>4} {'mb':>3} {'M':>4} {'sep_ms':>8} {'fuse_ms':>8} "
            f"{'outer_ms':>8} {'out/sep':>7} {'fuse/sep':>8} {'w_ratio':>7}"
        )
        gpu_rows = sweep_gpu_linear(
            b_steps=_parse_ints(args.b_steps),
            mbs=_parse_ints(args.mbs),
            k=args.k,
            n=args.n,
            dtype=dtype,
            device="cuda",
        )
        for r in gpu_rows:
            print(
                f"{r['b_step']:4d} {r['num_mb']:3d} {r['M']:4d} "
                f"{r['ms_separate']:8.3f} {r['ms_fused']:8.3f} "
                f"{r['ms_weight_outer']:8.3f} "
                f"{r['speedup_outer_vs_separate'] or 0:7.2f} "
                f"{r['speedup_fused_vs_separate'] or 0:8.2f} "
                f"{r['ideal_w_traffic_ratio_vs_separate']:7.3f}"
            )
        print(
            "\nNote: fused cuBLAS usually wins wall time; weight_outer proves "
            "doc control structure (W tile reused across mb). "
            "COALESCE_K≈num_mb on the Attn farm launch."
        )
    elif not args.cpu_only:
        print("\n(CUDA unavailable — skipped GPU sweep)")

    # Interpretation for next step (spin-wait).
    print("\n=== Plan gate ===")
    if gpu_rows:
        # Compare mb=1 vs max mb for separate vs outer
        by_b: Dict[int, List[Dict[str, Any]]] = {}
        for r in gpu_rows:
            by_b.setdefault(int(r["b_step"]), []).append(r)
        for b, rs in sorted(by_b.items()):
            r1 = min(rs, key=lambda x: x["num_mb"])
            rmax = max(rs, key=lambda x: x["num_mb"])
            if r1["ms_separate"]:
                sep_scale = rmax["ms_separate"] / r1["ms_separate"]
            else:
                sep_scale = None
            print(
                f"B_step={b}: separate latency scales ~{sep_scale:.2f}x from "
                f"mb={r1['num_mb']}→{rmax['num_mb']}; outer={rmax['ms_weight_outer']:.3f}ms "
                f"vs separate={rmax['ms_separate']:.3f}ms "
                f"(ideal W traffic {rmax['ideal_w_traffic_ratio_vs_separate']:.2f}x)."
            )
        print(
            "If coalesce (large mb) hurts TPOT in full farm serve, next is "
            "device spin-wait persistent grid. If tok/s already OK, stay here."
        )
    else:
        print("Run with CUDA to unlock the spin-wait go/no-go gate.")

    # Analytical TPOT proxy (no serve): arrival gap vs compute.
    print("\n=== TPOT proxy (analytic) ===")
    print(
        "Assume each B_step window arrives every T_arr µs; compute from GPU row.\n"
        "  sticky_separate: sum of per-mb separate GEMM (re-reads W)\n"
        "  coalesce_fused:  wait (K-1)*T_arr then one fused GEMM\n"
        "  spin_wait_ideal: no wait, per-mb compute ≈ fused(M=B)/K  (W amortized)"
    )
    tpot_rows: List[Dict[str, Any]] = []
    t_arr_us = 50.0  # F2A/ready gap proxy
    if gpu_rows:
        for r in gpu_rows:
            if r["num_mb"] < 2:
                continue
            k_mb = int(r["num_mb"])
            # Per-mb separate time ≈ total_separate / k
            t_sep_mb = r["ms_separate"] * 1e3 / k_mb  # µs
            t_fuse = r["ms_fused"] * 1e3  # µs for full M
            t_sticky = k_mb * t_sep_mb
            t_coal = (k_mb - 1) * t_arr_us + t_fuse
            t_spin = k_mb * (t_fuse / k_mb)  # ≈ fused, streamed
            row = {
                "b_step": r["b_step"],
                "K": k_mb,
                "T_arr_us": t_arr_us,
                "tpot_sticky_sep_us": round(t_sticky, 1),
                "tpot_coalesce_fused_us": round(t_coal, 1),
                "tpot_spin_ideal_us": round(t_spin, 1),
                "coalesce_vs_sticky": round(t_coal / t_sticky, 3) if t_sticky else None,
                "spin_vs_coalesce": round(t_spin / t_coal, 3) if t_coal else None,
            }
            tpot_rows.append(row)
            print(
                f"B={r['b_step']} K={k_mb}: sticky={t_sticky:.0f}µs "
                f"coalesce={t_coal:.0f}µs spin_ideal={t_spin:.0f}µs "
                f"(coal/sticky={row['coalesce_vs_sticky']}, "
                f"spin/coal={row['spin_vs_coalesce']})"
            )
        spin_rows: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            from sglang.srt.afd.farm.spin_wait_linear import (
                SpinWaitLinearSession,
                run_device_poller_rounds,
            )

            print("\n=== Spin-wait session (resident W, no coalesce wait) ===")
            dtype = getattr(torch, args.dtype)
            w = torch.randn(args.n, args.k, device="cuda", dtype=dtype)
            # Warmup JIT once.
            _warm = SpinWaitLinearSession(weight=w, b_step=_parse_ints(args.b_steps)[0])
            _warm.run_many(
                torch.randn(
                    2,
                    _warm.b_step,
                    args.k,
                    device="cuda",
                    dtype=dtype,
                )
            )
            torch.cuda.synchronize()
            for b_step in _parse_ints(args.b_steps):
                for num_mb in _parse_ints(args.mbs):
                    if num_mb < 2:
                        continue
                    xs = torch.randn(
                        num_mb, b_step, args.k, device="cuda", dtype=dtype
                    )
                    sess = SpinWaitLinearSession(weight=w, b_step=b_step)
                    # one dry run to populate kernel cache for this B_step
                    sess.run_many(xs[:1])
                    torch.cuda.synchronize()
                    y, ms = run_device_poller_rounds(sess, xs)
                    ref = xs.view(num_mb * b_step, args.k) @ w.T
                    err = (y.float() - ref.float()).abs().max().item()
                    xflat = xs.view(num_mb * b_step, args.k)

                    def _fused():
                        return xflat @ w.T

                    ms_f = _timed_ms(_fused)
                    t_coal = (num_mb - 1) * t_arr_us / 1000.0 + ms_f  # ms
                    spin_rows.append(
                        {
                            "b_step": b_step,
                            "num_mb": num_mb,
                            "ms_spin_session": round(ms, 4),
                            "ms_fused_only": round(ms_f, 4),
                            "ms_coalesce_proxy": round(t_coal, 4),
                            "err": err,
                            "spin_vs_coalesce_proxy": round(ms / t_coal, 3)
                            if t_coal
                            else None,
                        }
                    )
                    print(
                        f"B={b_step} K={num_mb}: spin_session={ms:.3f}ms "
                        f"coalesce_proxy={t_coal:.3f}ms "
                        f"fused_only={ms_f:.3f}ms err={err:.2e}"
                    )
        if tpot_rows:
            ratios = [r["coalesce_vs_sticky"] for r in tpot_rows if r["coalesce_vs_sticky"]]
            mean_r = sum(ratios) / len(ratios)
            if mean_r > 1.3:
                print(
                    f"Gate: coalesce TPOT proxy ~{mean_r:.2f}x sticky → "
                    "enable SGLANG_AFD_FARM_SPIN_WAIT=1 "
                    "(and prefer COALESCE_K=1 for low TPOT)."
                )
            else:
                print(
                    f"Gate: coalesce TPOT proxy ~{mean_r:.2f}x sticky → "
                    "stay on COALESCE_K + stock GEMM; defer spin-wait."
                )
    else:
        spin_rows = []

    out = {
        "cpu_coalesce": cpu_rows,
        "gpu_linear": gpu_rows,
        "tpot_proxy": tpot_rows if gpu_rows else [],
        "spin_wait": spin_rows if gpu_rows else [],
        "b_step_cpu": args.b_step,
        "t_arr_us": t_arr_us if gpu_rows else None,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")
    else:
        print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
