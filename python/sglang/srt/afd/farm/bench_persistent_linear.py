# SPDX-License-Identifier: Apache-2.0
"""Microbench: K separate GEMMs vs fused cuBLAS vs weight-outer Triton."""

from __future__ import annotations

import argparse
import time

import torch

from sglang.srt.afd.farm.persistent_linear import (
    reset_persistent_linear_stats,
    weight_outer_linear,
)
from sglang.srt.environ import envs


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1e3 / iters


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="total tokens (= num_mb * b_step)")
    p.add_argument("--b-step", type=int, default=16)
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=2112)
    p.add_argument("--dtype", type=str, default="float16")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    if args.m % args.b_step != 0:
        print("m must be divisible by b_step")
        return 1

    dtype = getattr(torch, args.dtype)
    device = "cuda"
    x = torch.randn(args.m, args.k, device=device, dtype=dtype)
    w = torch.randn(args.n, args.k, device=device, dtype=dtype)
    num_mb = args.m // args.b_step

    def separate():
        outs = []
        for i in range(num_mb):
            sl = x[i * args.b_step : (i + 1) * args.b_step]
            outs.append(sl @ w.T)
        return torch.cat(outs, dim=0)

    def fused():
        return x @ w.T

    def outer():
        return weight_outer_linear(x, w, b_step=args.b_step)

    envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(True)
    reset_persistent_linear_stats()
    y_s = separate()
    y_f = fused()
    y_o = outer()
    err_f = (y_s - y_f).float().abs().max().item()
    err_o = (y_s - y_o).float().abs().max().item()
    print(f"max_abs_err fused={err_f:.4e} weight_outer={err_o:.4e}")

    ms_s = _timed(separate)
    ms_f = _timed(fused)
    ms_o = _timed(outer)
    print(
        f"M={args.m} B_step={args.b_step} num_mb={num_mb} K={args.k} N={args.n}\n"
        f"  separate_gemm  {ms_s:.3f} ms\n"
        f"  fused_cublas   {ms_f:.3f} ms\n"
        f"  weight_outer   {ms_o:.3f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
