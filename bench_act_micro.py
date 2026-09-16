"""Microbench the MoE activation dispatch cost (progress.md §13, item #2).

Compares the JIT (tvm-ffi) elementwise activation against the sgl-kernel C++
op for the exact tensor shapes the farm hop uses, measuring CPU wall time of
the launch path (not GPU time) since the FFN stage is CPU-bound.
"""

import os
import time

import torch


def bench(fn, warmup=50, iters=2000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def main():
    dev = torch.device("cuda")
    # total_tokens = num_tokens * topk; DeepSeek-V2-Lite N = 1408 (2 * 704)
    rows, N = 128, 1408
    x = torch.randn(rows, N, dtype=torch.bfloat16, device=dev)
    out = torch.empty(rows, N // 2, dtype=torch.bfloat16, device=dev)

    from sglang.jit_kernel.activation import silu_and_mul as jit_silu

    results = {}

    results["jit silu_and_mul(x.view, out)"] = bench(
        lambda: jit_silu(x.view(-1, N), out)
    )

    x_flat = x.view(-1, N)
    results["jit silu_and_mul(preflat, out)"] = bench(lambda: jit_silu(x_flat, out))

    from sglang.jit_kernel.activation import _run_activation_inplace

    results["jit raw _run_activation_inplace"] = bench(
        lambda: _run_activation_inplace("silu", x_flat, out)
    )

    try:
        from sgl_kernel import silu_and_mul as sk_silu

        try:
            results["sgl_kernel silu_and_mul(x, out)"] = bench(lambda: sk_silu(x, out))
        except Exception as e:  # signature mismatch
            print("sgl_kernel(x, out) failed:", e)
            results["sgl_kernel silu_and_mul(out=, x=)"] = bench(
                lambda: sk_silu(out, x)
            )
    except Exception as e:
        print("sgl_kernel import failed:", e)

    # Reference: what a bare view + empty costs, to separate dispatcher floor.
    results["noop view"] = bench(lambda: x.view(-1, N))

    for k, v in results.items():
        print(f"{v:9.1f} us  {k}")


if __name__ == "__main__":
    main()
