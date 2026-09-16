"""Is the MoE combine (moe_sum_reduce) paying torch.compile guard cost per hop?

progress.md §13: num_tokens per farm hop is ~4-16, so `_use_moe_sum_reduce_torch_compile`
always picks the @torch.compile path. Dynamo re-guards on every call in eager
mode, which is a large fixed per-hop cost.
"""

import time

import torch


def bench(fn, warmup=50, iters=1000):
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
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        moe_sum_reduce,
        moe_sum_reduce_torch_compile,
    )

    for rows, topk, dim in ((4, 6, 2048), (8, 6, 2048), (16, 6, 2048), (64, 6, 2048)):
        x = torch.randn(rows, topk, dim, dtype=torch.bfloat16, device=dev)
        out = torch.empty(rows, dim, dtype=torch.bfloat16, device=dev)

        tc = bench(lambda: moe_sum_reduce_torch_compile(x, out, 1.0))
        eager = bench(lambda: moe_sum_reduce(x, out, 1.0))
        print(
            f"rows={rows:3d} topk={topk} dim={dim}: "
            f"torch.compile={tc:8.1f}us  sgl_kernel={eager:7.1f}us  "
            f"ratio={tc / eager:5.1f}x"
        )


if __name__ == "__main__":
    main()
