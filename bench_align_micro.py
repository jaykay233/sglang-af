"""Microbench moe_align_block_size in isolation (progress.md §13, moe_align_us).

298us/hop inside the farm vs a handful of tiny ops in isolation: the gap tells
us whether the in-situ figure is real CPU work or GPU/launch back-pressure.
"""

import time

import torch


def bench(fn, warmup=200, iters=3000):
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
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    out = {}
    for rows, topk, E, block in ((16, 6, 64, 64), (96, 6, 64, 64), (128, 8, 64, 64)):
        topk_ids = torch.randint(0, E, (rows, topk), dtype=torch.int32, device=dev)
        out[f"moe_align rows={rows} topk={topk} E={E} blk={block}"] = bench(
            lambda: moe_align_block_size(topk_ids, block, E)
        )

    # Floor: what just the 4 allocations + one trivial op cost.
    def allocs():
        torch.empty((100000,), dtype=torch.int32, device=dev)
        torch.empty((200,), dtype=torch.int32, device=dev)
        torch.empty((1,), dtype=torch.int32, device=dev)
        torch.empty((E + 2,), dtype=torch.int32, device=dev)

    E = 64
    out["4x torch.empty (floor)"] = bench(allocs)

    for k, v in out.items():
        print(f"{v:9.1f} us  {k}")


if __name__ == "__main__":
    main()
