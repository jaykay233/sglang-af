# SPDX-License-Identifier: Apache-2.0
"""P3 timeline bench: sequential vs pipelined A2F/FFN/F2A (Fake async).

Models Step-3 style overlap: simulated transfer latency runs concurrently for
multiple microbatches while a single FFN thread drains compute serially.

  sequential wall ≈ N_mb * (T_xfer + T_ffn)
  pipelined  wall ≈ T_xfer + N_mb * T_ffn   (transfers overlap)

Usage::

    python -m sglang.srt.afd.bench_pipeline
    python -m sglang.srt.afd.bench_pipeline --xfer-ms 25 --ffn-ms 25 --num-mb 3
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Tuple

import torch

from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.pipeline import remote_ffn_pipelined
from sglang.srt.afd.runtime import init_afd_runtime, shutdown_afd_runtime
from sglang.srt.afd.transport import AfdServerBatch
from sglang.srt.environ import envs


def _run_once(
    *,
    pipelined: bool,
    num_mb: int,
    num_tokens: int,
    hidden: int,
    ffn_ms: float,
    xfer_ms: float,
    rounds: int,
) -> Tuple[float, List[float]]:
    shutdown_afd_runtime()
    envs.SGLANG_AFD_MODE.set("attn")
    envs.SGLANG_AFD_TRANSPORT.set("fake")
    envs.SGLANG_AFD_NUM_MB.set(num_mb if pipelined else 1)
    envs.SGLANG_AFD_MAX_NUM_TOKEN.set(max(num_tokens, 64))
    envs.SGLANG_AFD_PIPELINE.set(pipelined)
    envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(True)
    envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(xfer_ms)

    def ffn(batch: AfdServerBatch):
        time.sleep(ffn_ms / 1000.0)
        t = batch.num_tokens
        out = batch.hidden.clone()
        out[:t] = batch.hidden[:t] + 1
        return [out]

    rt = init_afd_runtime(
        hidden_size=hidden,
        mode=AfdMode.ATTN,
        transport_name="fake",
        device="cpu",
        dtype=torch.float32,
        ffn_compute=ffn,
        moe_topk=0,
    )
    assert rt is not None
    x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    # Same token split as pipeline (sequential issues one mb at a time).
    ranges = []
    n = min(num_mb, num_tokens)
    base, rem = num_tokens // n, num_tokens % n
    s = 0
    for i in range(n):
        e = s + base + (1 if i < rem else 0)
        ranges.append((s, e))
        s = e
    # Warmup
    if pipelined:
        remote_ffn_pipelined(layer_id=0, hidden=x, runtime=rt)
    else:
        for _mb, (lo, hi) in enumerate(ranges):
            rt.remote_ffn(layer_id=0, hidden=x[lo:hi], mb_id=0)

    walls: List[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        if pipelined:
            y = remote_ffn_pipelined(layer_id=0, hidden=x, runtime=rt)
        else:
            outs = []
            for mb, (lo, hi) in enumerate(ranges):
                outs.append(rt.remote_ffn(layer_id=0, hidden=x[lo:hi], mb_id=0))
            y = torch.cat(outs, dim=0)
        walls.append((time.perf_counter() - t0) * 1000.0)
        if not torch.allclose(y, x + 1):
            raise AssertionError("parity failed in bench_pipeline")
    shutdown_afd_runtime()
    return statistics.median(walls), walls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AFD 3-stage pipeline overlap bench")
    p.add_argument("--num-mb", type=int, default=3)
    p.add_argument("--num-tokens", type=int, default=48)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--xfer-ms", type=float, default=30.0)
    p.add_argument("--ffn-ms", type=float, default=30.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument(
        "--min-speedup",
        type=float,
        default=1.25,
        help="Require seq_ms / pipe_ms >= this (default 1.25)",
    )
    args = p.parse_args(argv)

    n = args.num_mb
    t, c = args.xfer_ms, args.ffn_ms
    # Ideal: seq = n*(T+C), pipe = T + n*C
    ideal_seq = n * (t + c)
    ideal_pipe = t + n * c
    ideal_speedup = ideal_seq / ideal_pipe if ideal_pipe > 0 else 1.0

    seq_ms, _ = _run_once(
        pipelined=False,
        num_mb=n,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        ffn_ms=c,
        xfer_ms=t,
        rounds=args.rounds,
    )
    pipe_ms, _ = _run_once(
        pipelined=True,
        num_mb=n,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        ffn_ms=c,
        xfer_ms=t,
        rounds=args.rounds,
    )
    speedup = seq_ms / pipe_ms if pipe_ms > 0 else 0.0

    print(
        f"AFD pipeline bench num_mb={n} tokens={args.num_tokens} "
        f"xfer_ms={t} ffn_ms={c} rounds={args.rounds}"
    )
    print(f"  ideal_seq_ms={ideal_seq:.1f} ideal_pipe_ms={ideal_pipe:.1f} "
          f"ideal_speedup={ideal_speedup:.2f}x")
    print(f"  measured_seq_ms={seq_ms:.1f} measured_pipe_ms={pipe_ms:.1f} "
          f"speedup={speedup:.2f}x")

    if speedup < args.min_speedup:
        print(
            f"AFD_PIPELINE_BENCH_FAIL speedup {speedup:.2f} < {args.min_speedup}"
        )
        return 1
    print("AFD_PIPELINE_BENCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
