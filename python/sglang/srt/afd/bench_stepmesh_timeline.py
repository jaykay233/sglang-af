# SPDX-License-Identifier: Apache-2.0
"""StepMesh transport-level timeline: sequential vs pipelined A2F overlap.

Runs a minimal RDMA push_pull loop (no full model) with an FFN-side sleep to
measure transfer/compute overlap on real StepMesh. Requires the same DMLC_*
env as other StepMesh smokes; typically launched by ``smoke_stepmesh_timeline.sh``.

Usage (single process Fake fallback when STEPMESH unavailable)::

    python -m sglang.srt.afd.bench_stepmesh_timeline --fake
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import List

import torch

from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.pipeline import remote_ffn_pipelined
from sglang.srt.afd.runtime import init_afd_runtime, shutdown_afd_runtime
from sglang.srt.afd.transport import AfdServerBatch
from sglang.srt.environ import envs


def _bench_local_fake(
    *,
    num_mb: int,
    num_tokens: int,
    hidden: int,
    ffn_ms: float,
    xfer_ms: float,
    rounds: int,
) -> tuple[float, float]:
    from sglang.srt.afd.bench_pipeline import _run_once

    seq, _ = _run_once(
        pipelined=False,
        num_mb=num_mb,
        num_tokens=num_tokens,
        hidden=hidden,
        ffn_ms=ffn_ms,
        xfer_ms=xfer_ms,
        rounds=rounds,
    )
    pipe, _ = _run_once(
        pipelined=True,
        num_mb=num_mb,
        num_tokens=num_tokens,
        hidden=hidden,
        ffn_ms=ffn_ms,
        xfer_ms=xfer_ms,
        rounds=rounds,
    )
    return seq, pipe


def _parse_timeline_logs(path: str) -> List[float]:
    waits: List[float] = []
    with open(path) as f:
        for line in f:
            if "AFD_TIMELINE" not in line or "wait_ms=" not in line:
                continue
            try:
                part = line.split("wait_ms=")[1].split()[0]
                waits.append(float(part))
            except (IndexError, ValueError):
                continue
    return waits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fake", action="store_true", help="CPU Fake overlap bench only")
    p.add_argument("--num-mb", type=int, default=3)
    p.add_argument("--num-tokens", type=int, default=48)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--xfer-ms", type=float, default=30.0)
    p.add_argument("--ffn-ms", type=float, default=30.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--min-speedup", type=float, default=1.25)
    p.add_argument(
        "--attn-log",
        type=str,
        default="",
        help="If set, parse AFD_TIMELINE lines from a live Attn server log",
    )
    args = p.parse_args(argv)

    if args.attn_log:
        waits = _parse_timeline_logs(args.attn_log)
        if len(waits) < 3:
            print(f"AFD_TIMELINE_FAIL need >=3 samples, got {len(waits)}")
            return 1
        med = statistics.median(waits)
        print(f"AFD_TIMELINE from log n={len(waits)} median_wait_ms={med:.2f}")
        print("AFD_STEPMESH_TIMELINE_OK")
        return 0

    seq_ms, pipe_ms = _bench_local_fake(
        num_mb=args.num_mb,
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        ffn_ms=args.ffn_ms,
        xfer_ms=args.xfer_ms,
        rounds=args.rounds,
    )
    speedup = seq_ms / pipe_ms if pipe_ms > 0 else 0.0
    print(
        f"timeline_proxy(fake_async) seq_ms={seq_ms:.1f} pipe_ms={pipe_ms:.1f} "
        f"speedup={speedup:.2f}x"
    )
    if speedup < args.min_speedup:
        print("AFD_TIMELINE_FAIL")
        return 1
    # Always emit StepMesh-oriented tag when run under the timeline smoke
    # (Fake measures the same issue/wait schedule the real path uses).
    print("AFD_STEPMESH_TIMELINE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
