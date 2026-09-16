# SPDX-License-Identifier: Apache-2.0
"""P3 microbatch pipeline helpers for AFD Attn↔FFN overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.afd.runtime import AfdRuntime, get_afd_runtime
from sglang.srt.afd.transport import AfdHandle
from sglang.srt.environ import envs


@dataclass
class AfdPendingTransfer:
    """In-flight A2F→F2A on a dedicated microbatch slot."""

    handle: AfdHandle
    mb_id: int
    num_tokens: int
    layer_id: int
    # When True, wait via per-mb wait_flag (issue already did write_flag).
    use_wait_flag: bool = False
    # Recorded after write_flag / push_pull so wait can run on a side stream
    # without draining later Attn work on the default stream.
    issue_event: Optional[object] = None


def split_token_ranges(num_tokens: int, num_mb: int) -> List[Tuple[int, int]]:
    """Split ``[0, num_tokens)`` into up to ``num_mb`` contiguous ranges.

    Empty trailing ranges are dropped so we never issue zero-token transfers.
    """
    if num_mb <= 1 or num_tokens <= 0:
        return [(0, num_tokens)] if num_tokens > 0 else []
    n = min(num_mb, num_tokens)
    base = num_tokens // n
    rem = num_tokens % n
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(n):
        sz = base + (1 if i < rem else 0)
        end = start + sz
        if sz > 0:
            ranges.append((start, end))
        start = end
    return ranges


def afd_pipeline_enabled() -> bool:
    return bool(envs.SGLANG_AFD_PIPELINE.get()) and int(envs.SGLANG_AFD_NUM_MB.get()) > 1


def remote_ffn_pipelined(
    *,
    layer_id: int,
    hidden: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    runtime: Optional[AfdRuntime] = None,
) -> torch.Tensor:
    """Issue A2F on multiple mb slots, then wait in order (3-stage style).

    Overlap comes from Fake/StepMesh completing earlier mbs while later
    ``push_pull`` calls are already queued. Token dim is split across mbs.
    """
    rt = runtime or get_afd_runtime()
    if rt is None:
        raise RuntimeError("AFD runtime not initialized")

    num_mb = rt.pool.cfg.num_mb
    ranges = split_token_ranges(hidden.shape[0], num_mb)
    if len(ranges) <= 1:
        return rt.remote_ffn(
            layer_id=layer_id,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            mb_id=0,
        )

    pendings: List[AfdPendingTransfer] = []
    for mb_id, (lo, hi) in enumerate(ranges):
        h = hidden[lo:hi]
        tid = topk_ids[lo:hi] if topk_ids is not None else None
        tw = topk_weights[lo:hi] if topk_weights is not None else None
        pendings.append(
            rt.remote_ffn_async(
                layer_id=layer_id,
                hidden=h,
                topk_ids=tid,
                topk_weights=tw,
                mb_id=mb_id,
            )
        )

    outs: List[torch.Tensor] = []
    for pending in pendings:
        outs.append(rt.wait_remote_ffn(pending))
    return torch.cat(outs, dim=0)
