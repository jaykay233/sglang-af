# SPDX-License-Identifier: Apache-2.0
"""Which layers run FFN remotely vs locally on the Attn worker.

Policy (fixed for true AF):

* **All layers remote FFN** — ``REMOTE_FROM_LAYER`` is always ``0`` (ignored if set).
* ``SGLANG_AFD_REMOTE_MOE_ONLY`` — dense MLP may stay on Attn (no RPC); MoE remote.
* ``SGLANG_AFD_LAYER_MERGE_K`` — K layers share one A2F/F2A (layer-group).
  Group start Attn on Decode; interior Attn+FFN on FFN worker.

Former ``REMOTE_FROM_LAYER=K>0`` (local FFN on early layers) is removed: it was
a latency cheat, not true Attn/FFN disaggregation.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_merge_env_applied = False
_remote_from_warned = False


def remote_moe_only() -> bool:
    return bool(envs.SGLANG_AFD_REMOTE_MOE_ONLY.get())


def layer_merge_k() -> int:
    """Layers per A2F/F2A round-trip; ``1`` disables merge."""
    return max(1, int(envs.SGLANG_AFD_LAYER_MERGE_K.get() or 1))


def apply_layer_merge_env() -> int:
    """When merge_k>1: force from_layer=0, disable in-graph full CG.

    Keeps breakable CG. Prefers ``NUM_MB>=2`` so FFN can gather concurrent
    slots; true Attn∥FFN layer stagger still needs multi-seq pipeline.
    """
    global _merge_env_applied
    k = layer_merge_k()
    if k <= 1:
        return 1
    envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
    envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
    # Merge interiors are FFN-local eager; stagger still via TRUE_OVERLAP.
    if int(envs.SGLANG_AFD_NUM_MB.get() or 1) < 2:
        envs.SGLANG_AFD_NUM_MB.set(2)
    if not _merge_env_applied:
        logger.info(
            "AFD LAYER_MERGE_K=%s → REMOTE_FROM_LAYER=0 IN_GRAPH=0 NUM_MB=%s "
            "(FFN-local interior KV + breakable CG)",
            k,
            envs.SGLANG_AFD_NUM_MB.get(),
        )
        _merge_env_applied = True
    return k


def remote_from_layer() -> int:
    """Always ``0``: every layer uses remote FFN (subject to MoE-only / merge)."""
    global _remote_from_warned
    raw = int(envs.SGLANG_AFD_REMOTE_FROM_LAYER.get() or 0)
    if raw != 0 and not _remote_from_warned:
        logger.warning(
            "AFD SGLANG_AFD_REMOTE_FROM_LAYER=%s ignored; true AF requires 0 "
            "(all layers remote)",
            raw,
        )
        _remote_from_warned = True
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
    return 0


def layer_id_from_weight_name(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name.replace("/", "."))
    return int(m.group(1)) if m else None


def afd_should_remote_ffn(layer_id: int, *, is_moe: Optional[bool] = None) -> bool:
    """True when Attn should call remote FFN for this layer."""
    # Interior layers of a merge group are executed entirely on FFN — Attn skips.
    k = layer_merge_k()
    if k > 1 and int(layer_id) % k != 0:
        return False
    if remote_moe_only() and is_moe is False:
        return False
    return True


def afd_attn_keeps_local_ffn(layer_id: int, *, is_moe: bool) -> bool:
    """Attn must build/load real MLP/experts for this layer.

    Merge interiors are owned entirely by FFN — Attn stubs Attn+FFN.
    """
    if afd_is_merge_interior(int(layer_id)):
        return False
    return not afd_should_remote_ffn(layer_id, is_moe=is_moe)


def afd_is_merge_group_start(layer_id: int) -> bool:
    k = layer_merge_k()
    return k <= 1 or int(layer_id) % k == 0


def afd_is_merge_interior(layer_id: int) -> bool:
    """Interior layer of a merge group (Attn+FFN run on FFN worker)."""
    k = layer_merge_k()
    return k > 1 and int(layer_id) % k != 0


def afd_merge_group_end(layer_id: int, num_layers: int) -> int:
    """Inclusive end layer id for the group starting at ``layer_id``."""
    k = layer_merge_k()
    if k <= 1:
        return int(layer_id)
    return min(int(layer_id) + k - 1, num_layers - 1)
