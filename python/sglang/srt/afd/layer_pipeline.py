# SPDX-License-Identifier: Apache-2.0
"""P7 staggered multi-microbatch Attn↔FFN layer pipeline (StepMesh stages).

Recommended for Lite overlap: ``NUM_MB=2`` / ``SGLANG_AFD_STEPMESH_STAGES=2``.
StepMesh docs often use 3 stages; both work. Schedule::

    for layer L:
        for stage/mb in 0..S-1:
            if pending[mb]: recv FFN(mb); post_ffn(mb)   # like recv_ffn_output
            Attn+prep(mb, L); issue FFN(mb, L)           # send_attn_output
    for mb: drain recv

With S=2, FFN(mb0,L) overlaps Attn(mb1,L) (and FFN(mb1,L) overlaps Attn(mb0,L+1)).
Requires decode batch with enough sequences to split (else sequential fallback).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from sglang.srt.afd.attn_bridge import issue_remote_ffn, wait_remote_ffn
from sglang.srt.afd.pipeline import AfdPendingTransfer
from sglang.srt.afd.runtime import get_afd_runtime, is_afd_attn
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass
class AfdMbSlice:
    """One microbatch's tensors + sliced ForwardBatch."""

    mb_id: int
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]
    positions: torch.Tensor
    forward_batch: ForwardBatch
    token_lo: int
    token_hi: int


_stages_env_applied = False
_true_overlap_env_applied = False


def apply_true_overlap_env() -> bool:
    """Force TRUE_OVERLAP wiring: dual-mb layer pipe + breakable CG.

    Non-TRUE_OVERLAP (per-layer sync issue+wait / in-graph full CG) is removed
    as a supported Attn decode mode. Prefill still falls back to per-layer
    ``maybe_remote_ffn`` when the batch cannot be split into mbs.

    Decode farm (``SGLANG_AFD_FARM=1``) keeps TRUE_OVERLAP overlap hops but
    turns **off** lockstep layer-pipeline.

    Safe to call repeatedly. Always returns True.
    """
    global _true_overlap_env_applied
    if not bool(envs.SGLANG_AFD_TRUE_OVERLAP.get()):
        logger.warning(
            "AFD SGLANG_AFD_TRUE_OVERLAP=0 ignored; true AF decode requires "
            "TRUE_OVERLAP (dual-mb Attn∥FFN)"
        )
    envs.SGLANG_AFD_TRUE_OVERLAP.set(True)
    try:
        from sglang.srt.afd.farm.env import apply_farm_env, farm_enabled

        if farm_enabled():
            apply_farm_env()
            from sglang.srt.afd.remote_policy import remote_from_layer

            remote_from_layer()
            envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
            return True
    except Exception:
        pass
    envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
    envs.SGLANG_AFD_PIPELINE.set(False)
    envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
    # REMOTE_FROM_LAYER>0 is not true AF — always force 0.
    from sglang.srt.afd.remote_policy import remote_from_layer

    remote_from_layer()  # warn + clamp
    envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
    if int(envs.SGLANG_AFD_NUM_MB.get()) < 2:
        envs.SGLANG_AFD_NUM_MB.set(2)
    # Prefer dual-mb stagger; do not bump to 3 unless user set stages/num_mb.
    stages = int(envs.SGLANG_AFD_STEPMESH_STAGES.get() or 0)
    if stages < 2:
        envs.SGLANG_AFD_STEPMESH_STAGES.set(int(envs.SGLANG_AFD_NUM_MB.get()))
    if not _true_overlap_env_applied:
        logger.info(
            "AFD TRUE_OVERLAP (required) → LAYER_PIPELINE=1 NUM_MB=%s "
            "IN_GRAPH_WAIT=0 REMOTE_FROM_LAYER=0 "
            "(breakable CG + dual-mb mailbox overlap)",
            envs.SGLANG_AFD_NUM_MB.get(),
        )
        _true_overlap_env_applied = True
    return True


def apply_stepmesh_stages_env() -> int:
    """If ``SGLANG_AFD_STEPMESH_STAGES>=2``, wire LAYER_PIPELINE + NUM_MB.

    Returns the effective stage count (0 if unset). Safe to call repeatedly.
    """
    global _stages_env_applied
    apply_true_overlap_env()
    try:
        from sglang.srt.afd.farm.env import farm_enabled

        if farm_enabled():
            return 0
    except Exception:
        pass
    stages = int(envs.SGLANG_AFD_STEPMESH_STAGES.get() or 0)
    if stages < 2:
        return 0
    stages = min(stages, 8)
    envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
    envs.SGLANG_AFD_PIPELINE.set(False)
    envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
    if int(envs.SGLANG_AFD_NUM_MB.get()) < stages:
        envs.SGLANG_AFD_NUM_MB.set(stages)
    if not _stages_env_applied:
        logger.info(
            "AFD StepMesh stages=%s → LAYER_PIPELINE=1 NUM_MB=%s IN_GRAPH_WAIT=0",
            stages,
            envs.SGLANG_AFD_NUM_MB.get(),
        )
        _stages_env_applied = True
    return stages


def stepmesh_stage_count() -> int:
    """Effective number of AFD stages (2–3 typical); 0 if layer pipe off."""
    if not bool(envs.SGLANG_AFD_LAYER_PIPELINE.get()) and int(
        envs.SGLANG_AFD_STEPMESH_STAGES.get() or 0
    ) < 2:
        return 0
    stages = int(envs.SGLANG_AFD_STEPMESH_STAGES.get() or 0)
    num_mb = int(envs.SGLANG_AFD_NUM_MB.get())
    if stages >= 2:
        return max(2, min(stages, num_mb if num_mb >= 2 else stages, 8))
    if bool(envs.SGLANG_AFD_LAYER_PIPELINE.get()):
        # Dual-mb (2) is the overlap sweet spot for Lite; 3 only if explicitly set.
        return max(2, min(num_mb if num_mb >= 2 else 2, 8))
    return 0


def afd_layer_pipeline_enabled() -> bool:
    """True when P7 / StepMesh staggered layer pipeline should run."""
    try:
        from sglang.srt.afd.farm.env import farm_enabled

        if farm_enabled():
            return False
    except Exception:
        pass
    # AfPool MxN uses per-hop router credit; dual-mb layer pipe bypasses it.
    try:
        from sglang.srt.afd.pool.topology import pool_enabled

        if pool_enabled():
            return False
    except Exception:
        pass
    stages = int(envs.SGLANG_AFD_STEPMESH_STAGES.get() or 0)
    if stages >= 2:
        # Env already implies layer pipe; avoid re-entrant apply on hot path.
        pass
    elif not bool(envs.SGLANG_AFD_LAYER_PIPELINE.get()):
        return False
    # In-graph wait uses a single full CUDA Graph; incompatible with mb stagger.
    if bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get()):
        return False
    if not is_afd_attn():
        return False
    rt = get_afd_runtime()
    if rt is None or rt.num_mb < 2:
        return False
    # Token-pipeline (P3) inside one break conflicts with per-mb issue slots.
    if bool(envs.SGLANG_AFD_PIPELINE.get()):
        return False
    return bool(envs.SGLANG_AFD_LAYER_PIPELINE.get()) or stages >= 2


def _decode_token_num_per_seq(forward_batch: ForwardBatch) -> Optional[int]:
    if forward_batch.batch_size <= 0:
        return None
    n_tok = int(forward_batch.input_ids.shape[0])
    n_seq = int(forward_batch.batch_size)
    if n_tok % n_seq != 0:
        return None
    return n_tok // n_seq


def _balanced_seq_ranges(
    n_seq: int, num_mb: int, token_num_per_seq: int
) -> List[Tuple[int, int, int, int]]:
    """Return ``(seq0, seq1, tok0, tok1)`` ranges covering ``[0, n_seq)``."""
    ranges: List[Tuple[int, int, int, int]] = []
    for i in range(num_mb):
        s0 = (n_seq * i) // num_mb
        s1 = (n_seq * (i + 1)) // num_mb
        if s1 <= s0:
            continue
        t0 = s0 * token_num_per_seq
        t1 = s1 * token_num_per_seq
        ranges.append((s0, s1, t0, t1))
    return ranges


def split_decode_mbs(
    *,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    num_mb: int = 2,
) -> Optional[List[AfdMbSlice]]:
    """Split a decode ForwardBatch into ``num_mb`` contiguous seq ranges.

    Returns ``None`` when the batch cannot be split (caller must fall back).
    """
    if num_mb < 2:
        return None
    if not forward_batch.forward_mode.is_decode():
        return None

    from sglang.srt.batch_overlap.two_batch_overlap import (
        TboForwardBatchPreparer,
        get_token_num_per_seq,
    )

    token_num_per_seq = _decode_token_num_per_seq(forward_batch)
    if token_num_per_seq is None:
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=forward_batch.forward_mode,
            spec_info=getattr(forward_batch, "spec_info", None),
        )
    if token_num_per_seq is None or token_num_per_seq < 0:
        return None

    n_tok = int(hidden_states.shape[0])
    n_seq = int(forward_batch.batch_size)
    if n_seq < num_mb or n_tok < num_mb:
        return None

    num_mb = min(int(num_mb), n_seq)
    ranges = _balanced_seq_ranges(n_seq, num_mb, token_num_per_seq)
    if len(ranges) < 2:
        return None

    slices: List[AfdMbSlice] = []
    try:
        for mb_id, (s0, s1, t0, t1) in enumerate(ranges):
            # CG-safe: avoid torch.tensor(..., device=cuda) host→device copy.
            if forward_batch.num_token_non_padded is not None:
                out_non_padded = forward_batch.num_token_non_padded.new_empty(())
                out_non_padded.fill_(t1 - t0)
            else:
                out_non_padded = torch.empty(
                    (), dtype=torch.int32, device=hidden_states.device
                )
                out_non_padded.fill_(t1 - t0)
            child = TboForwardBatchPreparer.filter_batch(
                forward_batch,
                start_token_index=t0,
                end_token_index=t1,
                start_seq_index=s0,
                end_seq_index=s1,
                out_num_token_non_padded=out_non_padded,
            )
            res = residual[t0:t1] if residual is not None else None
            pos = positions[t0:t1] if positions.shape[0] == n_tok else positions
            slices.append(
                AfdMbSlice(
                    mb_id=mb_id,
                    hidden_states=hidden_states[t0:t1],
                    residual=res,
                    positions=pos,
                    forward_batch=child,
                    token_lo=t0,
                    token_hi=t1,
                )
            )
    except Exception as e:
        logger.warning(
            "AFD layer pipeline: ForwardBatch split failed (%s); falling back", e
        )
        return None

    if any(s.hidden_states.shape[0] == 0 for s in slices):
        return None
    return slices


def _merge_mb_states(
    slices: Sequence[AfdMbSlice],
    hidden_list: Sequence[torch.Tensor],
    residual_list: Sequence[Optional[torch.Tensor]],
    topk_list: Sequence[Optional[torch.Tensor]],
    original_num_tokens: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    hidden = torch.cat(list(hidden_list), dim=0)
    assert hidden.shape[0] == original_num_tokens

    residual: Optional[torch.Tensor] = None
    if residual_list[0] is not None:
        residual = torch.cat([r for r in residual_list if r is not None], dim=0)

    topk_indices: Optional[torch.Tensor] = None
    if topk_list[0] is not None:
        topk_indices = torch.cat([t for t in topk_list if t is not None], dim=0)

    return hidden, residual, topk_indices


def run_layers_pipelined(
    layers: Sequence[Any],
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
    zero_allocator,
    gemm_output_zero_allocator=None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    layers_to_capture: Optional[Sequence[int]] = None,
    aux_hidden_states: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run ``layers`` with staggered multi-mb remote FFN overlap.

    Returns ``(hidden_states, residual, topk_indices)`` matching a sequential
    layer loop. Falls back to sequential ``layer(...)`` when split fails.
    """
    rt = get_afd_runtime()
    want_mb = stepmesh_stage_count()
    if want_mb < 2:
        # Default dual-mb overlap when layer pipe is on but stages unset.
        n = 2 if rt is None else int(rt.num_mb)
        want_mb = 2 if n < 2 else min(n, 2)
    mb_slices = split_decode_mbs(
        hidden_states=hidden_states,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
        num_mb=want_mb,
    )
    if mb_slices is None:
        topk_indices = None
        for layer in layers:
            capture = None
            if (
                layers_to_capture is not None
                and aux_hidden_states is not None
                and layer.layer_id in layers_to_capture
            ):
                capture = aux_hidden_states
            hidden_states, residual, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
                gemm_output_zero_allocator,
                llama_4_scaling,
                prev_topk_indices=topk_indices,
                captured_last_layer_outputs=capture,
            )
        return hidden_states, residual, topk_indices

    logger.debug(
        "AFD StepMesh layer pipeline num_mb=%s tokens=%s seqs=%s",
        len(mb_slices),
        hidden_states.shape[0],
        int(getattr(forward_batch, "batch_size", len(mb_slices)) or len(mb_slices)),
    )

    original_n = int(hidden_states.shape[0])
    hiddens = [s.hidden_states for s in mb_slices]
    residuals = [s.residual for s in mb_slices]
    positions_mb = [s.positions for s in mb_slices]
    batches = [s.forward_batch for s in mb_slices]
    topk_prev: List[Optional[torch.Tensor]] = [None] * len(mb_slices)
    pending: List[Optional[AfdPendingTransfer]] = [None] * len(mb_slices)
    metas: List[Optional[dict]] = [None] * len(mb_slices)

    layer_list = list(layers)
    from sglang.srt.afd.remote_policy import afd_should_remote_ffn
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    for li, layer in enumerate(layer_list):
        is_moe = isinstance(getattr(layer, "mlp", None), DeepseekV2MoE)
        remote = afd_should_remote_ffn(int(layer.layer_id), is_moe=is_moe)
        for mb_id in range(len(mb_slices)):
            if pending[mb_id] is not None:
                # Consume F2A before next issue on this mb — no clone needed.
                mlp_out = wait_remote_ffn(pending[mb_id], clone=False)
                hiddens[mb_id], residuals[mb_id], topk_prev[mb_id] = (
                    layer_list[li - 1].forward_post_ffn(
                        mlp_out, residuals[mb_id], metas[mb_id]
                    )
                )
                pending[mb_id] = None
                metas[mb_id] = None

            capture = None
            if (
                layers_to_capture is not None
                and aux_hidden_states is not None
                and layer.layer_id in layers_to_capture
            ):
                capture = aux_hidden_states if mb_id == 0 else None

            (
                hidden_a2f,
                residuals[mb_id],
                topk_ids,
                topk_weights,
                meta,
            ) = layer.forward_pre_ffn(
                positions=positions_mb[mb_id],
                hidden_states=hiddens[mb_id],
                forward_batch=batches[mb_id],
                residual=residuals[mb_id],
                zero_allocator=zero_allocator,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=topk_prev[mb_id],
                captured_last_layer_outputs=capture,
            )
            if not remote:
                # Local FFN only when REMOTE_MOE_ONLY keeps dense MLP on Attn.
                fb = meta["forward_batch"]
                mlp_out = layer.mlp(
                    hidden_a2f,
                    fb,
                    meta["should_allreduce_fusion"],
                    meta["use_reduce_scatter"],
                    meta["gemm_output_zero_allocator"],
                )
                hiddens[mb_id], residuals[mb_id], topk_prev[mb_id] = (
                    layer.forward_post_ffn(mlp_out, residuals[mb_id], meta)
                )
                continue

            metas[mb_id] = meta
            pending[mb_id] = issue_remote_ffn(
                layer_id=layer.layer_id,
                hidden_states=hidden_a2f,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                mb_id=mb_id,
            )

    last_layer = layer_list[-1]
    for mb_id in range(len(mb_slices)):
        if pending[mb_id] is None:
            continue
        assert metas[mb_id] is not None
        mlp_out = wait_remote_ffn(pending[mb_id], clone=False)
        hiddens[mb_id], residuals[mb_id], topk_prev[mb_id] = last_layer.forward_post_ffn(
            mlp_out, residuals[mb_id], metas[mb_id]
        )
        pending[mb_id] = None

    return _merge_mb_states(
        mb_slices, hiddens, residuals, topk_prev, original_n
    )
