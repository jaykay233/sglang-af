# SPDX-License-Identifier: Apache-2.0
"""Contiguous decode-window slices for farm Attn (reuses TBO filter_batch)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import TOP_K_ALL


@dataclass
class FarmSlice:
    seq_lo: int
    seq_hi: int
    token_lo: int
    token_hi: int
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]
    positions: torch.Tensor
    forward_batch: ForwardBatch
    seq_idxs: List[int]


def decode_token_num_per_seq(forward_batch: ForwardBatch) -> Optional[int]:
    n_seq = int(getattr(forward_batch, "batch_size", 0) or 0)
    if n_seq <= 0:
        return None
    n_tok = int(forward_batch.input_ids.shape[0])
    if n_tok % n_seq != 0:
        return None
    return n_tok // n_seq


def _slice_optional_tensor(value, seq_lo: int, seq_hi: int):
    if value is None or not isinstance(value, torch.Tensor):
        return value
    return value[seq_lo:seq_hi]


def slice_sampling_info(
    sampling_info,
    *,
    seq_lo: int,
    seq_hi: int,
):
    """Build a non-mutating child view of ``SamplingBatchInfo``.

    ``SamplingBatchInfo.filter_batch`` mutates the parent and keeps the parent
    penalizer/grammar objects alive. Farm contexts sample independently, so
    every child gets own row slices and no shared mutable sampling state.
    """
    if sampling_info is None:
        return None

    if (
        getattr(sampling_info, "penalizer_orchestrator", None) is not None
        and getattr(sampling_info.penalizer_orchestrator, "is_required", False)
        and sampling_info.acc_additive_penalties is None
    ):
        # Materialize parent penalties once, then slice them per child. This
        # keeps child sampling independent while avoiding the parent
        # orchestrator being applied to only the parent's full row range.
        sampling_info.update_penalties()

    child = dataclasses.replace(sampling_info)
    idx = torch.arange(
        int(seq_lo),
        int(seq_hi),
        dtype=torch.long,
        device=sampling_info.temperatures.device,
    )
    for name in (
        "temperatures",
        "top_ps",
        "top_ks",
        "min_ps",
        "sampling_seed",
        "acc_additive_penalties",
        "acc_scaling_penalties",
        "logit_bias",
        "rids_int",
        "bootstrap_room_ids_int",
    ):
        setattr(
            child,
            name,
            _slice_optional_tensor(getattr(child, name), seq_lo, seq_hi),
        )

    grammars = getattr(child, "grammars", None)
    child.grammars = None if grammars is None else list(grammars[seq_lo:seq_hi])
    child.grammar_mask = None

    custom_params = getattr(child, "custom_params", None)
    child.custom_params = (
        None if custom_params is None else list(custom_params[seq_lo:seq_hi])
    )
    custom = getattr(child, "custom_logit_processor", None)
    if custom:
        sliced: Dict[int, Tuple[Any, torch.Tensor]] = {}
        for key, (processor, mask) in custom.items():
            child_mask = mask[seq_lo:seq_hi]
            if bool(torch.any(child_mask).item()):
                sliced[key] = (processor, child_mask)
        child.custom_logit_processor = sliced or None
        child.has_custom_logit_processor = bool(sliced)
    else:
        child.custom_logit_processor = None
        child.has_custom_logit_processor = False

    masks = getattr(child, "return_sampling_masks", None)
    child.return_sampling_masks = (
        None if masks is None else list(masks[seq_lo:seq_hi])
    )

    top_ks = child.top_ks
    top_ps = child.top_ps
    min_ps = child.min_ps
    child.is_all_greedy = bool(torch.all(top_ks <= 1).item())
    child.is_any_greedy = bool(torch.any(top_ks <= 1).item())
    child.need_top_p_sampling = bool(torch.any(top_ps != 1.0).item())
    child.need_top_k_sampling = bool(torch.any(top_ks != TOP_K_ALL).item())
    child.need_min_p_sampling = bool(torch.any(min_ps > 0).item())

    # The parent orchestrator operates on parent rows only. Child penalties
    # were materialized into the sliced accumulator tensors above.
    child.penalizer_orchestrator = None
    return child


def slice_decode_window(
    *,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    seq_idxs: Sequence[int],
    token_num_per_seq: int,
    with_sampling_info: bool = True,
) -> Optional[FarmSlice]:
    """Slice a *contiguous* seq run ``[seq_lo, seq_hi)`` out of a decode batch.

    ``with_sampling_info`` may be set to ``False`` for intermediate farm hops.
    ``filter_batch`` already yields a child with ``sampling_info=None``; only the
    hop that finishes a sequence is ever handed to the sampler (see
    ``_on_output_ready``), so building a sliced ``SamplingBatchInfo`` for the
    other 26 layers is dead work on the attn critical path.
    """
    if not seq_idxs:
        return None
    seq_lo = int(seq_idxs[0])
    seq_hi = int(seq_idxs[-1]) + 1
    if seq_hi - seq_lo != len(seq_idxs):
        return None
    if any(int(seq_idxs[i]) != seq_lo + i for i in range(len(seq_idxs))):
        return None

    tps = max(1, int(token_num_per_seq))
    t0 = seq_lo * tps
    t1 = seq_hi * tps
    n_tok = int(hidden_states.shape[0])
    if t0 < 0 or t1 > n_tok or t1 <= t0:
        return None

    from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer

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
        start_seq_index=seq_lo,
        end_seq_index=seq_hi,
        out_num_token_non_padded=out_non_padded,
    )
    child.sampling_info = (
        slice_sampling_info(
            getattr(forward_batch, "sampling_info", None),
            seq_lo=seq_lo,
            seq_hi=seq_hi,
        )
        if with_sampling_info
        else None
    )
    if child.sampling_info is not None:
        child.temperature = child.sampling_info.temperatures
        child.top_p = child.sampling_info.top_ps

    top_logprobs_nums = getattr(forward_batch, "top_logprobs_nums", None)
    child.top_logprobs_nums = (
        None if top_logprobs_nums is None else top_logprobs_nums[seq_lo:seq_hi]
    )
    token_ids_logprobs = getattr(forward_batch, "token_ids_logprobs", None)
    child.token_ids_logprobs = (
        None
        if token_ids_logprobs is None
        else token_ids_logprobs[seq_lo:seq_hi]
    )

    logits_buffer = getattr(forward_batch, "next_token_logits_buffer", None)
    if (
        isinstance(logits_buffer, torch.Tensor)
        and logits_buffer.shape[0] == int(forward_batch.batch_size)
    ):
        child.next_token_logits_buffer = logits_buffer[seq_lo:seq_hi]
    child.return_hidden_states_before_norm = bool(
        getattr(forward_batch, "return_hidden_states_before_norm", False)
    )
    child._afd_farm_parent_seq_idxs = tuple(int(x) for x in seq_idxs)
    child._afd_farm_parent_token_range = (t0, t1)
    res = residual[t0:t1] if residual is not None else None
    pos = positions[t0:t1] if positions.shape[0] == n_tok else positions
    return FarmSlice(
        seq_lo=seq_lo,
        seq_hi=seq_hi,
        token_lo=t0,
        token_hi=t1,
        hidden_states=hidden_states[t0:t1],
        residual=res,
        positions=pos,
        forward_batch=child,
        seq_idxs=[int(x) for x in seq_idxs],
    )


def scatter_rows(
    dst: torch.Tensor,
    src: torch.Tensor,
    token_lo: int,
    token_hi: int,
) -> None:
    n = token_hi - token_lo
    if n <= 0:
        return
    dst[token_lo:token_hi].copy_(src[:n])
