# SPDX-License-Identifier: Apache-2.0
"""Selective weight load / memory release for AFD Attn vs FFN roles (P4/P6).

Scheme A: Attn owns gate+topk; FFN owns experts / dense MLP projections.
Scheme B: FFN also owns gate; Attn skips gate weights.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, Tuple

import torch
import torch.nn as nn

from sglang.srt.afd.mode import AfdMode, get_afd_mode
from sglang.srt.afd.routing_scheme import AfdRoutingScheme, get_routing_scheme
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _norm_name(name: str) -> str:
    return name.replace("/", ".")


def is_moe_gate_weight(name: str) -> bool:
    """MoE router gate (not dense ``gate_up_proj``)."""
    n = _norm_name(name)
    if ".mlp.gate_up_proj" in n:
        return False
    if ".mlp.gate." in n:
        return True
    if n.endswith(".mlp.gate.weight") or n.endswith(".mlp.gate.bias"):
        return True
    if ".mlp.gate.e_score" in n:
        return True
    return False


def is_ffn_exclusive_weight(name: str) -> bool:
    """Weights that always live on the FFN worker (experts / dense MLP body)."""
    n = _norm_name(name)
    if ".mlp.experts" in n or "mlp.experts." in n:
        return True
    if ".mlp.shared_experts" in n:
        return True
    # Fused and unfused dense MLP projections (ckpt may still use gate/up_proj).
    if ".mlp.gate_up_proj" in n or ".mlp.down_proj" in n:
        return True
    if ".mlp.gate_proj" in n or ".mlp.up_proj" in n:
        return True
    return False


def _afd_collocated_ffn() -> bool:
    """Fake + no module stubs: Attn process also owns FFN weights (smoke)."""
    try:
        from sglang.srt.afd.mode import get_afd_transport_name

        if get_afd_transport_name() != "fake":
            return False
        # Default MODULE_STUBS=True means real disagg intent even on Fake queues.
        return not bool(envs.SGLANG_AFD_MODULE_STUBS.get())
    except Exception:
        return False


def should_load_weight(name: str, mode: AfdMode | None = None) -> bool:
    mode = mode or get_afd_mode()
    if mode == AfdMode.NULL:
        return True
    # ATTN+fake smoke: same process owns Attn + FFN weights.
    if mode == AfdMode.ATTN and _afd_collocated_ffn():
        return True
    scheme = get_routing_scheme()
    ffn_only = is_ffn_exclusive_weight(name)
    gate = is_moe_gate_weight(name)

    if mode == AfdMode.FFN:
        if ffn_only:
            return True
        # Scheme B: FFN also loads MoE gate.
        if scheme == AfdRoutingScheme.B and gate:
            return True
        # Layer-merge interiors: FFN owns full layer (Attn + FFN weights).
        from sglang.srt.afd.remote_policy import (
            afd_is_merge_interior,
            layer_id_from_weight_name,
            layer_merge_k,
        )

        if layer_merge_k() > 1:
            lid = layer_id_from_weight_name(name)
            if lid is not None and afd_is_merge_interior(lid):
                return True
        return False

    if mode == AfdMode.ATTN:
        from sglang.srt.afd.remote_policy import (
            afd_attn_keeps_local_ffn,
            afd_is_merge_interior,
            layer_id_from_weight_name,
            remote_moe_only,
        )

        lid = layer_id_from_weight_name(name)
        # Merge interiors: skip entire layer on Attn (runs on FFN).
        if lid is not None and afd_is_merge_interior(lid):
            return False
        if ffn_only:
            if lid is not None:
                # Dense MLP weights: MoE-only mode keeps all dense local.
                is_dense_mlp = (
                    ".mlp.gate_up_proj" in _norm_name(name)
                    or ".mlp.down_proj" in _norm_name(name)
                    or ".mlp.gate_proj" in _norm_name(name)
                    or ".mlp.up_proj" in _norm_name(name)
                )
                is_moe_body = (
                    ".mlp.experts" in _norm_name(name)
                    or ".mlp.shared_experts" in _norm_name(name)
                )
                if is_dense_mlp and (
                    remote_moe_only()
                    or afd_attn_keeps_local_ffn(lid, is_moe=False)
                ):
                    return True
                if is_moe_body and afd_attn_keeps_local_ffn(lid, is_moe=True):
                    return True
            return False
        # Scheme B: gate lives on FFN — skip on Attn.
        if scheme == AfdRoutingScheme.B and gate:
            return False
        return True
    return True


def filter_weights_for_afd(
    weights: Iterable[Tuple[str, torch.Tensor]],
    mode: AfdMode | None = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    mode = mode or get_afd_mode()
    if mode == AfdMode.NULL:
        yield from weights
        return
    if mode == AfdMode.ATTN and _afd_collocated_ffn():
        yield from weights
        return

    kept = skipped = 0
    for name, tensor in weights:
        if should_load_weight(name, mode):
            kept += 1
            yield name, tensor
        else:
            skipped += 1
    logger.info(
        "AFD weight filter mode=%s scheme=%s kept=%s skipped=%s",
        mode.value,
        get_routing_scheme().value,
        kept,
        skipped,
    )


def release_unneeded_parameter_storage(
    model: nn.Module,
    mode: AfdMode | None = None,
) -> int:
    """Move unused role params off GPU to reclaim memory (opt-in)."""
    mode = mode or get_afd_mode()
    if mode == AfdMode.NULL:
        return 0
    if not envs.SGLANG_AFD_RELEASE_UNUSED_PARAMS.get():
        return 0

    moved = 0
    bytes_freed = 0
    for name, param in model.named_parameters():
        if should_load_weight(name, mode):
            continue
        if not param.is_cuda:
            continue
        bytes_freed += param.numel() * param.element_size()
        param.data = torch.empty(param.shape, dtype=param.dtype, device="cpu")
        moved += 1
    if moved:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "AFD released %s unused GPU params (~%.2f GiB) for mode=%s",
            moved,
            bytes_freed / (1024**3),
            mode.value,
        )
    return moved
