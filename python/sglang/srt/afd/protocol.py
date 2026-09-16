# SPDX-License-Identifier: Apache-2.0
"""A2F / F2A tensor contract and StepMesh-compatible key packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


# Slot ids (private_key) for A2F / F2A — see RFC §5.
SLOT_HIDDEN = 0
SLOT_NUM_TOKENS = 1
SLOT_LAYER_ID = 2
SLOT_TOPK_IDS = 3
SLOT_TOPK_WEIGHTS = 4
SLOT_HIDDEN_SCALE = 5  # present when A2F wire dtype is fp8
SLOT_RESIDUAL = 6  # layer-merge: residual stream [T, H]
SLOT_POSITIONS = 7  # layer-merge: positions [T]
SLOT_MLP_OUT = 0  # F2A
SLOT_F2A_RESIDUAL = 1  # layer-merge F2A residual


def pack_layer_merge_meta(layer_id: int, merge_k: int) -> int:
    """Pack start layer + merge_k into the int32 layer_id slot."""
    k = max(1, int(merge_k))
    assert 0 <= int(layer_id) < 65536
    assert 1 <= k < 256
    return int(layer_id) | (k << 16)


def unpack_layer_merge_meta(packed: int) -> Tuple[int, int]:
    p = int(packed)
    layer_id = p & 0xFFFF
    merge_k = (p >> 16) & 0xFF
    if merge_k <= 0:
        merge_k = 1
    return layer_id, merge_k


def pack_key(
    private_key: int,
    *,
    microbatch: int = 0,
    worker_rank: int = 0,
    is_pull: bool = False,
) -> int:
    """Pack a StepMesh-style 32-bit key (matches StepMesh test helpers)."""
    assert 0 <= private_key < 256
    assert 0 <= microbatch < 256
    assert 0 <= worker_rank < 256
    key = private_key + (microbatch << 8) + (worker_rank << 16)
    if is_pull:
        key += 1 << 24
    return key


def gen_push_key(private_key: int, microbatch: int = 0, worker_rank: int = 0) -> int:
    return pack_key(
        private_key, microbatch=microbatch, worker_rank=worker_rank, is_pull=False
    )


def gen_pull_key(private_key: int, microbatch: int = 0, worker_rank: int = 0) -> int:
    return pack_key(
        private_key, microbatch=microbatch, worker_rank=worker_rank, is_pull=True
    )


@dataclass
class AfdA2FPayload:
    """Attn → FFN activation bundle (logical view over registered buffers)."""

    hidden: torch.Tensor  # [T, H] (may be fp8 when quantized)
    num_tokens: torch.Tensor  # [1] int32
    layer_id: torch.Tensor  # [1] int32
    topk_ids: Optional[torch.Tensor] = None  # [T, K]
    topk_weights: Optional[torch.Tensor] = None  # [T, K]
    hidden_scale: Optional[torch.Tensor] = None  # [1] fp32 when fp8 A2F
    residual: Optional[torch.Tensor] = None  # [T, H] layer-merge
    positions: Optional[torch.Tensor] = None  # [T] int64 layer-merge

    def as_tensor_list(self) -> List[torch.Tensor]:
        out = [self.hidden, self.num_tokens, self.layer_id]
        if self.topk_ids is not None:
            out.append(self.topk_ids)
        if self.topk_weights is not None:
            out.append(self.topk_weights)
        if self.hidden_scale is not None:
            out.append(self.hidden_scale)
        if self.residual is not None:
            out.append(self.residual)
        if self.positions is not None:
            out.append(self.positions)
        return out


@dataclass
class AfdF2APayload:
    mlp_out: torch.Tensor  # [T, H]
    residual: Optional[torch.Tensor] = None  # layer-merge residual out

    def as_tensor_list(self) -> List[torch.Tensor]:
        out = [self.mlp_out]
        if self.residual is not None:
            out.append(self.residual)
        return out


@dataclass
class AfdServerBatch:
    """One worker's request as seen by an FFN server."""

    handler: int
    worker_rank: int
    tensors: Sequence[torch.Tensor]
    keys: Sequence[int]
    compute_dtype: torch.dtype = torch.bfloat16
    host_num_tokens: Optional[int] = None
    host_layer_id: Optional[int] = None

    @property
    def hidden_wire(self) -> torch.Tensor:
        return self.tensors[SLOT_HIDDEN]

    @property
    def hidden(self) -> torch.Tensor:
        from sglang.srt.afd.a2f_quant import dequantize_hidden_from_a2f

        return dequantize_hidden_from_a2f(
            self.hidden_wire, self.hidden_scale, self.compute_dtype
        )

    @property
    def num_tokens(self) -> int:
        if self.host_num_tokens is not None and self.host_num_tokens >= 0:
            return int(self.host_num_tokens)
        return int(self.tensors[SLOT_NUM_TOKENS].item())

    def _packed_layer_meta(self) -> int:
        if self.host_layer_id is not None and self.host_layer_id >= 0:
            return int(self.host_layer_id)
        return int(self.tensors[SLOT_LAYER_ID].item())

    @property
    def layer_id(self) -> int:
        lid, _ = unpack_layer_merge_meta(self._packed_layer_meta())
        return lid

    @property
    def merge_k(self) -> int:
        _, k = unpack_layer_merge_meta(self._packed_layer_meta())
        return k

    @property
    def topk_ids(self) -> Optional[torch.Tensor]:
        if len(self.tensors) >= 5 and self.tensors[SLOT_TOPK_IDS].dim() == 2:
            return self.tensors[SLOT_TOPK_IDS]
        return None

    @property
    def topk_weights(self) -> Optional[torch.Tensor]:
        if len(self.tensors) >= 5 and self.tensors[SLOT_TOPK_WEIGHTS].dim() == 2:
            return self.tensors[SLOT_TOPK_WEIGHTS]
        return None

    @property
    def hidden_scale(self) -> Optional[torch.Tensor]:
        for t in self.tensors[3:]:
            if t.numel() == 1 and t.dtype == torch.float32 and t.dim() <= 1:
                if t is not self.tensors[SLOT_NUM_TOKENS] and t is not self.tensors[SLOT_LAYER_ID]:
                    return t
        return None

    @property
    def residual(self) -> Optional[torch.Tensor]:
        h = self.hidden_wire.shape[-1]
        for t in self.tensors[3:]:
            if (
                t.dim() == 2
                and t.shape[-1] == h
                and t is not self.tensors[SLOT_HIDDEN]
                and t.dtype in (torch.bfloat16, torch.float16, torch.float32)
            ):
                return t
        return None

    @property
    def positions(self) -> Optional[torch.Tensor]:
        for t in self.tensors:
            if t.dim() == 1 and t.dtype in (torch.int32, torch.int64) and t.numel() > 1:
                return t
        return None
