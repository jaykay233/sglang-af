# SPDX-License-Identifier: Apache-2.0
"""Pre-registered A2F / F2A buffer pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.afd.a2f_quant import (
    a2f_needs_scale,
    quantize_hidden_for_a2f,
    resolve_a2f_wire_dtype,
)
from sglang.srt.afd.protocol import (
    AfdA2FPayload,
    AfdF2APayload,
    pack_layer_merge_meta,
)
from sglang.srt.environ import envs


@dataclass
class AfdBufferPoolConfig:
    num_mb: int = 1
    max_num_token: int = 256
    hidden_size: int = 7168
    dtype: torch.dtype = torch.bfloat16  # compute / F2A dtype
    device: str = "cuda"
    moe_topk: int = 0  # 0 = dense / no routing buffers
    worker_rank: int = 0
    a2f_wire_dtype: Optional[torch.dtype] = None  # None → resolve from env
    layer_merge: bool = False  # residual + positions + F2A residual


class AfdBufferPool:
    """Fixed-shape buffers for decode AFD (no hot-path alloc)."""

    def __init__(self, cfg: AfdBufferPoolConfig):
        self.cfg = cfg
        if not cfg.layer_merge:
            cfg.layer_merge = int(envs.SGLANG_AFD_LAYER_MERGE_K.get() or 1) > 1
        self.compute_dtype = cfg.dtype
        self.wire_dtype = cfg.a2f_wire_dtype or resolve_a2f_wire_dtype(cfg.dtype)
        self._needs_scale = a2f_needs_scale(self.wire_dtype)
        # When True, skip GPU num_tokens/layer_id fills (host mailbox carries meta).
        self.host_meta_only: bool = False
        self._a2f_hidden: List[torch.Tensor] = []
        self._a2f_num_tokens: List[torch.Tensor] = []
        self._a2f_layer_id: List[torch.Tensor] = []
        self._a2f_topk_ids: List[Optional[torch.Tensor]] = []
        self._a2f_topk_weights: List[Optional[torch.Tensor]] = []
        self._a2f_hidden_scale: List[Optional[torch.Tensor]] = []
        self._a2f_residual: List[Optional[torch.Tensor]] = []
        self._a2f_positions: List[Optional[torch.Tensor]] = []
        self._f2a_out: List[torch.Tensor] = []
        self._f2a_residual: List[Optional[torch.Tensor]] = []
        self._allocate()

    def _allocate(self) -> None:
        c = self.cfg
        for _ in range(c.num_mb):
            self._a2f_hidden.append(
                torch.empty(
                    c.max_num_token,
                    c.hidden_size,
                    dtype=self.wire_dtype,
                    device=c.device,
                )
            )
            self._a2f_num_tokens.append(
                torch.zeros(1, dtype=torch.int32, device=c.device)
            )
            self._a2f_layer_id.append(
                torch.zeros(1, dtype=torch.int32, device=c.device)
            )
            if c.moe_topk > 0:
                self._a2f_topk_ids.append(
                    torch.empty(
                        c.max_num_token,
                        c.moe_topk,
                        dtype=torch.int32,
                        device=c.device,
                    )
                )
                self._a2f_topk_weights.append(
                    torch.empty(
                        c.max_num_token,
                        c.moe_topk,
                        dtype=torch.float32,
                        device=c.device,
                    )
                )
            else:
                self._a2f_topk_ids.append(None)
                self._a2f_topk_weights.append(None)
            if self._needs_scale:
                self._a2f_hidden_scale.append(
                    torch.ones(1, dtype=torch.float32, device=c.device)
                )
            else:
                self._a2f_hidden_scale.append(None)
            if c.layer_merge:
                self._a2f_residual.append(
                    torch.empty(
                        c.max_num_token,
                        c.hidden_size,
                        dtype=self.compute_dtype,
                        device=c.device,
                    )
                )
                self._a2f_positions.append(
                    torch.zeros(
                        c.max_num_token, dtype=torch.int64, device=c.device
                    )
                )
                self._f2a_residual.append(
                    torch.empty(
                        c.max_num_token,
                        c.hidden_size,
                        dtype=self.compute_dtype,
                        device=c.device,
                    )
                )
            else:
                self._a2f_residual.append(None)
                self._a2f_positions.append(None)
                self._f2a_residual.append(None)
            self._f2a_out.append(
                torch.empty(
                    c.max_num_token,
                    c.hidden_size,
                    dtype=self.compute_dtype,
                    device=c.device,
                )
            )
        self._rebuild_tensor_lists()

    def _rebuild_tensor_lists(self) -> None:
        """Cache A2F/F2A tensor lists so hot-path hops skip list rebuilds."""
        n = self.cfg.num_mb
        self._a2f_lists: List[List[torch.Tensor]] = [
            self.a2f_tensor_list(i) for i in range(n)
        ]
        self._f2a_lists: List[List[torch.Tensor]] = [
            self.f2a_tensor_list(i) for i in range(n)
        ]

    def fill_a2f(
        self,
        mb_id: int,
        *,
        hidden: torch.Tensor,
        layer_id: int,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> AfdA2FPayload:
        """Quantize/copy / pad ``hidden`` into the registered slot."""
        t = int(hidden.shape[0])
        dst = self._a2f_hidden[mb_id]
        capacity = int(dst.shape[0])
        if t > capacity:
            raise RuntimeError(
                "AFD A2F message exceeds the registered slot: "
                f"tokens={t} capacity={capacity} mb={mb_id}. "
                "Increase SGLANG_AFD_MAX_NUM_TOKEN before server startup; "
                "decode farm callers should normally split into B_step-sized "
                "messages."
            )
        if self.wire_dtype == hidden.dtype:
            wire, scale = hidden, None
        else:
            wire, scale = quantize_hidden_for_a2f(hidden, self.wire_dtype)
        # Only the live prefix matters; FFN uses host_num_tokens / [:t] and CG
        # zeros its own pad. Skipping dst[t:].zero_() saves a pad-wide kernel/hop.
        dst[:t].copy_(wire[:t], non_blocking=True)
        if not self.host_meta_only:
            # StepMesh / readers that .item() GPU meta still need these fills.
            self._a2f_num_tokens[mb_id].fill_(t)
            self._a2f_layer_id[mb_id].fill_(
                pack_layer_merge_meta(layer_id, merge_k)
            )

        scale_buf = self._a2f_hidden_scale[mb_id]
        if scale is not None:
            scale_buf.copy_(scale.to(device=scale_buf.device).view(1), non_blocking=True)

        topk_ids_buf = self._a2f_topk_ids[mb_id]
        topk_w_buf = self._a2f_topk_weights[mb_id]
        if topk_ids is not None and topk_ids_buf is not None:
            topk_ids_buf[:t].copy_(topk_ids[:t], non_blocking=True)
        if topk_weights is not None and topk_w_buf is not None:
            topk_w_buf[:t].copy_(topk_weights[:t], non_blocking=True)

        residual_buf = self._a2f_residual[mb_id]
        positions_buf = self._a2f_positions[mb_id]
        if residual is not None and residual_buf is not None:
            residual_buf[:t].copy_(
                residual[:t].to(dtype=residual_buf.dtype), non_blocking=True
            )
        if positions is not None and positions_buf is not None:
            positions_buf[:t].copy_(
                positions[:t].to(dtype=positions_buf.dtype), non_blocking=True
            )

        return AfdA2FPayload(
            hidden=dst,
            num_tokens=self._a2f_num_tokens[mb_id],
            layer_id=self._a2f_layer_id[mb_id],
            topk_ids=topk_ids_buf,
            topk_weights=topk_w_buf,
            hidden_scale=scale_buf,
            residual=residual_buf,
            positions=positions_buf,
        )

    def get_f2a(self, mb_id: int) -> AfdF2APayload:
        return AfdF2APayload(
            mlp_out=self._f2a_out[mb_id],
            residual=self._f2a_residual[mb_id],
        )

    def a2f_tensor_list(self, mb_id: int) -> List[torch.Tensor]:
        payload = AfdA2FPayload(
            hidden=self._a2f_hidden[mb_id],
            num_tokens=self._a2f_num_tokens[mb_id],
            layer_id=self._a2f_layer_id[mb_id],
            topk_ids=self._a2f_topk_ids[mb_id],
            topk_weights=self._a2f_topk_weights[mb_id],
            hidden_scale=self._a2f_hidden_scale[mb_id],
            residual=self._a2f_residual[mb_id],
            positions=self._a2f_positions[mb_id],
        )
        return payload.as_tensor_list()

    def f2a_tensor_list(self, mb_id: int) -> List[torch.Tensor]:
        return self.get_f2a(mb_id).as_tensor_list()
