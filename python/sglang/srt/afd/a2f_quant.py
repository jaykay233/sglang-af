# SPDX-License-Identifier: Apache-2.0
"""Optional A2F wire-dtype quantization (Attn→FFN hidden).

Residual always stays on the Attn worker (never travels on A2F).
Default wire dtype matches compute (bf16/fp16). Opt-in FP8::

    SGLANG_AFD_A2F_DTYPE=fp8
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

# Rough max of float8_e4m3fn finite values.
_FP8_E4M3_MAX = 448.0


def get_a2f_dtype_name() -> str:
    """Return normalized wire dtype name: auto | bf16 | fp16 | fp8."""
    raw = (envs.SGLANG_AFD_A2F_DTYPE.get() or "auto").strip().lower()
    if raw in ("", "auto", "default"):
        return "auto"
    if raw in ("bf16", "bfloat16"):
        return "bf16"
    if raw in ("fp16", "float16", "half"):
        return "fp16"
    if raw in ("fp8", "fp8_e4m3", "float8_e4m3fn", "e4m3"):
        return "fp8"
    raise ValueError(
        f"Unknown SGLANG_AFD_A2F_DTYPE={raw!r}; expected auto|bf16|fp16|fp8"
    )


def resolve_a2f_wire_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    name = get_a2f_dtype_name()
    if name == "auto":
        return compute_dtype
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn unavailable; cannot use A2F fp8")
        return torch.float8_e4m3fn
    return compute_dtype


def a2f_needs_scale(wire_dtype: torch.dtype) -> bool:
    return wire_dtype == getattr(torch, "float8_e4m3fn", object())


def quantize_hidden_for_a2f(
    hidden: torch.Tensor,
    wire_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (wire_hidden, optional per-tensor scale fp32[1])."""
    if wire_dtype == hidden.dtype:
        return hidden, None
    if a2f_needs_scale(wire_dtype):
        # Per-tensor absmax scale → e4m3.
        flat = hidden.detach().float()
        amax = flat.abs().amax().clamp(min=1e-12)
        scale = (amax / _FP8_E4M3_MAX).to(torch.float32)
        q = (flat / scale.clamp(min=1e-12)).to(wire_dtype)
        return q, scale.view(1)
    return hidden.to(wire_dtype), None


def dequantize_hidden_from_a2f(
    wire_hidden: torch.Tensor,
    scale: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if scale is None:
        return wire_hidden.to(compute_dtype)
    return (wire_hidden.float() * scale.float().reshape(()).item()).to(compute_dtype)
