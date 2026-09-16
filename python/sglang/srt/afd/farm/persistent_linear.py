# SPDX-License-Identifier: Apache-2.0
"""P3 true-ish persistent linear: weight-tile outer, microbatch inner (Triton).

Doc shape::

    for n_tile in weight_tiles:
        for k_tile:
            w = load W[n_tile, k_tile]     # once
            for mb in range(num_mb):      # reuse w across B_step windows
                for m_tile:
                    Y[mb] += X[mb] @ w

When farm coalesce packs ``num_mb * B_step`` tokens, this keeps each weight
fragment live across the mb loop (unlike K separate tiny GEMMs).

Honesty: plain FP16/BF16 only; MLA FlashInfer path untouched; not a
device-side spin-wait persistent grid.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_stats = {
    "calls": 0,
    "tokens": 0,
    "microbatches": 0,
    "fallback": 0,
}

_triton_kernel = None
_wrap_installed = False


def persistent_linear_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.get())


def _spin_wait_on() -> bool:
    return bool(envs.SGLANG_AFD_FARM_SPIN_WAIT.get())


def persistent_linear_stats() -> dict:
    return dict(_stats)


def reset_persistent_linear_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _get_kernel():
    global _triton_kernel
    if _triton_kernel is not None:
        return _triton_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _weight_outer_gemm(
        X,
        W,
        Y,  # fp32 accumulator [num_mb, B_STEP, N]
        B_STEP,
        NUM_MB,
        K,
        N,
        stride_x_mb,
        stride_x_m,
        stride_x_k,
        stride_w_n,
        stride_w_k,
        stride_y_mb,
        stride_y_m,
        stride_y_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        n0 = pid_n * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        m0 = 0
        while m0 < B_STEP:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < B_STEP
            # Zero fp32 Y strip.
            mb = 0
            while mb < NUM_MB:
                y_ptrs = (
                    Y
                    + mb * stride_y_mb
                    + offs_m[:, None] * stride_y_m
                    + offs_n[None, :] * stride_y_n
                )
                tl.store(
                    y_ptrs,
                    tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32),
                    mask=mask_m[:, None] & mask_n[None, :],
                )
                mb += 1
            # Weight-tile outer: load each W[k_tile] once, reuse across mb.
            k0 = 0
            while k0 < K:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                w_ptrs = (
                    W
                    + offs_k[:, None] * stride_w_k
                    + offs_n[None, :] * stride_w_n
                )
                w = tl.load(
                    w_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                mb = 0
                while mb < NUM_MB:
                    x_ptrs = (
                        X
                        + mb * stride_x_mb
                        + offs_m[:, None] * stride_x_m
                        + offs_k[None, :] * stride_x_k
                    )
                    x = tl.load(
                        x_ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    partial = tl.dot(x, w)
                    y_ptrs = (
                        Y
                        + mb * stride_y_mb
                        + offs_m[:, None] * stride_y_m
                        + offs_n[None, :] * stride_y_n
                    )
                    cur = tl.load(
                        y_ptrs,
                        mask=mask_m[:, None] & mask_n[None, :],
                        other=0.0,
                    )
                    tl.store(
                        y_ptrs,
                        cur + partial,
                        mask=mask_m[:, None] & mask_n[None, :],
                    )
                    mb += 1
                k0 += BLOCK_K
            m0 += BLOCK_M

    _triton_kernel = _weight_outer_gemm
    return _triton_kernel


def weight_outer_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    b_step: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``Y = X @ W.T (+ bias)`` with weight-fragment outer / mb inner."""
    if x.dim() != 2 or weight.dim() != 2:
        raise ValueError("expected 2D x and weight")
    m, kin = x.shape
    n, kw = weight.shape
    if kin != kw:
        raise ValueError(f"K mismatch x={kin} w={kw}")
    b_step = max(1, int(b_step))
    if m % b_step != 0 or m // b_step < 1:
        raise ValueError(f"M={m} not divisible by b_step={b_step}")
    num_mb = m // b_step

    if (
        (not x.is_cuda)
        or x.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != x.dtype
    ):
        return _python_weight_outer(x, weight, b_step=b_step, bias=bias)

    x_c = x.contiguous()
    w_c = weight.contiguous()
    # fp32 accumulator avoids intermediate fp16 rounding across k-tiles.
    y_f = torch.empty(m, n, device=x.device, dtype=torch.float32)
    x_mb = x_c.view(num_mb, b_step, kin)
    y_mb = y_f.view(num_mb, b_step, n)

    import triton

    # Drop cached kernel if signature changed across reloads.
    global _triton_kernel
    kernel = _get_kernel()
    block_m = 16 if b_step >= 16 else max(1, int(b_step))
    block_n = 64 if n >= 64 else (32 if n > 32 else 16)
    block_k = 32
    grid = (triton.cdiv(n, block_n),)
    kernel[grid](
        x_mb,
        w_c,
        y_mb,
        b_step,
        num_mb,
        kin,
        n,
        x_mb.stride(0),
        x_mb.stride(1),
        x_mb.stride(2),
        w_c.stride(0),
        w_c.stride(1),
        y_mb.stride(0),
        y_mb.stride(1),
        y_mb.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    y = y_f.to(dtype=x.dtype)
    if bias is not None:
        y = y + bias
    _stats["calls"] += 1
    _stats["tokens"] += m
    _stats["microbatches"] += num_mb
    return y


def _python_weight_outer(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    b_step: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    m, kin = x.shape
    n = weight.shape[0]
    num_mb = m // b_step
    x_mb = x.view(num_mb, b_step, kin)
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    y_mb = y.view(num_mb, b_step, n)
    tile_n = 128
    for n0 in range(0, n, tile_n):
        n1 = min(n, n0 + tile_n)
        w_tile = weight[n0:n1, :]
        for mb in range(num_mb):
            y_mb[mb, :, n0:n1] = x_mb[mb] @ w_tile.T
    if bias is not None:
        y = y + bias
    _stats["calls"] += 1
    _stats["tokens"] += m
    _stats["microbatches"] += num_mb
    return y


def try_persistent_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    b_step: int,
    bias: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not persistent_linear_enabled():
        return None
    b_step = max(1, int(b_step))
    if x.dim() != 2 or weight.dim() != 2:
        _stats["fallback"] += 1
        return None
    if x.shape[0] < 2 * b_step or x.shape[0] % b_step != 0:
        _stats["fallback"] += 1
        return None
    if weight.shape[1] != x.shape[1]:
        _stats["fallback"] += 1
        return None
    try:
        return weight_outer_linear(x, weight, b_step=b_step, bias=bias)
    except Exception as e:
        _stats["fallback"] += 1
        logger.debug("persistent linear fallback: %s", e)
        return None


def _plain_weight(module: Any) -> Optional[torch.Tensor]:
    w = getattr(module, "weight", None)
    if not torch.is_tensor(w) or w.dim() != 2:
        return None
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    qm = getattr(module, "quant_method", None)
    if qm is not None:
        name = type(qm).__name__.lower()
        if any(k in name for k in ("fp8", "fp4", "awq", "gptq", "marlin")):
            return None
    return w


def wrap_linear_module(module: Any, *, b_step_fn) -> bool:
    if getattr(module, "_farm_persist_wrapped", False):
        return False
    if _plain_weight(module) is None:
        return False
    orig_forward = module.forward
    cls_name = module.__class__.__name__
    session_box: dict = {"sess": None}

    def _wrapped(x, *args, **kwargs):
        if args or kwargs:
            return orig_forward(x, *args, **kwargs)
        w = _plain_weight(module)
        if w is None or not torch.is_tensor(x) or x.dim() != 2:
            return orig_forward(x)
        bs = int(b_step_fn())
        # Prefer spin-wait session for single-window or multi-mb streams.
        if _spin_wait_on() and x.is_cuda and x.shape[0] % bs == 0:
            try:
                from sglang.srt.afd.farm.spin_wait_linear import SpinWaitLinearSession

                sess = session_box["sess"]
                if (
                    sess is None
                    or sess.b_step != bs
                    or sess.k_in != w.shape[1]
                    or sess.n_out != w.shape[0]
                    or sess._w.data_ptr() != w.data_ptr()
                ):
                    sess = SpinWaitLinearSession(weight=w, b_step=bs)
                    session_box["sess"] = sess
                out = sess.run_many(x)
                bias = getattr(module, "bias", None)
                if bias is not None and not getattr(module, "skip_bias_add", False):
                    out = out + bias
                if cls_name in ("ReplicatedLinear", "ColumnParallelLinear"):
                    if getattr(module, "skip_bias_add", False):
                        return out, bias
                    return out, None
                return out
            except Exception as e:
                logger.debug("spin-wait linear fallback: %s", e)

        out = try_persistent_linear(x, w, b_step=bs, bias=None)
        if out is None:
            return orig_forward(x)
        bias = getattr(module, "bias", None)
        if bias is not None and not getattr(module, "skip_bias_add", False):
            out = out + bias
        if cls_name in ("ReplicatedLinear", "ColumnParallelLinear"):
            if getattr(module, "skip_bias_add", False):
                return out, bias
            return out, None
        return out

    module.forward = _wrapped  # type: ignore[method-assign]
    module._farm_persist_wrapped = True
    module._farm_persist_orig_forward = orig_forward
    return True


_ATTN_LINEAR_ATTRS = (
    "fused_qkv_a_proj_with_mqa",
    "kv_a_proj_with_mqa",
    "q_a_proj",
    "q_b_proj",
)


def install_persistent_linear_on_layers(
    layers: Sequence[Any], *, b_step_fn=None
) -> int:
    global _wrap_installed
    if not (persistent_linear_enabled() or _spin_wait_on()):
        return 0
    from sglang.srt.afd.farm.env import farm_b_step

    b_step_fn = b_step_fn or farm_b_step
    n = 0
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for name in _ATTN_LINEAR_ATTRS:
            mod = getattr(attn, name, None)
            if mod is None:
                continue
            if wrap_linear_module(mod, b_step_fn=b_step_fn):
                n += 1
    if n and not _wrap_installed:
        logger.info(
            "AFD farm wrapped %s MLA proj modules (persistent=%s spin_wait=%s)",
            n,
            persistent_linear_enabled(),
            _spin_wait_on(),
        )
        _wrap_installed = True
    return n
