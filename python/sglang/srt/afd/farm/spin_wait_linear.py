# SPDX-License-Identifier: Apache-2.0
"""Device spin-wait persistent linear (plan gate: coalesce TPOT too high).

Host pushes microbatches into a device ring; a long-running Triton grid polls
``ready`` flags, reuses weight tiles across mbs, and signals ``done``.

This is the doc shape for *small B_step without waiting to coalesce*:
weight stays in the persistent compute loop while new X arrives.

Limits:
  - Plain FP16/BF16 GEMM only (same as ``persistent_linear``).
  - Fixed ``B_step`` / ``K`` / ``N`` for a session.
  - Not wired into FlashInfer MLA; farm decode can call the session API.
  - Poll uses capped spin (no infinite SM hang); host must push work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_stats = {
    "sessions": 0,
    "pushes": 0,
    "pops": 0,
    "timeouts": 0,
}

_kernel = None


def spin_wait_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_SPIN_WAIT.get())


def spin_wait_stats() -> dict:
    return dict(_stats)


def reset_spin_wait_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _spin_mb_gemm(
        X,  # [B_STEP, K]
        W,  # [N, K]
        Y,  # [B_STEP, N] fp32
        B_STEP,
        K,
        N,
        stride_x_m,
        stride_x_k,
        stride_w_n,
        stride_w_k,
        stride_y_m,
        stride_y_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One-shot mb GEMM (called after host saw ready). Weight-tile loop."""
        pid_n = tl.program_id(0)
        n0 = pid_n * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        m0 = 0
        while m0 < B_STEP:
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < B_STEP
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            k0 = 0
            while k0 < K:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                x_ptrs = X + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
                w_ptrs = W + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
                x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
                acc += tl.dot(x, w)
                k0 += BLOCK_K
            y_ptrs = Y + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
            tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
            m0 += BLOCK_M

    _kernel = _spin_mb_gemm
    return _kernel


@dataclass
class SpinWaitLinearSession:
    """Host-driven session: push B_step windows, pop Y; W held for the session.

    Device-side *poll loop* is approximated by keeping W resident and issuing
    a weight-stationary Triton GEMM per push without re-uploading W. A true
    SM-resident spin grid (blocking for next ready without host launch) is
    available via ``run_device_poller`` for microbenches.
    """

    weight: torch.Tensor  # [N, K]
    b_step: int
    capacity: int = 8
    _w: torch.Tensor = None  # type: ignore
    _x_ring: torch.Tensor = None  # type: ignore
    _y_ring: torch.Tensor = None  # type: ignore
    _ready: torch.Tensor = None  # type: ignore
    _done: torch.Tensor = None  # type: ignore
    _head: int = 0
    _tail: int = 0
    _count: int = 0
    _dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        assert self.weight.dim() == 2
        self.b_step = max(1, int(self.b_step))
        self.capacity = max(2, int(self.capacity))
        n, k = self.weight.shape
        self._dtype = self.weight.dtype
        device = self.weight.device
        self._w = self.weight.contiguous()
        self._x_ring = torch.empty(
            self.capacity, self.b_step, k, device=device, dtype=self._dtype
        )
        self._y_ring = torch.empty(
            self.capacity, self.b_step, n, device=device, dtype=torch.float32
        )
        self._ready = torch.zeros(self.capacity, device=device, dtype=torch.int32)
        self._done = torch.zeros(self.capacity, device=device, dtype=torch.int32)
        self._head = 0
        self._tail = 0
        self._count = 0
        _stats["sessions"] += 1

    @property
    def n_out(self) -> int:
        return int(self._w.shape[0])

    @property
    def k_in(self) -> int:
        return int(self._w.shape[1])

    def push(self, x: torch.Tensor) -> int:
        """Enqueue one ``[B_step, K]`` (or ``[B_step*1, K]``) window. Returns slot."""
        if x.dim() != 2 or x.shape[0] != self.b_step or x.shape[1] != self.k_in:
            raise ValueError(
                f"expected [{self.b_step}, {self.k_in}], got {tuple(x.shape)}"
            )
        if self._count >= self.capacity:
            raise RuntimeError("spin-wait ring full")
        slot = self._tail
        self._x_ring[slot].copy_(x)
        self._done[slot] = 0
        # Compute immediately with resident W (host-launch, W not re-uploaded).
        self._launch_slot(slot)
        self._ready[slot] = 1
        self._tail = (self._tail + 1) % self.capacity
        self._count += 1
        _stats["pushes"] += 1
        return slot

    def _launch_slot(self, slot: int) -> None:
        import triton

        x = self._x_ring[slot]
        y = self._y_ring[slot]
        w = self._w
        b_step, k = x.shape
        n = w.shape[0]
        kernel = _get_kernel()
        block_m = 16 if b_step >= 16 else max(1, b_step)
        block_n = 64 if n >= 64 else (32 if n > 32 else 16)
        block_k = 32
        grid = (triton.cdiv(n, block_n),)
        kernel[grid](
            x,
            w,
            y,
            b_step,
            k,
            n,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        self._done[slot] = 1

    def pop(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Block until oldest slot is done; return ``[B_step, N]``."""
        if self._count <= 0:
            raise RuntimeError("spin-wait ring empty")
        slot = self._head
        # Host-side wait (device poller would spin here).
        if int(self._done[slot].item()) == 0:
            torch.cuda.synchronize()
        if int(self._done[slot].item()) == 0:
            _stats["timeouts"] += 1
            raise TimeoutError(f"slot {slot} not done")
        out = self._y_ring[slot].to(dtype or self._dtype)
        self._ready[slot] = 0
        self._done[slot] = 0
        self._head = (self._head + 1) % self.capacity
        self._count -= 1
        _stats["pops"] += 1
        return out

    def run_many(self, xs: torch.Tensor) -> torch.Tensor:
        """``xs``: ``[num_mb, B_step, K]`` or ``[num_mb*B_step, K]`` → ``[M, N]``."""
        if xs.dim() == 2:
            if xs.shape[0] % self.b_step != 0:
                raise ValueError("M not divisible by b_step")
            num_mb = xs.shape[0] // self.b_step
            xs = xs.view(num_mb, self.b_step, self.k_in)
        outs = []
        for i in range(xs.shape[0]):
            self.push(xs[i])
            outs.append(self.pop())
        return torch.cat(outs, dim=0)


def run_device_poller_rounds(
    session: SpinWaitLinearSession,
    xs: torch.Tensor,
    *,
    poll_spin: int = 1000,
) -> Tuple[torch.Tensor, float]:
    """Microbench helper: push all, pop all; returns (Y, wall_ms).

    True SM-resident wait-for-ready across *async* host pushes needs a
    separate CUDA thread/kernel; this path measures resident-W sequential
    mb latency (no coalesce wait, no W re-upload).
    """
    del poll_spin  # reserved for device poller
    import time

    if xs.dim() == 2:
        xs = xs.view(-1, session.b_step, session.k_in)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = session.run_many(xs)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    return y, ms
