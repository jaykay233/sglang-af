# SPDX-License-Identifier: Apache-2.0
"""P3 soft-persistent Attn proxy (optional A/B): weight-outer tax on first B_win window.

Prefer P2 ``COALESCE_K`` (real stock-kernel amortize) and P3 ``LAYER_CG``
(launch amortize). This tax does **not** skip HBM weight reads.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_bufs: dict = {}
_stats = {
    "tax_calls": 0,
    "skip_calls": 0,
    "tax_us": 0.0,
    "tax_bytes": 0,
}


def soft_persistent_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.get())


def weight_tax_us() -> float:
    return max(0.0, float(envs.SGLANG_AFD_FARM_WEIGHT_TAX_US.get() or 0.0))


def weight_tax_bytes() -> int:
    return max(0, int(envs.SGLANG_AFD_FARM_WEIGHT_TAX_BYTES.get() or 0))


def soft_persistent_stats() -> dict:
    return dict(_stats)


def reset_soft_persistent_stats() -> None:
    _stats["tax_calls"] = 0
    _stats["skip_calls"] = 0
    _stats["tax_us"] = 0.0
    _stats["tax_bytes"] = 0


def _busy_us(us: float) -> None:
    if us <= 0:
        return
    deadline = time.perf_counter() + us * 1e-6
    while time.perf_counter() < deadline:
        pass


def _copy_burn(nbytes: int, device: torch.device) -> None:
    if nbytes <= 0 or not str(device).startswith("cuda"):
        if nbytes > 0:
            # CPU fallback: busy proportional to bytes @ ~50GB/s host proxy.
            _busy_us((nbytes / (50.0 * (1 << 30))) * 1e6)
        return
    n_elem = max(1, (nbytes + 1) // 2)
    key = ("wtax", str(device))
    buf = _bufs.get(key)
    if buf is None or buf.numel() < n_elem or buf.device != device:
        buf = torch.empty(n_elem, dtype=torch.float16, device=device)
        _bufs[key] = buf
    n = max(1, n_elem // 2)
    a = buf[:n]
    b = buf[n : 2 * n] if 2 * n <= buf.numel() else buf[:n]
    b.copy_(a)
    a.copy_(b)


class WeightOuterTax:
    """Pay weight-outer tax only on ``win_index==0`` (first window of B_win)."""

    def maybe_tax(
        self,
        *,
        layer_id: int,
        win_index: int,
        device: Optional[torch.device] = None,
    ) -> float:
        """Return microseconds charged (0 if skipped / disabled)."""
        del layer_id  # reserved for per-layer profiles
        if not soft_persistent_enabled():
            return 0.0
        if int(win_index) > 0:
            _stats["skip_calls"] += 1
            return 0.0
        us = weight_tax_us()
        nbytes = weight_tax_bytes()
        t0 = time.perf_counter()
        if nbytes > 0:
            dev = device if device is not None else torch.device("cpu")
            _copy_burn(nbytes, dev)
        if us > 0:
            _busy_us(us)
        charged = (time.perf_counter() - t0) * 1e6
        _stats["tax_calls"] += 1
        _stats["tax_us"] += charged
        _stats["tax_bytes"] += nbytes
        return charged


_default_tax = WeightOuterTax()


def maybe_weight_outer_tax(
    *,
    layer_id: int,
    win_index: int,
    device: Optional[torch.device] = None,
) -> float:
    return _default_tax.maybe_tax(
        layer_id=layer_id, win_index=win_index, device=device
    )
