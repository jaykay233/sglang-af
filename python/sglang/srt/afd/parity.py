# SPDX-License-Identifier: Apache-2.0
"""Eager parity helper: collocated Linear FFN vs AFD Fake remote (P5)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.afd.runtime import init_afd_runtime, shutdown_afd_runtime
from sglang.srt.environ import envs


class _SharedLinearFfn(nn.Module):
    def __init__(self, hidden: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def compare_local_vs_afd_fake(
    *,
    hidden_size: int = 8,
    num_tokens: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (local_out, afd_out) for the same Linear weights / input.

    Caller should assert ``torch.allclose(local, afd)``.
    """
    torch.manual_seed(seed)
    dtype = torch.float32
    ffn = _SharedLinearFfn(hidden_size, device, dtype)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    local = ffn(x)

    def compute(batch: AfdServerBatch):
        t = batch.num_tokens
        y = ffn(batch.hidden[:t])
        out = batch.hidden.clone()
        out[:t] = y
        return [out]

    prev_mode = envs.SGLANG_AFD_MODE.get()
    prev_transport = envs.SGLANG_AFD_TRANSPORT.get()
    prev_mb = envs.SGLANG_AFD_NUM_MB.get()
    prev_tok = envs.SGLANG_AFD_MAX_NUM_TOKEN.get()
    try:
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(max(16, num_tokens))
        shutdown_afd_runtime()
        rt = init_afd_runtime(
            hidden_size=hidden_size,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device=device,
            dtype=dtype,
            ffn_compute=compute,
        )
        assert rt is not None
        afd = rt.remote_ffn(layer_id=0, hidden=x)
    finally:
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set(prev_mode)
        envs.SGLANG_AFD_TRANSPORT.set(prev_transport)
        envs.SGLANG_AFD_NUM_MB.set(prev_mb)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(prev_tok)

    return local, afd
