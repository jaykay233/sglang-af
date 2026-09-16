# SPDX-License-Identifier: Apache-2.0
"""FFN server process entry.

Usage (Fake / same-host smoke)::

    SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=fake \\
      python -m sglang.srt.afd.ffn_server --hidden-size 1024

For StepMesh, set ``SGLANG_AFD_TRANSPORT=stepmesh`` and the usual DMLC_* env
vars, then run this module on FFN ranks.

Prefer launching a full SGLang server with ``SGLANG_AFD_MODE=ffn`` so
``maybe_init_afd_from_model_runner`` binds real MoE weights + poll loop.
This CLI remains useful for transport bring-up (identity / Linear FFN).
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
import torch.nn as nn

from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.afd.runtime import get_afd_runtime, init_afd_runtime, shutdown_afd_runtime
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class IdentityFfn(nn.Module):
    """Placeholder FFN for transport bring-up (y = x)."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class LinearFfn(nn.Module):
    def __init__(self, hidden_size: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden)


def _build_compute(ffn: nn.Module):
    def compute(batch: AfdServerBatch):
        t = batch.num_tokens
        x = batch.hidden[:t]
        y = ffn(x)
        # Pad back to registered length for respond copy.
        out = batch.hidden.clone()
        out[:t] = y
        return [out]

    return compute


def run_ffn_server(
    *,
    hidden_size: int,
    use_linear: bool = False,
    max_iters: int = -1,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ffn: nn.Module
    if use_linear:
        ffn = LinearFfn(hidden_size, device, dtype)
    else:
        ffn = IdentityFfn()

    # For fake transport in FFN-only process without auto thread, we drive
    # get_batch ourselves. Attn process must use the same Fake instance —
    # that only works in-process. For multi-process Fake is insufficient;
    # use stepmesh.
    transport_name = (envs.SGLANG_AFD_TRANSPORT.get() or "fake").strip().lower()
    if transport_name == "fake":
        logger.warning(
            "FFN server with fake transport only works in-process with a shared "
            "FakeAfdTransport (unit tests). Prefer stepmesh for multi-process."
        )

    rt = init_afd_runtime(
        hidden_size=hidden_size,
        mode=AfdMode.FFN,
        device=device,
        dtype=dtype,
        ffn_compute=_build_compute(ffn) if transport_name == "fake" else None,
    )
    assert rt is not None

    if transport_name == "fake":
        # Auto-serve via ffn_compute if Attn shares this process; otherwise spin.
        logger.info("FFN server idle (fake); waiting on shared transport")
        try:
            while max_iters != 0:
                time.sleep(1.0)
                if max_iters > 0:
                    max_iters -= 1
        finally:
            shutdown_afd_runtime()
        return

    # StepMesh path: poll get_batch / respond.
    compute = _build_compute(ffn)
    iters = 0
    logger.info("FFN server loop (stepmesh) started")
    try:
        while max_iters < 0 or iters < max_iters:
            batches = rt.transport.get_batch(timeout_s=1.0)
            for batch in batches:
                outs = compute(batch)
                rt.transport.respond(batch, outs)
                iters += 1
    finally:
        shutdown_afd_runtime()


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="SGLang AFD FFN server (transport smoke)")
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--linear", action="store_true", help="Use Linear FFN instead of identity")
    p.add_argument("--max-iters", type=int, default=-1)
    args = p.parse_args()
    # Ensure mode is ffn even if env unset.
    envs.SGLANG_AFD_MODE.set("ffn")
    run_ffn_server(
        hidden_size=args.hidden_size,
        use_linear=args.linear,
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
