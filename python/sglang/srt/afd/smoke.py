# SPDX-License-Identifier: Apache-2.0
"""AFD smoke / bring-up checks (no RDMA required for Fake path).

Usage::

    python -m sglang.srt.afd.smoke
    python -m sglang.srt.afd.smoke --with-fp8   # needs CUDA + float8
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

logger = logging.getLogger(__name__)


def _check_imports() -> None:
    from sglang.srt.afd import (
        AfdMode,
        apply_afd_pd_policy,
        get_afd_mode,
        should_load_weight,
    )
    from sglang.srt.afd.a2f_quant import get_a2f_dtype_name
    from sglang.srt.afd.routing_scheme import get_routing_scheme

    logger.info(
        "imports ok mode=%s scheme=%s a2f=%s",
        get_afd_mode().value,
        get_routing_scheme().value,
        get_a2f_dtype_name(),
    )
    del AfdMode, apply_afd_pd_policy, should_load_weight


def _check_parity() -> None:
    from sglang.srt.afd.parity import compare_local_vs_afd_fake

    local, afd = compare_local_vs_afd_fake(
        hidden_size=16, num_tokens=7, seed=0, device="cpu"
    )
    if not torch.allclose(local, afd, atol=1e-5):
        raise AssertionError("AFD Fake parity mismatch")
    logger.info("parity ok max_abs=%.3e", (local - afd).abs().max().item())


def _check_fp8() -> None:
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("CUDA + float8_e4m3fn required for --with-fp8")
    from sglang.srt.environ import envs
    from sglang.srt.afd.parity import compare_local_vs_afd_fake

    prev = envs.SGLANG_AFD_A2F_DTYPE.get()
    try:
        envs.SGLANG_AFD_A2F_DTYPE.set("fp8")
        local, afd = compare_local_vs_afd_fake(
            hidden_size=16, num_tokens=5, seed=1, device="cuda"
        )
        # FP8 round-trip is lossy — loose tolerance.
        if not torch.allclose(local.float(), afd.float(), rtol=0.05, atol=0.05):
            raise AssertionError(
                f"FP8 A2F parity too far max_abs="
                f"{(local.float() - afd.float()).abs().max().item()}"
            )
        logger.info(
            "fp8 a2f ok max_abs=%.3e",
            (local.float() - afd.float()).abs().max().item(),
        )
    finally:
        envs.SGLANG_AFD_A2F_DTYPE.set(prev)


def _check_stepmesh_import() -> None:
    try:
        import fserver_lib  # noqa: F401

        logger.info("fserver_lib importable — StepMesh transport available")
    except ImportError:
        logger.warning(
            "fserver_lib not importable; use Fake transport or "
            "`pip install -e /path/to/StepMesh` with CUDA_HOME set"
        )


def print_stepmesh_checklist() -> None:
    print(
        """
StepMesh multi-process checklist
--------------------------------
1. Build/install StepMesh (CUDA_HOME matching torch CUDA).
2. Export RDMA / scheduler env (example)::

     export DMLC_ROLE=...          # per StepMesh docs
     export DMLC_NUM_WORKER=1
     export DMLC_NUM_SERVER=1
     export DMLC_PS_ROOT_URI=<ip>
     export DMLC_PS_ROOT_PORT=8000

3. FFN rank (loads experts; no KV)::

     SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=stepmesh \\
       SGLANG_AFD_MODULE_STUBS=1 SGLANG_AFD_RELEASE_UNUSED_PARAMS=1 \\
       python -m sglang.launch_server --model-path <ckpt> ...

   Or transport-only::

     SGLANG_AFD_MODE=ffn SGLANG_AFD_TRANSPORT=stepmesh \\
       python -m sglang.srt.afd.ffn_server --hidden-size <H>

4. Attn / decode rank::

     SGLANG_AFD_MODE=attn SGLANG_AFD_TRANSPORT=stepmesh \\
       SGLANG_AFD_USE_WAIT_FLAG=1 \\   # optional CG sync
       python -m sglang.launch_server --model-path <ckpt> \\
         --disaggregation-mode decode   # optional PD

5. Optional knobs::

     SGLANG_AFD_ROUTING_SCHEME=a|b     # default a (Attn topk)
     SGLANG_AFD_A2F_DTYPE=auto|fp8
     SGLANG_AFD_PIPELINE=1 SGLANG_AFD_NUM_MB=3
""".strip()
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="SGLang AFD smoke checks")
    p.add_argument("--with-fp8", action="store_true")
    p.add_argument("--checklist-only", action="store_true")
    args = p.parse_args(argv)

    if args.checklist_only:
        print_stepmesh_checklist()
        return 0

    _check_imports()
    _check_parity()
    _check_stepmesh_import()
    if args.with_fp8:
        _check_fp8()
    print_stepmesh_checklist()
    logger.info("AFD smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
