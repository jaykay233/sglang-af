# SPDX-License-Identifier: Apache-2.0
"""Compose AFD roles with Prefill–Decode disaggregation (P4)."""

from __future__ import annotations

import logging

from sglang.srt.afd.mode import AfdMode, get_afd_mode


logger = logging.getLogger(__name__)


def apply_afd_pd_policy(server_args) -> None:
    """Validate / document AFD × PD disaggregation combinations.

    Supported:
      - AFD null + any PD (unchanged)
      - AFD attn + PD null   (collocated prefill+decode Attn, remote FFN)
      - AFD attn + PD decode (decode Attn receives KV; FFN remote)  ← recommended
      - AFD ffn  + PD null   (FFN-only process)

    Rejected:
      - AFD ffn + PD prefill/decode
      - AFD attn + PD prefill (prefill Attn without local FFN is unusual; refuse
        unless ``SGLANG_AFD_ALLOW_PREFILL_ATTN=1``)
    """
    from sglang.srt.environ import envs

    mode = get_afd_mode()
    if mode == AfdMode.NULL:
        return

    # Layer-merge: force policy before CUDA-graph / PD checks.
    try:
        from sglang.srt.afd.remote_policy import apply_layer_merge_env

        apply_layer_merge_env()
    except Exception:
        pass

    pd = getattr(server_args, "disaggregation_mode", "null") or "null"

    if mode == AfdMode.FFN and pd != "null":
        raise ValueError(
            f"AFD mode=ffn is incompatible with disaggregation_mode={pd!r}. "
            "Run FFN workers with --disaggregation-mode=null."
        )

    if mode == AfdMode.ATTN and pd == "prefill":
        if not envs.SGLANG_AFD_ALLOW_PREFILL_ATTN.get():
            raise ValueError(
                "AFD mode=attn with disaggregation_mode=prefill is disabled by "
                "default (prefill usually keeps local FFN). Set "
                "SGLANG_AFD_ALLOW_PREFILL_ATTN=1 to override, or use "
                "disaggregation_mode=decode|null."
            )
        logger.warning(
            "AFD attn + PD prefill enabled via SGLANG_AFD_ALLOW_PREFILL_ATTN=1"
        )

    if mode == AfdMode.ATTN and pd == "decode":
        logger.info(
            "AFD×PD: decode Attn worker (KV from prefill) + remote FFN via StepMesh"
        )
    elif mode == AfdMode.ATTN and pd == "null":
        logger.info("AFD: unified Attn (local prefill+decode) + remote FFN")
    elif mode == AfdMode.FFN:
        logger.info("AFD: FFN-only worker (PD=null)")
