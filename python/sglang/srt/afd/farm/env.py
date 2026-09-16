# SPDX-License-Identifier: Apache-2.0
"""Decode-farm env helpers (P0/P1). No CUDA / model imports."""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_farm_env_applied = False


def farm_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM.get())


def farm_b_step() -> int:
    return max(1, int(envs.SGLANG_AFD_FARM_B_STEP.get() or 1))


def farm_num_contexts() -> int:
    return max(1, int(envs.SGLANG_AFD_FARM_NUM_CONTEXTS.get() or 1))


def farm_context_stagger_layers() -> int:
    """Layer lead required before injecting the next sequence context."""
    return max(
        0,
        int(envs.SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS.get() or 0),
    )


def farm_contexts_per_stage() -> int:
    """Contexts that enter together and share same-layer A2F groups."""
    return max(
        1,
        int(envs.SGLANG_AFD_FARM_CONTEXTS_PER_STAGE.get() or 1),
    )


def farm_b_win_k() -> int:
    return max(1, int(envs.SGLANG_AFD_FARM_B_WIN_K.get() or 1))


def farm_soft_persistent_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.get())


def farm_coalesce_k() -> int:
    return max(1, int(envs.SGLANG_AFD_FARM_COALESCE_K.get() or 1))


def farm_layer_burst() -> int:
    return max(0, int(envs.SGLANG_AFD_FARM_LAYER_BURST.get() or 0))


def farm_sched() -> str:
    value = str(envs.SGLANG_AFD_FARM_SCHED.get() or "max").strip().lower()
    return value if value in ("max", "oldest", "deepest") else "max"


def farm_layer_cg_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_LAYER_CG.get())


def farm_persistent_linear_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.get())


def farm_spin_wait_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_SPIN_WAIT.get())


def farm_max_inflight() -> int:
    return max(1, int(envs.SGLANG_AFD_FARM_MAX_INFLIGHT.get() or 1))


def farm_max_inflight_per_layer() -> int:
    """Cap concurrent A2F hops from one layer.

    Without this cap the first ready layer can occupy every MB slot before any
    later layer exists, recreating lockstep. Zero selects a conservative cap of
    4 for the legacy path and the full farm limit when the limit is smaller.
    """
    configured = int(envs.SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER.get() or 0)
    max_inf = farm_max_inflight()
    if configured <= 0:
        configured = min(max_inf, 4)
    return max(1, min(max_inf, configured))


def farm_wait_slice_us() -> int:
    """Bounded poll window before re-entering the farm scheduler."""
    return max(0, int(envs.SGLANG_AFD_FARM_WAIT_SLICE_US.get() or 0))


def farm_persistent_enabled() -> bool:
    """True when contexts must survive across ``model.forward()`` calls."""
    return bool(envs.SGLANG_AFD_FARM_PERSISTENT.get())


def farm_persistent_max_age() -> int:
    """Max consecutive forwards one context may stay live (0 = unbounded)."""
    return max(0, int(envs.SGLANG_AFD_FARM_PERSISTENT_MAX_AGE.get() or 0))


def farm_persistent_poll_us() -> int:
    """Bounded poll budget per forward for pending hops (0 = never sleep)."""
    return max(0, int(envs.SGLANG_AFD_FARM_PERSISTENT_POLL_US.get() or 0))


def farm_persistent_groups() -> int:
    """Target number of layer-synchronised groups in the persistent farm.

    ``0`` keeps one context per row (one token per A2F hop).
    """
    return max(0, int(envs.SGLANG_AFD_FARM_PERSISTENT_GROUPS.get() or 0))


def apply_farm_env() -> bool:
    """Wire farm: drop lockstep layer-pipe, bump NUM_MB, optional LPU gather.

    Safe to call repeatedly. Returns True when farm is on.
    """
    global _farm_env_applied
    if not farm_enabled():
        return False

    envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
    envs.SGLANG_AFD_PIPELINE.set(False)
    envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)

    max_inf = farm_max_inflight()
    try:
        from sglang.srt.afd.pool.topology import pool_enabled

        if pool_enabled():
            pool_inf = max(1, int(envs.SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN.get() or 1))
            max_inf = max(max_inf, pool_inf)
            if pool_inf < max_inf:
                envs.SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN.set(max_inf)
    except Exception:
        pass

    # Keep the farm scheduler on the same effective limit as the pool.
    # Without this, startup may print 8 while later reads fall back to 4.
    if farm_max_inflight() < max_inf:
        envs.SGLANG_AFD_FARM_MAX_INFLIGHT.set(max_inf)

    if int(envs.SGLANG_AFD_NUM_MB.get() or 1) < max_inf:
        envs.SGLANG_AFD_NUM_MB.set(max_inf)

    gather_us = int(envs.SGLANG_AFD_FFN_GATHER_US.get() or 0)
    lpu_us = int(envs.SGLANG_AFD_FARM_LPU_GATHER_US.get() or 0)
    if gather_us <= 0 and lpu_us > 0:
        envs.SGLANG_AFD_FFN_GATHER_US.set(lpu_us)
        gather_max = max(int(envs.SGLANG_AFD_FFN_GATHER_MAX.get() or 1), max_inf)
        envs.SGLANG_AFD_FFN_GATHER_MAX.set(gather_max)

    transport = (envs.SGLANG_AFD_TRANSPORT.get() or "fake").strip().lower()
    if transport not in ("cuda_ipc", "fake", "nvlink"):
        logger.warning(
            "AFD farm prefers cuda_ipc (got transport=%s); hops may not poll",
            transport,
        )

    if not _farm_env_applied:
        logger.info(
            "AFD farm on B_step=%s num_contexts=%s context_stagger=%s "
            "contexts_per_stage=%s B_win_k=%s "
            "coalesce_k=%s layer_burst=%s sched=%s max_inflight=%s "
            "max_inflight_per_layer=%s "
            "NUM_MB=%s gather_us=%s soft_persistent=%s layer_cg=%s "
            "persistent_linear=%s spin_wait=%s persistent=%s (layer_pipeline=0)",
            farm_b_step(),
            farm_num_contexts(),
            farm_context_stagger_layers(),
            farm_contexts_per_stage(),
            farm_b_win_k(),
            farm_coalesce_k(),
            farm_layer_burst(),
            farm_sched(),
            max_inf,
            farm_max_inflight_per_layer(),
            envs.SGLANG_AFD_NUM_MB.get(),
            envs.SGLANG_AFD_FFN_GATHER_US.get(),
            envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.get(),
            envs.SGLANG_AFD_FARM_LAYER_CG.get(),
            envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.get(),
            envs.SGLANG_AFD_FARM_SPIN_WAIT.get(),
            farm_persistent_enabled(),
        )
        _farm_env_applied = True
    return True


def afd_farm_enabled(forward_batch=None) -> bool:
    """True when decode farm should replace the lockstep layer loop."""
    if not farm_enabled():
        return False
    try:
        from sglang.srt.afd.runtime import is_afd_attn

        if not is_afd_attn():
            return False
    except Exception:
        return False
    if forward_batch is None:
        return True
    try:
        if not forward_batch.forward_mode.is_decode():
            return False
        if getattr(forward_batch, "can_run_tbo", False):
            return False
        n_tok = int(forward_batch.input_ids.shape[0])
        n_seq = int(forward_batch.batch_size)
        if n_seq <= 0 or n_tok <= 0 or n_tok % n_seq != 0:
            return False
    except Exception:
        return False
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        pass
    return True
