# SPDX-License-Identifier: Apache-2.0
"""Decode farm: cross-layer token queues + LPU-sim gather (P0–P3).

Enable with ``SGLANG_AFD_FARM=1`` (cuda_ipc). See ``README.md``.
"""

from __future__ import annotations

from sglang.srt.afd.farm.decode_farm_loop import run_farm_layers
from sglang.srt.afd.farm.env import (
    afd_farm_enabled,
    apply_farm_env,
    farm_b_step,
    farm_b_win_k,
    farm_coalesce_k,
    farm_context_stagger_layers,
    farm_enabled,
    farm_layer_burst,
    farm_layer_cg_enabled,
    farm_max_inflight,
    farm_num_contexts,
    farm_persistent_linear_enabled,
    farm_sched,
    farm_soft_persistent_enabled,
    farm_spin_wait_enabled,
)
from sglang.srt.afd.farm.layer_cuda_graph import (
    layer_cg_enabled,
    layer_cg_stats,
    run_layer_forward_pre_ffn,
)
from sglang.srt.afd.farm.persistent_linear import (
    install_persistent_linear_on_layers,
    persistent_linear_enabled,
    weight_outer_linear,
)
from sglang.srt.afd.farm.scheduler import (
    FarmContinuousScheduler,
    FarmContext,
    FarmTicket,
    get_continuous_scheduler,
    plan_context_ranges,
    reset_continuous_schedulers,
)
from sglang.srt.afd.farm.soft_persistent import (
    maybe_weight_outer_tax,
    soft_persistent_enabled,
)
from sglang.srt.afd.farm.spin_wait_linear import SpinWaitLinearSession
from sglang.srt.afd.farm.token_queue import (
    BWinStats,
    LayerReadyQueues,
    TokenKey,
    simulate_bwin_kpi,
    simulate_farm_occupancy,
    take_contiguous_run,
)

__all__ = [
    "BWinStats",
    "FarmContinuousScheduler",
    "FarmContext",
    "FarmTicket",
    "LayerReadyQueues",
    "TokenKey",
    "SpinWaitLinearSession",
    "afd_farm_enabled",
    "apply_farm_env",
    "farm_b_step",
    "farm_b_win_k",
    "farm_coalesce_k",
    "farm_context_stagger_layers",
    "farm_enabled",
    "farm_layer_burst",
    "farm_layer_cg_enabled",
    "farm_max_inflight",
    "farm_num_contexts",
    "farm_persistent_linear_enabled",
    "farm_sched",
    "farm_soft_persistent_enabled",
    "farm_spin_wait_enabled",
    "get_continuous_scheduler",
    "install_persistent_linear_on_layers",
    "layer_cg_enabled",
    "layer_cg_stats",
    "maybe_weight_outer_tax",
    "persistent_linear_enabled",
    "plan_context_ranges",
    "reset_continuous_schedulers",
    "run_farm_layers",
    "run_layer_forward_pre_ffn",
    "simulate_bwin_kpi",
    "simulate_farm_occupancy",
    "soft_persistent_enabled",
    "take_contiguous_run",
    "weight_outer_linear",
]
