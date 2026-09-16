# SPDX-License-Identifier: Apache-2.0
"""Attention–FFN Disaggregation (AFD) via StepMesh.

See ``RFC.md`` in this package for the interface contract.

Bring-up::

    python -m sglang.srt.afd.smoke
    # multi-process example: bringup_stepmesh_example.sh
"""

from sglang.srt.afd.bootstrap import (
    apply_afd_cuda_graph_policy,
    maybe_init_afd_from_model_runner,
)
from sglang.srt.afd.mode import AfdMode, get_afd_mode, get_afd_transport_name
from sglang.srt.afd.module_stubs import (
    AfdExpertsStub,
    AfdMissingModule,
    afd_module_stubs_enabled,
    afd_skip_dense_mlp,
    afd_skip_entire_moe,
    afd_skip_experts,
    afd_skip_self_attn,
)
from sglang.srt.afd.pd_policy import apply_afd_pd_policy
from sglang.srt.afd.pipeline import (
    afd_pipeline_enabled,
    remote_ffn_pipelined,
    split_token_ranges,
)
from sglang.srt.afd.layer_pipeline import (
    afd_layer_pipeline_enabled,
    apply_stepmesh_stages_env,
    run_layers_pipelined,
    split_decode_mbs,
)
from sglang.srt.afd.routing_scheme import (
    AfdRoutingScheme,
    get_routing_scheme,
    is_scheme_a,
    is_scheme_b,
)
from sglang.srt.afd.runtime import (
    AfdRuntime,
    get_afd_runtime,
    init_afd_runtime,
    is_afd_attn,
    is_afd_enabled,
    is_afd_ffn,
    shutdown_afd_runtime,
)
from sglang.srt.afd.weight_filter import (
    filter_weights_for_afd,
    is_ffn_exclusive_weight,
    should_load_weight,
)

# AfPool MxN (optional) — light imports only (avoid pulling cuda_ipc at import).
from sglang.srt.afd.pool.topology import pool_enabled
from sglang.srt.afd.pool import get_af_attn_client, shutdown_af_pool
from sglang.srt.afd.farm.env import afd_farm_enabled, farm_enabled

__all__ = [
    "AfdExpertsStub",
    "AfdMissingModule",
    "AfdMode",
    "AfdRoutingScheme",
    "AfdRuntime",
    "afd_module_stubs_enabled",
    "afd_layer_pipeline_enabled",
    "afd_farm_enabled",
    "apply_stepmesh_stages_env",
    "afd_pipeline_enabled",
    "afd_skip_dense_mlp",
    "afd_skip_entire_moe",
    "afd_skip_experts",
    "afd_skip_self_attn",
    "apply_afd_cuda_graph_policy",
    "apply_afd_pd_policy",
    "filter_weights_for_afd",
    "farm_enabled",
    "get_af_attn_client",
    "get_afd_mode",
    "get_afd_runtime",
    "get_afd_transport_name",
    "get_routing_scheme",
    "init_afd_runtime",
    "is_afd_attn",
    "is_afd_enabled",
    "is_afd_ffn",
    "is_ffn_exclusive_weight",
    "is_scheme_a",
    "is_scheme_b",
    "maybe_init_afd_from_model_runner",
    "pool_enabled",
    "remote_ffn_pipelined",
    "run_layers_pipelined",
    "should_load_weight",
    "shutdown_af_pool",
    "shutdown_afd_runtime",
    "split_decode_mbs",
    "split_token_ranges",
]
