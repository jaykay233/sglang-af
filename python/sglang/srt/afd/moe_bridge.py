# SPDX-License-Identifier: Apache-2.0
"""MoE routing on Attn + experts on FFN (AFD scheme A)."""

from __future__ import annotations

import os
import time
import logging
from typing import TYPE_CHECKING, Tuple

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def _section_timing_enabled() -> bool:
    from sglang.srt.afd.detail_profile import profile_detail_enabled

    if not profile_detail_enabled():
        return False
    return os.environ.get("SGLANG_AFD_FARM_FFN_SECTION_TIME", "0") in (
        "1",
        "true",
        "on",
        "yes",
    )


def _timed_call(name: str, fn):
    from sglang.srt.afd.detail_profile import record_us

    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        record_us(name, (time.perf_counter() - t0) * 1e6)


_PROBE_STATE = {"done": False, "calls": 0}


def _probe_enabled() -> bool:
    """One-shot CUDA-kernel census for ``ffn_routed`` (see progress.md §8).

    Answers whether the ~1.6 ms host cost is CUDA launch overhead or Python /
    dispatch work: it reports the kernel count and the CPU time per launch.
    """
    from sglang.srt.afd.detail_profile import profile_detail_enabled

    if not profile_detail_enabled():
        return False
    return os.environ.get("SGLANG_AFD_FARM_FFN_PROBE", "0") in (
        "1",
        "true",
        "on",
        "yes",
    )


def _probe_warmup() -> int:
    """How many ``ffn_routed`` calls to skip before the census.

    Profiling the *first* call reports a cold, un-warmed process (lazy kernel
    loading, autotune, caching allocator growth), which is why the first probe
    run showed an unusable wall time. Default 200.
    """
    try:
        return max(0, int(os.environ.get("SGLANG_AFD_FARM_FFN_PROBE_WARMUP", "200")))
    except ValueError:
        return 200


def _run_routed_probe(fn):
    """Profile one ``ffn_routed`` call and log a kernel/CPU census."""
    import torch

    _PROBE_STATE["calls"] += 1
    if _PROBE_STATE["calls"] <= _probe_warmup():
        return fn()
    _PROBE_STATE["done"] = True
    try:
        torch.cuda.synchronize()
        c0 = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            out = fn()
            torch.cuda.synchronize()
        c1 = time.perf_counter()
    except Exception:
        _PROBE_STATE["done"] = True
        logger.exception("[ffn-probe] profiler failed; falling back to plain call")
        return fn()

    try:
        from torch.autograd import DeviceType

        wall_us = (c1 - c0) * 1e6
        events = list(prof.events())
        kerns = [e for e in events if e.device_type == DeviceType.CUDA]
        n_k = len(kerns)
        gpu_us = sum(float(e.self_device_time_total) for e in kerns)
        ka = list(prof.key_averages())
        cpu_us = sum(float(e.self_cpu_time_total) for e in ka)
        logger.info(
            "[ffn-probe] ffn_routed wall=%.0fus kernels=%d "
            "gpu_kernel_total=%.0fus cpu_total=%.0fus "
            "cpu_per_launch=%.1fus gpu_idle_in_region=%.0fus",
            wall_us,
            n_k,
            gpu_us,
            cpu_us,
            cpu_us / max(1, n_k),
            max(0.0, wall_us - gpu_us),
        )
        for e in sorted(kerns, key=lambda x: -float(x.self_device_time_total))[:8]:
            logger.info(
                "[ffn-probe] gpu %-58s %8.1fus",
                str(e.name)[:58],
                float(e.self_device_time_total),
            )
        for e in sorted(ka, key=lambda x: -float(x.self_cpu_time_total))[:10]:
            logger.info(
                "[ffn-probe] cpu %-58s %8.1fus x%d (cuda %.1f)",
                str(e.key)[:58],
                float(e.self_cpu_time_total),
                int(e.count),
                float(e.self_device_time_total),
            )
    except Exception:
        logger.exception("[ffn-probe] summary failed")
    return out


def attn_compute_routing(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run gate + topk on the Attn worker. Returns (topk_ids, topk_weights)."""
    if hidden_states.shape[0] == 0:
        empty = moe.topk.empty_topk_output(hidden_states.device, layer_id=moe.layer_id)
        return empty.topk_ids, empty.topk_weights

    try:
        from sglang.srt.afd.detail_profile import (
            profile_detail_enabled as _pd_on,
            record_span as _pd_span,
        )

        _pd_timing = bool(_pd_on())
    except Exception:
        _pd_timing = False

    _t_gate = time.perf_counter() if _pd_timing else 0.0
    router_logits = moe.gate(hidden_states)
    if _pd_timing:
        _pd_span("attn_gate_us", _t_gate)

    _t_topk = time.perf_counter() if _pd_timing else 0.0
    topk_output = moe.topk(hidden_states, router_logits)
    if _pd_timing:
        _pd_span("attn_topk_us", _t_topk)
    if not hasattr(topk_output, "topk_ids"):
        raise TypeError(
            f"AFD expects StandardTopKOutput-like routing, got {type(topk_output)}"
        )
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    num_fused_shared = int(getattr(moe, "num_fused_shared_experts", 0) or 0)
    if num_fused_shared > 0:
        # The generic DeepSeek TopK path pads every shared slot with the same
        # expert id and derives its weight from the routed sum. AF scheme A
        # carries the shared experts as real MoE slots, so make the ids and
        # weights explicit here. The current AF enablement is restricted to
        # DeepSeek-V2-Lite with routed_scaling_factor == 1.0.
        expected_topk = int(moe.config.num_experts_per_tok) + num_fused_shared
        if topk_ids.shape[-1] != expected_topk:
            raise RuntimeError(
                "AFD fused shared-expert routing width mismatch: "
                f"got {topk_ids.shape[-1]}, expected {expected_topk}"
            )
        first_shared = int(moe.config.n_routed_experts)
        shared_ids = torch.arange(
            first_shared,
            first_shared + num_fused_shared,
            device=topk_ids.device,
            dtype=topk_ids.dtype,
        )
        topk_ids[:, -num_fused_shared:] = shared_ids
        topk_weights[:, -num_fused_shared:] = 1.0

    return topk_ids, topk_weights


def ffn_apply_experts(
    moe: "DeepseekV2MoE",
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    should_allreduce_fusion: bool = False,
    use_reduce_scatter: bool = False,
) -> torch.Tensor:
    """Run shared experts + routed experts with precomputed topk (FFN worker)."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.moe import should_skip_post_experts_all_reduce
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        maybe_fuse_routed_scale_and_shared_add,
    )

    if hidden_states.shape[0] == 0:
        return hidden_states

    section_timing = _section_timing_enabled()

    def _prepare_topk():
        n_routed = int(moe.config.n_routed_experts)
        # Reuse a per-module buffer to avoid allocating [T, n_routed] every call
        # (matters for eager path and CUDA-graph capture noise).
        buf = getattr(moe, "_afd_router_logits_buf", None)
        if (
            buf is None
            or buf.device != hidden_states.device
            or buf.dtype != hidden_states.dtype
            or buf.shape[0] < hidden_states.shape[0]
            or buf.shape[1] != n_routed
        ):
            buf = hidden_states.new_zeros(hidden_states.shape[0], n_routed)
            moe._afd_router_logits_buf = buf
        router_logits = buf[: hidden_states.shape[0]]
        router_logits.zero_()
        return StandardTopKOutput(
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            router_logits,
        )

    topk_output = (
        _timed_call("ffn_prep_us", _prepare_topk)
        if section_timing
        else _prepare_topk()
    )

    skip_shared = False
    shared_output = None
    if (
        hasattr(moe, "shared_experts")
        and moe.shared_experts is not None
        and int(getattr(moe, "num_fused_shared_experts", 0) or 0) == 0
        and not getattr(moe, "_fuse_shared_experts_inside_sbo", False)
        and not skip_shared
    ):
        shared_output = (
            _timed_call(
                "ffn_shared_us",
                lambda: moe._forward_shared_experts(hidden_states),
            )
            if section_timing
            else moe._forward_shared_experts(hidden_states)
        )

    if hasattr(moe.experts, "forward_impl"):
        routed_fn = lambda: moe.experts.forward_impl(hidden_states, topk_output)
    else:
        routed_fn = lambda: moe.experts(hidden_states, topk_output)
    if _probe_enabled() and not _PROBE_STATE["done"]:
        final_hidden_states = _run_routed_probe(routed_fn)
    elif section_timing:
        # Split wall / process CPU / *calling-thread* CPU for the routed region.
        # progress.md §15: process CPU exceeding wall only proves *something* in
        # the process ran; thread CPU isolates whether the FFN compute thread
        # itself is the busy one. If thread ~= wall, no thread competition and
        # the in-situ inflation is GPU-queue back-pressure instead.
        from sglang.srt.afd.detail_profile import record_us

        _c0, _w0 = time.process_time(), time.perf_counter()
        _t0 = time.thread_time()
        final_hidden_states = routed_fn()
        _w1 = time.perf_counter()
        _t1, _c1 = time.thread_time(), time.process_time()
        record_us("ffn_routed_us", (_w1 - _w0) * 1e6)
        record_us("ffn_routed_cpu_us", (_c1 - _c0) * 1e6)
        record_us("ffn_routed_thr_us", (_t1 - _t0) * 1e6)
    else:
        final_hidden_states = routed_fn()

    fuse_fn = lambda: maybe_fuse_routed_scale_and_shared_add(
        moe.experts,
        final_hidden_states,
        None if getattr(moe, "_shared_expert_tp1", False) else shared_output,
        moe.routed_scaling_factor,
    )
    final_hidden_states = (
        _timed_call("ffn_fuse_us", fuse_fn) if section_timing else fuse_fn()
    )

    def _postprocess():
        nonlocal final_hidden_states
        if moe.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        if getattr(moe, "_shared_expert_tp1", False) and shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states

    if section_timing:
        return _timed_call("ffn_post_us", _postprocess)
    return _postprocess()


def detect_moe_topk_from_config(hf_config) -> int:
    """Return the top-k width carried by scheme-A A2F buffers."""
    n_routed = getattr(hf_config, "n_routed_experts", None)
    if not n_routed:
        return 0
    per_tok = int(getattr(hf_config, "num_experts_per_tok", 0) or 0)
    n_shared = int(getattr(hf_config, "n_shared_experts", 0) or 0)
    if (
        int(n_routed) == 64
        and per_tok == 6
        and n_shared == 2
        and float(getattr(hf_config, "routed_scaling_factor", 0.0) or 0.0) == 1.0
    ):
        # DeepSeek-V2-Lite AF path fuses both shared experts into the MoE
        # kernel, so A2F must carry 6 routed + 2 shared slots.
        return per_tok + n_shared
    return per_tok


def get_decoder_layers(model: torch.nn.Module):
    """Best-effort locate transformer layers list on common SGLang wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find .layers on {type(model)}")
