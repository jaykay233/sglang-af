# SPDX-License-Identifier: Apache-2.0
"""Bootstrap AFD from ModelRunner / ServerArgs after weights are loaded."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.afd.mode import AfdMode, get_afd_mode, get_afd_transport_name
from sglang.srt.afd.moe_bridge import detect_moe_topk_from_config
from sglang.srt.afd.runtime import (
    AfdRuntime,
    get_afd_runtime,
    init_afd_runtime,
    is_afd_enabled,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_ffn_loop_thread: Optional[threading.Thread] = None
_ffn_loop_stop: Optional[threading.Event] = None


def _resolve_max_num_token(model_runner: "ModelRunner") -> int:
    """Prefer decode CUDA-graph bucket size; fall back to env.

    FFN workers do not run model-level decode CG (stubs), so honor
    ``SGLANG_AFD_MAX_NUM_TOKEN`` only — do not inflate from unused CG max_bs.
    """
    env_max = max(1, int(envs.SGLANG_AFD_MAX_NUM_TOKEN.get()))
    if get_afd_mode() == AfdMode.FFN:
        return env_max
    try:
        cfg = model_runner.server_args.cuda_graph_config
        max_bs = cfg.decode.max_bs if cfg is not None else None
    except Exception:
        max_bs = None
    if max_bs is None:
        max_bs = getattr(model_runner.server_args, "cuda_graph_max_bs", None) or 160
    derived = int(max_bs)
    return max(env_max, derived)


def _resolve_dtype(model_runner: "ModelRunner") -> torch.dtype:
    dtype = getattr(model_runner, "dtype", None) or model_runner.model_config.dtype
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return torch.bfloat16


def _start_ffn_poll_loop(rt: AfdRuntime, compute) -> None:
    """Background get_batch/respond for FFN + StepMesh (or Fake without auto)."""
    global _ffn_loop_thread, _ffn_loop_stop
    if _ffn_loop_thread is not None and _ffn_loop_thread.is_alive():
        return
    _ffn_loop_stop = threading.Event()

    def _loop():
        assert _ffn_loop_stop is not None
        # Bind CUDA context on this thread before graph replay / MoE kernels.
        try:
            dev = getattr(rt.pool.cfg, "device", None)
            if torch.cuda.is_available() and dev is not None and str(dev).startswith("cuda"):
                torch.cuda.set_device(dev)
        except Exception:
            pass
        parallel_mb = bool(envs.SGLANG_AFD_FFN_PARALLEL_MB.get())
        num_mb = max(1, int(getattr(rt, "num_mb", 1) or 1))
        mb_streams: list = []
        if parallel_mb and torch.cuda.is_available() and num_mb > 1:
            # Prefer CudaGraphFfnCompute per-mb streams when present.
            get_st = getattr(compute, "_stream_for_mb", None)
            if callable(get_st):
                mb_streams = [get_st(i) for i in range(num_mb)]
            else:
                mb_streams = [
                    torch.cuda.Stream(device=torch.cuda.current_device())
                    for _ in range(num_mb)
                ]
        logger.info(
            "AFD FFN poll loop started gather_us=%s gather_max=%s parallel_mb=%s streams=%s",
            envs.SGLANG_AFD_FFN_GATHER_US.get(),
            envs.SGLANG_AFD_FFN_GATHER_MAX.get(),
            parallel_mb,
            len(mb_streams),
        )
        while not _ffn_loop_stop.is_set():
            try:
                batches = rt.transport.get_batch(timeout_s=0.05)
            except Exception as e:
                logger.exception("AFD FFN get_batch failed: %s", e)
                continue
            if not batches:
                continue
            # Same-layer gather: fuse MoE when multiple slots share layer_id.
            from collections import defaultdict

            from sglang.srt.afd.ffn_compute import try_compute_same_layer_group

            by_layer = defaultdict(list)
            for batch in batches:
                by_layer[batch.layer_id].append(batch)

            for layer_id, group in by_layer.items():
                try:
                    from sglang.srt.afd.detail_profile import (
                        profile_detail_enabled,
                        record_span,
                    )

                    _detail = profile_detail_enabled()
                    # Parallel mb path: skip fused concat so each mb keeps its stream.
                    fused = None
                    if not (parallel_mb and len(group) > 1 and mb_streams):
                        fused = try_compute_same_layer_group(compute, group)
                    if fused is not None:
                        tl = getattr(rt.transport, "_timeline", None)
                        for batch, outs in zip(group, fused):
                            if tl is not None:
                                tl.mark_ffn_compute(
                                    int(batch.handler), time.perf_counter()
                                )
                            rt.transport.respond(batch, outs)
                        continue

                    def _run_one(batch, st=None):
                        tl = getattr(rt.transport, "_timeline", None)
                        t0 = time.perf_counter() if _detail else 0.0
                        if st is not None:
                            with torch.cuda.stream(st):
                                outs = compute(batch)
                            # Respond only after this mb's stream finishes — do not
                            # defer across get_batch polls (that inflated ffn_to_respond).
                            st.synchronize()
                        else:
                            outs = compute(batch)
                        if _detail:
                            t_wall = time.perf_counter()
                            record_span("compute_wall_us", t0, t_wall)
                        if tl is not None:
                            tl.mark_ffn_compute(
                                int(batch.handler), time.perf_counter()
                            )
                        if _detail:
                            try:
                                torch.cuda.synchronize()
                            except Exception:
                                pass
                            record_span("compute_cuda_us", t0)
                        rt.transport.respond(batch, outs)

                    if parallel_mb and len(group) > 1 and mb_streams:
                        # Both ready: launch on separate streams, sync+respond each.
                        launched = []
                        for batch in group:
                            mb = int(getattr(batch, "_mb_id", 0) or 0)
                            st = mb_streams[mb % len(mb_streams)]
                            t0 = time.perf_counter() if _detail else 0.0
                            with torch.cuda.stream(st):
                                outs = compute(batch)
                            launched.append((batch, outs, st, t0))
                        for batch, outs, st, t0 in launched:
                            st.synchronize()
                            if _detail:
                                record_span("compute_wall_us", t0)
                            tl = getattr(rt.transport, "_timeline", None)
                            if tl is not None:
                                tl.mark_ffn_compute(
                                    int(batch.handler), time.perf_counter()
                                )
                            rt.transport.respond(batch, outs)
                    else:
                        for batch in group:
                            _run_one(batch)
                except Exception:
                    logger.exception(
                        "AFD FFN compute/respond failed layer=%s n=%s",
                        layer_id,
                        len(group),
                    )

    _ffn_loop_thread = threading.Thread(
        target=_loop, name="afd-ffn-poll", daemon=True
    )
    _ffn_loop_thread.start()


def maybe_init_afd_from_model_runner(model_runner: "ModelRunner") -> None:
    """Initialize AFD runtime after ``load_model`` when ``SGLANG_AFD_MODE`` != null.

    - ATTN + fake: in-process FFN using the same loaded weights (smoke / CI).
    - ATTN + stepmesh: RDMA worker; remote FFN process must be running.
    - FFN + *: serve experts via poll loop (stepmesh) or Fake auto-thread.
    """
    if not is_afd_enabled():
        return
    if get_afd_runtime() is not None:
        logger.info("AFD runtime already initialized; skip re-init")
        return

    try:
        from sglang.srt.afd.farm.env import apply_farm_env

        apply_farm_env()
    except Exception:
        pass

    # AfPool MxN (experimental): takes over when SGLANG_AFD_POOL=1 + cuda_ipc.
    try:
        from sglang.srt.afd.pool.bootstrap_pool import (
            maybe_init_af_pool_from_model_runner,
        )
        from sglang.srt.afd.pool.topology import pool_enabled

        if pool_enabled() and maybe_init_af_pool_from_model_runner(model_runner):
            logger.info("AFD bootstrap: AfPool path active (classic 1A1F skipped)")
            return
    except Exception:
        logger.exception("AfPool bootstrap failed; falling back to classic AFD")

    # StepMesh stages=3 → NUM_MB=3 + layer pipeline before buffer alloc.
    try:
        from sglang.srt.afd.layer_pipeline import apply_stepmesh_stages_env

        apply_stepmesh_stages_env()
    except Exception:
        pass

    # Layer-merge forces from_layer=0 and disables in-graph / true-overlap.
    try:
        from sglang.srt.afd.remote_policy import apply_layer_merge_env

        apply_layer_merge_env()
    except Exception:
        pass

    mode = get_afd_mode()
    transport = get_afd_transport_name()
    hidden_size = int(model_runner.model_config.hidden_size)
    device = model_runner.device
    if device == "cuda":
        device = f"cuda:{model_runner.gpu_id}"
    dtype = _resolve_dtype(model_runner)
    max_num_token = _resolve_max_num_token(model_runner)
    envs.SGLANG_AFD_MAX_NUM_TOKEN.set(max_num_token)

    hf_cfg = model_runner.model_config.hf_config
    from sglang.srt.afd.routing_scheme import is_scheme_b

    # Scheme B: routing runs on FFN; A2F is hidden (+ meta) only.
    moe_topk = 0 if is_scheme_b() else detect_moe_topk_from_config(hf_cfg)

    worker_rank = int(envs.SGLANG_AFD_WORKER_RANK.get())
    if worker_rank < 0:
        worker_rank = int(getattr(model_runner, "tp_rank", 0) or 0)

    from sglang.srt.afd.ffn_compute import (
        make_identity_ffn_compute,
        make_model_ffn_compute,
    )

    try:
        if mode == AfdMode.FFN:
            from sglang.srt.afd.ffn_cuda_graph import maybe_wrap_ffn_cuda_graph

            model_compute = maybe_wrap_ffn_cuda_graph(
                model_runner.model,
                max_num_token=max_num_token,
                device=device,
                dtype=dtype,
                hidden_size=hidden_size,
            )
        else:
            model_compute = make_model_ffn_compute(model_runner.model)
    except Exception as e:
        logger.warning(
            "AFD could not bind model FFN compute (%s); using identity FFN", e
        )
        model_compute = make_identity_ffn_compute()

    ffn_compute = None
    if transport == "fake":
        ffn_compute = model_compute
        if mode == AfdMode.ATTN:
            logger.warning(
                "AFD ATTN+fake: in-process FFN (smoke). "
                "Use SGLANG_AFD_TRANSPORT=stepmesh for real remote FFN."
            )

    rt = init_afd_runtime(
        hidden_size=hidden_size,
        mode=mode,
        transport_name=transport,
        device=device,
        dtype=dtype,
        ffn_compute=ffn_compute,
        moe_topk=moe_topk,
        worker_rank=worker_rank,
    )
    assert rt is not None

    # FFN process: bind IPC staging (private MoE + in-graph F2A copy) then poll.
    if mode == AfdMode.FFN and transport in ("stepmesh", "cuda_ipc"):
        can_bind_ffn_graph = hasattr(model_compute, "bind_shared_slot")
        if (
            transport == "cuda_ipc"
            and hasattr(rt.transport, "bind_ffn_shared_io")
            and can_bind_ffn_graph
        ):
            try:
                rt.transport.bind_ffn_shared_io(model_compute)
                # Warm every decode-sized bucket so the first real batch does not
                # pay on-demand capture (was only [1, max] → mid-bench bs=8 capture).
                if hasattr(model_compute, "capture_buckets"):
                    buckets = list(getattr(model_compute, "_buckets", None) or [])
                    if not buckets:
                        buckets = [1, max_num_token]
                    model_compute.capture_buckets(buckets=buckets)
            except Exception as e:
                logger.warning("AFD FFN IPC staging bind/warm failed: %s", e)
        elif transport == "cuda_ipc":
            logger.info(
                "AFD FFN eager compute: IPC CUDA-graph staging bind skipped "
                "(SGLANG_AFD_FFN_CUDA_GRAPH=0)"
            )
        _start_ffn_poll_loop(rt, model_compute)

    # Optional wait_flag (TRUE_OVERLAP is always forced for Attn decode).
    try:
        from sglang.srt.afd.layer_pipeline import apply_true_overlap_env

        if mode == AfdMode.ATTN:
            apply_true_overlap_env()
    except Exception:
        pass
    want_flags = (
        bool(envs.SGLANG_AFD_USE_WAIT_FLAG.get())
        or bool(envs.SGLANG_AFD_TRUE_OVERLAP.get())
    )
    if want_flags and mode == AfdMode.ATTN and transport in ("stepmesh", "cuda_ipc"):
        try:
            rt.enable_wait_flag_sync(gpu_id=model_runner.gpu_id)
        except Exception as e:
            logger.warning(
                "AFD wait_flag enable failed, falling back to mailbox soft wait: %s",
                e,
            )

    logger.info(
        "AFD auto-init done mode=%s transport=%s hidden=%s max_token=%s "
        "moe_topk=%s device=%s module_stubs=%s scheme=%s a2f_dtype=%s "
        "worker_rank=%s in_graph_wait=%s remote_moe_only=%s remote_from_layer=%s "
        "layer_merge_k=%s",
        mode.value,
        transport,
        hidden_size,
        max_num_token,
        moe_topk,
        device,
        envs.SGLANG_AFD_MODULE_STUBS.get(),
        envs.SGLANG_AFD_ROUTING_SCHEME.get(),
        envs.SGLANG_AFD_A2F_DTYPE.get(),
        worker_rank,
        envs.SGLANG_AFD_IN_GRAPH_WAIT.get(),
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.get(),
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.get(),
        envs.SGLANG_AFD_LAYER_MERGE_K.get(),
    )

    # merge_kv needs token_to_kv_pool — deferred to maybe_init_merge_kv_after_pool.


def apply_afd_cuda_graph_policy(server_args) -> None:
    """AFD CUDA Graph policy for Attn vs FFN processes.

    * Attn + in-graph wait: ``full`` decode CG (GPU flags inside the graph).
    * Attn + layer pipeline / StepMesh stages: ``breakable`` (stagger is eager).
    * Attn + layer-merge: ``breakable`` (group RTT is eager).
    * Attn default: ``breakable`` (remote FFN is an eager graph break).
    * FFN: ``disabled`` — model CG hits attn stubs; use FFN compute graphs.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    if not is_afd_enabled():
        return

    try:
        from sglang.srt.afd.remote_policy import apply_layer_merge_env

        apply_layer_merge_env()
    except Exception:
        pass

    try:
        from sglang.srt.afd.layer_pipeline import apply_stepmesh_stages_env

        apply_stepmesh_stages_env()
    except Exception:
        pass

    try:
        from sglang.srt.afd.farm.env import apply_farm_env

        apply_farm_env()
    except Exception:
        pass

    cfg = server_args.cuda_graph_config
    if cfg is None:
        return

    mode = get_afd_mode()
    locked = getattr(server_args, "_cuda_graph_config_locked", None)

    if mode == AfdMode.FFN:
        if cfg.decode.backend != Backend.DISABLED:
            logger.warning(
                "AFD FFN: overriding decode CUDA graph backend %s -> disabled "
                "(model CG hits attn stubs); enable SGLANG_AFD_FFN_CUDA_GRAPH "
                "for FFN compute graphs",
                cfg.decode.backend,
            )
            cfg.decode.backend = Backend.DISABLED
            if locked is not None:
                locked.add((Phase.DECODE, "backend"))
        else:
            logger.info(
                "AFD FFN: model decode CUDA graph disabled; "
                "FFN compute graphs via SGLANG_AFD_FFN_CUDA_GRAPH=%s",
                envs.SGLANG_AFD_FFN_CUDA_GRAPH.get(),
            )
        return

    # Layer-merge: always breakable (in-graph disabled by apply_layer_merge_env).
    from sglang.srt.afd.remote_policy import layer_merge_k

    if layer_merge_k() > 1:
        if cfg.decode.backend not in (Backend.BREAKABLE, Backend.DISABLED):
            logger.warning(
                "AFD layer-merge: overriding decode CUDA graph backend %s -> breakable",
                cfg.decode.backend,
            )
            cfg.decode.backend = Backend.BREAKABLE
            if locked is not None:
                locked.add((Phase.DECODE, "backend"))
        return

    # P7 / StepMesh stages: Python stagger every forward. Prefer breakable so
    # Attn kernels can still hit CUDA-graph segments around eager issue/wait.
    # Full CG is incompatible; disabled loses too much on Lite.
    if bool(envs.SGLANG_AFD_LAYER_PIPELINE.get()) and not bool(
        envs.SGLANG_AFD_IN_GRAPH_WAIT.get()
    ):
        if cfg.decode.backend not in (Backend.BREAKABLE, Backend.DISABLED):
            logger.warning(
                "AFD layer pipeline (StepMesh stages): overriding decode CUDA "
                "graph backend %s -> breakable",
                cfg.decode.backend,
            )
            cfg.decode.backend = Backend.BREAKABLE
            if locked is not None:
                locked.add((Phase.DECODE, "backend"))
        elif cfg.decode.backend == Backend.DISABLED:
            logger.info(
                "AFD layer pipeline: decode CUDA graph left disabled "
                "(StepMesh %s-stage stagger is eager)",
                envs.SGLANG_AFD_STEPMESH_STAGES.get() or envs.SGLANG_AFD_NUM_MB.get(),
            )
        return

    # P8: in-graph wait was an alternate to TRUE_OVERLAP; no longer supported
    # for AFD Attn decode (TRUE_OVERLAP is required and clears IN_GRAPH_WAIT).
    if bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get()):
        logger.warning(
            "AFD SGLANG_AFD_IN_GRAPH_WAIT=1 ignored; TRUE_OVERLAP requires "
            "breakable CG (forcing IN_GRAPH_WAIT=0)"
        )
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)

    if cfg.decode.backend == Backend.DISABLED:
        logger.info("AFD enabled but decode CUDA graph disabled; leave disabled")
        return
    if cfg.decode.backend != Backend.BREAKABLE:
        logger.warning(
            "AFD enabled: overriding decode CUDA graph backend %s -> breakable",
            cfg.decode.backend,
        )
        cfg.decode.backend = Backend.BREAKABLE
        if locked is not None:
            locked.add((Phase.DECODE, "backend"))
