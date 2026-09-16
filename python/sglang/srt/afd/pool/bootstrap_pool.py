# SPDX-License-Identifier: Apache-2.0
"""Bootstrap AfPool from ModelRunner when SGLANG_AFD_POOL=1."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.afd.mode import AfdMode, get_afd_mode, get_afd_transport_name
from sglang.srt.afd.pool import (
    AfAttnClient,
    AfFfnWorker,
    load_topology_from_env,
    pool_enabled,
    resolve_local_rank,
    set_af_attn_client,
    set_af_ffn_worker,
    shutdown_af_pool,
)
from sglang.srt.afd.pool.topology import ensure_endpoint_dir
from sglang.srt.afd.runtime import AfdRuntime, get_afd_runtime
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def maybe_init_af_pool(
    *,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
    moe_topk: int = 0,
    ffn_compute=None,
    num_mb: Optional[int] = None,
    max_num_token: Optional[int] = None,
) -> bool:
    """Init AfPool client/worker. Returns True if pool path was taken."""
    if not pool_enabled():
        return False
    transport = get_afd_transport_name()
    if transport not in ("cuda_ipc", "nvlink"):
        logger.warning(
            "SGLANG_AFD_POOL=1 requires cuda_ipc (got %s); skipping pool init",
            transport,
        )
        return False

    mode = get_afd_mode()
    if mode == AfdMode.NULL:
        return False

    topo = load_topology_from_env()
    ensure_endpoint_dir(topo)
    local = resolve_local_rank(role=mode.value)
    num_mb = int(num_mb if num_mb is not None else envs.SGLANG_AFD_NUM_MB.get())
    max_tok = int(
        max_num_token
        if max_num_token is not None
        else envs.SGLANG_AFD_MAX_NUM_TOKEN.get()
    )

    if mode == AfdMode.ATTN:
        if local >= topo.num_attn:
            raise RuntimeError(
                f"AfPool Attn local_rank={local} >= num_attn={topo.num_attn}"
            )
        client = AfAttnClient(
            topo,
            attn_rank=local,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            num_mb=num_mb,
            max_num_token=max_tok,
            moe_topk=moe_topk,
        )
        set_af_attn_client(client)
        # Compat: expose link0 as global AfdRuntime for non-pool call sites.
        from sglang.srt.afd import runtime as rt_mod

        link = client._links[0]
        compat = AfdRuntime(
            mode=AfdMode.ATTN, transport=link.transport, pool=link.pool
        )
        with rt_mod._lock:
            if rt_mod._runtime is not None:
                try:
                    rt_mod._runtime.close()
                except Exception:
                    pass
            rt_mod._runtime = compat
        logger.info(
            "AfPool Attn ready rank=%s num_ffn=%s endpoints=%s",
            local,
            topo.num_ffn,
            list(topo.endpoints_for_attn(local)),
        )
        return True

    if mode == AfdMode.FFN:
        if local >= topo.num_ffn:
            raise RuntimeError(
                f"AfPool FFN local_rank={local} >= num_ffn={topo.num_ffn}"
            )
        if ffn_compute is None:
            from sglang.srt.afd.pool.ffn_worker import make_identity_compute

            ffn_compute = make_identity_compute(0.0)
            logger.warning("AfPool FFN: no model compute bound; using identity")
        worker = AfFfnWorker(
            topo,
            ffn_rank=local,
            hidden_size=hidden_size,
            compute=ffn_compute,
            device=device,
            dtype=dtype,
            num_mb=num_mb,
            max_num_token=max_tok,
            moe_topk=moe_topk,
        )
        worker.start_background()
        set_af_ffn_worker(worker)
        from sglang.srt.afd import runtime as rt_mod

        link_tr = worker._transports[0]
        link_pool = worker._pools[0]
        compat = AfdRuntime(
            mode=AfdMode.FFN, transport=link_tr, pool=link_pool
        )
        with rt_mod._lock:
            if rt_mod._runtime is not None:
                try:
                    rt_mod._runtime.close()
                except Exception:
                    pass
            rt_mod._runtime = compat
        logger.info(
            "AfPool FFN ready rank=%s num_attn=%s endpoints=%s",
            local,
            topo.num_attn,
            list(topo.endpoints_for_ffn(local)),
        )
        return True

    return False


def maybe_init_af_pool_from_model_runner(model_runner: "ModelRunner") -> bool:
    """Called from AFD bootstrap when pool env is on."""
    if not pool_enabled():
        return False
    if get_afd_runtime() is not None:
        logger.info("AFD runtime already set; skip AfPool re-init")
        return False

    hidden_size = int(model_runner.model_config.hidden_size)
    device = model_runner.device
    if device == "cuda":
        device = f"cuda:{model_runner.gpu_id}"
    dtype = getattr(model_runner, "dtype", None) or model_runner.model_config.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        dtype = torch.bfloat16

    from sglang.srt.afd.moe_bridge import detect_moe_topk_from_config
    from sglang.srt.afd.routing_scheme import is_scheme_b

    moe_topk = (
        0
        if is_scheme_b()
        else detect_moe_topk_from_config(model_runner.model_config.hf_config)
    )
    ffn_compute = None
    mode = get_afd_mode()
    if mode == AfdMode.FFN:
        try:
            from sglang.srt.afd.ffn_compute import make_model_ffn_compute
            from sglang.srt.afd.ffn_cuda_graph import maybe_wrap_ffn_cuda_graph

            max_tok = int(envs.SGLANG_AFD_MAX_NUM_TOKEN.get())
            ffn_compute = maybe_wrap_ffn_cuda_graph(
                model_runner.model,
                max_num_token=max_tok,
                device=device,
                dtype=dtype,
                hidden_size=hidden_size,
            )
        except Exception as e:
            logger.warning("AfPool FFN model compute bind failed: %s", e)
            from sglang.srt.afd.ffn_compute import make_model_ffn_compute

            try:
                ffn_compute = make_model_ffn_compute(model_runner.model)
            except Exception:
                ffn_compute = None

    return maybe_init_af_pool(
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
        moe_topk=moe_topk,
        ffn_compute=ffn_compute,
    )
