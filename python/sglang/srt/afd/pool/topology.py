# SPDX-License-Identifier: Apache-2.0
"""Parse AfPool topology from env."""

from __future__ import annotations

import os
from typing import Optional

from sglang.srt.afd.pool.types import PoolTopology
from sglang.srt.environ import envs


def pool_enabled() -> bool:
    return bool(envs.SGLANG_AFD_POOL.get())


def load_topology_from_env() -> PoolTopology:
    num_attn = max(1, int(envs.SGLANG_AFD_POOL_NUM_ATTN.get()))
    num_ffn = max(1, int(envs.SGLANG_AFD_POOL_NUM_FFN.get()))
    endpoint_dir = (envs.SGLANG_AFD_POOL_ENDPOINT_DIR.get() or "/tmp/afd_pool").strip()
    max_inf = max(1, int(envs.SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN.get()))
    route = (envs.SGLANG_AFD_POOL_ROUTE.get() or "least_inflight").strip()
    return PoolTopology(
        num_attn=num_attn,
        num_ffn=num_ffn,
        endpoint_dir=endpoint_dir,
        max_inflight_per_ffn=max_inf,
        route=route,
    )


def resolve_local_rank(*, role: str) -> int:
    """Return this process's rank inside Attn or FFN pool."""
    local = int(envs.SGLANG_AFD_POOL_LOCAL_RANK.get())
    if local >= 0:
        return local
    wr = int(envs.SGLANG_AFD_WORKER_RANK.get())
    if wr >= 0:
        return wr
    del role
    return 0


def ensure_endpoint_dir(topo: PoolTopology) -> None:
    os.makedirs(topo.endpoint_dir, exist_ok=True)


def apply_pool_env(
    *,
    num_attn: int,
    num_ffn: int,
    endpoint_dir: Optional[str] = None,
    max_inflight: Optional[int] = None,
    route: Optional[str] = None,
    enabled: bool = True,
) -> PoolTopology:
    """Set pool env vars (bench / launcher helpers)."""
    envs.SGLANG_AFD_POOL.set(bool(enabled))
    envs.SGLANG_AFD_POOL_NUM_ATTN.set(int(num_attn))
    envs.SGLANG_AFD_POOL_NUM_FFN.set(int(num_ffn))
    if endpoint_dir is not None:
        envs.SGLANG_AFD_POOL_ENDPOINT_DIR.set(str(endpoint_dir))
    if max_inflight is not None:
        envs.SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN.set(int(max_inflight))
    if route is not None:
        envs.SGLANG_AFD_POOL_ROUTE.set(str(route))
    envs.SGLANG_AFD_TRANSPORT.set("cuda_ipc")
    topo = load_topology_from_env()
    ensure_endpoint_dir(topo)
    return topo
