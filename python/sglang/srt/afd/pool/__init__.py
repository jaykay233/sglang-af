# SPDX-License-Identifier: Apache-2.0
"""AfPool: MxN Attn↔FFN work pool over real cuda_ipc.

Enable with ``SGLANG_AFD_POOL=1``. Default off — classic 1A1F TRUE_OVERLAP
unchanged. See ``README.md`` in this package.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from sglang.srt.afd.pool.attn_client import AfAttnClient
from sglang.srt.afd.pool.credit import FfnCreditWindow
from sglang.srt.afd.pool.ffn_worker import AfFfnWorker, make_identity_compute
from sglang.srt.afd.pool.router import AfRouter
from sglang.srt.afd.pool.scheduler import AfScheduler
from sglang.srt.afd.pool.topology import (
    apply_pool_env,
    ensure_endpoint_dir,
    load_topology_from_env,
    pool_enabled,
    resolve_local_rank,
)
from sglang.srt.afd.pool.types import (
    A2FWorkItem,
    AfTaskId,
    F2ACompletion,
    PoolStats,
    PoolTopology,
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_attn_client: Optional[AfAttnClient] = None
_ffn_worker: Optional[AfFfnWorker] = None


def get_af_attn_client() -> Optional[AfAttnClient]:
    return _attn_client


def get_af_ffn_worker() -> Optional[AfFfnWorker]:
    return _ffn_worker


def set_af_attn_client(client: Optional[AfAttnClient]) -> None:
    global _attn_client
    with _lock:
        if _attn_client is not None and _attn_client is not client:
            try:
                _attn_client.close()
            except Exception:
                pass
        _attn_client = client


def set_af_ffn_worker(worker: Optional[AfFfnWorker]) -> None:
    global _ffn_worker
    with _lock:
        if _ffn_worker is not None and _ffn_worker is not worker:
            try:
                _ffn_worker.close()
            except Exception:
                pass
        _ffn_worker = worker


def shutdown_af_pool() -> None:
    set_af_attn_client(None)
    set_af_ffn_worker(None)


__all__ = [
    "A2FWorkItem",
    "AfAttnClient",
    "AfFfnWorker",
    "AfRouter",
    "AfScheduler",
    "AfTaskId",
    "F2ACompletion",
    "FfnCreditWindow",
    "PoolStats",
    "PoolTopology",
    "apply_pool_env",
    "ensure_endpoint_dir",
    "get_af_attn_client",
    "get_af_ffn_worker",
    "load_topology_from_env",
    "make_identity_compute",
    "pool_enabled",
    "resolve_local_rank",
    "set_af_attn_client",
    "set_af_ffn_worker",
    "shutdown_af_pool",
]
