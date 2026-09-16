# SPDX-License-Identifier: Apache-2.0
"""Same-node CUDA IPC transfer engine (NVLink/P2P, no host bounce)."""

from __future__ import annotations

import base64
import logging
import threading

from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress, get_free_port

logger = logging.getLogger(__name__)


def _cudart():
    from cuda.bindings import runtime as cudart

    return cudart


def _check(ret):
    cudart = _cudart()
    err = ret[0] if isinstance(ret, tuple) else ret
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA IPC error: {err}")
    return ret[1] if isinstance(ret, tuple) and len(ret) > 1 else None


class CudaIpcTransferEngine(BaseTransferEngine):
    """GPU↔GPU transfer via cudaIpc* (uses NVLink when peer-accessible)."""

    def __init__(
        self,
        hostname: str = "127.0.0.1",
        gpu_id: int = 0,
        ib_device: str | None = None,
    ):
        self.hostname = hostname
        self.gpu_id = gpu_id
        self._session_id = NetworkAddress(hostname, get_free_port()).to_host_port_str()
        self._pool_ptr: int | None = None
        self._pool_size: int = 0
        self._ipc_raw: bytes | None = None
        self._lock = threading.Lock()
        # session_id -> (mapped_base_ptr, remote_pool_ptr)
        self._remotes: dict[str, tuple[int, int]] = {}
        logger.info(
            "CudaIpcTransferEngine initialized: session_id=%s gpu_id=%s",
            self._session_id,
            gpu_id,
        )

    @property
    def supports_gpu_direct(self) -> bool:
        return True

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def ipc_handle_b64(self) -> str | None:
        if not self._ipc_raw:
            return None
        return base64.b64encode(self._ipc_raw).decode("ascii")

    def register_buffer(self, ptr: int, length: int) -> None:
        cudart = _cudart()
        _check(cudart.cudaSetDevice(self.gpu_id))
        handle = _check(cudart.cudaIpcGetMemHandle(ptr))
        self._pool_ptr = ptr
        self._pool_size = length
        self._ipc_raw = bytes(bytearray(handle.reserved))
        logger.info(
            "CudaIpcTransferEngine: exported IPC handle for pool %#x (%d bytes)",
            ptr,
            length,
        )

    def deregister_buffer(self, ptr: int) -> None:
        # Local pool lifetime is owned by TransferTensorBuffer; just drop handle.
        if self._pool_ptr == ptr:
            self._pool_ptr = None
            self._ipc_raw = None

    def note_remote(
        self, session_id: str, ipc_handle_b64: str | None, pool_ptr: int
    ) -> None:
        """Cache receiver IPC handle so transfer_sync can map it."""
        if not ipc_handle_b64:
            return
        with self._lock:
            existing = self._remotes.get(session_id)
            if existing and existing[1] == pool_ptr:
                return
            # Close previous mapping if session changes handle/pool.
            if existing:
                try:
                    _check(_cudart().cudaIpcCloseMemHandle(existing[0]))
                except Exception:
                    pass
                self._remotes.pop(session_id, None)

            cudart = _cudart()
            _check(cudart.cudaSetDevice(self.gpu_id))
            raw = base64.b64decode(ipc_handle_b64)
            handle = cudart.cudaIpcMemHandle_t()
            handle.reserved = raw
            base = _check(
                cudart.cudaIpcOpenMemHandle(
                    handle, cudart.cudaIpcMemLazyEnablePeerAccess
                )
            )
            self._remotes[session_id] = (int(base), int(pool_ptr))
            logger.debug(
                "CudaIpcTransferEngine: opened remote %s pool=%#x -> %#x",
                session_id,
                pool_ptr,
                base,
            )

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        try:
            with self._lock:
                remote = self._remotes.get(dst_session_id)
            if remote is None:
                logger.error(
                    "CudaIpcTransferEngine: no IPC mapping for %s", dst_session_id
                )
                return -1
            base, pool_ptr = remote
            offset = int(dst_addr) - int(pool_ptr)
            if offset < 0:
                logger.error(
                    "CudaIpcTransferEngine: bad offset dst=%#x pool=%#x",
                    dst_addr,
                    pool_ptr,
                )
                return -1
            cudart = _cudart()
            _check(cudart.cudaSetDevice(self.gpu_id))
            _check(
                cudart.cudaMemcpy(
                    base + offset,
                    int(src_addr),
                    int(length),
                    cudart.cudaMemcpyKind.cudaMemcpyDefault,
                )
            )
            return 0
        except Exception:
            logger.exception("CudaIpcTransferEngine.transfer_sync failed")
            return -1

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        for s, d, n in zip(src_addrs, dst_addrs, lengths):
            ret = self.transfer_sync(dst_session_id, s, d, n)
            if ret != 0:
                return ret
        return 0
