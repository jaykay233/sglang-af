# SPDX-License-Identifier: Apache-2.0
"""Transfer engine abstraction for tensor transfer between role instances."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_MOONCAKE_AVAILABLE = None


def _check_mooncake() -> bool:
    global _MOONCAKE_AVAILABLE
    if _MOONCAKE_AVAILABLE is None:
        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (  # noqa: F401
                MooncakeTransferEngine as _MTE,
            )

            _MOONCAKE_AVAILABLE = True
        except ImportError:
            _MOONCAKE_AVAILABLE = False
    return _MOONCAKE_AVAILABLE


class BaseTransferEngine(ABC):
    """Abstract transfer engine for data movement between roles."""

    @property
    def supports_gpu_direct(self) -> bool:
        return False

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @abstractmethod
    def register_buffer(self, ptr: int, length: int) -> None: ...

    @abstractmethod
    def deregister_buffer(self, ptr: int) -> None: ...

    @abstractmethod
    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        """Returns 0 on success, negative on failure."""

    @abstractmethod
    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int: ...


class MooncakeDiffusionEngine(BaseTransferEngine):
    """Production engine backed by MooncakeTransferEngine (RDMA/TCP)."""

    @property
    def supports_gpu_direct(self) -> bool:
        # GPUDirect only works with RDMA/GPU transports. Mooncake TCP cannot
        # reliably cudaMemcpy from a GPU pool (invalid argument / broken pipe
        # on single-node), so force pinned-CPU staging for TCP.
        return self._supports_gpu_direct

    def __init__(
        self,
        hostname: str,
        gpu_id: int = 0,
        ib_device: str | None = None,
    ):
        import os

        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            MooncakeTransferEngine,
        )
        from sglang.srt.environ import envs

        protocol = (
            "tcp"
            if os.environ.get("MC_FORCE_TCP") == "1"
            else envs.MOONCAKE_PROTOCOL.get()
        ).lower()
        self._supports_gpu_direct = protocol not in ("tcp", "ascend")

        self._engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )
        logger.info(
            "MooncakeDiffusionEngine initialized: session_id=%s protocol=%s "
            "gpu_direct=%s",
            self._engine.session_id,
            protocol,
            self._supports_gpu_direct,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    def register_buffer(self, ptr: int, length: int) -> None:
        self._engine.register(ptr, length)

    def deregister_buffer(self, ptr: int) -> None:
        self._engine.deregister(ptr)

    def transfer_sync(
        self, dst_session_id: str, src_addr: int, dst_addr: int, length: int
    ) -> int:
        return self._engine.transfer_sync(dst_session_id, src_addr, dst_addr, length)

    def batch_transfer_sync(
        self,
        dst_session_id: str,
        src_addrs: list[int],
        dst_addrs: list[int],
        lengths: list[int],
    ) -> int:
        return self._engine.batch_transfer_sync(
            dst_session_id, src_addrs, dst_addrs, lengths
        )


def create_transfer_engine(
    hostname: str = "127.0.0.1",
    gpu_id: int = 0,
    ib_device: str | None = None,
) -> BaseTransferEngine:
    """Factory for the role-to-role tensor transfer engine.

    Same-node NVLink path: set ``MOONCAKE_PROTOCOL=nvlink_intra`` (or
    ``cuda_ipc`` / ``DISAGG_CUDA_IPC=1``). The stock Mooncake wheel is often
    built without ``USE_INTRA_NVLINK``, so we use a CUDA-IPC engine that
    performs GPU↔GPU copies over NVLink/P2P instead.
    """
    import os

    from sglang.srt.environ import envs

    protocol = (
        "tcp"
        if os.environ.get("MC_FORCE_TCP") == "1"
        else envs.MOONCAKE_PROTOCOL.get()
    ).lower()
    use_cuda_ipc = (
        os.environ.get("DISAGG_CUDA_IPC") == "1"
        or protocol in ("nvlink_intra", "nvlink", "cuda_ipc")
    )
    if use_cuda_ipc:
        from sglang.multimodal_gen.runtime.disaggregation.transport.cuda_ipc_engine import (
            CudaIpcTransferEngine,
        )

        return CudaIpcTransferEngine(
            hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
        )

    if not _check_mooncake():
        raise RuntimeError(
            "Mooncake transfer engine is required for disaggregated diffusion "
            "but is not installed. Please install mooncake first."
        )
    return MooncakeDiffusionEngine(
        hostname=hostname, gpu_id=gpu_id, ib_device=ib_device
    )
