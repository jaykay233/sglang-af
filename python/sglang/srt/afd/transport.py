# SPDX-License-Identifier: Apache-2.0
"""AFD transport backends: Fake (CI) and StepMesh (RDMA)."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, List, Optional, Sequence

import torch

from sglang.srt.afd.buffers import AfdBufferPool
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.protocol import (
    SLOT_HIDDEN,
    SLOT_LAYER_ID,
    SLOT_NUM_TOKENS,
    AfdServerBatch,
    gen_pull_key,
    gen_push_key,
)

logger = logging.getLogger(__name__)


@dataclass
class AfdHandle:
    id: int


FfnComputeFn = Callable[[AfdServerBatch], Sequence[torch.Tensor]]


class AfdTransport(ABC):
    @abstractmethod
    def init(self, role: AfdMode, *, worker_rank: int = 0, num_workers: int = 1) -> None:
        ...

    @abstractmethod
    def register_buffers(self, pool: AfdBufferPool) -> None:
        ...

    @abstractmethod
    def push_pull(
        self,
        *,
        layer_id: int,
        mb_id: int,
        a2f: Sequence[torch.Tensor],
        f2a: Sequence[torch.Tensor],
        num_tokens: Optional[int] = None,
    ) -> AfdHandle:
        ...

    @abstractmethod
    def wait(self, handle: AfdHandle, timeout_ms: int = 5000) -> None:
        ...

    @abstractmethod
    def get_batch(self, timeout_s: float = 1.0) -> List[AfdServerBatch]:
        ...

    @abstractmethod
    def respond(
        self, batch: AfdServerBatch, tensors: Sequence[torch.Tensor]
    ) -> None:
        ...

    def close(self) -> None:
        return None

    def poll_done(self, handle: AfdHandle) -> bool:
        """Non-blocking: True if F2A is ready (does not consume the handle)."""
        return False


class FakeAfdTransport(AfdTransport):
    """Same-process queue transport.

    If ``ffn_compute`` is set, a background thread serves FFN automatically
    (Attn-only unit tests). Otherwise the caller runs an explicit FFN loop
    via ``get_batch`` / ``respond``.
    """

    def __init__(self, ffn_compute: Optional[FfnComputeFn] = None):
        self._role = AfdMode.NULL
        self._worker_rank = 0
        self._num_workers = 1
        self._pool: Optional[AfdBufferPool] = None
        self._req_q: Queue = Queue()
        self._done: dict[int, threading.Event] = {}
        self._handle_counter = 0
        self._lock = threading.Lock()
        self._ffn_compute = ffn_compute
        self._server_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _fake_async_ffn_enabled(self) -> bool:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_AFD_FAKE_ASYNC_FFN.get())

    def _fake_transfer_s(self) -> float:
        from sglang.srt.environ import envs

        return max(0.0, float(envs.SGLANG_AFD_FAKE_TRANSFER_MS.get()) / 1000.0)

    def _ensure_ffn_poll_thread(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return
        self._stop.clear()
        self._server_thread = threading.Thread(
            target=self._auto_ffn_loop, name="afd-fake-ffn", daemon=True
        )
        self._server_thread.start()

    def init(self, role: AfdMode, *, worker_rank: int = 0, num_workers: int = 1) -> None:
        self._role = role
        self._worker_rank = worker_rank
        self._num_workers = num_workers
        # Default collocated Fake serves FFN synchronously in push_pull (CUDA-safe).
        # FFN-role, or opt-in FAKE_ASYNC_FFN, uses a poll thread for overlap benches.
        if self._ffn_compute is not None and (
            role == AfdMode.FFN or self._fake_async_ffn_enabled()
        ):
            self._ensure_ffn_poll_thread()

    def register_buffers(self, pool: AfdBufferPool) -> None:
        self._pool = pool

    def set_ffn_compute(self, fn: FfnComputeFn) -> None:
        self._ffn_compute = fn

    def push_pull(
        self,
        *,
        layer_id: int,
        mb_id: int,
        a2f: Sequence[torch.Tensor],
        f2a: Sequence[torch.Tensor],
        num_tokens: Optional[int] = None,
    ) -> AfdHandle:
        with self._lock:
            hid = self._handle_counter
            self._handle_counter += 1
            done = threading.Event()
            self._done[hid] = done

        # Clone A2F so server can read without racing Attn overwrites.
        a2f_clone = [t.detach().clone() for t in a2f]
        compute_dtype = torch.bfloat16
        if self._pool is not None:
            compute_dtype = self._pool.compute_dtype
        batch = AfdServerBatch(
            handler=hid,
            worker_rank=self._worker_rank,
            tensors=a2f_clone,
            keys=[
                gen_push_key(i, mb_id, self._worker_rank) for i in range(len(a2f_clone))
            ],
            compute_dtype=compute_dtype,
            host_num_tokens=num_tokens,
            # Prefer packed meta from A2F layer_id slot (may include merge_k).
            host_layer_id=None,
        )
        # Stash pull buffers for respond().
        batch._f2a_bufs = f2a  # type: ignore[attr-defined]
        batch._mb_id = mb_id  # type: ignore[attr-defined]
        batch._layer_id_arg = layer_id  # type: ignore[attr-defined]

        use_async = (
            self._ffn_compute is not None
            and self._role != AfdMode.FFN
            and self._fake_async_ffn_enabled()
        )
        if self._ffn_compute is not None and self._role != AfdMode.FFN and not use_async:
            # Collocated Fake: run FFN on the calling thread (avoids CUDA+thread hangs).
            try:
                outs = self._ffn_compute(batch)
                self.respond(batch, outs)
            except Exception:
                logger.exception(
                    "AFD Fake sync FFN failed layer=%s handle=%s", layer_id, hid
                )
                raise
            return AfdHandle(id=hid)

        transfer_s = self._fake_transfer_s() if use_async else 0.0
        if transfer_s > 0:

            def _delayed_enqueue(b=batch, delay=transfer_s) -> None:
                time.sleep(delay)
                self._req_q.put(b)

            if use_async:
                self._ensure_ffn_poll_thread()
            threading.Thread(
                target=_delayed_enqueue, name="afd-fake-xfer", daemon=True
            ).start()
            return AfdHandle(id=hid)

        if use_async:
            self._ensure_ffn_poll_thread()
        self._req_q.put(batch)
        return AfdHandle(id=hid)

    def wait(self, handle: AfdHandle, timeout_ms: int = 5000) -> None:
        ev = self._done.get(handle.id)
        if ev is None:
            raise KeyError(f"unknown AFD handle {handle.id}")
        ok = ev.wait(timeout_ms / 1000.0)
        if not ok:
            raise TimeoutError(f"AFD wait timed out handle={handle.id}")
        with self._lock:
            self._done.pop(handle.id, None)

    def poll_done(self, handle: AfdHandle) -> bool:
        ev = self._done.get(handle.id)
        return ev is not None and ev.is_set()

    def get_batch(self, timeout_s: float = 1.0) -> List[AfdServerBatch]:
        try:
            batch = self._req_q.get(timeout=timeout_s)
        except Empty:
            return []
        return [batch]

    def respond(
        self, batch: AfdServerBatch, tensors: Sequence[torch.Tensor]
    ) -> None:
        f2a_bufs: Sequence[torch.Tensor] = getattr(batch, "_f2a_bufs")
        assert len(tensors) == len(f2a_bufs)
        for dst, src in zip(f2a_bufs, tensors):
            n = min(dst.shape[0], src.shape[0])
            dst[:n].copy_(src[:n])
        ev = self._done.get(batch.handler)
        if ev is not None:
            ev.set()

    def _auto_ffn_loop(self) -> None:
        assert self._ffn_compute is not None
        while not self._stop.is_set():
            batches = self.get_batch(timeout_s=0.05)
            for batch in batches:
                try:
                    outs = self._ffn_compute(batch)
                    self.respond(batch, outs)
                except Exception:
                    logger.exception(
                        "AFD Fake FFN poll failed handle=%s", batch.handler
                    )
                    # Unblock waiter so failures surface as bad data / next error.
                    ev = self._done.get(batch.handler)
                    if ev is not None:
                        ev.set()

    def close(self) -> None:
        self._stop.set()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None


class StepMeshAfdTransport(AfdTransport):
    """Thin wrapper over ``fserver_lib`` (requires StepMesh + RDMA)."""

    def __init__(self):
        self._f = None
        self._role = AfdMode.NULL
        self._worker_rank = 0
        self._pool: Optional[AfdBufferPool] = None
        self._pending: dict[int, object] = {}

    def init(self, role: AfdMode, *, worker_rank: int = 0, num_workers: int = 1) -> None:
        try:
            import fserver_lib as f
        except ImportError as e:
            raise ImportError(
                "StepMeshAfdTransport requires fserver_lib (pip install -e StepMesh). "
                "Use SGLANG_AFD_TRANSPORT=fake for local/CI."
            ) from e
        self._f = f
        self._role = role
        self._worker_rank = worker_rank
        # Align StepMesh node rank with AFD key packing when not already set.
        # preferred_rank = DMLC_GROUP_SIZE * DMLC_NODE_RANK + STEPMESH_GPU (+ offset).
        import os

        if role == AfdMode.ATTN and "DMLC_NODE_RANK" not in os.environ:
            os.environ["DMLC_NODE_RANK"] = str(int(worker_rank))
        f.init()
        logger.info(
            "StepMesh AFD transport init role=%s rank=%s dmlc_node_rank=%s",
            role,
            worker_rank,
            os.environ.get("DMLC_NODE_RANK", ""),
        )

    def register_buffers(self, pool: AfdBufferPool) -> None:
        # StepMesh registers MRs on first push_pull of each buffer; keep pool ref.
        self._pool = pool

    def push_pull(
        self,
        *,
        layer_id: int,
        mb_id: int,
        a2f: Sequence[torch.Tensor],
        f2a: Sequence[torch.Tensor],
        num_tokens: Optional[int] = None,
    ) -> AfdHandle:
        del num_tokens  # host meta is cuda_ipc-only
        assert self._f is not None
        push_keys = [
            gen_push_key(i, mb_id, self._worker_rank) for i in range(len(a2f))
        ]
        pull_keys = [
            gen_pull_key(i, mb_id, self._worker_rank) for i in range(len(f2a))
        ]
        handler = self._f.push_pull(
            list(a2f), push_keys, list(f2a), pull_keys, need_event=False
        )
        self._pending[handler] = time.time()
        return AfdHandle(id=int(handler))

    def wait(self, handle: AfdHandle, timeout_ms: int = 5000) -> None:
        assert self._f is not None
        self._f.wait(handle.id, timeout_ms)
        self._pending.pop(handle.id, None)

    def get_batch(self, timeout_s: float = 1.0) -> List[AfdServerBatch]:
        assert self._f is not None
        # StepMesh get_batch blocks until all workers present; no timeout API.
        raw = self._f.get_batch()
        out: List[AfdServerBatch] = []
        for i, item in enumerate(raw):
            # item: (handler, tensors, keys) per C++ ServerDataBatch binding
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                handler, tensors = item[0], item[1]
                keys = item[2] if len(item) > 2 else []
            else:
                raise TypeError(f"unexpected get_batch item: {type(item)}")
            out.append(
                AfdServerBatch(
                    handler=int(handler),
                    worker_rank=i,
                    tensors=list(tensors),
                    keys=list(keys) if keys else [],
                    compute_dtype=(
                        self._pool.compute_dtype
                        if self._pool is not None
                        else torch.bfloat16
                    ),
                )
            )
        return out

    def respond(
        self, batch: AfdServerBatch, tensors: Sequence[torch.Tensor]
    ) -> None:
        assert self._f is not None
        self._f.respond(list(tensors), batch.handler, False)

    def close(self) -> None:
        self._f = None


def create_transport(
    name: str,
    *,
    ffn_compute: Optional[FfnComputeFn] = None,
    endpoint: Optional[str] = None,
) -> AfdTransport:
    if name == "fake":
        return FakeAfdTransport(ffn_compute=ffn_compute)
    if name == "stepmesh":
        return StepMeshAfdTransport()
    if name in ("cuda_ipc", "nvlink"):
        from sglang.srt.afd.cuda_ipc_transport import CudaIpcAfdTransport

        return CudaIpcAfdTransport(endpoint=endpoint)
    raise ValueError(f"unknown AFD transport: {name}")


# Silence unused import warnings for slot constants used by docs/callers.
_ = (SLOT_HIDDEN, SLOT_LAYER_ID, SLOT_NUM_TOKENS)
