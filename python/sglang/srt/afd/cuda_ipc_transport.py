# SPDX-License-Identifier: Apache-2.0
"""Same-host AFD transport via CUDA IPC (NVLink / P2P when available).

Hot-path optimizations vs v1:
* Host shared-memory mailbox for posted/done seq (no GPU ``.item()`` spin).
* Attn remaps ``AfdBufferPool`` onto imported IPC tensors so ``fill_a2f`` /
  ``get_f2a`` already touch shared memory — ``push_pull``/``wait`` skip copies
  when pointers match.
* CUDA IPC events fence data; CPU mailbox wakes the peer.
* Optional GPU doorbell (``SGLANG_AFD_GPU_DOORBELL=1``): ``cudaHostRegister``
  the POSIX mailbox and ``cuStreamWriteValue64`` into ``done[]`` from the FFN
  stream so Attn can observe completion without waiting on FFN's CPU
  ``set_done``. Cross-GPU ``cuStreamWaitValue64`` on device IPC memory hangs
  here — not used. Default off; A/B on Lite first.
"""

from __future__ import annotations

import array
import ctypes
import logging
import os
import pickle
import select
import socket
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.afd.buffers import AfdBufferPool
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.protocol import AfdServerBatch, gen_push_key
from sglang.srt.afd.transport import AfdHandle, AfdTransport
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HDR = struct.Struct("!Q")

# cudaStreamWaitValue* / WriteValue* flags (cuda.h driver API)
_CU_STREAM_WAIT_VALUE_EQ = 0x1
_CU_STREAM_WRITE_VALUE_DEFAULT = 0x0

_libcuda = None
_libcuda_ok: Optional[bool] = None


def _ipc_endpoint() -> str:
    return (envs.SGLANG_AFD_IPC_ENDPOINT.get() or "/tmp/afd_cuda_ipc.sock").strip()


def _mailbox_name(endpoint: str) -> str:
    base = os.path.basename(endpoint).replace(".", "_")
    # SharedMemory adds the POSIX leading '/'; do not include it here.
    return f"afd_mb_{base}"[:200]


def _eventfd_wake_enabled() -> bool:
    # Default off: measured TRUE_OVERLAP Lite TPOT regressed vs sleep(0)+hot-spin
    # (~36ms vs ~31.5ms). Opt in with SGLANG_AFD_EVENTFD_WAKE=1 to A/B.
    v = os.environ.get("SGLANG_AFD_EVENTFD_WAKE", "0").strip().lower()
    return v in ("1", "true", "on", "yes")


def _gpu_doorbell_enabled() -> bool:
    v = os.environ.get("SGLANG_AFD_GPU_DOORBELL", "0").strip().lower()
    return v in ("1", "true", "on", "yes")


def _load_libcuda():
    """Driver API: cuStreamWriteValue64 / cuStreamWaitValue64 (not in cudart)."""
    global _libcuda, _libcuda_ok
    if _libcuda_ok is not None:
        return _libcuda if _libcuda_ok else None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        write_sym = None
        wait_sym = None
        for w, wt in (
            ("cuStreamWriteValue64_v2", "cuStreamWaitValue64_v2"),
            ("cuStreamWriteValue64", "cuStreamWaitValue64"),
        ):
            if hasattr(lib, w) and hasattr(lib, wt):
                write_sym, wait_sym = w, wt
                break
        if write_sym is None:
            continue
        # cuInit once
        if hasattr(lib, "cuInit"):
            lib.cuInit.argtypes = [ctypes.c_uint]
            lib.cuInit.restype = ctypes.c_int
            lib.cuInit(0)
        getattr(lib, write_sym).argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,  # CUdeviceptr
            ctypes.c_uint64,
            ctypes.c_uint,
        ]
        getattr(lib, write_sym).restype = ctypes.c_int
        getattr(lib, wait_sym).argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint,
        ]
        getattr(lib, wait_sym).restype = ctypes.c_int
        lib._afd_write_sym = write_sym  # type: ignore[attr-defined]
        lib._afd_wait_sym = wait_sym  # type: ignore[attr-defined]
        _libcuda = lib
        _libcuda_ok = True
        return lib
    _libcuda_ok = False
    _libcuda = None
    return None


def _stream_ptr(stream: torch.cuda.Stream) -> int:
    # torch.cuda.Stream.cuda_stream is the raw cudaStream_t / CUstream.
    return int(stream.cuda_stream)


def _cuda_write_value64(stream: torch.cuda.Stream, data_ptr: int, value: int) -> bool:
    lib = _load_libcuda()
    if lib is None:
        return False
    fn = getattr(lib, lib._afd_write_sym)  # type: ignore[attr-defined]
    err = fn(
        ctypes.c_void_p(_stream_ptr(stream)),
        ctypes.c_uint64(int(data_ptr)),
        ctypes.c_uint64(int(value) & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_uint(_CU_STREAM_WRITE_VALUE_DEFAULT),
    )
    return err == 0


def _cuda_wait_value64(stream: torch.cuda.Stream, data_ptr: int, value: int) -> bool:
    lib = _load_libcuda()
    if lib is None:
        return False
    fn = getattr(lib, lib._afd_wait_sym)  # type: ignore[attr-defined]
    err = fn(
        ctypes.c_void_p(_stream_ptr(stream)),
        ctypes.c_uint64(int(data_ptr)),
        ctypes.c_uint64(int(value) & 0xFFFFFFFFFFFFFFFF),
        ctypes.c_uint(_CU_STREAM_WAIT_VALUE_EQ),
    )
    return err == 0


def _send_msg(conn: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(_HDR.pack(len(payload)) + payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("AFD cuda_ipc socket closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> Any:
    (length,) = _HDR.unpack(_recv_exact(conn, _HDR.size))
    return pickle.loads(_recv_exact(conn, length))


def _send_fd(conn: socket.socket, fd: int) -> None:
    """Pass an open FD over a Unix stream (SCM_RIGHTS)."""
    conn.sendmsg(
        [b"\x01"],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [int(fd)]))],
    )


def _recv_fd(conn: socket.socket) -> int:
    fds = array.array("i")
    _msg, ancdata, _flags, _addr = conn.recvmsg(1, socket.CMSG_LEN(4))
    for level, typ, data in ancdata:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds.frombytes(data[: fds.itemsize])
            if len(fds) < 1:
                break
            return int(fds[0])
    raise RuntimeError("AFD cuda_ipc expected wake eventfd via SCM_RIGHTS")


def _create_wake_eventfd() -> Optional[int]:
    if not _eventfd_wake_enabled():
        return None
    if not hasattr(os, "eventfd") or not hasattr(os, "eventfd_write"):
        return None
    flags = int(getattr(os, "EFD_CLOEXEC", 0))
    if hasattr(os, "EFD_NONBLOCK"):
        flags |= int(os.EFD_NONBLOCK)
    try:
        return int(os.eventfd(0, flags))
    except OSError as e:
        logger.warning("AFD cuda_ipc eventfd create failed: %s", e)
        return None


def _share_tensor(t: torch.Tensor) -> Dict[str, Any]:
    if not t.is_cuda:
        raise ValueError("cuda_ipc only shares CUDA tensors")
    t = t.contiguous()
    storage = t.untyped_storage()
    (
        device,
        handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()
    return {
        "dtype": str(t.dtype).replace("torch.", ""),
        "size": tuple(t.size()),
        "stride": tuple(t.stride()),
        "storage_offset": int(t.storage_offset()),
        "device": device,
        "handle": handle,
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": ref_counter_offset,
        "event_handle": event_handle,
        "event_sync_required": event_sync_required,
    }


def _open_tensor(meta: Dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype = getattr(torch, meta["dtype"])
    storage = torch.UntypedStorage._new_shared_cuda(
        meta["device"],
        meta["handle"],
        meta["storage_size_bytes"],
        meta["storage_offset_bytes"],
        meta["ref_counter_handle"],
        meta["ref_counter_offset"],
        meta["event_handle"],
        meta["event_sync_required"],
    )
    t = torch.empty(0, dtype=dtype, device=device)
    t.set_(storage, meta["storage_offset"], torch.Size(meta["size"]))
    if tuple(t.stride()) != tuple(meta["stride"]):
        t = torch.as_strided(t, meta["size"], meta["stride"], meta["storage_offset"])
    return t


def _share_event(ev: torch.cuda.Event) -> bytes:
    return ev.ipc_handle()


def _open_event(handle: bytes, device: torch.device) -> torch.cuda.Event:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.Event.from_ipc_handle(idx, handle)


class _HostMailbox:
    """posted/done/num_tokens/layer_id as int64 arrays in POSIX shared memory."""

    def __init__(self, shm: shared_memory.SharedMemory, num_mb: int, *, create: bool):
        self.shm = shm
        self.num_mb = num_mb
        self._create = create
        self._host_registered = False
        self._done_dev_ptr: Optional[int] = None  # mapped device ptr to done[0]
        self._posted_dev_ptr: Optional[int] = None
        import numpy as np

        # Layout: [posted | done | num_tokens | layer_id]
        nbytes = num_mb * 8 * 4
        if len(shm.buf) < nbytes:
            raise RuntimeError(
                f"AFD mailbox shm too small: {len(shm.buf)} < {nbytes}"
            )
        self._nbytes = nbytes
        self._posted = np.ndarray(
            (num_mb,), dtype=np.int64, buffer=shm.buf, offset=0
        )
        self._done = np.ndarray(
            (num_mb,), dtype=np.int64, buffer=shm.buf, offset=num_mb * 8
        )
        self._num_tokens = np.ndarray(
            (num_mb,), dtype=np.int64, buffer=shm.buf, offset=num_mb * 16
        )
        self._layer_id = np.ndarray(
            (num_mb,), dtype=np.int64, buffer=shm.buf, offset=num_mb * 24
        )
        if create:
            self._posted.fill(0)
            self._done.fill(0)
            self._num_tokens.fill(-1)
            self._layer_id.fill(-1)

    @classmethod
    def create(cls, name: str, num_mb: int) -> "_HostMailbox":
        nbytes = num_mb * 8 * 4
        try:
            old = shared_memory.SharedMemory(name=name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        return cls(shm, num_mb, create=True)

    @classmethod
    def attach(cls, name: str, num_mb: int) -> "_HostMailbox":
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm, num_mb, create=False)

    def enable_mapped_gpu_doorbell(self) -> bool:
        """cudaHostRegister SHM so GPU WriteValue can publish posted/done.

        Cross-GPU AFD cannot use cuStreamWaitValue on peer-device IPC memory
        (hangs). Mapped host SHM lets FFN/Attn WriteValue into the same pages
        the peer already spins on via numpy — no WaitValue needed.
        """
        lib = _load_libcuda()
        if lib is None:
            return False
        # Resolve host base address of the SHM mapping.
        try:
            host_ptr = int(self._posted.ctypes.data)
        except Exception:
            return False
        # cudaHostRegister / cudaHostGetDevicePointer live in cudart.
        try:
            cudart = ctypes.CDLL("libcudart.so")
        except OSError:
            try:
                cudart = ctypes.CDLL("libcudart.so.11.0")
            except OSError:
                return False
        cudaHostRegister = cudart.cudaHostRegister
        cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        cudaHostRegister.restype = ctypes.c_int
        cudaHostGetDevicePointer = cudart.cudaHostGetDevicePointer
        cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        cudaHostGetDevicePointer.restype = ctypes.c_int
        # cudaHostRegisterMapped | cudaHostRegisterPortable
        flags = 0x02 | 0x01
        err = cudaHostRegister(
            ctypes.c_void_p(host_ptr), ctypes.c_size_t(self._nbytes), ctypes.c_uint(flags)
        )
        if err != 0:
            # Already registered / unsupported — try get-device-pointer anyway.
            logger.warning("AFD mailbox cudaHostRegister returned %s", err)
        dev = ctypes.c_void_p()
        err2 = cudaHostGetDevicePointer(
            ctypes.byref(dev), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
        )
        if err2 != 0 or not dev.value:
            logger.warning("AFD mailbox cudaHostGetDevicePointer failed err=%s", err2)
            return False
        base = int(dev.value)
        self._posted_dev_ptr = base
        self._done_dev_ptr = base + self.num_mb * 8
        self._host_registered = True
        return True

    def set_posted(
        self,
        mb: int,
        hid: int,
        *,
        num_tokens: int = -1,
        layer_id: int = -1,
    ) -> None:
        self._num_tokens[mb] = int(num_tokens)
        self._layer_id[mb] = int(layer_id)
        # posted last so FFN observing posted also sees meta.
        self._posted[mb] = int(hid)

    def get_posted(self, mb: int) -> int:
        return int(self._posted[mb])

    def get_meta(self, mb: int) -> Tuple[int, int]:
        return int(self._num_tokens[mb]), int(self._layer_id[mb])

    def set_done(self, mb: int, hid: int) -> None:
        self._done[mb] = int(hid)

    def get_done(self, mb: int) -> int:
        return int(self._done[mb])

    def close(self, *, unlink: bool = False) -> None:
        if self._host_registered:
            try:
                cudart = ctypes.CDLL("libcudart.so")
            except OSError:
                try:
                    cudart = ctypes.CDLL("libcudart.so.11.0")
                except OSError:
                    cudart = None
            if cudart is not None:
                try:
                    host_ptr = int(self._posted.ctypes.data)
                    cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
                    cudart.cudaHostUnregister.restype = ctypes.c_int
                    cudart.cudaHostUnregister(ctypes.c_void_p(host_ptr))
                except Exception:
                    pass
            self._host_registered = False
            self._done_dev_ptr = None
            self._posted_dev_ptr = None
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass


class CudaIpcAfdTransport(AfdTransport):
    """Cross-process AFD over CUDA IPC shared GPU buffers."""

    def __init__(self, endpoint: Optional[str] = None):
        self._role = AfdMode.NULL
        self._worker_rank = 0
        self._pool: Optional[AfdBufferPool] = None
        # Optional per-link endpoint for AfPool MxN (Na x Nf sock matrix).
        self._endpoint = (endpoint or _ipc_endpoint()).strip()
        self._mailbox_name = _mailbox_name(self._endpoint)
        self._listen_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._device = torch.device("cuda")
        self._num_mb = 0
        self._moe_topk = 0
        self._needs_scale = False

        self._a2f: List[List[torch.Tensor]] = []
        self._f2a: List[List[torch.Tensor]] = []
        self._a2f_ready: List[torch.cuda.Event] = []
        self._f2a_done: List[torch.cuda.Event] = []
        self._mailbox: Optional[_HostMailbox] = None
        self._zero_copy = False
        # Pinned meta for in-graph: GPU writes before write_flag; host reads after signal.
        self._meta_pin: Optional[torch.Tensor] = None  # [num_mb, 2] int32 host
        self._meta_dev: Optional[torch.Tensor] = None  # mapped view for Attn GPU writes
        self._ffn_runner = None  # optional FFN CG runner for shared-IO bind
        # Optional mapped-host GPU doorbell; see _gpu_doorbell_enabled.
        self._gpu_doorbell = False

        self._ffn_seen: List[int] = []
        self._pending: Dict[int, Tuple[int, Sequence[torch.Tensor]]] = {}
        self._handle_counter = 1
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._timeline = None  # Optional[AfdRtTimeline]
        # After a live FFN batch, busy-poll briefly so the next A2F post avoids
        # sleep(0) wake latency (~50–200µs) on the critical path.
        self._ffn_hot = False
        # Cross-process eventfd: Attn signals on set_posted; FFN parks on it
        # instead of bare sleep(0) when outside the hot window.
        self._wake_fd: Optional[int] = None

    def init(self, role: AfdMode, *, worker_rank: int = 0, num_workers: int = 1) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda_ipc transport requires CUDA")
        self._role = role
        self._worker_rank = worker_rank
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if role == AfdMode.FFN:
            self._start_listen()
        logger.info(
            "AFD cuda_ipc init role=%s endpoint=%s device=%s mailbox=%s",
            role.value,
            self._endpoint,
            self._device,
            self._mailbox_name,
        )

    def _start_listen(self) -> None:
        path = self._endpoint
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(path)
        sock.listen(1)
        sock.settimeout(120.0)
        self._listen_sock = sock

    def register_buffers(self, pool: AfdBufferPool) -> None:
        self._pool = pool
        self._num_mb = pool.cfg.num_mb
        self._moe_topk = int(pool.cfg.moe_topk or 0)
        self._needs_scale = bool(pool._needs_scale)
        self._ffn_seen = [0] * self._num_mb

        if self._role == AfdMode.FFN:
            # Create timeline before accept so Attn can attach during handshake.
            from sglang.srt.afd.timeline import AfdRtTimeline, timeline_enabled

            if timeline_enabled():
                self._timeline = AfdRtTimeline.try_open(
                    create=True, endpoint=self._endpoint
                )
            self._ffn_export_and_accept(pool)
        elif self._role == AfdMode.ATTN:
            self._attn_connect_and_import(pool)
            from sglang.srt.afd.timeline import AfdRtTimeline, timeline_enabled

            if timeline_enabled():
                # FFN creates shm; retry briefly in case of handshake ordering.
                deadline = time.time() + 30.0
                while time.time() < deadline:
                    self._timeline = AfdRtTimeline.try_open(
                        create=False, endpoint=self._endpoint
                    )
                    if self._timeline is not None:
                        break
                    time.sleep(0.05)
        else:
            raise RuntimeError(f"cuda_ipc unsupported role {self._role}")
        if self._timeline is not None:
            logger.info(
                "AFD RT timeline enabled role=%s",
                self._role.value,
            )
        self._ready.set()
        logger.info(
            "AFD cuda_ipc buffers ready role=%s num_mb=%s moe_topk=%s zero_copy=%s",
            self._role.value,
            self._num_mb,
            self._moe_topk,
            self._zero_copy,
        )

    def _build_slot_tensors(self, pool: AfdBufferPool, mb: int) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for s in pool.a2f_tensor_list(mb):
            t = torch.empty_like(s)
            t.copy_(s)
            out.append(t)
        return out

    def _build_f2a(self, pool: AfdBufferPool, mb: int) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for s in pool.f2a_tensor_list(mb):
            t = torch.empty_like(s)
            t.zero_()
            out.append(t)
        return out

    def _remap_pool_to_shared(self, pool: AfdBufferPool) -> None:
        """Point Attn pool slots at imported IPC tensors (zero-copy fill/get)."""
        for mb in range(self._num_mb):
            tensors = self._a2f[mb]
            i = 0
            pool._a2f_hidden[mb] = tensors[i]
            i += 1
            pool._a2f_num_tokens[mb] = tensors[i]
            i += 1
            pool._a2f_layer_id[mb] = tensors[i]
            i += 1
            if self._moe_topk > 0:
                pool._a2f_topk_ids[mb] = tensors[i]
                i += 1
                pool._a2f_topk_weights[mb] = tensors[i]
                i += 1
            if self._needs_scale:
                pool._a2f_hidden_scale[mb] = tensors[i]
                i += 1
            # Layer-merge optional residual + positions (after scale).
            if getattr(pool.cfg, "layer_merge", False) and i < len(tensors):
                # residual is 2-D hidden-sized; positions is 1-D int.
                for j in range(i, len(tensors)):
                    t = tensors[j]
                    if (
                        t.dim() == 2
                        and t.shape[-1] == pool.cfg.hidden_size
                        and t.dtype
                        in (torch.bfloat16, torch.float16, torch.float32)
                    ):
                        pool._a2f_residual[mb] = t
                    elif t.dim() == 1 and t.dtype in (torch.int32, torch.int64):
                        pool._a2f_positions[mb] = t
            f2a = self._f2a[mb]
            pool._f2a_out[mb] = f2a[0]
            if getattr(pool.cfg, "layer_merge", False) and len(f2a) > 1:
                pool._f2a_residual[mb] = f2a[1]
        self._zero_copy = True
        pool._rebuild_tensor_lists()

    def _ffn_export_and_accept(self, pool: AfdBufferPool) -> None:
        assert self._listen_sock is not None
        a2f_local: List[List[torch.Tensor]] = []
        f2a_local: List[List[torch.Tensor]] = []
        a2f_events: List[torch.cuda.Event] = []
        f2a_events: List[torch.cuda.Event] = []
        for mb in range(self._num_mb):
            a2f_local.append(self._build_slot_tensors(pool, mb))
            f2a_local.append(self._build_f2a(pool, mb))
            a2f_events.append(torch.cuda.Event(interprocess=True, enable_timing=False))
            f2a_events.append(torch.cuda.Event(interprocess=True, enable_timing=False))

        mailbox = _HostMailbox.create(self._mailbox_name, self._num_mb)
        wake_fd = _create_wake_eventfd()
        want_db = _gpu_doorbell_enabled() and _load_libcuda() is not None
        mapped_ok = False
        if want_db:
            mapped_ok = mailbox.enable_mapped_gpu_doorbell()
            if mapped_ok:
                self._gpu_doorbell = True
                logger.info(
                    "AFD cuda_ipc FFN mapped-host GPU doorbell enabled "
                    "(WriteValue → POSIX mailbox)"
                )
            else:
                logger.warning(
                    "AFD cuda_ipc GPU doorbell: cudaHostRegister/map failed; "
                    "using host mailbox CPU publish only"
                )

        payload = {
            "num_mb": self._num_mb,
            "moe_topk": self._moe_topk,
            "needs_scale": self._needs_scale,
            "mailbox": self._mailbox_name,
            "wake_eventfd": wake_fd is not None,
            "gpu_doorbell": bool(mapped_ok),
            "a2f": [[_share_tensor(t) for t in slot] for slot in a2f_local],
            "f2a": [[_share_tensor(t) for t in slot] for slot in f2a_local],
            "a2f_ready": [_share_event(e) for e in a2f_events],
            "f2a_done": [_share_event(e) for e in f2a_events],
        }

        logger.info("AFD cuda_ipc FFN waiting for Attn on %s", self._endpoint)
        conn, _addr = self._listen_sock.accept()
        conn.settimeout(120.0)
        self._conn = conn
        _send_msg(conn, payload)
        if wake_fd is not None:
            try:
                _send_fd(conn, wake_fd)
                self._wake_fd = wake_fd
            except OSError as e:
                logger.warning("AFD cuda_ipc wake fd send failed: %s", e)
                try:
                    os.close(wake_fd)
                except Exception:
                    pass
                self._wake_fd = None
        self._a2f = a2f_local
        self._f2a = f2a_local
        self._a2f_ready = a2f_events
        self._f2a_done = f2a_events
        self._mailbox = mailbox
        # FFN also serves from shared owner buffers (same pointers as exported).
        self._remap_pool_to_shared(pool)
        ack = _recv_msg(conn)
        if not isinstance(ack, dict) or ack.get("status") != "ok":
            raise RuntimeError(f"AFD cuda_ipc bad Attn ack: {ack!r}")
        if self._wake_fd is not None:
            logger.info("AFD cuda_ipc FFN wake eventfd enabled")

    def _attn_connect_and_import(self, pool: AfdBufferPool) -> None:
        deadline = time.time() + 180.0
        last_err: Optional[BaseException] = None
        conn: Optional[socket.socket] = None
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect(self._endpoint)
                conn = s
                break
            except OSError as e:
                last_err = e
                time.sleep(0.2)
        if conn is None:
            raise RuntimeError(
                f"AFD cuda_ipc Attn failed to connect {self._endpoint}: {last_err}"
            )
        self._conn = conn
        payload = _recv_msg(conn)
        self._num_mb = int(payload["num_mb"])
        self._moe_topk = int(payload["moe_topk"])
        self._needs_scale = bool(payload["needs_scale"])
        self._ffn_seen = [0] * self._num_mb
        self._mailbox_name = str(payload["mailbox"])
        if payload.get("wake_eventfd"):
            try:
                self._wake_fd = _recv_fd(conn)
            except Exception as e:
                logger.warning("AFD cuda_ipc wake fd recv failed: %s", e)
                self._wake_fd = None

        self._a2f = [
            [_open_tensor(m, self._device) for m in slot] for slot in payload["a2f"]
        ]
        self._f2a = [
            [_open_tensor(m, self._device) for m in slot] for slot in payload["f2a"]
        ]
        self._a2f_ready = [_open_event(h, self._device) for h in payload["a2f_ready"]]
        self._f2a_done = [_open_event(h, self._device) for h in payload["f2a_done"]]
        self._mailbox = _HostMailbox.attach(self._mailbox_name, self._num_mb)
        # Attn only needs to know doorbell is on; FFN owns the mapped WriteValue.
        if payload.get("gpu_doorbell"):
            self._gpu_doorbell = True
            logger.info("AFD cuda_ipc Attn: peer FFN mapped-host GPU doorbell on")
        self._remap_pool_to_shared(pool)
        _send_msg(conn, {"status": "ok"})
        if self._wake_fd is not None:
            logger.info("AFD cuda_ipc Attn wake eventfd enabled")

    def push_pull(
        self,
        *,
        layer_id: int,
        mb_id: int,
        a2f: Sequence[torch.Tensor],
        f2a: Sequence[torch.Tensor],
        num_tokens: Optional[int] = None,
    ) -> AfdHandle:
        if self._role != AfdMode.ATTN:
            raise RuntimeError("cuda_ipc push_pull is Attn-only")
        if not self._ready.is_set():
            raise RuntimeError("cuda_ipc buffers not ready")
        assert self._mailbox is not None
        if mb_id < 0 or mb_id >= self._num_mb:
            raise IndexError(f"mb_id={mb_id} out of range n={self._num_mb}")

        shared = self._a2f[mb_id]
        if len(a2f) != len(shared):
            raise ValueError(
                f"A2F arity mismatch local={len(a2f)} shared={len(shared)}"
            )
        stream = torch.cuda.current_stream(self._device)
        # Zero-copy: fill_a2f already wrote into shared IPC tensors.
        if not self._zero_copy:
            for dst, src in zip(shared, a2f):
                if dst.data_ptr() == src.data_ptr():
                    continue
                if dst.shape == src.shape:
                    dst.copy_(src, non_blocking=True)
                else:
                    slices = tuple(
                        slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape)
                    )
                    dst[slices].copy_(src[slices], non_blocking=True)

        stream.record_event(self._a2f_ready[mb_id])
        # Publish mailbox before taking the pending lock's long path — FFN can
        # observe posted as soon as meta+hid are visible; pending is Attn-local.
        with self._lock:
            hid = self._handle_counter
            self._handle_counter += 1
            self._pending[hid] = (mb_id, f2a)
        n_tok = -1 if num_tokens is None else int(num_tokens)
        self._mailbox.set_posted(
            mb_id, hid, num_tokens=n_tok, layer_id=int(layer_id)
        )
        self._signal_wake()
        if self._timeline is not None:
            self._timeline.begin(
                hid, layer=int(layer_id), mb=int(mb_id), t_post=time.perf_counter()
            )
        return AfdHandle(id=hid)

    def _signal_wake(self) -> None:
        """Notify FFN poll that mailbox posted (cross-process eventfd)."""
        fd = self._wake_fd
        if fd is None:
            return
        try:
            os.eventfd_write(fd, 1)
        except BlockingIOError:
            # Counter saturated — FFN will still see SHM posted.
            pass
        except OSError:
            pass

    def _park_until_wake(self, timeout_s: float = 0.00005) -> None:
        """Block until Attn signals or timeout; drain pending wake counts.

        Replaces bare ``sleep(0)`` so a ``set_posted`` can wake FFN without
        waiting for a full scheduler slice. Keep the timeout short so a missed
        signal still falls back quickly to another mailbox scan.
        """
        fd = self._wake_fd
        if fd is None:
            time.sleep(0)
            return
        try:
            r, _, _ = select.select([fd], [], [], timeout_s)
            if r:
                while True:
                    try:
                        os.eventfd_read(fd)
                    except BlockingIOError:
                        break
                    except OSError:
                        break
            else:
                # No signal within timeout — same yield as legacy path.
                time.sleep(0)
        except Exception:
            time.sleep(0)

    def write_meta_dev(
        self,
        mb_id: int,
        *,
        num_tokens: int = -1,
        layer_id: int = -1,
        num_tokens_t: Optional[torch.Tensor] = None,
        layer_id_t: Optional[torch.Tensor] = None,
    ) -> None:
        """GPU write of (tokens, layer) into pinned meta (before write_flag).

        Prefer copying from already-on-device A2F meta tensors so this is safe
        inside CUDA graph capture (no unpinned CPU scalar assignment).
        """
        if self._meta_dev is None:
            return
        dst = self._meta_dev[mb_id]
        if num_tokens_t is not None:
            src = num_tokens_t.reshape(-1)[0]
            if src.dtype != dst.dtype:
                src = src.to(dtype=dst.dtype)
            dst[0].copy_(src)
        elif num_tokens >= 0:
            # Eager / non-capture fallback only.
            dst[0].fill_(int(num_tokens))
        if layer_id_t is not None:
            src = layer_id_t.reshape(-1)[0]
            if src.dtype != dst.dtype:
                src = src.to(dtype=dst.dtype)
            dst[1].copy_(src)
        elif layer_id >= 0:
            dst[1].fill_(int(layer_id))

    def enable_meta_map(self, gpu_id: int = 0) -> None:
        """Map pinned meta for in-graph path (Attn only; needs fserver_lib)."""
        if self._role != AfdMode.ATTN or self._num_mb <= 0:
            return
        if self._meta_pin is None:
            self._meta_pin = torch.zeros(
                (self._num_mb, 2), dtype=torch.int32, pin_memory=True
            )
        try:
            import fserver_lib as f

            self._meta_dev = f.map_pinned_tensor(self._meta_pin, int(gpu_id))
            logger.info(
                "AFD cuda_ipc meta map enabled gpu=%s mb=%s", gpu_id, self._num_mb
            )
        except Exception as e:
            logger.warning("AFD cuda_ipc meta map failed: %s", e)
            self._meta_dev = None

    def bind_ffn_shared_io(self, runner) -> None:
        """Bind FFN-local IPC A2F/F2A as CUDA-graph staging (not MoE workspace)."""
        self._ffn_runner = runner
        if self._role != AfdMode.FFN or not self._a2f or not self._f2a:
            return
        try:
            for mb in range(self._num_mb):
                hidden = self._a2f[mb][0]
                f2a = self._f2a[mb][0]
                topk_ids = None
                topk_w = None
                if self._moe_topk > 0 and len(self._a2f[mb]) >= 5:
                    topk_ids = self._a2f[mb][3]
                    topk_w = self._a2f[mb][4]
                runner.bind_shared_slot(
                    mb,
                    hidden=hidden,
                    mlp_out=f2a,
                    topk_ids=topk_ids,
                    topk_weights=topk_w,
                )
            logger.info(
                "AFD FFN IPC staging registered num_mb=%s moe_topk=%s shared_moe=%s",
                self._num_mb,
                self._moe_topk,
                bool(envs.SGLANG_AFD_FFN_SHARED_IO.get()),
            )
        except Exception as e:
            logger.warning("AFD FFN IPC staging bind failed: %s", e)

    def post_mailbox_only(
        self,
        mb_id: int,
        f2a: Sequence[torch.Tensor],
        *,
        layer_id: int = -1,
        num_tokens: int = -1,
    ) -> AfdHandle:
        """In-graph path: A2F already filled; CPU publishes mailbox + host meta."""
        if self._role != AfdMode.ATTN:
            raise RuntimeError("cuda_ipc post_mailbox_only is Attn-only")
        assert self._mailbox is not None
        if self._meta_pin is not None and (num_tokens < 0 or layer_id < 0):
            try:
                tok = int(self._meta_pin[mb_id, 0])
                lyr = int(self._meta_pin[mb_id, 1])
                if num_tokens < 0:
                    num_tokens = tok
                if layer_id < 0:
                    layer_id = lyr
            except Exception:
                pass
        with self._lock:
            hid = self._handle_counter
            self._handle_counter += 1
            self._pending[hid] = (mb_id, f2a)
        self._mailbox.set_posted(
            mb_id, hid, num_tokens=int(num_tokens), layer_id=int(layer_id)
        )
        self._signal_wake()
        if self._timeline is not None:
            self._timeline.begin(
                hid, layer=int(layer_id), mb=int(mb_id), t_post=time.perf_counter()
            )
        return AfdHandle(id=hid)

    def wait(
        self, handle: AfdHandle, timeout_ms: int = 5000, *, soft: bool = False
    ) -> None:
        if self._role != AfdMode.ATTN:
            return
        assert self._mailbox is not None
        pending = self._pending.get(handle.id)
        if pending is None:
            raise KeyError(f"unknown AFD cuda_ipc handle {handle.id}")
        mb_id, f2a_bufs = pending
        deadline = time.time() + timeout_ms / 1000.0
        stream = torch.cuda.current_stream(self._device)
        if self._timeline is not None:
            # Record when Attn actually arrives at the wait, separating "Attn was
            # still busy on other contexts" from "Attn was parked here spinning".
            self._timeline.mark_attn_wait_enter(handle.id, time.perf_counter())
        # NOTE: cuStreamWaitValue64 on IPC doorbell memory that lives on the
        # *peer* GPU hangs on this platform (Attn GPU ≠ FFN GPU). Keep the
        # host-mailbox spin for generation-safe completion; GPU WriteValue on
        # FFN still populates done_db for optional D2H peek / future same-GPU.
        spins = 0
        while self._mailbox.get_done(mb_id) != handle.id:
            if time.time() > deadline:
                raise TimeoutError(
                    f"AFD cuda_ipc wait timed out handle={handle.id} mb={mb_id} "
                    f"done={self._mailbox.get_done(mb_id)}"
                )
            spins += 1
            if spins > 64:
                time.sleep(0)

        shared_f2a = self._f2a[mb_id]
        if len(shared_f2a) != len(f2a_bufs):
            raise ValueError("F2A arity mismatch on wait")
        if soft:
            # Attn stream: fence + async copy; consumer kernels on this stream
            # see completed F2A without blocking the CPU thread (true overlap).
            stream.wait_event(self._f2a_done[mb_id])
            if not self._zero_copy:
                for dst, src in zip(f2a_bufs, shared_f2a):
                    if dst.data_ptr() == src.data_ptr():
                        continue
                    n = min(dst.shape[0], src.shape[0])
                    dst[:n].copy_(src[:n], non_blocking=True)
        else:
            # Flag-CPU / sync callers: ack must imply F2A fully resident.
            self._f2a_done[mb_id].synchronize()
            if not self._zero_copy:
                for dst, src in zip(f2a_bufs, shared_f2a):
                    if dst.data_ptr() == src.data_ptr():
                        continue
                    n = min(dst.shape[0], src.shape[0])
                    dst[:n].copy_(src[:n], non_blocking=False)
        if self._timeline is not None:
            self._timeline.mark_attn_done(handle.id, time.perf_counter())
            # Periodic host summary (survives SIGKILL of peer).
            self._tl_done = getattr(self, "_tl_done", 0) + 1
            if self._tl_done % 512 == 0:
                try:
                    from sglang.srt.afd.timeline import summarize_samples

                    summary = summarize_samples(self._timeline.samples())
                    logger.info("\n%s", summary)
                    out = os.environ.get("SGLANG_AFD_TIMELINE_OUT", "")
                    if out:
                        with open(out, "w", encoding="utf-8") as f:
                            f.write(summary + "\n")
                except Exception:
                    pass
        with self._lock:
            self._pending.pop(handle.id, None)

    def poll_done(self, handle: AfdHandle) -> bool:
        pending = self._pending.get(handle.id)
        if pending is None or self._mailbox is None:
            return False
        mb_id, _f2a_bufs = pending
        return self._mailbox.get_done(mb_id) == handle.id

    def park_idle(self, timeout_s: float = 0.00005) -> None:
        """Public: yield/park the calling thread waiting for a peer post.

        Used by the NA1F FFN serve loop, which parks once for all links instead
        of blocking on a single link (see ``AfFfnWorker._poll_ready``).
        """
        self._park_until_wake(timeout_s)

    def get_batch(
        self, timeout_s: float = 1.0, *, nonblocking: bool = False
    ) -> List[AfdServerBatch]:
        if self._role != AfdMode.FFN:
            return []
        if not self._ready.wait(timeout=timeout_s):
            return []
        assert self._mailbox is not None
        from sglang.srt.environ import envs as _envs

        gather_us = max(0, int(_envs.SGLANG_AFD_FFN_GATHER_US.get() or 0))
        gather_max = max(1, int(_envs.SGLANG_AFD_FFN_GATHER_MAX.get() or 1))
        parallel_mb = bool(_envs.SGLANG_AFD_FFN_PARALLEL_MB.get())
        if parallel_mb and self._num_mb > 1:
            # Progressive: return every *already* ready slot in one scan, but do
            # not wait for the peer. Waiting for gather kills TRUE_OVERLAP stagger
            # (delays FFN(mb0) until Attn(mb1) posts).
            gather_max = max(gather_max, int(self._num_mb))
            # Keep caller-set gather_us; default stays 0 under PARALLEL_MB.
        in_graph = bool(_envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        deadline = time.time() + timeout_s

        def _make_batch(mb: int, posted: int, *, t_hit: float) -> AfdServerBatch:
            from sglang.srt.afd.detail_profile import (
                profile_detail_enabled,
                record_span,
                record_us,
            )

            _detail = profile_detail_enabled()
            if not in_graph:
                t_sync0 = time.perf_counter() if _detail else 0.0
                self._a2f_ready[mb].synchronize()
                if _detail:
                    record_span("a2f_sync_us", t_sync0)
            self._ffn_seen[mb] = posted
            tensors = self._a2f[mb]
            keys = [
                gen_push_key(i, mb, self._worker_rank) for i in range(len(tensors))
            ]
            compute_dtype = (
                self._pool.compute_dtype
                if self._pool is not None
                else torch.bfloat16
            )
            n_tok, lyr = self._mailbox.get_meta(mb)
            batch = AfdServerBatch(
                handler=posted,
                worker_rank=self._worker_rank,
                tensors=tensors,
                keys=keys,
                compute_dtype=compute_dtype,
                host_num_tokens=n_tok if n_tok >= 0 else None,
                host_layer_id=lyr if lyr >= 0 else None,
            )
            batch._mb_id = mb  # type: ignore[attr-defined]
            batch._f2a_bufs = self._f2a[mb]  # type: ignore[attr-defined]
            t_seen = time.perf_counter()
            if _detail and self._timeline is not None:
                try:
                    post_ts = self._timeline.get_attn_post(int(posted))
                    if post_ts > 0:
                        record_us("post_poll_us", (t_hit - post_ts) * 1e6)
                except Exception:
                    pass
            if self._timeline is not None:
                self._timeline.mark_ffn_seen(int(posted), t_seen)
            return batch

        spins = 0
        # Hot window: after recent work, pure-busy ~3ms so post→seen skips sleep(0).
        hot_until = time.perf_counter() + (0.003 if self._ffn_hot else 0.0)
        while True:
            ready: List[AfdServerBatch] = []
            for mb in range(self._num_mb):
                posted = self._mailbox.get_posted(mb)
                if posted <= self._ffn_seen[mb]:
                    continue
                t_hit = time.perf_counter()
                ready.append(_make_batch(mb, posted, t_hit=t_hit))
                if len(ready) >= gather_max:
                    break
            if ready:
                if gather_us > 0 and len(ready) < gather_max and self._num_mb > 1:
                    gather_deadline = time.time() + gather_us * 1e-6
                    while time.time() < gather_deadline and len(ready) < gather_max:
                        for mb in range(self._num_mb):
                            posted = self._mailbox.get_posted(mb)
                            if posted <= self._ffn_seen[mb]:
                                continue
                            t_hit = time.perf_counter()
                            ready.append(_make_batch(mb, posted, t_hit=t_hit))
                            if len(ready) >= gather_max:
                                break
                        if len(ready) >= gather_max:
                            break
                        time.sleep(0)
                self._ffn_hot = True
                return ready
            # Scan first, then decide whether to wait. Always scan at least
            # once: the old `while time.time() < deadline` guard skipped the
            # scan entirely for timeout_s == 0 and returned [], but callers use
            # timeout_s=0 to mean "one non-blocking pass" — the NA1F FFN worker
            # draining several Attn links, and extra_gather. That silently made
            # both no-ops. Never park on a pass that must not block.
            if nonblocking or time.time() >= deadline:
                return []
            spins += 1
            now = time.perf_counter()
            if now < hot_until:
                continue
            self._ffn_hot = False
            # Prefer eventfd park over sleep(0): Attn signals on set_posted.
            self._park_until_wake(0.00005)
            # Micro-busy after wake/timeout to catch a post that raced the park.
            hot_until = time.perf_counter() + 0.0002
            spins = 0

    def respond(
        self, batch: AfdServerBatch, tensors: Sequence[torch.Tensor]
    ) -> None:
        if self._role != AfdMode.FFN:
            raise RuntimeError("cuda_ipc respond is FFN-only")
        assert self._mailbox is not None
        from sglang.srt.afd.detail_profile import profile_detail_enabled, record_span

        _detail = profile_detail_enabled()
        t0 = time.perf_counter() if _detail else 0.0
        mb_id = int(getattr(batch, "_mb_id"))
        f2a_bufs: Sequence[torch.Tensor] = getattr(batch, "_f2a_bufs")
        if len(tensors) != len(f2a_bufs):
            # Layer-merge may emit residual while older pools only have mlp_out
            # (or vice versa). Copy the common prefix so Attn is not stuck.
            logger.warning(
                "AFD cuda_ipc respond F2A arity mismatch compute=%s bufs=%s; "
                "copying min",
                len(tensors),
                len(f2a_bufs),
            )
            n_copy = min(len(tensors), len(f2a_bufs))
            tensors = list(tensors)[:n_copy]
            f2a_bufs = list(f2a_bufs)[:n_copy]
        stream = torch.cuda.current_stream(self._device)
        for dst, src in zip(f2a_bufs, tensors):
            if dst.data_ptr() == src.data_ptr():
                continue
            # Attn only consumes [:num_tokens]; pad may be stale — avoid an
            # extra GPU .item() sync here on the critical path.
            n = min(dst.shape[0], src.shape[0])
            dst[:n].copy_(src[:n], non_blocking=True)
        # Mapped-host GPU doorbell: WriteValue into POSIX done[] on the FFN
        # stream after F2A copy. Attn already spins on the same SHM pages —
        # avoids waiting for FFN CPU to reach set_done(). Do NOT use
        # cuStreamWaitValue64 on peer-GPU IPC device memory (hangs).
        hid = int(batch.handler)
        if self._timeline is not None:
            # AfPool path: nothing else marks ffn_compute, and samples() drops
            # any slot with t2<=0 — leaving the timeline empty (n=0).
            self._timeline.mark_ffn_compute(hid, time.perf_counter())
        if self._gpu_doorbell and self._mailbox is not None:
            done_dev = self._mailbox._done_dev_ptr
            if done_dev is not None:
                ptr = int(done_dev) + int(mb_id) * 8
                _cuda_write_value64(stream, ptr, hid)
        stream.record_event(self._f2a_done[mb_id])
        # CPU publish remains as a coherence/fallback path.
        self._mailbox.set_done(mb_id, hid)
        if self._timeline is not None:
            self._timeline.mark_ffn_respond(hid, time.perf_counter())
        if _detail:
            record_span("respond_us", t0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        try:
            if self._listen_sock is not None:
                self._listen_sock.close()
        except Exception:
            pass
        if self._wake_fd is not None:
            try:
                os.close(self._wake_fd)
            except Exception:
                pass
            self._wake_fd = None
        if self._mailbox is not None:
            self._mailbox.close(unlink=(self._role == AfdMode.FFN))
            self._mailbox = None
        if self._timeline is not None:
            try:
                from sglang.srt.afd.timeline import summarize_samples

                logger.info("\n%s", summarize_samples(self._timeline.samples()))
            except Exception:
                pass
            self._timeline.close(unlink=(self._role == AfdMode.FFN))
            self._timeline = None
        try:
            from sglang.srt.afd.detail_profile import dump_detail_summary

            dump_detail_summary()
        except Exception:
            pass
        if self._role == AfdMode.FFN:
            try:
                if os.path.exists(self._endpoint):
                    os.unlink(self._endpoint)
            except OSError:
                pass
        self._conn = None
        self._listen_sock = None
