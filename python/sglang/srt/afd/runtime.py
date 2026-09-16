# SPDX-License-Identifier: Apache-2.0
"""Process-wide AFD runtime (buffers + transport)."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import torch

from sglang.srt.afd.buffers import AfdBufferPool, AfdBufferPoolConfig
from sglang.srt.afd.cuda_graph_breaks import AfdGraphSyncFlags
from sglang.srt.afd.mode import AfdMode, get_afd_mode, get_afd_transport_name
from sglang.srt.afd.transport import AfdHandle, AfdTransport, FfnComputeFn, create_transport
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.afd.pipeline import AfdPendingTransfer

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_runtime: Optional["AfdRuntime"] = None


class AfdRuntime:
    def __init__(
        self,
        mode: AfdMode,
        transport: AfdTransport,
        pool: AfdBufferPool,
    ):
        self.mode = mode
        self.transport = transport
        self.pool = pool
        self._mb_cursor = 0
        self._mb_in_flight: Set[int] = set()
        # Compat: first / only flag set (in-graph always uses index 0).
        self._wait_flag: Optional[AfdGraphSyncFlags] = None
        # Per-mb flags for true-overlap deferred issue/wait (independent seq).
        self._wait_flags: List[AfdGraphSyncFlags] = []
        self._flag_expected: List[int] = []
        self._flag_thread: Optional[threading.Thread] = None
        self._flag_stop = threading.Event()
        self._pending_job: Optional[dict] = None  # single-slot legacy
        self._pending_jobs: Dict[int, dict] = {}  # mb_id -> push_pull kwargs
        self._job_lock = threading.Lock()
        self._ffn_wait_stream: Optional[torch.cuda.Stream] = None
        self._gpu_id: int = 0
        # cuda_ipc mailbox path never uses deferred write_flag (see below).
        from sglang.srt.afd.cuda_ipc_transport import CudaIpcAfdTransport

        self._force_mailbox_async: bool = isinstance(transport, CudaIpcAfdTransport)
        self._timeline_log: bool = bool(envs.SGLANG_AFD_TIMELINE.get())
        # cuda_ipc mailbox path must use soft wait or TRUE_OVERLAP collapses
        # (CPU synchronize on F2A blocks the peer-mb Attn window).
        self._wait_soft: bool = bool(self._force_mailbox_async)
        if not self._wait_soft:
            try:
                import inspect

                self._wait_soft = "soft" in inspect.signature(transport.wait).parameters
            except Exception:
                self._wait_soft = False

    @property
    def num_mb(self) -> int:
        return self.pool.cfg.num_mb

    def next_mb(self) -> int:
        """Allocate a free mb slot (blocks logically if all in flight — raises)."""
        for _ in range(self.num_mb):
            mb = self._mb_cursor
            self._mb_cursor = (self._mb_cursor + 1) % self.num_mb
            if mb not in self._mb_in_flight:
                return mb
        raise RuntimeError(
            f"All {self.num_mb} AFD microbatch slots in flight; "
            "wait_remote_ffn before issuing more"
        )

    def enable_wait_flag_sync(self, gpu_id: int = 0) -> None:
        """GPU write_flag/wait_flag + CPU push_pull (fserver_lib flag kernels).

        Works for StepMesh *and* cuda_ipc transports: only the flag kernels
        come from fserver_lib (``map_pinned_tensor`` needs no ``f.init()``).

        In-graph mode keeps a single flag pair (mb 0). True-overlap / layer
        pipeline allocates one flag pair per microbatch so deferred waits do
        not share a sequence counter across outstanding issues.
        """
        import fserver_lib as f

        device = f"cuda:{gpu_id}"
        self._gpu_id = int(gpu_id)
        in_graph = bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        n_flags = 1 if in_graph else max(1, self.num_mb)
        self._wait_flags = []
        for _ in range(n_flags):
            flags = AfdGraphSyncFlags(device=device)
            flags.map_with_stepmesh(f, gpu_id)
            self._wait_flags.append(flags)
        self._wait_flag = self._wait_flags[0]
        self._flag_expected = [1] * n_flags
        self._flag_stop.clear()
        self._in_graph_ready_event = torch.cuda.Event(enable_timing=False)
        if torch.cuda.is_available() and (
            bool(envs.SGLANG_AFD_TRUE_OVERLAP.get())
            or bool(envs.SGLANG_AFD_LAYER_PIPELINE.get())
        ):
            # Secondary stream so wait_flag does not block Attn kernels already
            # queued on the default stream (TRUE_OVERLAP dual-stream).
            self._ffn_wait_stream = torch.cuda.Stream(device=device)
        if hasattr(self.transport, "enable_meta_map"):
            try:
                self.transport.enable_meta_map(gpu_id)
            except Exception as e:
                logger.warning("AFD meta map enable failed: %s", e)
        self._flag_thread = threading.Thread(
            target=self._flag_cpu_loop, name="afd-wait-flag", daemon=True
        )
        self._flag_thread.start()
        logger.info(
            "AFD wait_flag sync enabled on gpu=%s transport=%s in_graph=%s "
            "n_flags=%s dual_stream=%s",
            gpu_id,
            get_afd_transport_name(),
            in_graph,
            n_flags,
            self._ffn_wait_stream is not None,
        )

    def _flag_cpu_loop(self) -> None:
        assert self._wait_flag is not None
        in_graph = bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        # Bind CUDA context in this thread for transport.wait copies.
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self._gpu_id)
            except Exception:
                pass
        while not self._flag_stop.is_set():
            if in_graph:
                flags = self._wait_flags[0]
                expected = self._flag_expected[0]
                signal_value = int(flags.signal_host.item())
                if signal_value < expected:
                    continue
                # Full-CG replay never re-runs Python job posting — rebuild from
                # the pre-registered pool. Stream order is fill → write_flag, so
                # when the host sees ``signal_host`` the fills are done on Attn.
                # Do NOT event.synchronize()/cuda.synchronize() here: syncing a
                # captured event or the wait_flag stream deadlocks capture.
                mb_id = 0
                f2a = self.pool.f2a_tensor_list(mb_id)
                transport = self.transport
                # Never GPU .item()/.cpu() here: the Attn stream is blocked in
                # wait_flag waiting for this thread to ack — a device sync deadlocks.
                if hasattr(transport, "post_mailbox_only"):
                    handle = transport.post_mailbox_only(mb_id, f2a, layer_id=-1)
                else:
                    a2f = self.pool.a2f_tensor_list(mb_id)
                    handle = transport.push_pull(
                        layer_id=0,
                        mb_id=mb_id,
                        a2f=a2f,
                        f2a=f2a,
                    )
                self.transport.wait(handle, timeout_ms=60000)
                self._flag_expected[0] = signal_value + 1
                flags.ack_host.fill_(signal_value)
                continue

            progressed = False
            for mb_id, flags in enumerate(self._wait_flags):
                expected = self._flag_expected[mb_id]
                signal_value = int(flags.signal_host.item())
                if signal_value < expected:
                    continue
                with self._job_lock:
                    job = self._pending_jobs.pop(mb_id, None)
                    if job is None and mb_id == 0:
                        # Legacy single-job slot (sync remote_ffn).
                        job = self._pending_job
                        self._pending_job = None
                if job is None:
                    continue
                handle = self.transport.push_pull(**job)
                # Warmup/capture can trigger on-demand FFN CUDA graphs (seconds).
                self.transport.wait(handle, timeout_ms=120000)
                self._flag_expected[mb_id] = signal_value + 1
                flags.ack_host.fill_(signal_value)
                progressed = True
            if not progressed:
                # Avoid busy-spin burning a core when idle.
                time.sleep(0)

    def _use_deferred_flag_async(self) -> bool:
        """Issue write_flag only; caller waits later (true Attn∥FFN overlap).

        cuda_ipc: prefer mailbox ``push_pull`` (post without wait). A flag-CPU
        hop races FFN's on-demand CUDA-graph capture during warmup (default
        wait was 5s) and is unnecessary for overlap — soft stream wait already
        hides F2A behind the next mb's Attn.
        """
        if self._force_mailbox_async or not self._wait_flags:
            return False
        if bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get()):
            return False
        if self._in_breakable_cudagraph_capture():
            return False
        try:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return False
        except Exception:
            pass
        return bool(envs.SGLANG_AFD_TRUE_OVERLAP.get()) or (
            bool(envs.SGLANG_AFD_USE_WAIT_FLAG.get())
            and bool(envs.SGLANG_AFD_LAYER_PIPELINE.get())
        )

    def _remote_ffn_chunked(
        self,
        *,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        merge_k: int,
        residual: Optional[torch.Tensor],
        positions: Optional[torch.Tensor],
    ):
        """Serve an oversized prefill by splitting on the token dimension.

        Decode farm traffic is already B_step-sized. Splitting here keeps the
        registered IPC buffers fixed while allowing batched prefill/chunked
        prefill to use the same remote-FFN contract.
        """
        chunk_size = max(1, int(self.pool.cfg.max_num_token))
        t = int(hidden.shape[0])
        out = torch.empty_like(hidden)
        residual_out = torch.empty_like(residual) if residual is not None else None
        saw_tuple = False
        logger.info(
            "AFD remote FFN splitting token batch=%s into chunks=%s (%s tokens)",
            t,
            (t + chunk_size - 1) // chunk_size,
            chunk_size,
        )
        for start in range(0, t, chunk_size):
            end = min(t, start + chunk_size)
            chunk = self.remote_ffn(
                layer_id=layer_id,
                hidden=hidden[start:end],
                topk_ids=topk_ids[start:end] if topk_ids is not None else None,
                topk_weights=(
                    topk_weights[start:end] if topk_weights is not None else None
                ),
                merge_k=merge_k,
                residual=residual[start:end] if residual is not None else None,
                positions=positions[start:end] if positions is not None else None,
            )
            if isinstance(chunk, tuple):
                saw_tuple = True
                out[start:end].copy_(chunk[0])
                if residual_out is None:
                    residual_out = torch.empty_like(chunk[1])
                residual_out[start:end].copy_(chunk[1])
            else:
                out[start:end].copy_(chunk)
        return (out, residual_out) if saw_tuple else out

    def remote_ffn_async(
        self,
        *,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        mb_id: Optional[int] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> "AfdPendingTransfer":
        """Issue A2F without waiting (P3). Caller must ``wait_remote_ffn``."""
        from sglang.srt.afd.pipeline import AfdPendingTransfer
        from sglang.srt.afd.protocol import pack_layer_merge_meta

        if mb_id is None:
            mb_id = 0 if self.num_mb == 1 else self.next_mb()
        if mb_id in self._mb_in_flight:
            raise RuntimeError(f"AFD mb_id={mb_id} already in flight")

        t = hidden.shape[0]
        a2f = self.pool.fill_a2f(
            mb_id,
            hidden=hidden,
            layer_id=layer_id,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )
        f2a = self.pool.get_f2a(mb_id)
        packed_layer = pack_layer_merge_meta(layer_id, merge_k)
        t_issue = time.perf_counter()

        if self._use_deferred_flag_async():
            import fserver_lib as f

            flag_i = mb_id if mb_id < len(self._wait_flags) else 0
            flags = self._wait_flags[flag_i]
            if hasattr(self.transport, "write_meta_dev"):
                self.transport.write_meta_dev(
                    mb_id,
                    num_tokens_t=a2f.num_tokens,
                    layer_id_t=a2f.layer_id,
                )
            job = dict(
                layer_id=packed_layer,
                mb_id=mb_id,
                a2f=a2f.as_tensor_list(),
                f2a=f2a.as_tensor_list(),
                num_tokens=t,
            )
            with self._job_lock:
                self._pending_jobs[mb_id] = job
            f.seq_add_one(flags.sequence)
            f.write_flag(flags.signal_dev, flags.sequence)
            issue_event = None
            if self._ffn_wait_stream is not None and torch.cuda.is_available():
                issue_event = torch.cuda.Event()
                issue_event.record()
            self._mb_in_flight.add(mb_id)
            pending = AfdPendingTransfer(
                handle=AfdHandle(id=-1 - mb_id),
                mb_id=mb_id,
                num_tokens=t,
                layer_id=layer_id,
                use_wait_flag=True,
                issue_event=issue_event,
            )
            if self._timeline_log:
                pending._t_issue = t_issue  # type: ignore[attr-defined]
            pending._merge_k = merge_k  # type: ignore[attr-defined]
            return pending

        handle = self.transport.push_pull(
            layer_id=packed_layer,
            mb_id=mb_id,
            a2f=self.pool._a2f_lists[mb_id],
            f2a=self.pool._f2a_lists[mb_id],
            num_tokens=t,
        )
        self._mb_in_flight.add(mb_id)
        pending = AfdPendingTransfer(
            handle=handle, mb_id=mb_id, num_tokens=t, layer_id=layer_id
        )
        if self._timeline_log:
            pending._t_issue = t_issue  # type: ignore[attr-defined]
        pending._merge_k = merge_k  # type: ignore[attr-defined]
        return pending

    def wait_remote_ffn(
        self, pending: "AfdPendingTransfer", *, clone: bool = True
    ):
        """Wait for a previously issued ``remote_ffn_async``.

        ``clone=False`` is safe when the caller consumes ``mlp_out`` before the
        next ``issue`` on the same ``mb_id`` (layer-pipeline stagger).

        When ``merge_k>1``, returns ``(mlp_out, residual)``.
        """
        if pending.use_wait_flag:
            import fserver_lib as f

            flag_i = (
                pending.mb_id if pending.mb_id < len(self._wait_flags) else 0
            )
            flags = self._wait_flags[flag_i]
            attn_stream = torch.cuda.current_stream()
            wait_stream = self._ffn_wait_stream
            issue_event = pending.issue_event
            if wait_stream is not None and issue_event is not None:
                wait_stream.wait_event(issue_event)
                with torch.cuda.stream(wait_stream):
                    f.wait_flag(flags.ack_dev, flags.sequence)
                attn_stream.wait_stream(wait_stream)
            else:
                f.wait_flag(flags.ack_dev, flags.sequence)
        elif self._wait_soft:
            self.transport.wait(pending.handle, timeout_ms=120000, soft=True)
        else:
            wait_fn = self.transport.wait
            try:
                wait_fn(pending.handle, timeout_ms=120000, soft=True)
            except TypeError:
                try:
                    wait_fn(pending.handle, timeout_ms=120000)
                except TypeError:
                    wait_fn(pending.handle)
        self._mb_in_flight.discard(pending.mb_id)
        if self._timeline_log:
            t_done = time.perf_counter()
            t_issue = getattr(pending, "_t_issue", t_done)
            logger.info(
                "AFD_TIMELINE layer=%s mb=%s tokens=%s wait_ms=%.3f flag=%s",
                pending.layer_id,
                pending.mb_id,
                pending.num_tokens,
                (t_done - t_issue) * 1000.0,
                pending.use_wait_flag,
            )
        f2a = self.pool.get_f2a(pending.mb_id)
        out = f2a.mlp_out[: pending.num_tokens]
        out = out.clone() if clone else out
        merge_k = int(getattr(pending, "_merge_k", 1) or 1)
        if merge_k > 1 and f2a.residual is not None:
            res = f2a.residual[: pending.num_tokens]
            return out, (res.clone() if clone else res)
        return out

    def remote_ffn(
        self,
        *,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        mb_id: Optional[int] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ):
        """Attn-side: send post-MLP-prep hidden, wait for FFN output.

        When ``merge_k>1``, returns ``(mlp_out, residual)``.
        """
        capacity = int(self.pool.cfg.max_num_token)
        if int(hidden.shape[0]) > capacity:
            return self._remote_ffn_chunked(
                layer_id=layer_id,
                hidden=hidden,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                merge_k=merge_k,
                residual=residual,
                positions=positions,
            )
        in_graph = bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        # wait_flag inside full CUDA Graph (in_graph): always use GPU flags.
        # Deferred flag async: issue+wait via remote_ffn_async path.
        # Sync wait_flag outside breakable capture: OK for single-shot remote_ffn.
        use_wait_flag = self._wait_flag is not None and (
            in_graph
            or (
                not self._use_deferred_flag_async()
                and not self._in_breakable_cudagraph_capture()
                and not self._in_breakable_cudagraph_replay()
            )
        )
        if use_wait_flag:
            if mb_id is None:
                mb_id = 0 if self.num_mb == 1 else self.next_mb()
            # In-graph path always uses slot 0 (single outstanding FFN per stream).
            if in_graph:
                mb_id = 0
            t = hidden.shape[0]
            a2f = self.pool.fill_a2f(
                mb_id,
                hidden=hidden,
                layer_id=layer_id,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                merge_k=merge_k,
                residual=residual,
                positions=positions,
            )
            f2a = self.pool.get_f2a(mb_id)
            import fserver_lib as f
            from sglang.srt.afd.protocol import pack_layer_merge_meta

            packed_layer = pack_layer_merge_meta(layer_id, merge_k)

            flag_i = 0 if in_graph else (mb_id if mb_id < len(self._wait_flags) else 0)
            flags = self._wait_flags[flag_i] if self._wait_flags else self._wait_flag
            assert flags is not None
            # Publish tokens/layer on mapped pin before write_flag (same stream).
            # Copy from GPU A2F meta tensors — safe under CUDA graph capture.
            if hasattr(self.transport, "write_meta_dev"):
                self.transport.write_meta_dev(
                    mb_id,
                    num_tokens_t=a2f.num_tokens,
                    layer_id_t=a2f.layer_id,
                )
            if not in_graph:
                job = dict(
                    layer_id=packed_layer,
                    mb_id=mb_id,
                    a2f=a2f.as_tensor_list(),
                    f2a=f2a.as_tensor_list(),
                    num_tokens=t,
                )
                with self._job_lock:
                    self._pending_jobs[mb_id] = job
                    if mb_id == 0:
                        self._pending_job = job
            f.seq_add_one(flags.sequence)
            f.write_flag(flags.signal_dev, flags.sequence)
            f.wait_flag(flags.ack_dev, flags.sequence)
            out = f2a.mlp_out[:t]
            if merge_k > 1 and f2a.residual is not None:
                res = f2a.residual[:t]
                if in_graph:
                    return out, res
                torch.cuda.current_stream().synchronize()
                return out.clone(), res.clone()
            if in_graph:
                return out
            torch.cuda.current_stream().synchronize()
            return out.clone()

        pending = self.remote_ffn_async(
            layer_id=layer_id,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            mb_id=mb_id,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )
        return self.wait_remote_ffn(pending)

    @staticmethod
    def _in_breakable_cudagraph_capture() -> bool:
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
                _current_capture_var,
            )

            return _current_capture_var.get() is not None
        except Exception:
            return False

    @staticmethod
    def _in_breakable_cudagraph_replay() -> bool:
        try:
            from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
                is_in_breakable_cuda_graph,
            )

            return bool(is_in_breakable_cuda_graph())
        except Exception:
            return False

    def close(self) -> None:
        self._flag_stop.set()
        if self._flag_thread is not None:
            self._flag_thread.join(timeout=2.0)
            self._flag_thread = None
        self.transport.close()


def init_afd_runtime(
    *,
    hidden_size: int,
    mode: Optional[AfdMode] = None,
    transport_name: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    ffn_compute: Optional[FfnComputeFn] = None,
    moe_topk: int = 0,
    worker_rank: int = 0,
) -> Optional[AfdRuntime]:
    """Initialize global AFD runtime. No-op when mode is NULL."""
    global _runtime
    mode = mode or get_afd_mode()
    if mode == AfdMode.NULL:
        return None

    transport_name = transport_name or get_afd_transport_name()
    num_mb = int(envs.SGLANG_AFD_NUM_MB.get())
    max_num_token = int(envs.SGLANG_AFD_MAX_NUM_TOKEN.get())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pool = AfdBufferPool(
        AfdBufferPoolConfig(
            num_mb=num_mb,
            max_num_token=max_num_token,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            moe_topk=moe_topk,
            worker_rank=worker_rank,
            layer_merge=int(envs.SGLANG_AFD_LAYER_MERGE_K.get() or 1) > 1,
        )
    )
    transport = create_transport(transport_name, ffn_compute=ffn_compute)
    transport.init(mode, worker_rank=worker_rank)
    transport.register_buffers(pool)

    rt = AfdRuntime(mode=mode, transport=transport, pool=pool)
    with _lock:
        if _runtime is not None:
            _runtime.close()
        _runtime = rt
    from sglang.srt.afd.a2f_quant import get_a2f_dtype_name
    from sglang.srt.afd.routing_scheme import get_routing_scheme

    logger.info(
        "AFD runtime ready mode=%s transport=%s mb=%s max_token=%s H=%s "
        "moe_topk=%s scheme=%s a2f_dtype=%s wire=%s layer_merge=%s "
        "a2f_arity=%s f2a_arity=%s",
        mode.value,
        transport_name,
        num_mb,
        max_num_token,
        hidden_size,
        moe_topk,
        get_routing_scheme().value,
        get_a2f_dtype_name(),
        pool.wire_dtype,
        pool.cfg.layer_merge,
        len(pool.a2f_tensor_list(0)),
        len(pool.f2a_tensor_list(0)),
    )
    return rt


def get_afd_runtime() -> Optional[AfdRuntime]:
    return _runtime


def shutdown_afd_runtime() -> None:
    global _runtime
    with _lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None


def is_afd_enabled() -> bool:
    return get_afd_mode() != AfdMode.NULL


def is_afd_attn() -> bool:
    return get_afd_mode() == AfdMode.ATTN


def is_afd_ffn() -> bool:
    return get_afd_mode() == AfdMode.FFN
