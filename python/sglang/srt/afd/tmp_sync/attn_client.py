# SPDX-License-Identifier: Apache-2.0
"""Attn-side AfPool client: multi cuda_ipc links + scheduler."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from sglang.srt.afd.buffers import AfdBufferPool, AfdBufferPoolConfig
from sglang.srt.afd.cuda_ipc_transport import CudaIpcAfdTransport
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.pool.scheduler import AfScheduler
from sglang.srt.afd.pool.types import A2FWorkItem, PoolTopology
from sglang.srt.afd.protocol import pack_layer_merge_meta
from sglang.srt.afd.transport import AfdHandle

logger = logging.getLogger(__name__)

RemoteOut = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class _Link:
    ffn_rank: int
    endpoint: str
    transport: CudaIpcAfdTransport
    pool: AfdBufferPool
    mb_cursor: int = 0
    in_flight: set = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.in_flight is None:
            self.in_flight = set()

    def next_mb(self) -> int:
        n = self.pool.cfg.num_mb
        for _ in range(n):
            mb = self.mb_cursor
            self.mb_cursor = (self.mb_cursor + 1) % n
            if mb not in self.in_flight:
                return mb
        raise RuntimeError(
            f"AfPool link ffn={self.ffn_rank}: all {n} mb slots in flight"
        )


class AfAttnClient:
    """One Attn rank with Na→Nf cuda_ipc transports."""

    def __init__(
        self,
        topo: PoolTopology,
        *,
        attn_rank: int,
        hidden_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_mb: int = 2,
        max_num_token: int = 256,
        moe_topk: int = 0,
    ):
        self.topo = topo
        self.attn_rank = int(attn_rank)
        self.scheduler = AfScheduler(topo, attn_rank=self.attn_rank)
        self.device = device
        self.dtype = dtype
        self._links: List[_Link] = []
        self._lock = threading.Lock()
        self._pending: Dict[int, Tuple[_Link, AfdHandle, A2FWorkItem, float]] = {}
        self._handle_seq = 0

        for j in range(topo.num_ffn):
            ep = topo.endpoint(self.attn_rank, j)
            pool = AfdBufferPool(
                AfdBufferPoolConfig(
                    num_mb=num_mb,
                    max_num_token=max_num_token,
                    hidden_size=hidden_size,
                    dtype=dtype,
                    device=device,
                    moe_topk=moe_topk,
                    worker_rank=self.attn_rank,
                )
            )
            tr = CudaIpcAfdTransport(endpoint=ep)
            tr.init(AfdMode.ATTN, worker_rank=self.attn_rank)
            tr.register_buffers(pool)
            self._links.append(
                _Link(ffn_rank=j, endpoint=ep, transport=tr, pool=pool)
            )
            logger.info(
                "AfPool Attn rank=%s link ffn=%s endpoint=%s ready",
                self.attn_rank,
                j,
                ep,
            )

    def close(self) -> None:
        for link in self._links:
            try:
                link.transport.close()
            except Exception:
                pass

    def _max_num_token(self) -> int:
        return max(1, int(self._links[0].pool.cfg.max_num_token))

    def _require_single_hop_capacity(
        self, method: str, hidden: torch.Tensor
    ) -> None:
        t = int(hidden.shape[0])
        capacity = self._max_num_token()
        if t > capacity:
            raise RuntimeError(
                f"AfPool {method} does not split oversized batches: "
                f"tokens={t} capacity={capacity}. "
                "Use blocking remote_ffn() for prefill-sized batches."
            )

    def issue_capacity(self, *, num_tokens: int = 0) -> int:
        """Return how many single-hop A2F requests can be issued immediately.

        This is a lock-safe hint used by farm queues to avoid repeatedly
        concatenating tensors and allocating scheduler tasks while every FFN
        link is out of credit or MB slots. The real issue path still performs
        the authoritative acquire.
        """
        if int(num_tokens) > self._max_num_token():
            return 0
        capacity = 0
        for link in list(self._links):
            credit = self.scheduler.credit.window_remaining(link.ffn_rank)
            if credit <= 0:
                continue
            with self._lock:
                free_mb = int(link.pool.cfg.num_mb) - len(link.in_flight)
            if free_mb > 0:
                capacity += min(int(credit), int(free_mb))
        return capacity

    def remote_ffn(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> RemoteOut:
        """Blocking A2F→F2A via routed FFN (credit + least_inflight)."""
        t = int(hidden.shape[0])
        capacity = self._max_num_token()
        if t > capacity:
            logger.info(
                "AfPool remote FFN splitting token batch=%s into chunks=%s (%s tokens)",
                t,
                (t + capacity - 1) // capacity,
                capacity,
            )
            chunks = [
                self._remote_ffn_single(
                    req_id=req_id,
                    layer_id=layer_id,
                    hidden=hidden[start : start + capacity],
                    topk_ids=(
                        topk_ids[start : start + capacity]
                        if topk_ids is not None
                        else None
                    ),
                    topk_weights=(
                        topk_weights[start : start + capacity]
                        if topk_weights is not None
                        else None
                    ),
                    merge_k=merge_k,
                    residual=(
                        residual[start : start + capacity]
                        if residual is not None
                        else None
                    ),
                    positions=(
                        positions[start : start + capacity]
                        if positions is not None
                        else None
                    ),
                )
                for start in range(0, t, capacity)
            ]
            if any(isinstance(chunk, tuple) for chunk in chunks):
                if not all(isinstance(chunk, tuple) for chunk in chunks):
                    raise RuntimeError(
                        "AfPool FFN split chunks returned inconsistent outputs"
                    )
                return tuple(
                    torch.cat([chunk[i] for chunk in chunks], dim=0)
                    for i in (0, 1)
                )  # type: ignore[return-value]
            return torch.cat(chunks, dim=0)  # type: ignore[arg-type]

        return self._remote_ffn_single(
            req_id=req_id,
            layer_id=layer_id,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )

    def _remote_ffn_single(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> RemoteOut:
        """Issue one A2F hop that fits the registered slot."""
        t = int(hidden.shape[0])
        item = self.scheduler.next_task(req_id, layer_id, t)
        ffn = self.scheduler.assign_ffn(item)
        link = self._links[ffn]
        with self._lock:
            mb = link.next_mb()
            link.in_flight.add(mb)
        item.mb_id = mb

        a2f = link.pool.fill_a2f(
            mb,
            hidden=hidden,
            layer_id=int(layer_id),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )
        f2a = link.pool.get_f2a(mb)
        t0 = time.perf_counter()
        handle = link.transport.push_pull(
            layer_id=pack_layer_merge_meta(layer_id, merge_k),
            mb_id=mb,
            a2f=a2f.as_tensor_list(),
            f2a=f2a.as_tensor_list(),
            num_tokens=t,
        )
        try:
            wait = link.transport.wait
            try:
                wait(handle, timeout_ms=120000, soft=True)
            except TypeError:
                wait(handle, timeout_ms=120000)
        finally:
            compute_s = time.perf_counter() - t0
            with self._lock:
                link.in_flight.discard(mb)
            self.scheduler.complete(item, compute_s=compute_s)

        out = link.pool.get_f2a(mb)
        if merge_k > 1 and out.residual is not None:
            return out.mlp_out[:t].clone(), out.residual[:t].clone()
        return out.mlp_out[:t].clone()

    def issue_remote_ffn(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> int:
        """Non-blocking issue; returns local pending id for ``wait_remote_ffn``."""
        self._require_single_hop_capacity("issue_remote_ffn", hidden)
        t = int(hidden.shape[0])
        item = self.scheduler.next_task(req_id, layer_id, t)
        ffn = self.scheduler.assign_ffn(item)
        link = self._links[ffn]
        with self._lock:
            mb = link.next_mb()
            link.in_flight.add(mb)
            self._handle_seq += 1
            pid = self._handle_seq
        item.mb_id = mb
        a2f = link.pool.fill_a2f(
            mb,
            hidden=hidden,
            layer_id=int(layer_id),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )
        f2a = link.pool.get_f2a(mb)
        t0 = time.perf_counter()
        handle = link.transport.push_pull(
            layer_id=pack_layer_merge_meta(layer_id, merge_k),
            mb_id=mb,
            a2f=a2f.as_tensor_list(),
            f2a=f2a.as_tensor_list(),
            num_tokens=t,
        )
        with self._lock:
            self._pending[pid] = (link, handle, item, t0)
        return pid

    def try_issue_remote_ffn(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        merge_k: int = 1,
        residual: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> Optional[int]:
        """Non-blocking issue; None if no credit or mb slot (farm smooth send)."""
        self._require_single_hop_capacity("try_issue_remote_ffn", hidden)
        t = int(hidden.shape[0])
        item = self.scheduler.next_task(req_id, layer_id, t)
        ffn = self.scheduler.try_assign_ffn(item)
        if ffn is None:
            return None
        link = self._links[ffn]
        with self._lock:
            try:
                mb = link.next_mb()
            except RuntimeError:
                mb = None
            if mb is not None:
                link.in_flight.add(mb)
                self._handle_seq += 1
                pid = self._handle_seq
        if mb is None:
            self.scheduler.abort_assign(item)
            return None
        item.mb_id = mb
        a2f = link.pool.fill_a2f(
            mb,
            hidden=hidden,
            layer_id=int(layer_id),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            merge_k=merge_k,
            residual=residual,
            positions=positions,
        )
        f2a = link.pool.get_f2a(mb)
        t0 = time.perf_counter()
        handle = link.transport.push_pull(
            layer_id=pack_layer_merge_meta(layer_id, merge_k),
            mb_id=mb,
            a2f=a2f.as_tensor_list(),
            f2a=f2a.as_tensor_list(),
            num_tokens=t,
        )
        with self._lock:
            self._pending[pid] = (link, handle, item, t0)
        return pid

    def poll_remote_ffn(
        self, pending_id: int, *, merge_k: int = 1
    ) -> Optional[RemoteOut]:
        with self._lock:
            rec = self._pending.get(int(pending_id))
            if rec is None:
                raise KeyError(f"unknown AfPool pending id {pending_id}")
            link, handle, _item, _t0 = rec
        poll = getattr(link.transport, "poll_done", None)
        if not callable(poll) or not poll(handle):
            return None
        return self.wait_remote_ffn(pending_id, merge_k=merge_k)

    def wait_remote_ffn(self, pending_id: int, *, merge_k: int = 1) -> RemoteOut:
        with self._lock:
            link, handle, item, t0 = self._pending.pop(int(pending_id))
        try:
            wait = link.transport.wait
            try:
                wait(handle, timeout_ms=120000, soft=True)
            except TypeError:
                wait(handle, timeout_ms=120000)
        finally:
            compute_s = time.perf_counter() - t0
            with self._lock:
                link.in_flight.discard(item.mb_id)
            self.scheduler.complete(item, compute_s=compute_s)
            try:
                from sglang.srt.afd.detail_profile import (
                    profile_detail_enabled,
                    record_us,
                )

                if profile_detail_enabled():
                    record_us("rtt_us", compute_s * 1e6)
            except Exception:
                pass
        t = item.num_tokens
        out = link.pool.get_f2a(item.mb_id)
        if merge_k > 1 and out.residual is not None:
            return out.mlp_out[:t].clone(), out.residual[:t].clone()
        return out.mlp_out[:t].clone()

    def run_multi_req_layers(
        self,
        *,
        num_reqs: int,
        layers: int,
        tokens: int,
        hidden_size: int,
        attn_burn_us: float,
        dual_mb: bool = True,
    ) -> None:
        """Throughput-oriented loop: many reqs × layers with dual-mb stagger."""

        def _burn(us: float) -> None:
            if us <= 0:
                return
            deadline = time.perf_counter() + us * 1e-6
            while time.perf_counter() < deadline:
                pass

        hs = [
            torch.empty(tokens, hidden_size, device=self.device, dtype=self.dtype)
            for _ in range(num_reqs)
        ]
        for h in hs:
            h.normal_()

        self.scheduler.begin_wall()
        if dual_mb and num_reqs >= 2:
            # Stagger: keep 2 reqs in flight across layers.
            layer_idx = [0] * num_reqs
            pending: Dict[int, int] = {}  # req -> pending_id
            active = list(range(num_reqs))
            while active:
                progressed = False
                for r in list(active):
                    if r in pending:
                        continue
                    if layer_idx[r] >= layers:
                        active.remove(r)
                        continue
                    if len(pending) >= min(2, num_reqs):
                        break
                    _burn(attn_burn_us)
                    pid = self.issue_remote_ffn(
                        req_id=r,
                        layer_id=layer_idx[r],
                        hidden=hs[r],
                    )
                    pending[r] = pid
                    progressed = True
                if not pending:
                    if not progressed:
                        break
                    continue
                # Wait oldest pending.
                r = next(iter(pending))
                pid = pending.pop(r)
                self.wait_remote_ffn(pid)
                layer_idx[r] += 1
        else:
            for r in range(num_reqs):
                for L in range(layers):
                    _burn(attn_burn_us)
                    self.remote_ffn(req_id=r, layer_id=L, hidden=hs[r])
        self.scheduler.end_wall()
