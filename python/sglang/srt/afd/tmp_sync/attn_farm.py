# SPDX-License-Identifier: Apache-2.0
"""Attn-side farm hops: async A2F issue + poll F2A (pool credit or classic)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import torch

from sglang.srt.afd.pipeline import AfdPendingTransfer
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

RemoteOut = Union[torch.Tensor, tuple]


@dataclass
class FarmHop:
    """One in-flight A2F window."""

    layer_i: int
    seq_idxs: List[int]
    token_lo: int
    token_hi: int
    residual: Optional[torch.Tensor]
    meta: Dict[str, Any]
    pool_pid: Optional[int] = None
    classic: Optional[AfdPendingTransfer] = None
    send_queue: Optional["AttnSendQueue"] = None
    send_item: Optional["_QueuedSend"] = None
    send_group: Optional["_SendGroup"] = None
    # Host timestamp set when the hop is issued (stage-stats instrumentation).
    t_issue: float = 0.0


@dataclass
class _QueuedSend:
    req_id: int
    layer_id: int
    hidden: torch.Tensor
    topk_ids: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]
    hop: FarmHop
    num_tokens: int
    enqueued_ts: float
    eligible_ts: Optional[float] = None
    blocked_us: float = 0.0
    blocked_since: Optional[float] = None


@dataclass
class _SendGroup:
    queue: "AttnSendQueue"
    layer_id: int
    items: List[_QueuedSend]
    pid: Optional[int] = None
    output: Optional[RemoteOut] = None

    def slice_for(self, hop: FarmHop) -> Optional[RemoteOut]:
        if self.output is None:
            return None
        start = 0
        for item in self.items:
            end = start + item.num_tokens
            if item.hop is hop:
                return _slice_remote_out(self.output, start, end)
            start = end
        raise KeyError("hop does not belong to Attn send group")


class AttnSendQueue:
    """Per-layer outbound queue that coalesces same-layer A2F calls.

    Tensors are snapshotted at enqueue time because the next Attn window may
    overwrite its source buffers. A group consumes an A2F slot only when it is
    actually issued.
    """

    def __init__(
        self,
        client,
        *,
        target_tokens: int = 64,
        max_wait_us: int = 200,
        max_hops: int = 4,
        stats_every: int = 0,
    ) -> None:
        self.client = client
        self.target_tokens = max(1, int(target_tokens))
        self.max_wait_us = max(0, int(max_wait_us))
        self.max_hops = max(1, int(max_hops))
        self.stats_every = max(0, int(stats_every))
        self._layers: Dict[int, Deque[_QueuedSend]] = defaultdict(deque)
        self._layer_tokens: Dict[int, int] = defaultdict(int)
        self._stats: Dict[str, Any] = {
            "groups": 0,
            "hops": 0,
            "tokens": 0,
            "active_wait_us_sum": 0.0,
            "active_wait_us_max": 0.0,
            "blocked_us_sum": 0.0,
            "blocked_us_max": 0.0,
            "blocked_events": 0,
            "capacity_empty": 0,
            "group_hist": {},
            "layer_hist": {},
            "flush_reason_hist": {},
        }

    def pending(self) -> bool:
        return any(self._layers.values())

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def enqueue(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        layer_i: int,
        seq_idxs: Sequence[int],
        token_lo: int,
        token_hi: int,
        residual: Optional[torch.Tensor],
        meta: Dict[str, Any],
        now: Optional[float] = None,
    ) -> FarmHop:
        now = time.perf_counter() if now is None else float(now)
        hop = FarmHop(
            layer_i=int(layer_i),
            seq_idxs=list(seq_idxs),
            token_lo=int(token_lo),
            token_hi=int(token_hi),
            residual=_clone_optional(residual),
            meta=meta,
        )
        item = _QueuedSend(
            req_id=int(req_id),
            layer_id=int(layer_id),
            hidden=hidden.detach().clone(),
            topk_ids=_clone_optional(topk_ids),
            topk_weights=_clone_optional(topk_weights),
            hop=hop,
            num_tokens=int(hidden.shape[0]),
            enqueued_ts=now,
        )
        hop.send_queue = self
        hop.send_item = item
        self._layers[item.layer_id].append(item)
        self._layer_tokens[item.layer_id] += item.num_tokens
        self.flush_due(now=now)
        return hop

    def flush_due(self, *, now: Optional[float] = None) -> int:
        now = time.perf_counter() if now is None else float(now)
        ready = []
        for layer_id, q in self._layers.items():
            if not q:
                continue
            waited_us = max(0.0, (now - q[0].enqueued_ts) * 1e6)
            if (
                self._layer_tokens[layer_id] >= self.target_tokens
                or len(q) >= self.max_hops
                or waited_us >= self.max_wait_us
            ):
                ready.append(layer_id)
        ready.sort(key=lambda layer_id: self._layers[layer_id][0].enqueued_ts)
        if not ready:
            return 0

        capacity = self._issue_capacity(ready)
        if capacity <= 0:
            self._account_blocked(ready, now)
            self._stats["capacity_empty"] = int(self._stats["capacity_empty"]) + 1
            return 0

        issued = 0
        for layer_id in ready:
            if capacity <= 0:
                self._account_blocked(ready[issued:], now)
                break
            if self._issue_layer(layer_id, force=False, now=now):
                issued += 1
                capacity -= 1
        return issued

    def _issue_capacity(self, layer_ids: Sequence[int]) -> int:
        if not layer_ids:
            return 0
        min_tokens = min(
            sum(item.num_tokens for item in list(self._layers[layer_id])[: self.max_hops])
            for layer_id in layer_ids
            if self._layers.get(layer_id)
        )
        check = getattr(self.client, "issue_capacity", None)
        if not callable(check):
            return 1
        return max(0, int(check(num_tokens=min_tokens)))

    def _account_blocked(self, layer_ids: Sequence[int], now: float) -> None:
        event = False
        for layer_id in layer_ids:
            q = self._layers.get(layer_id)
            if not q:
                continue
            for item in q:
                if item.eligible_ts is None:
                    item.eligible_ts = now
                if item.blocked_since is None:
                    item.blocked_since = now
                delta = max(0.0, now - item.blocked_since)
                item.blocked_us += delta * 1e6
                item.blocked_since = now
                event = event or delta > 0.0
        if event:
            self._stats["blocked_events"] = int(self._stats["blocked_events"]) + 1

    def _mark_eligible(self, items: Sequence[_QueuedSend], now: float) -> None:
        for item in items:
            if item.eligible_ts is None:
                item.eligible_ts = now

    def _issue_layer(self, layer_id: int, *, force: bool, now: float) -> bool:
        q = self._layers.get(layer_id)
        if not q:
            return False
        items = list(q)[: self.max_hops]
        if len(items) < self.max_hops:
            waited_us = max(0.0, (now - items[0].enqueued_ts) * 1e6)
            if not (
                force
                or self._layer_tokens[layer_id] >= self.target_tokens
                or waited_us >= self.max_wait_us
            ):
                return False
        capacity = self._issue_capacity((layer_id,))
        if capacity <= 0:
            self._account_blocked((layer_id,), now)
            self._stats["capacity_empty"] = int(self._stats["capacity_empty"]) + 1
            return False
        self._mark_eligible(items, now)
        for item in items:
            if item.blocked_since is None:
                continue
            item.blocked_us += max(0.0, now - item.blocked_since) * 1e6
            item.blocked_since = None
        hidden = torch.cat([item.hidden for item in items], dim=0)
        topk_ids = _cat_optional([item.topk_ids for item in items])
        topk_weights = _cat_optional([item.topk_weights for item in items])
        pid = self.client.try_issue_remote_ffn(
            req_id=items[0].req_id,
            layer_id=layer_id,
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        if pid is None:
            return False

        group = _SendGroup(queue=self, layer_id=layer_id, items=items, pid=int(pid))
        for _ in items:
            q.popleft()
        if not q:
            self._layers.pop(layer_id, None)
        self._layer_tokens.pop(layer_id, None)
        for item in items:
            item.hop.send_group = group

        active_wait_us = max(
            0.0,
            (
                (items[0].eligible_ts or now)
                - items[0].enqueued_ts
            )
            * 1e6,
        )
        blocked_us = sum(item.blocked_us for item in items) / max(1, len(items))
        group_tokens = sum(item.num_tokens for item in items)
        if force:
            reason = "force"
        elif self._layer_tokens.get(layer_id, group_tokens) >= self.target_tokens:
            reason = "target_tokens"
        elif len(items) >= self.max_hops:
            reason = "max_hops"
        else:
            reason = "timeout"
        self._stats["groups"] = int(self._stats["groups"]) + 1
        self._stats["hops"] = int(self._stats["hops"]) + len(items)
        self._stats["tokens"] = int(self._stats["tokens"]) + group_tokens
        self._stats["active_wait_us_sum"] = float(
            self._stats["active_wait_us_sum"]
        ) + (
            active_wait_us * len(items)
        )
        self._stats["active_wait_us_max"] = max(
            float(self._stats["active_wait_us_max"]), active_wait_us
        )
        self._stats["blocked_us_sum"] = float(
            self._stats["blocked_us_sum"]
        ) + (
            blocked_us * len(items)
        )
        self._stats["blocked_us_max"] = max(
            float(self._stats["blocked_us_max"]), blocked_us
        )
        hist = self._stats["group_hist"]
        hist[len(items)] = int(hist.get(len(items), 0)) + 1
        layer_hist = self._stats["layer_hist"]
        layer_hist[layer_id] = int(layer_hist.get(layer_id, 0)) + 1
        reason_hist = self._stats["flush_reason_hist"]
        reason_hist[reason] = int(reason_hist.get(reason, 0)) + 1
        self._maybe_log()
        return True

    def _maybe_log(self) -> None:
        if self.stats_every <= 0:
            return
        groups = int(self._stats["groups"])
        if groups % self.stats_every != 0:
            return
        hops = max(1, int(self._stats["hops"]))
        logger.info(
            "AFD Attn queue stats groups=%s hops=%s tokens=%s "
            "mean_hops=%.2f active_wait_mean_us=%.1f active_wait_max_us=%.1f "
            "blocked_mean_us=%.1f blocked_max_us=%.1f blocked_events=%s "
            "capacity_empty=%s reason_hist=%s group_hist=%s layer_hist=%s",
            groups,
            self._stats["hops"],
            self._stats["tokens"],
            float(self._stats["hops"]) / groups,
            float(self._stats["active_wait_us_sum"]) / hops,
            self._stats["active_wait_us_max"],
            float(self._stats["blocked_us_sum"]) / hops,
            self._stats["blocked_us_max"],
            self._stats["blocked_events"],
            self._stats["capacity_empty"],
            self._stats["flush_reason_hist"],
            self._stats["group_hist"],
            self._stats["layer_hist"],
        )

    def poll_group(self, group: _SendGroup) -> Optional[RemoteOut]:
        if group.pid is None:
            self.flush_due()
            if group.pid is None:
                return None
        if group.output is None:
            output = self.client.poll_remote_ffn(group.pid)
            if output is None:
                return None
            group.output = output
        return group.output

    def wait_group(self, group: _SendGroup) -> RemoteOut:
        while group.pid is None:
            self._issue_layer(
                group.layer_id,
                force=True,
                now=time.perf_counter(),
            )
            if group.pid is None:
                time.sleep(0.00005)
        if group.output is None:
            group.output = self.client.wait_remote_ffn(group.pid)
        return group.output

    def wait_pending(self, hop: FarmHop) -> RemoteOut:
        if hop.send_item is None:
            raise RuntimeError("Attn queued hop is missing its queue item")
        while hop.send_group is None:
            self._issue_layer(
                hop.send_item.layer_id,
                force=True,
                now=time.perf_counter(),
            )
            if hop.send_group is None:
                time.sleep(0.00005)
        return self.wait_group(hop.send_group)


_ATTN_SEND_QUEUE: Optional[AttnSendQueue] = None
_ATTN_SEND_QUEUE_CLIENT_ID: Optional[int] = None


def _attn_queue_enabled() -> bool:
    return bool(envs.SGLANG_AFD_ATTN_QUEUE_ENABLE.get())


def _attn_send_queue(client) -> AttnSendQueue:
    global _ATTN_SEND_QUEUE, _ATTN_SEND_QUEUE_CLIENT_ID
    client_id = id(client)
    if _ATTN_SEND_QUEUE is None or _ATTN_SEND_QUEUE_CLIENT_ID != client_id:
        _ATTN_SEND_QUEUE = AttnSendQueue(
            client,
            target_tokens=envs.SGLANG_AFD_ATTN_QUEUE_TARGET_TOKENS.get(),
            max_wait_us=envs.SGLANG_AFD_ATTN_QUEUE_MAX_WAIT_US.get(),
            max_hops=envs.SGLANG_AFD_ATTN_QUEUE_MAX_HOPS.get(),
            stats_every=envs.SGLANG_AFD_ATTN_QUEUE_STATS_EVERY.get(),
        )
        _ATTN_SEND_QUEUE_CLIENT_ID = client_id
    return _ATTN_SEND_QUEUE


def _clone_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone()


def _cat_optional(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    if not tensors or any(tensor is None for tensor in tensors):
        return None
    return torch.cat(tensors, dim=0)


def _slice_remote_out(output: RemoteOut, start: int, end: int) -> RemoteOut:
    if isinstance(output, tuple):
        return tuple(part[start:end] for part in output)
    return output[start:end]


def _pool_client():
    try:
        from sglang.srt.afd.pool import get_af_attn_client, pool_enabled

        if pool_enabled():
            return get_af_attn_client()
    except Exception:
        return None
    return None


def try_issue_hop(
    *,
    layer_id: int,
    hidden: torch.Tensor,
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    layer_i: int,
    seq_idxs: List[int],
    token_lo: int,
    token_hi: int,
    residual: Optional[torch.Tensor],
    meta: Dict[str, Any],
    req_id: int = 0,
) -> Optional[FarmHop]:
    """Non-blocking A2F. None = no credit / no mb slot."""
    client = _pool_client()
    if client is not None:
        queue = None
        if _attn_queue_enabled():
            bypass_first = bool(
                envs.SGLANG_AFD_ATTN_QUEUE_BYPASS_FIRST_LAYER.get()
            )
            if not (bypass_first and int(layer_i) == 0):
                queue = _attn_send_queue(client)
        if queue is not None:
            return queue.enqueue(
                req_id=int(req_id),
                layer_id=int(layer_id),
                hidden=hidden,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                layer_i=int(layer_i),
                seq_idxs=seq_idxs,
                token_lo=token_lo,
                token_hi=token_hi,
                residual=residual,
                meta=meta,
            )

    hop = FarmHop(
        layer_i=layer_i,
        seq_idxs=list(seq_idxs),
        token_lo=int(token_lo),
        token_hi=int(token_hi),
        residual=residual,
        meta=meta,
    )
    if client is not None:
        pid = client.try_issue_remote_ffn(
            req_id=int(req_id),
            layer_id=int(layer_id),
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        if pid is None:
            return None
        hop.pool_pid = int(pid)
        return hop

    from sglang.srt.afd.runtime import get_afd_runtime

    rt = get_afd_runtime()
    if rt is None:
        return None
    if len(rt._mb_in_flight) >= rt.num_mb:
        return None
    mb = rt.next_mb()
    hop.classic = rt.remote_ffn_async(
        layer_id=int(layer_id),
        hidden=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mb_id=mb,
    )
    return hop


def poll_hop(hop: FarmHop) -> Optional[RemoteOut]:
    """Return F2A payload if done, else None (does not block)."""
    if hop.send_queue is not None and hop.send_group is None:
        hop.send_queue.flush_due()
        if hop.send_group is None:
            return None
    if hop.send_group is not None:
        output = hop.send_group.queue.poll_group(hop.send_group)
        if output is None:
            return None
        return hop.send_group.slice_for(hop)
    client = _pool_client()
    if hop.pool_pid is not None and client is not None:
        return client.poll_remote_ffn(hop.pool_pid)
    if hop.classic is None:
        return None
    from sglang.srt.afd.runtime import get_afd_runtime

    rt = get_afd_runtime()
    if rt is None:
        return None
    tr = rt.transport
    poll = getattr(tr, "poll_done", None)
    if callable(poll):
        if not poll(hop.classic.handle):
            return None
    else:
        return None
    return rt.wait_remote_ffn(hop.classic, clone=False)


def wait_hop(hop: FarmHop) -> RemoteOut:
    """Blocking F2A wait."""
    if hop.send_queue is not None and hop.send_group is None:
        return hop.send_queue.wait_pending(hop)
    if hop.send_group is not None:
        output = hop.send_group.queue.wait_group(hop.send_group)
        return hop.send_group.slice_for(hop)
    client = _pool_client()
    if hop.pool_pid is not None and client is not None:
        return client.wait_remote_ffn(hop.pool_pid)
    if hop.classic is None:
        raise RuntimeError("farm hop has no pending transfer")
    from sglang.srt.afd.runtime import get_afd_runtime

    rt = get_afd_runtime()
    if rt is None:
        raise RuntimeError("AFD runtime missing while waiting farm hop")
    return rt.wait_remote_ffn(hop.classic, clone=False)
