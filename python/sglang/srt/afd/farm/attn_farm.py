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
    batch_id: int = 0
    sched_ticket: int = 0
    ctx_id: int = 0
    mb_id: int = 0
    step_id: int = 0

    @property
    def owner(self) -> Tuple[int, int, int]:
        return (int(self.ctx_id), int(self.mb_id), int(self.step_id))


@dataclass
class _QueuedSend:
    req_id: int
    layer_id: int
    owner: Tuple[int, int, int]
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


def _send_queue_key(
    owner: Tuple[int, int, int], layer_id: int
) -> int:
    """Queue by layer so independent contexts share one A2F group."""
    del owner
    return int(layer_id)


class AttnSendQueue:
    """Per-layer outbound queue that coalesces same-layer A2F calls.

    Tensors are snapshotted at enqueue time because the next Attn window may
    overwrite its source buffers. A group consumes an A2F slot only when it is
    actually issued. The queue uses continuous-batching semantics: any ready
    layer is eligible immediately, while ``target_tokens`` and ``max_hops``
    only bound the size of one group. Items from independent decode contexts
    share a queue at the same layer and are concatenated into one remote FFN
    call; their outputs are sliced back by hop.
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
        self._active_groups: List[_SendGroup] = []
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
            "multi_owner_groups": 0,
            "active_groups_peak": 0,
            "active_layers_peak": 0,
            "active_tokens_peak": 0,
            "group_hist": {},
            "unique_owners_hist": {},
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
        owner: Optional[Tuple[int, int, int]] = None,
        now: Optional[float] = None,
        defer_flush: bool = False,
    ) -> FarmHop:
        now = time.perf_counter() if now is None else float(now)
        owner = (0, 0, 0) if owner is None else tuple(int(x) for x in owner)
        hop = FarmHop(
            layer_i=int(layer_i),
            seq_idxs=list(seq_idxs),
            token_lo=int(token_lo),
            token_hi=int(token_hi),
            residual=_clone_optional(residual),
            meta=meta,
            ctx_id=owner[0],
            mb_id=owner[1],
            step_id=owner[2],
        )
        item = _QueuedSend(
            req_id=int(req_id),
            layer_id=int(layer_id),
            owner=owner,
            hidden=hidden.detach().clone(),
            topk_ids=_clone_optional(topk_ids),
            topk_weights=_clone_optional(topk_weights),
            hop=hop,
            num_tokens=int(hidden.shape[0]),
            enqueued_ts=now,
        )
        hop.send_queue = self
        hop.send_item = item
        key = _send_queue_key(item.owner, item.layer_id)
        self._layers[key].append(item)
        self._layer_tokens[key] += item.num_tokens
        if not defer_flush:
            self.flush_due(now=now)
        return hop

    def cancel(self, hop: FarmHop) -> bool:
        """Remove an unissued hop so the scheduler can put it back in waiting."""
        if hop.send_group is not None or hop.send_item is None:
            return False
        item = hop.send_item
        key = _send_queue_key(item.owner, item.layer_id)
        q = self._layers.get(key)
        if not q:
            return False
        kept = deque(x for x in q if x is not item)
        if len(kept) == len(q):
            return False
        before = self._layer_tokens.get(key, 0)
        self._layer_tokens[key] = max(0, before - item.num_tokens)
        if kept:
            self._layers[key] = kept
        else:
            self._layers.pop(key, None)
            self._layer_tokens.pop(key, None)
        hop.send_item = None
        hop.send_queue = None
        return True

    def flush_due(self, *, now: Optional[float] = None) -> int:
        now = time.perf_counter() if now is None else float(now)
        ready = self._ready_layers()
        if not ready:
            return 0

        capacity = self._issue_capacity(ready)
        if capacity <= 0:
            self._account_blocked(ready, now)
            self._stats["capacity_empty"] = int(self._stats["capacity_empty"]) + 1
            return 0

        issued = 0
        while capacity > 0:
            ready = self._ready_layers()
            if not ready:
                break
            if not self._issue_layer(
                ready[0],
                now=now,
                capacity_checked=True,
            ):
                break
            issued += 1
            capacity -= 1
        return issued

    def _ready_layers(self) -> List[int]:
        """All non-empty layer queues, oldest head first."""
        ready = [key for key, q in self._layers.items() if q]
        ready.sort(key=lambda key: self._layers[key][0].enqueued_ts)
        return ready

    def _issue_capacity(self, queue_keys: Sequence[int]) -> int:
        if not queue_keys:
            return 0
        min_tokens = min(
            sum(item.num_tokens for item in list(self._layers[key])[: self.max_hops])
            for key in queue_keys
            if self._layers.get(key)
        )
        check = getattr(self.client, "issue_capacity", None)
        if not callable(check):
            return 1
        return max(0, int(check(num_tokens=min_tokens)))

    def _account_blocked(self, queue_keys: Sequence[int], now: float) -> None:
        event = False
        for key in queue_keys:
            q = self._layers.get(key)
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

    def _issue_layer(
        self,
        layer_id: int,
        *,
        now: float,
        capacity_checked: bool = False,
    ) -> bool:
        layer_id = int(layer_id)
        q = self._layers.get(layer_id)
        if not q:
            return False
        items, group_tokens = self._take_group(list(q))
        if not items:
            return False
        if not capacity_checked:
            capacity = self._issue_capacity((layer_id,))
            if capacity <= 0:
                self._account_blocked((layer_id,), now)
                self._stats["capacity_empty"] = int(
                    self._stats["capacity_empty"]
                ) + 1
                return False
        if group_tokens >= self.target_tokens:
            reason = "target_tokens"
        elif len(items) >= self.max_hops:
            reason = "max_hops"
        else:
            reason = "ready"
        self._mark_eligible(items, now)
        for item in items:
            if item.blocked_since is None:
                continue
            item.blocked_us += max(0.0, now - item.blocked_since) * 1e6
            item.blocked_since = None
        if len(items) == 1:
            # Single-hop group: torch.cat of one tensor still allocates and
            # launches a copy kernel. fill_a2f copies into the A2F slot anyway,
            # so pass the view through (py-spy §19: send path ~16% of farm time).
            hidden = items[0].hidden
            topk_ids = items[0].topk_ids
            topk_weights = items[0].topk_weights
        else:
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
        self._active_groups.append(group)
        self._stats["active_groups_peak"] = max(
            int(self._stats["active_groups_peak"]),
            len(self._active_groups),
        )
        self._stats["active_layers_peak"] = max(
            int(self._stats["active_layers_peak"]),
            len({active.layer_id for active in self._active_groups}),
        )
        self._stats["active_tokens_peak"] = max(
            int(self._stats["active_tokens_peak"]),
            sum(
                item.num_tokens
                for active in self._active_groups
                for item in active.items
            ),
        )
        for _ in items:
            q.popleft()
        if not q:
            self._layers.pop(layer_id, None)
        self._layer_tokens.pop(layer_id, None)
        for item in items:
            item.hop.send_group = group

        unique_owners = len({item.owner for item in items})
        active_wait_us = max(
            0.0,
            (
                (items[0].eligible_ts or now)
                - items[0].enqueued_ts
            )
            * 1e6,
        )
        blocked_us = sum(item.blocked_us for item in items) / max(1, len(items))
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
        owner_hist = self._stats["unique_owners_hist"]
        owner_hist[unique_owners] = int(owner_hist.get(unique_owners, 0)) + 1
        if unique_owners > 1:
            self._stats["multi_owner_groups"] = (
                int(self._stats["multi_owner_groups"]) + 1
            )
        layer_hist = self._stats["layer_hist"]
        layer_hist[layer_id] = int(layer_hist.get(layer_id, 0)) + 1
        reason_hist = self._stats["flush_reason_hist"]
        reason_hist[reason] = int(reason_hist.get(reason, 0)) + 1
        self._maybe_log()
        return True

    def _take_group(self, queued: List[_QueuedSend]) -> tuple[List[_QueuedSend], int]:
        """Take a bounded prefix without delaying a ready layer."""
        items: List[_QueuedSend] = []
        tokens = 0
        for item in queued[: self.max_hops]:
            if items and tokens + item.num_tokens > self.target_tokens:
                break
            items.append(item)
            tokens += item.num_tokens
            if tokens >= self.target_tokens:
                break
        return items, tokens

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
            "capacity_empty=%s multi_owner_groups=%s reason_hist=%s "
            "active_groups_peak=%s active_layers_peak=%s "
            "active_tokens_peak=%s group_hist=%s "
            "unique_owners_hist=%s layer_hist=%s",
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
            self._stats["multi_owner_groups"],
            self._stats["flush_reason_hist"],
            self._stats["active_groups_peak"],
            self._stats["active_layers_peak"],
            self._stats["active_tokens_peak"],
            self._stats["group_hist"],
            self._stats["unique_owners_hist"],
            self._stats["layer_hist"],
        )

    def poll_group(self, group: _SendGroup) -> Optional[RemoteOut]:
        if group.pid is None:
            self.flush_due()
            if group.pid is None:
                return None
        if group.output is None:
            self._poll_active()
        return group.output

    def wait_group(self, group: _SendGroup) -> RemoteOut:
        while group.output is None:
            # Keep issuing ready groups (and reaping other active groups) while
            # this hop is in flight. Blocking directly on one remote call here
            # serialized the whole layer pipeline.
            self.flush_due()
            self._poll_active()
            if group.output is None:
                time.sleep(0.00005)
        return group.output

    def wait_pending(self, hop: FarmHop) -> RemoteOut:
        if hop.send_item is None:
            raise RuntimeError("Attn queued hop is missing its queue item")
        while hop.send_group is None:
            self.flush_due()
            if hop.send_group is None:
                time.sleep(0.00005)
        return self.wait_group(hop.send_group)

    def _poll_active(self) -> None:
        still: List[_SendGroup] = []
        for group in self._active_groups:
            if group.output is None and group.pid is not None:
                group.output = self.client.poll_remote_ffn(group.pid)
            if group.output is None:
                still.append(group)
        self._active_groups = still


_ATTN_SEND_QUEUE: Optional[AttnSendQueue] = None
_ATTN_SEND_QUEUE_CLIENT_ID: Optional[int] = None
_CLASSIC_SEND_CLIENT = None
_CLASSIC_SEND_CLIENT_RUNTIME_ID: Optional[int] = None


class _ClassicAttnClient:
    """Adapter exposing the pool client's non-blocking credit API for classic AFD.

    The classic 1A1F runtime already has per-microbatch buffers. Wrapping it here
    keeps farm queue accounting on the same slot count as the transport instead
    of maintaining a second, independent max-inflight budget.
    """

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._pending = {}
        self._handle_seq = 0

    def issue_capacity(self, *, num_tokens: int = 0) -> int:
        max_tokens = int(self.runtime.pool.cfg.max_num_token)
        if int(num_tokens) > max_tokens:
            return 0
        return max(0, int(self.runtime.num_mb) - len(self.runtime._mb_in_flight))

    def try_issue_remote_ffn(
        self,
        *,
        req_id: int,
        layer_id: int,
        hidden: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> Optional[int]:
        if int(hidden.shape[0]) > int(self.runtime.pool.cfg.max_num_token):
            return None
        if len(self.runtime._mb_in_flight) >= int(self.runtime.num_mb):
            return None
        try:
            mb_id = self.runtime.next_mb()
        except RuntimeError:
            return None
        pending = self.runtime.remote_ffn_async(
            layer_id=int(layer_id),
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            mb_id=mb_id,
        )
        self._handle_seq += 1
        pid = self._handle_seq
        self._pending[pid] = pending
        return pid

    def poll_remote_ffn(self, pending_id: int):
        pid = int(pending_id)
        pending = self._pending.get(pid)
        if pending is None:
            return None
        if not self.runtime.transport.poll_done(pending.handle):
            return None
        out = self.runtime.wait_remote_ffn(pending, clone=False)
        self._pending.pop(pid, None)
        return out

    def wait_remote_ffn(self, pending_id: int):
        pid = int(pending_id)
        pending = self._pending.get(pid)
        if pending is None:
            return None
        out = self.runtime.wait_remote_ffn(pending, clone=False)
        self._pending.pop(pid, None)
        return out


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


def _classic_send_client():
    global _CLASSIC_SEND_CLIENT, _CLASSIC_SEND_CLIENT_RUNTIME_ID
    from sglang.srt.afd.runtime import get_afd_runtime

    runtime = get_afd_runtime()
    if runtime is None:
        return None
    runtime_id = id(runtime)
    if (
        _CLASSIC_SEND_CLIENT is None
        or _CLASSIC_SEND_CLIENT_RUNTIME_ID != runtime_id
    ):
        _CLASSIC_SEND_CLIENT = _ClassicAttnClient(runtime)
        _CLASSIC_SEND_CLIENT_RUNTIME_ID = runtime_id
    return _CLASSIC_SEND_CLIENT


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
    owner: Optional[Tuple[int, int, int]] = None,
    defer_flush: bool = False,
) -> Optional[FarmHop]:
    """Non-blocking A2F. None = no credit / no mb slot."""
    owner = (0, 0, 0) if owner is None else tuple(int(x) for x in owner)
    pool_client = _pool_client()
    client = pool_client
    if client is None and (_attn_queue_enabled() or defer_flush):
        client = _classic_send_client()
    if client is not None:
        queue = None
        if _attn_queue_enabled() or defer_flush:
            bypass_first = bool(
                envs.SGLANG_AFD_ATTN_QUEUE_BYPASS_FIRST_LAYER.get()
            )
            if defer_flush or not (bypass_first and int(layer_i) == 0):
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
                owner=owner,
                defer_flush=bool(defer_flush),
            )

    hop = FarmHop(
        layer_i=layer_i,
        seq_idxs=list(seq_idxs),
        token_lo=int(token_lo),
        token_hi=int(token_hi),
        residual=residual,
        meta=meta,
        ctx_id=owner[0],
        mb_id=owner[1],
        step_id=owner[2],
    )
    if pool_client is not None:
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


def flush_hop_queues() -> int:
    """Flush all queued A2F work after a farm scheduling round."""
    if _ATTN_SEND_QUEUE is None:
        return 0
    return _ATTN_SEND_QUEUE.flush_due()


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
