# SPDX-License-Identifier: Apache-2.0
"""Persistent decode-farm scheduler with independent context wavefronts."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.afd.farm.token_queue import LayerReadyQueues, TokenKey


def plan_context_ranges(
    n_seq: int,
    *,
    b_step: int,
    num_contexts: int,
) -> Tuple[Tuple[int, int], ...]:
    """Split a decode batch into contiguous context ranges.

    Contexts are the unit of independent layer progress. A context may be
    smaller than ``b_step``: the queue can coalesce adjacent windows when
    possible, but correctness must not depend on a minimum sequence count.
    """
    total = max(0, int(n_seq))
    if total <= 0:
        return ()
    requested = max(1, int(num_contexts))
    n_ctx = min(requested, total)
    base, rem = divmod(total, n_ctx)
    ranges: List[Tuple[int, int]] = []
    lo = 0
    for ctx_idx in range(n_ctx):
        width = base + (1 if ctx_idx < rem else 0)
        hi = lo + width
        ranges.append((lo, hi))
        lo = hi
    return tuple(ranges)


def context_stage_ready(
    *,
    next_context_idx: int,
    contexts_per_stage: int,
    context_stagger_layers: int,
    active_depths: Sequence[int],
    num_layers: int,
    max_inflight: int,
) -> bool:
    """Return whether the next context may enter the scheduler.

    Contexts in one stage enter together so their same-layer work can be
    coalesced. Stages remain staggered, which lets one stage compute while the
    previous stage's remote FFN result is still in flight.
    """
    next_idx = max(0, int(next_context_idx))
    if next_idx == 0:
        return True

    stage_size = max(1, int(contexts_per_stage))
    stagger = max(0, int(context_stagger_layers))
    if (
        stagger <= 0
        or int(num_layers) <= 1
        or int(max_inflight) <= 1
        or next_idx % stage_size != 0
    ):
        return True

    if not active_depths:
        return True
    required_depth = (next_idx // stage_size) * stagger
    return max(int(depth) for depth in active_depths) >= required_depth


@dataclass(frozen=True)
class FarmTicket:
    """A reserved layer window owned by one context/microbatch."""

    ticket_id: int
    ctx_id: int
    mb_id: int
    step_id: int
    layer_i: int
    keys: Tuple[Any, ...]
    win_index: int

    @property
    def batch_id(self) -> int:
        """Backward-compatible alias used by older call sites."""
        return self.ctx_id

    @property
    def seq_idxs(self) -> Tuple[int, ...]:
        def _seq_idx(key: Any) -> int:
            if isinstance(key, TokenKey):
                return int(key.seq_idx)
            if isinstance(key, tuple):
                return int(key[-1])
            return int(key)

        return tuple(_seq_idx(key) for key in self.keys)


@dataclass
class FarmContext:
    """One independently schedulable decode dependency graph."""

    ctx_id: int
    mb_id: int
    step_id: int
    seq_idxs: Tuple[int, ...]
    keys: Tuple[TokenKey, ...]
    expected_tokens: int
    # ``output_ready`` means the final layer's FFN result was scattered.
    # ``sampled`` means norm + lm_head + sampler produced next_token_ids.
    output_ready_tokens: int = 0
    sampled_tokens: int = 0
    # Backward-compatible alias for last-layer completion metrics.
    finished_tokens: int = 0

    def output_ready(self) -> bool:
        return self.output_ready_tokens >= self.expected_tokens

    def sampled(self) -> bool:
        return self.sampled_tokens >= self.expected_tokens

    def complete(self) -> bool:
        """Return whether the full sampling chain completed."""
        return self.sampled()

    def ffn_complete(self) -> bool:
        """Return whether all final-layer FFN outputs were scattered."""
        return self.output_ready()

    def finished(self) -> bool:
        """Backward-compatible alias for final-layer FFN completion."""
        return self.output_ready()


@dataclass
class FarmSchedulerStats:
    batches_started: int = 0
    batches_finished: int = 0
    batches_aborted: int = 0
    reservations: int = 0
    commits: int = 0
    rollbacks: int = 0
    completions: int = 0
    stale_completions: int = 0
    layer_cap_blocks: int = 0
    global_cap_blocks: int = 0
    waiting_by_layer: Dict[int, int] = field(default_factory=dict)
    running_by_layer: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "batches_started": self.batches_started,
            "batches_finished": self.batches_finished,
            "batches_aborted": self.batches_aborted,
            "reservations": self.reservations,
            "commits": self.commits,
            "rollbacks": self.rollbacks,
            "completions": self.completions,
            "stale_completions": self.stale_completions,
            "layer_cap_blocks": self.layer_cap_blocks,
            "global_cap_blocks": self.global_cap_blocks,
            "waiting_by_layer": dict(self.waiting_by_layer),
            "running_by_layer": dict(self.running_by_layer),
        }


class FarmContinuousScheduler:
    """Owns queue and dependency state across scheduling rounds.

    Contexts are independent: completing a ticket only advances the context
    that owns its ``TokenKey`` values. Multiple contexts may therefore be at
    different layers at the same time, while one context remains strictly
    layer-ordered.
    """

    def __init__(
        self,
        num_layers: int,
        *,
        b_win_k: int = 8,
        coalesce_k: int = 1,
        layer_burst: int = 0,
        sched: str = "max",
    ) -> None:
        self.num_layers = max(0, int(num_layers))
        self.queues = LayerReadyQueues(
            self.num_layers,
            b_win_k=b_win_k,
            coalesce_k=coalesce_k,
            layer_burst=layer_burst,
            sched=sched,
        )
        self.stats = FarmSchedulerStats()
        self._lock = RLock()
        self._next_ticket_id = 1
        self._ctx_seq = 0
        self._contexts: Dict[int, FarmContext] = {}
        self._context_order: List[int] = []
        self._reserved: Dict[int, FarmTicket] = {}
        self._running: Dict[int, FarmTicket] = {}
        self._last_context_id: Optional[int] = None
        self._closed_expected = 0
        self._closed_finished = 0
        self._closed_sampled = 0

    @property
    def batch_id(self) -> Optional[int]:
        if self._last_context_id in self._contexts:
            return self._last_context_id
        return self._context_order[0] if self._context_order else None

    @property
    def context_ids(self) -> Tuple[int, ...]:
        with self._lock:
            return tuple(self._context_order)

    @property
    def finished_tokens(self) -> int:
        with self._lock:
            return self._closed_finished + sum(
                c.finished_tokens for c in self._contexts.values()
            )

    @property
    def output_ready_tokens(self) -> int:
        with self._lock:
            return self._closed_finished + sum(
                c.output_ready_tokens for c in self._contexts.values()
            )

    @property
    def sampled_tokens(self) -> int:
        with self._lock:
            return self._closed_sampled + sum(
                c.sampled_tokens for c in self._contexts.values()
            )

    @property
    def expected_tokens(self) -> int:
        with self._lock:
            return self._closed_expected + sum(
                c.expected_tokens for c in self._contexts.values()
            )

    @property
    def running_count(self) -> int:
        with self._lock:
            return len(self._running) + len(self._reserved)

    def begin_batch(
        self,
        seq_idxs: Sequence[int],
        *,
        ctx_id: Optional[int] = None,
        mb_id: int = 0,
        step_id: Optional[int] = None,
    ) -> int:
        """Start one context and enqueue its layer-0 work.

        Existing contexts are intentionally left untouched.
        """
        with self._lock:
            if not self._contexts:
                self._closed_expected = 0
                self._closed_finished = 0
                self._closed_sampled = 0
            idxs = tuple(sorted({int(x) for x in seq_idxs}))
            if not idxs:
                raise ValueError("farm context requires at least one sequence")
            blocked = {
                seq_idx
                for ctx in self._contexts.values()
                if not ctx.sampled()
                for seq_idx in ctx.seq_idxs
                if seq_idx in idxs
            }
            if blocked:
                raise RuntimeError(
                    "farm context for sequence(s) "
                    f"{sorted(blocked)} must be sampled before starting the next token"
                )
            if ctx_id is None:
                self._ctx_seq += 1
                cid = self._ctx_seq
            else:
                cid = int(ctx_id)
                if cid <= 0:
                    raise ValueError("farm ctx_id must be positive")
                if cid in self._contexts:
                    raise RuntimeError(f"farm context {cid} is already active")
                self._ctx_seq = max(self._ctx_seq, cid)
            if step_id is None:
                self._ctx_seq += 1
                sid = self._ctx_seq
            else:
                sid = int(step_id)
                self._ctx_seq = max(self._ctx_seq, sid)
            keys = tuple(TokenKey(cid, int(mb_id), sid, s) for s in idxs)
            self._contexts[cid] = FarmContext(
                ctx_id=cid,
                mb_id=int(mb_id),
                step_id=sid,
                seq_idxs=idxs,
                keys=keys,
                expected_tokens=len(idxs),
            )
            self._context_order.append(cid)
            self._last_context_id = cid
            self.queues.enqueue(0, keys)
            self.stats.batches_started += 1
            self._snapshot_depths()
            return cid

    def reserve(
        self,
        b_step: int,
        *,
        max_inflight: int,
        max_inflight_per_layer: int,
        ctx_id: Optional[int] = None,
    ) -> Optional[FarmTicket]:
        """Reserve the next eligible context window or return ``None``."""
        with self._lock:
            if not self._context_order:
                return None
            if len(self._running) + len(self._reserved) >= max(1, int(max_inflight)):
                self.stats.global_cap_blocks += 1
                return None

            cap = max(1, int(max_inflight_per_layer))
            order = self._candidate_context_order(ctx_id)
            blocked_by_cap = False
            allow_legacy = ctx_id is not None or len(self._contexts) == 1
            for cid in order:
                ctx = self._contexts.get(cid)
                if ctx is None or ctx.output_ready():
                    continue

                def _can_reserve(layer: int, key: Any) -> bool:
                    nonlocal blocked_by_cap
                    if isinstance(key, TokenKey):
                        if key.ctx_id != cid:
                            return False
                    elif not allow_legacy:
                        return False
                    if self._layer_count(layer) >= cap:
                        blocked_by_cap = True
                        return False
                    return True

                picked = self.queues.pick(
                    b_step,
                    skip_layers=None,
                    key_filter=_can_reserve,
                )
                if picked is None:
                    continue
                layer_i, keys = picked
                ticket_keys = tuple(keys)
                ticket = FarmTicket(
                    ticket_id=self._next_ticket_id,
                    ctx_id=cid,
                    mb_id=ctx.mb_id,
                    step_id=ctx.step_id,
                    layer_i=int(layer_i),
                    keys=ticket_keys,
                    win_index=int(self.queues.last_win_index),
                )
                self._next_ticket_id += 1
                self._reserved[ticket.ticket_id] = ticket
                self._last_context_id = cid
                self.stats.reservations += 1
                self._snapshot_depths()
                return ticket
            if blocked_by_cap:
                self.stats.layer_cap_blocks += 1
            return None

    def commit(self, ticket_id: int) -> bool:
        """Turn a successful remote issue reservation into running work."""
        with self._lock:
            ticket = self._reserved.pop(int(ticket_id), None)
            if ticket is None or not self._ticket_is_current(ticket):
                self.stats.stale_completions += 1
                return False
            self.queues.mark_running(ticket.layer_i, ticket.keys)
            self._running[ticket.ticket_id] = ticket
            self.stats.commits += 1
            self._snapshot_depths()
            return True

    def rollback(self, ticket_id: int) -> bool:
        """Return a reservation that could not acquire transport credit."""
        with self._lock:
            ticket = self._reserved.pop(int(ticket_id), None)
            if ticket is None:
                return False
            if self._ticket_is_current(ticket):
                self.queues.enqueue(ticket.layer_i, ticket.keys)
            self.stats.rollbacks += 1
            self._snapshot_depths()
            return True

    def complete(self, ticket_id: int) -> int:
        """Advance the owning context and return tokens that reached the end."""
        with self._lock:
            ticket = self._running.pop(int(ticket_id), None)
            if ticket is None or not self._ticket_is_current(ticket):
                self.stats.stale_completions += 1
                return 0
            self.queues.complete_running(ticket.layer_i, ticket.keys)
            ctx = self._contexts[ticket.ctx_id]
            nxt = ticket.layer_i + 1
            if nxt >= self.num_layers:
                ctx.output_ready_tokens += len(ticket.keys)
                ctx.finished_tokens += len(ticket.keys)
                finished = len(ticket.keys)
            else:
                self.queues.enqueue(nxt, ticket.keys)
                finished = 0
            self.stats.completions += 1
            self._snapshot_depths()
            return finished

    def mark_context_sampled(
        self,
        ctx_id: int,
        token_count: Optional[int] = None,
    ) -> int:
        """Mark a context's output-ready rows as sampled.

        Returns the number of rows newly marked. The context must have reached
        the output-ready state for those rows first; sampling before the final
        layer exists is a dependency violation, not an optimistic state
        transition. The final layer may complete in multiple windows, so this
        only marks rows that are already output-ready.
        """
        with self._lock:
            cid = int(ctx_id)
            ctx = self._contexts.get(cid)
            if ctx is None:
                return 0
            ready_unmarked = ctx.output_ready_tokens - ctx.sampled_tokens
            if ready_unmarked <= 0:
                return 0
            amount = ready_unmarked if token_count is None else int(token_count)
            if amount < 0 or amount > ready_unmarked:
                raise RuntimeError(
                    f"farm context {cid} cannot mark {amount} rows sampled; "
                    f"only {ready_unmarked} output-ready rows are unmarked"
                )
            ctx.sampled_tokens += amount
            self._snapshot_depths()
            return amount

    def finish_context(self, ctx_id: int) -> bool:
        """Close one context after all of its keys were sampled."""
        with self._lock:
            cid = int(ctx_id)
            ctx = self._contexts.get(cid)
            if ctx is None or not ctx.sampled():
                return False
            if any(t.ctx_id == cid for t in self._reserved.values()):
                return False
            if any(t.ctx_id == cid for t in self._running.values()):
                return False
            if self._queue_has_context(cid):
                return False
            del self._contexts[cid]
            self._context_order = [x for x in self._context_order if x != cid]
            self._closed_expected += ctx.expected_tokens
            self._closed_finished += ctx.finished_tokens
            self._closed_sampled += ctx.sampled_tokens
            self._last_context_id = self._context_order[0] if self._context_order else None
            self.stats.batches_finished += 1
            self._snapshot_depths()
            return True

    def finish_batch(self, ctx_id: Optional[int] = None) -> bool:
        """Backward-compatible close helper.

        With one active context this behaves like the old method. With multiple
        contexts, passing ``ctx_id`` is the precise form; omitting it succeeds
        only when every active context is ready to close.
        """
        with self._lock:
            if ctx_id is not None:
                return self.finish_context(int(ctx_id))
            ids = list(self._context_order)
            if not ids:
                return False
            if not all(
                self._contexts[cid].complete()
                and not any(t.ctx_id == cid for t in self._reserved.values())
                and not any(t.ctx_id == cid for t in self._running.values())
                and not self._queue_has_context(cid)
                for cid in ids
            ):
                return False
            return all(self.finish_context(cid) for cid in ids)

    def abort_batch(self, ctx_id: Optional[int] = None) -> None:
        """Drop host-side state after an error or a stale forward."""
        with self._lock:
            if ctx_id is None:
                count = len(self._contexts)
                self._contexts.clear()
                self._context_order.clear()
                self._reserved.clear()
                self._running.clear()
                self.queues.ready = [[] for _ in range(self.num_layers)]
                self.queues.running = [[] for _ in range(self.num_layers)]
                self.queues.running_hops = [0] * self.num_layers
                self.queues.sticky_layer = -1
                self.queues.sticky_left = 0
                self._last_context_id = None
                self._closed_expected = 0
                self._closed_finished = 0
                self._closed_sampled = 0
                if count:
                    self.stats.batches_aborted += count
                self._snapshot_depths()
                return

            cid = int(ctx_id)
            if cid in self._contexts:
                self.stats.batches_aborted += 1
            self._contexts.pop(cid, None)
            self._context_order = [x for x in self._context_order if x != cid]
            self._reserved = {
                tid: ticket
                for tid, ticket in self._reserved.items()
                if ticket.ctx_id != cid
            }
            running_to_remove = [
                ticket
                for ticket in self._running.values()
                if ticket.ctx_id == cid
            ]
            self._running = {
                tid: ticket
                for tid, ticket in self._running.items()
                if ticket.ctx_id != cid
            }
            for ticket in running_to_remove:
                self.queues.running_hops[ticket.layer_i] = max(
                    0, self.queues.running_hops[ticket.layer_i] - 1
                )
            for layer_i in range(self.num_layers):
                self.queues.ready[layer_i] = [
                    key
                    for key in self.queues.ready[layer_i]
                    if not (isinstance(key, TokenKey) and key.ctx_id == cid)
                ]
                self.queues.running[layer_i] = [
                    key
                    for key in self.queues.running[layer_i]
                    if not (isinstance(key, TokenKey) and key.ctx_id == cid)
                ]
            self._last_context_id = self._context_order[0] if self._context_order else None
            self._snapshot_depths()

    def context_complete(self, ctx_id: int) -> bool:
        with self._lock:
            ctx = self._contexts.get(int(ctx_id))
            return bool(ctx is not None and ctx.complete())

    def batch_complete(self) -> bool:
        with self._lock:
            return bool(self._contexts) and all(
                ctx.complete() for ctx in self._contexts.values()
            )

    def idle(self) -> bool:
        with self._lock:
            return (
                not self._contexts
                and not self._reserved
                and not self._running
                and self.queues.empty()
            )

    def waiting_empty(self, ctx_id: Optional[int] = None) -> bool:
        with self._lock:
            if ctx_id is None:
                return self.queues.waiting_empty()
            cid = int(ctx_id)
            return not self._queue_has_context(cid, include_running=False)

    def occupancy(self) -> List[int]:
        with self._lock:
            return self.queues.occupancy()

    def running_hops(self, layer_i: int) -> int:
        with self._lock:
            return sum(
                1
                for ticket in (*self._reserved.values(), *self._running.values())
                if ticket.layer_i == int(layer_i)
            )

    def context_deepest_layer(self, ctx_id: int) -> int:
        """Return the deepest layer currently owned by one context.

        Ready, reserved, and running work are all included. ``-1`` means the
        context has no layer work left at this instant. An output-ready
        context also returns ``-1`` because its remaining work is sampling,
        not another transformer layer.
        """
        cid = int(ctx_id)
        with self._lock:
            if cid not in self._contexts:
                return -1
            for layer_i in range(self.num_layers - 1, -1, -1):
                if any(
                    isinstance(key, TokenKey) and key.ctx_id == cid
                    for key in self.queues.ready[layer_i]
                ) or any(
                    isinstance(key, TokenKey) and key.ctx_id == cid
                    for key in self.queues.running[layer_i]
                ):
                    return layer_i
            for ticket in (*self._reserved.values(), *self._running.values()):
                if ticket.ctx_id == cid:
                    return ticket.layer_i
            return -1

    def context_output_ready(self, ctx_id: int) -> bool:
        with self._lock:
            ctx = self._contexts.get(int(ctx_id))
            return bool(ctx is not None and ctx.output_ready())

    def _candidate_context_order(self, ctx_id: Optional[int]) -> List[int]:
        if ctx_id is not None:
            cid = int(ctx_id)
            return [cid] if cid in self._contexts else []
        if not self._context_order:
            return []
        start = 0
        if self._last_context_id in self._context_order:
            start = (self._context_order.index(self._last_context_id) + 1) % len(
                self._context_order
            )
        return self._context_order[start:] + self._context_order[:start]

    def _layer_count(self, layer_i: int) -> int:
        return sum(
            1
            for ticket in (*self._reserved.values(), *self._running.values())
            if ticket.layer_i == int(layer_i)
        )

    def _queue_has_context(
        self, ctx_id: int, *, include_running: bool = True
    ) -> bool:
        for layer_i in range(self.num_layers):
            for key in self.queues.ready[layer_i]:
                if isinstance(key, TokenKey) and key.ctx_id == int(ctx_id):
                    return True
            if include_running:
                for key in self.queues.running[layer_i]:
                    if isinstance(key, TokenKey) and key.ctx_id == int(ctx_id):
                        return True
        return False

    def _ticket_is_current(self, ticket: FarmTicket) -> bool:
        ctx = self._contexts.get(ticket.ctx_id)
        return (
            ctx is not None
            and ctx.mb_id == ticket.mb_id
            and ctx.step_id == ticket.step_id
        )

    def _snapshot_depths(self) -> None:
        self.stats.waiting_by_layer = {
            i: len(q) for i, q in enumerate(self.queues.ready) if q
        }
        self.stats.running_by_layer = {
            i: len(q) for i, q in enumerate(self.queues.running) if q
        }


_SCHEDULERS: Dict[Tuple[str, int], FarmContinuousScheduler] = {}
_SCHEDULER_LOCK = RLock()


def get_continuous_scheduler(
    num_layers: int,
    *,
    device_key: str = "cuda",
    b_win_k: int = 8,
    coalesce_k: int = 1,
    layer_burst: int = 0,
    sched: str = "max",
) -> FarmContinuousScheduler:
    """Return the process-local scheduler for one model/device shape."""
    key = (str(device_key), int(num_layers))
    with _SCHEDULER_LOCK:
        scheduler = _SCHEDULERS.get(key)
        if scheduler is None:
            scheduler = FarmContinuousScheduler(
                num_layers,
                b_win_k=b_win_k,
                coalesce_k=coalesce_k,
                layer_burst=layer_burst,
                sched=sched,
            )
            _SCHEDULERS[key] = scheduler
        scheduler.queues.b_win_k = max(1, int(b_win_k))
        scheduler.queues.coalesce_k = max(1, int(coalesce_k))
        scheduler.queues.layer_burst = max(0, int(layer_burst))
        scheduler.queues.sched = str(sched or "max").strip().lower()
        if scheduler.queues.sched not in ("max", "oldest", "deepest"):
            scheduler.queues.sched = "max"
        return scheduler


def reset_continuous_schedulers() -> None:
    with _SCHEDULER_LOCK:
        _SCHEDULERS.clear()
