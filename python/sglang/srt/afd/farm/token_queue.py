# SPDX-License-Identifier: Apache-2.0
"""Per-layer waiting/running queues + sticky B_win pick (decode farm P0/P2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True, order=True)
class TokenKey:
    """Stable identity for one decode token inside the farm.

    ``ctx_id`` is unique for one active dependency context. ``mb_id`` and
    ``step_id`` are carried for tracing and for keeping contexts from different
    decode forwards distinct even if a scheduler instance is process-local.
    Only keys with the same first three fields may be coalesced.
    """

    ctx_id: int
    mb_id: int
    step_id: int
    seq_idx: int

    @property
    def owner(self) -> Tuple[int, int, int]:
        return (int(self.ctx_id), int(self.mb_id), int(self.step_id))


def _queue_order(value) -> Tuple[int, int, int, int]:
    if isinstance(value, TokenKey):
        return (
            int(value.ctx_id),
            int(value.mb_id),
            int(value.step_id),
            int(value.seq_idx),
        )
    if isinstance(value, tuple):
        return (int(value[0]), 0, 0, int(value[1]))
    return (0, 0, 0, int(value))


def _same_run(left, right) -> bool:
    if isinstance(left, TokenKey) or isinstance(right, TokenKey):
        return (
            isinstance(left, TokenKey)
            and isinstance(right, TokenKey)
            and left.owner == right.owner
            and int(left.seq_idx) + 1 == int(right.seq_idx)
        )
    if isinstance(left, tuple) or isinstance(right, tuple):
        return (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and int(left[0]) == int(right[0])
            and int(left[1]) + 1 == int(right[1])
        )
    return int(left) + 1 == int(right)


def take_contiguous_run(sorted_idxs: Sequence, b_step: int) -> List:
    """Longest contiguous prefix of ``sorted_idxs``, capped at ``b_step``."""
    if not sorted_idxs or b_step <= 0:
        return []
    run = [sorted_idxs[0]]
    for x in sorted_idxs[1:]:
        if not _same_run(run[-1], x):
            break
        run.append(x)
        if len(run) >= b_step:
            break
    return run[:b_step]


def _filtered_run(
    sorted_idxs: Sequence,
    b_step: int,
    key_filter: Optional[Callable[[int, Any], bool]],
    layer: int,
) -> List:
    if key_filter is None:
        return take_contiguous_run(sorted_idxs, b_step)
    filtered = [x for x in sorted_idxs if key_filter(layer, x)]
    return take_contiguous_run(filtered, b_step)


@dataclass
class BWinStats:
    """P2 sticky-window + coalesce accounting."""

    layer_switches: int = 0
    picks: int = 0
    completed_win_lens: List[int] = field(default_factory=list)
    cur_win_len: int = 0
    last_layer: int = -1
    last_win_index: int = 0
    # Coalesce (true stock-kernel amortize): one launch may cover several B_steps.
    launch_tokens: int = 0
    micro_windows: int = 0  # sum of ceil(tokens / b_step) across picks
    b_step: int = 1

    def note_pick(self, layer: int, *, n_tokens: int = 0, b_step: int = 1) -> int:
        """Record a pick; return 0-based index within the sticky run."""
        self.picks += 1
        bs = max(1, int(b_step))
        self.b_step = bs
        nt = max(0, int(n_tokens))
        self.launch_tokens += nt
        micros = max(1, (nt + bs - 1) // bs) if nt > 0 else 1
        self.micro_windows += micros
        if int(layer) != self.last_layer:
            if self.cur_win_len > 0:
                self.completed_win_lens.append(self.cur_win_len)
            self.layer_switches += 1
            self.cur_win_len = micros
            self.last_layer = int(layer)
            self.last_win_index = 0
            return 0
        self.last_win_index = self.cur_win_len
        self.cur_win_len += micros
        return self.last_win_index

    def finalize(self) -> None:
        if self.cur_win_len > 0:
            self.completed_win_lens.append(self.cur_win_len)
            self.cur_win_len = 0

    @property
    def mean_win_len(self) -> float:
        xs = list(self.completed_win_lens)
        if self.cur_win_len > 0:
            xs.append(self.cur_win_len)
        if not xs:
            return 0.0
        return float(sum(xs)) / float(len(xs))

    @property
    def mean_tokens_per_launch(self) -> float:
        if self.picks <= 0:
            return 0.0
        return float(self.launch_tokens) / float(self.picks)

    @property
    def amortize_factor(self) -> float:
        """Mean launch tokens / B_step (1.0 = no coalesce)."""
        bs = max(1, int(self.b_step))
        return self.mean_tokens_per_launch / float(bs)

    def as_dict(self) -> Dict[str, object]:
        self.finalize()
        return {
            "layer_switches": self.layer_switches,
            "picks": self.picks,
            "mean_win_len": self.mean_win_len,
            "completed_wins": len(self.completed_win_lens),
            "completed_win_lens": list(self.completed_win_lens),
            "launch_tokens": self.launch_tokens,
            "micro_windows": self.micro_windows,
            "mean_tokens_per_launch": self.mean_tokens_per_launch,
            "amortize_factor": self.amortize_factor,
            "b_step": self.b_step,
        }


@dataclass
class LayerReadyQueues:
    """Sorted unique seq/token indices ready for Attn at each layer.

    ``ready`` is the waiting queue: work that has reached the layer but has
    not acquired an A2F slot. ``running`` is work whose Attn pre-FFN has been
    issued and is waiting for F2A. A completion moves directly from ``running``
    to the next layer's ``ready`` queue.

    ``pick`` prefers a sticky layer for ``b_win_k`` windows so FFN can gather
    same-``layer_id`` hops (LPU-sim). Falls back to the layer with the most
    ready work (not always the lowest layer — that would drain wavefronts).
    """

    num_layers: int
    b_win_k: int = 8
    coalesce_k: int = 1
    layer_burst: int = 0
    sched: str = "max"
    ready: List[List[Any]] = field(init=False)
    running: List[List[Any]] = field(init=False)
    running_hops: List[int] = field(init=False)
    sticky_layer: int = -1
    sticky_left: int = 0
    picks: int = 0
    last_win_index: int = 0
    last_micro_windows: int = 1
    bwin: BWinStats = field(default_factory=BWinStats)

    def __post_init__(self) -> None:
        n = max(0, int(self.num_layers))
        self.ready = [[] for _ in range(n)]
        self.running = [[] for _ in range(n)]
        self.running_hops = [0] * n
        self.b_win_k = max(1, int(self.b_win_k))
        self.coalesce_k = max(1, int(self.coalesce_k))
        self.layer_burst = max(0, int(self.layer_burst))
        self.sched = str(self.sched or "max").strip().lower()
        if self.sched not in ("max", "oldest", "deepest"):
            self.sched = "max"

    def empty(self) -> bool:
        return self.running_total() == 0 and all(not q for q in self.ready)

    def waiting_empty(self) -> bool:
        return all(not q for q in self.ready)

    def depth(self, layer: int) -> int:
        if layer < 0 or layer >= len(self.ready):
            return 0
        return len(self.ready[layer])

    def running_depth(self, layer: int) -> int:
        if layer < 0 or layer >= len(self.running):
            return 0
        return len(self.running[layer])

    def running_hop_depth(self, layer: int) -> int:
        """Number of in-flight hops/windows at ``layer``."""
        if layer < 0 or layer >= len(self.running_hops):
            return 0
        return int(self.running_hops[layer])

    def running_total(self) -> int:
        return sum(len(q) for q in self.running)

    def occupancy(self) -> List[int]:
        return [
            len(self.ready[i]) + len(self.running[i])
            for i in range(len(self.ready))
        ]

    def mark_running(self, layer: int, idxs: Sequence[int]) -> None:
        """Move ``idxs`` from waiting to running for ``layer``."""
        if layer < 0 or layer >= len(self.running) or not idxs:
            return
        run = self.running[layer]
        seen = set(run)
        for x in idxs:
            if x not in seen:
                run.append(x)
                seen.add(x)
        run.sort(key=_queue_order)
        self.running_hops[layer] += 1
        taken = set(idxs)
        self.ready[layer] = [x for x in self.ready[layer] if x not in taken]

    def complete_running(self, layer: int, idxs: Sequence[int]) -> None:
        """Remove completed work from the running queue for ``layer``."""
        if layer < 0 or layer >= len(self.running) or not idxs:
            return
        done = set(idxs)
        self.running[layer] = [x for x in self.running[layer] if x not in done]
        self.running_hops[layer] = max(0, self.running_hops[layer] - 1)

    def requeue_running(self, layer: int, idxs: Sequence[int]) -> None:
        """Return an issue failure from running to waiting."""
        self.complete_running(layer, idxs)
        self.enqueue(layer, idxs)

    def enqueue(self, layer: int, idxs: Sequence[int]) -> None:
        if layer < 0 or layer >= len(self.ready) or not idxs:
            return
        q = self.ready[layer]
        seen = set(q)
        for x in idxs:
            if x not in seen:
                q.append(x)
                seen.add(x)
        q.sort(key=_queue_order)

    def pick(
        self,
        b_step: int,
        coalesce_k: Optional[int] = None,
        *,
        skip_layers: Optional[Set[int]] = None,
        key_filter: Optional[Callable[[int, Any], bool]] = None,
    ) -> Optional[Tuple[int, List[int]]]:
        """Return ``(layer, contiguous_indices)`` or None if idle.

        When ``coalesce_k > 1``, dequeue up to ``coalesce_k * b_step`` contiguous
        tokens so one Attn launch amortizes weights (larger M). Snaps down to a
        multiple of ``b_step`` when that still leaves ≥ one full window.

        ``skip_layers`` lets the pipeline cap how many windows a single layer
        may keep in flight. A blocked sticky layer is abandoned immediately so
        another ready layer can advance the dependency wavefront.

        Sets ``last_win_index`` (0 = first micro-window of a sticky B_win run).
        """
        b_step = max(1, int(b_step))
        ck = max(1, int(self.coalesce_k if coalesce_k is None else coalesce_k))
        max_tok = b_step * ck
        if self.layer_burst > 0:
            max_tok = min(max_tok, self.layer_burst)
        skip = {int(x) for x in (skip_layers or ())}
        li = self._choose_layer(skip_layers=skip, key_filter=key_filter)
        if li is None:
            return None
        run = _filtered_run(self.ready[li], max_tok, key_filter, li)
        if not run:
            return None
        if ck > 1 and len(run) > b_step:
            snapped = (len(run) // b_step) * b_step
            if snapped >= b_step:
                run = run[:snapped]
        taken = set(run)
        self.ready[li] = [x for x in self.ready[li] if x not in taken]
        micros = max(1, (len(run) + b_step - 1) // b_step)
        self.last_micro_windows = micros
        self.sticky_layer = li
        self.sticky_left = max(0, self.sticky_left - micros)
        self.picks += 1
        self.last_win_index = self.bwin.note_pick(
            li, n_tokens=len(run), b_step=b_step
        )
        return li, run

    def _choose_layer(
        self,
        *,
        skip_layers: Optional[Set[int]] = None,
        key_filter: Optional[Callable[[int, Any], bool]] = None,
    ) -> Optional[int]:
        skip = {int(x) for x in (skip_layers or ())}
        def count_ready(layer: int) -> int:
            if key_filter is None:
                return len(self.ready[layer])
            return sum(1 for x in self.ready[layer] if key_filter(layer, x))

        candidates = [
            (i, count_ready(i))
            for i, q in enumerate(self.ready)
            if q and count_ready(i) > 0 and i not in skip
        ]
        if not candidates:
            if self.sticky_layer in skip:
                self.sticky_left = 0
            elif not self.empty():
                # Every ready layer is temporarily capped. Keep the sticky
                # preference for after one of those layers completes.
                return None
            else:
                self.sticky_layer = -1
            self.sticky_left = 0
            return None
        if self.sched == "deepest":
            # Drive the dependency wavefront instead of draining one layer at a
            # time. This exposes cross-layer overlap for the async FFN hops.
            best = max(candidates, key=lambda x: x[0])[0]
            self.sticky_layer = best
            self.sticky_left = self.b_win_k
            return best
        if self.sched == "oldest":
            best = min(candidates, key=lambda x: x[0])[0]
            self.sticky_layer = best
            self.sticky_left = self.b_win_k
            return best
        if (
            self.sticky_left > 0
            and 0 <= self.sticky_layer < len(self.ready)
            and self.ready[self.sticky_layer]
            and self.sticky_layer not in skip
            and count_ready(self.sticky_layer) > 0
        ):
            return self.sticky_layer
        # New B_win: if sticky quota expired and another layer has work, switch
        # away from the previous sticky layer (P2 orifice semantics).
        if len(candidates) > 1 and self.sticky_layer >= 0:
            others = [(i, n) for i, n in candidates if i != self.sticky_layer]
            if others:
                candidates = others
        best = max(candidates, key=lambda x: x[1])[0]
        self.sticky_layer = best
        self.sticky_left = self.b_win_k
        return best


@dataclass
class FarmSimStats:
    ticks: int = 0
    finished: int = 0
    windows: int = 0
    mean_layers_busy: float = 0.0
    peak_layers_busy: int = 0
    occupancy_sum: List[int] = field(default_factory=list)
    samples: int = 0
    layer_switches: int = 0
    mean_win_len: float = 0.0
    b_win_k: int = 0
    coalesce_k: int = 1
    mean_tokens_per_launch: float = 0.0
    amortize_factor: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "ticks": self.ticks,
            "finished": self.finished,
            "windows": self.windows,
            "mean_layers_busy": self.mean_layers_busy,
            "peak_layers_busy": self.peak_layers_busy,
            "samples": self.samples,
            "layer_switches": self.layer_switches,
            "mean_win_len": self.mean_win_len,
            "b_win_k": self.b_win_k,
            "coalesce_k": self.coalesce_k,
            "mean_tokens_per_launch": self.mean_tokens_per_launch,
            "amortize_factor": self.amortize_factor,
        }


def simulate_bwin_kpi(
    *,
    n_layers: int = 8,
    n_per_layer: int = 64,
    b_step: int = 8,
    b_win_k: int = 8,
    coalesce_k: int = 1,
    max_picks: int = 64,
) -> Dict[str, object]:
    """P2 KPI: sticky K + optional coalesce amortize factor."""
    queues = LayerReadyQueues(n_layers, b_win_k=b_win_k, coalesce_k=coalesce_k)
    base = 0
    for L in range(n_layers):
        queues.enqueue(L, range(base, base + n_per_layer))
        base += n_per_layer
    for _ in range(max_picks):
        if queues.pick(b_step) is None:
            break
    out = queues.bwin.as_dict()
    out["b_win_k"] = int(b_win_k)
    out["b_step"] = int(b_step)
    out["coalesce_k"] = int(coalesce_k)
    out["n_layers"] = int(n_layers)
    return out


def simulate_farm_occupancy(
    *,
    n_tokens: int,
    n_layers: int,
    b_step: int,
    b_win_k: int = 8,
    coalesce_k: int = 1,
    max_inflight: int = 4,
    attn_ticks: int = 1,
    ffn_ticks: int = 3,
    max_ticks: int = 100000,
) -> FarmSimStats:
    """Discrete-event occupancy sim (no GPU). Tokens start at layer 0.

    A window spends ``attn_ticks`` then ``ffn_ticks`` in flight at that layer,
    then moves to ``layer+1``. Occupancy counts ready + in-flight tokens/layer.
    """
    n_tokens = max(0, int(n_tokens))
    n_layers = max(1, int(n_layers))
    queues = LayerReadyQueues(n_layers, b_win_k=b_win_k, coalesce_k=coalesce_k)
    queues.enqueue(0, range(n_tokens))
    inflight: List[Tuple[int, int, List[int]]] = []  # (done_tick, layer, idxs)
    finished = 0
    tick = 0
    busy_sum = 0
    peak = 0
    samples = 0
    windows = 0
    occ_sum = [0] * n_layers
    attn_free_at = 0

    def _complete_ready(now: int) -> None:
        nonlocal finished, inflight
        still: List[Tuple[int, int, List[int]]] = []
        for done_t, li, idxs in inflight:
            if done_t > now:
                still.append((done_t, li, idxs))
                continue
            nxt = li + 1
            if nxt >= n_layers:
                finished += len(idxs)
            else:
                queues.enqueue(nxt, idxs)
        inflight = still

    while finished < n_tokens and tick < max_ticks:
        _complete_ready(tick)
        # One Attn GPU: at most one B_step window per attn_ticks.
        if tick >= attn_free_at and len(inflight) < max_inflight:
            picked = queues.pick(b_step)
            if picked is not None:
                li, idxs = picked
                done = tick + max(0, int(attn_ticks)) + max(0, int(ffn_ticks))
                inflight.append((done, li, idxs))
                windows += 1
                attn_free_at = tick + max(1, int(attn_ticks))
        occ = queues.occupancy()
        inflight_occ = [0] * n_layers
        for _, li, idxs in inflight:
            inflight_occ[li] += len(idxs)
        layers_busy = 0
        for i in range(n_layers):
            n = occ[i] + inflight_occ[i]
            occ_sum[i] += n
            if n > 0:
                layers_busy += 1
        busy_sum += layers_busy
        peak = max(peak, layers_busy)
        samples += 1
        if not inflight and queues.empty():
            break
        tick += 1

    bwin = queues.bwin.as_dict()
    stats = FarmSimStats(
        ticks=tick,
        finished=finished,
        windows=windows,
        mean_layers_busy=(busy_sum / samples) if samples else 0.0,
        peak_layers_busy=peak,
        occupancy_sum=occ_sum,
        samples=samples,
        layer_switches=int(bwin["layer_switches"]),
        mean_win_len=float(bwin["mean_win_len"]),
        b_win_k=int(b_win_k),
        coalesce_k=int(coalesce_k),
        mean_tokens_per_launch=float(bwin["mean_tokens_per_launch"]),
        amortize_factor=float(bwin["amortize_factor"]),
    )
    return stats
