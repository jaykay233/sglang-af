# SPDX-License-Identifier: Apache-2.0
"""LPU-sim: same-layer gather on FFN (one replica / GPU, fused MoE)."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from sglang.srt.afd.ffn_compute import try_compute_same_layer_group
from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.environ import envs

TransportBatch = Tuple[Any, AfdServerBatch]


@dataclass
class _QueuedLayer:
    items: Deque[Tuple[float, TransportBatch]] = field(default_factory=deque)
    tokens: int = 0
    first_ts: float = 0.0
    max_wait_us: float = 0.0

    def push(self, item: TransportBatch, *, now: float) -> None:
        if not self.items:
            self.first_ts = float(now)
        self.items.append((float(now), item))
        self.tokens += int(getattr(item[1], "num_tokens", 0))

    def pop(self, limit: int, max_tokens: int = 0) -> List[TransportBatch]:
        out: List[TransportBatch] = []
        tokens = 0
        while self.items and len(out) < limit:
            _ts, item = self.items.popleft()
            n = int(getattr(item[1], "num_tokens", 0))
            if out and max_tokens > 0 and tokens + n > max_tokens:
                self.items.appendleft((_ts, item))
                break
            out.append(item)
            tokens += n
            self.tokens -= n
            if max_tokens > 0 and tokens >= max_tokens:
                break
        if not self.items:
            self.tokens = 0
            self.first_ts = 0.0
            self.max_wait_us = 0.0
        return out


class PerLayerBatchQueue:
    """Bounded same-layer batching queue for the FFN serve loop.

    Items remain valid while their attached transport slot is in flight. The
    global hop cap therefore also bounds how many transport slots are held.
    Continuous-batching semantics mean every queued layer is ready now:
    ``target_tokens`` and ``max_hops_per_layer`` bound one compute group only.
    """

    def __init__(
        self,
        *,
        target_tokens: int = 64,
        max_wait_us: int = 300,
        max_hops_per_layer: int = 4,
        max_global_hops: int = 16,
    ) -> None:
        self.target_tokens = max(1, int(target_tokens))
        self.max_wait_us = max(0, int(max_wait_us))
        self.max_hops_per_layer = max(1, int(max_hops_per_layer))
        self.max_global_hops = max(1, int(max_global_hops))
        self._layers: Dict[int, _QueuedLayer] = {}
        self.global_hops = 0
        self.total_tokens = 0

    def push(
        self, items: Sequence[TransportBatch], *, now: Optional[float] = None
    ) -> int:
        if not items:
            return 0
        now = time.perf_counter() if now is None else float(now)
        added = 0
        for item in items:
            layer = int(getattr(item[1], "layer_id", -1))
            q = self._layers.setdefault(layer, _QueuedLayer())
            q.push(item, now=now)
            n = int(getattr(item[1], "num_tokens", 0))
            self.global_hops += 1
            self.total_tokens += n
            added += 1
        return added

    def empty(self) -> bool:
        return self.global_hops <= 0

    def time_until_due_us(self, *, now: Optional[float] = None) -> float:
        if not self._layers:
            return 0.0
        return 0.0

    def _ready_layers(self, now: float, *, force: bool) -> List[int]:
        ready: List[int] = []
        for layer, q in self._layers.items():
            if not q.items:
                continue
            waited_us = max(0.0, (now - q.first_ts) * 1e6)
            q.max_wait_us = max(q.max_wait_us, waited_us)
            ready.append(layer)
        if ready:
            # Fairness: oldest head first, then larger token groups.
            ready.sort(
                key=lambda layer: (
                    self._layers[layer].first_ts,
                    -self._layers[layer].tokens,
                )
            )
            return ready
        return []

    def pop_ready(
        self,
        *,
        now: Optional[float] = None,
        force: bool = False,
        stats: Optional[dict] = None,
    ) -> List[TransportBatch]:
        if not self._layers:
            return []
        now = time.perf_counter() if now is None else float(now)
        ready = self._ready_layers(now, force=force)
        if not ready:
            return []
        out: List[TransportBatch] = []
        for layer in ready:
            q = self._layers.get(layer)
            if q is None or not q.items:
                continue
            waited_us = max(0.0, (now - q.first_ts) * 1e6)
            group_items = q.pop(self.max_hops_per_layer, self.target_tokens)
            if not group_items:
                continue
            group_tokens = sum(
                int(getattr(batch, "num_tokens", 0)) for _tr, batch in group_items
            )
            self.global_hops -= len(group_items)
            self.total_tokens -= group_tokens
            out.extend(group_items)
            if stats is not None:
                stats["queue_groups"] = int(stats.get("queue_groups", 0)) + 1
                stats["queue_hops"] = int(stats.get("queue_hops", 0)) + len(
                    group_items
                )
                stats["queue_tokens"] = int(stats.get("queue_tokens", 0)) + group_tokens
                stats["queue_wait_us_sum"] = float(
                    stats.get("queue_wait_us_sum", 0.0)
                ) + waited_us * len(group_items)
                stats["queue_wait_us_max"] = max(
                    float(stats.get("queue_wait_us_max", 0.0)), waited_us
                )
                hist = stats.setdefault("queue_group_hist", {})
                hist[len(group_items)] = int(hist.get(len(group_items), 0)) + 1
            if not q.items:
                self._layers.pop(layer, None)
        return out


def dispatch_lpu_batches(
    compute,
    items: Sequence[TransportBatch],
    stats: Optional[dict] = None,
) -> int:
    """Group by layer_id, fused MoE when possible. Returns hops served."""
    if not items:
        return 0
    split_time = _ffn_time_split_enabled()
    by_layer: dict = defaultdict(list)
    for tr, batch in items:
        by_layer[int(getattr(batch, "layer_id", -1))].append((tr, batch))
    if stats is not None:
        stats["calls"] = int(stats.get("calls", 0)) + 1
        stats["batches"] = int(stats.get("batches", 0)) + len(items)
        stats["tokens"] = int(stats.get("tokens", 0)) + sum(
            int(getattr(batch, "num_tokens", 0)) for _, batch in items
        )
        stats["layer_groups"] = int(stats.get("layer_groups", 0)) + len(by_layer)
    served = 0
    for _lid, group in by_layer.items():
        batches = [b for _, b in group]
        group_size = len(group)
        if stats is not None:
            stats["max_group"] = max(int(stats.get("max_group", 0)), group_size)
            hist = stats.setdefault("group_hist", {})
            hist[group_size] = int(hist.get(group_size, 0)) + 1
            if group_size == 1:
                stats["singleton_groups"] = int(
                    stats.get("singleton_groups", 0)
                ) + 1
            else:
                stats["multi_groups"] = int(stats.get("multi_groups", 0)) + 1
        fused = None
        if len(group) > 1:
            fused = try_compute_same_layer_group(compute, batches)
        if fused is not None:
            if stats is not None:
                stats["fused_groups"] = int(stats.get("fused_groups", 0)) + 1
                stats["fused_batches"] = int(
                    stats.get("fused_batches", 0)
                ) + group_size
                stats["fused_tokens"] = int(stats.get("fused_tokens", 0)) + sum(
                    int(getattr(batch, "num_tokens", 0)) for batch in batches
                )
            for (tr, batch), outs in zip(group, fused):
                tr.respond(batch, outs)
                served += 1
            continue
        if stats is not None and group_size > 1:
            stats["unfused_groups"] = int(stats.get("unfused_groups", 0)) + 1
            stats["unfused_batches"] = int(
                stats.get("unfused_batches", 0)
            ) + group_size
        for tr, batch in group:
            outs = _compute_timed(compute, batch) if split_time else compute(batch)
            tr.respond(batch, outs)
            served += 1
    return served


def _ffn_time_split_enabled() -> bool:
    """Host-launch vs GPU-execution split for FFN compute (diagnostic)."""
    from sglang.srt.afd.detail_profile import profile_detail_enabled

    if not profile_detail_enabled():
        return False
    return os.environ.get("SGLANG_AFD_FARM_FFN_TIME_SPLIT", "0") in ("1", "true", "on", "yes")


def _compute_timed(compute, batch):
    """Run ``compute`` and record host wall vs launch+GPU wall separately."""
    import torch

    from sglang.srt.afd.detail_profile import record_us

    t0 = time.perf_counter()
    outs = compute(batch)
    t_wall = time.perf_counter()
    record_us("compute_wall_us", (t_wall - t0) * 1e6)
    try:
        torch.cuda.synchronize()
    except Exception:
        return outs
    record_us("compute_cuda_us", (time.perf_counter() - t0) * 1e6)
    return outs


def extra_gather(
    poll_one,
    collected: List[TransportBatch],
    *,
    gather_us: Optional[int] = None,
    gather_max: Optional[int] = None,
) -> List[TransportBatch]:
    """Spin briefly to pull more ready hops for same-layer concat."""
    if gather_us is None:
        gather_us = max(0, int(envs.SGLANG_AFD_FFN_GATHER_US.get() or 0))
    if gather_max is None:
        gather_max = max(1, int(envs.SGLANG_AFD_FFN_GATHER_MAX.get() or 1))
    if gather_us <= 0 or len(collected) >= gather_max:
        return list(collected)
    deadline = time.perf_counter() + gather_us * 1e-6
    out = list(collected)
    while time.perf_counter() < deadline and len(out) < gather_max:
        extra = poll_one()
        if extra:
            out.extend(extra)
        else:
            time.sleep(0)
    return out
