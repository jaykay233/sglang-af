# SPDX-License-Identifier: Apache-2.0
"""AfScheduler: route + credit + completion stats for MxN pool."""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Tuple

from sglang.srt.afd.pool.credit import FfnCreditWindow
from sglang.srt.afd.pool.router import AfRouter
from sglang.srt.afd.pool.types import (
    A2FWorkItem,
    AfTaskId,
    F2ACompletion,
    PoolStats,
    PoolTopology,
)

logger = logging.getLogger(__name__)


class AfScheduler:
    """Process-local scheduler (Attn side owns routing; FFN only serves)."""

    def __init__(self, topo: PoolTopology, *, attn_rank: int = 0):
        self.topo = topo
        self.attn_rank = int(attn_rank)
        self.credit = FfnCreditWindow(topo.num_ffn, topo.max_inflight_per_ffn)
        self.router = AfRouter(topo.num_ffn, topo.route)
        self.stats = PoolStats()
        self.stats.ensure_ffn(topo.num_ffn)
        self._seq = 0
        self._lock = threading.Lock()
        self._inflight: Dict[Tuple[int, int, int, int], A2FWorkItem] = {}
        self._ffn_compute_start: Dict[int, float] = {}

    def begin_wall(self) -> None:
        self.stats.wall_start = time.perf_counter()

    def end_wall(self) -> None:
        self.stats.wall_end = time.perf_counter()

    def next_task(self, req_id: int, layer_id: int, num_tokens: int) -> A2FWorkItem:
        with self._lock:
            self._seq += 1
            seq = self._seq
        task = AfTaskId(
            req_id=int(req_id),
            layer_id=int(layer_id),
            attn_rank=self.attn_rank,
            seq=seq,
        )
        item = A2FWorkItem(task=task, num_tokens=int(num_tokens), t_submit=time.perf_counter())
        return item

    def assign_ffn(self, item: A2FWorkItem, *, timeout_s: float = 30.0) -> int:
        """Pick FFN, block on credit, record pick."""
        while True:
            ffn = self.router.pick(self.credit)
            if self.credit.try_acquire(ffn):
                break
            # Wait specifically for this rank's credit (or any).
            if not self.credit.acquire(ffn, timeout_s=min(0.05, timeout_s)):
                # Retry pick in case another FFN freed up.
                if timeout_s <= 0:
                    raise TimeoutError("AfScheduler credit acquire timed out")
                timeout_s -= 0.05
                continue
            break
        item.ffn_rank = int(ffn)
        with self._lock:
            self.stats.tasks_submitted += 1
            self.stats.route_picks.append(item.ffn_rank)
            self._inflight[item.task.key()] = item
        return item.ffn_rank

    def try_assign_ffn(self, item: A2FWorkItem) -> Optional[int]:
        """Non-blocking credit acquire (farm smooth send). None if all full."""
        order = sorted(
            range(self.topo.num_ffn), key=lambda j: self.credit.inflight(j)
        )
        for ffn in order:
            if not self.credit.try_acquire(ffn):
                continue
            item.ffn_rank = int(ffn)
            with self._lock:
                self.stats.tasks_submitted += 1
                self.stats.route_picks.append(item.ffn_rank)
                self._inflight[item.task.key()] = item
            return item.ffn_rank
        return None

    def abort_assign(self, item: A2FWorkItem) -> None:
        """Release credit without counting a completion (issue failed)."""
        ffn = int(item.ffn_rank)
        if ffn >= 0:
            self.credit.release(ffn)
        with self._lock:
            self._inflight.pop(item.task.key(), None)
            if self.stats.tasks_submitted > 0:
                self.stats.tasks_submitted -= 1
        item.ffn_rank = -1

    def note_ffn_compute_start(self, ffn_rank: int) -> None:
        self._ffn_compute_start[int(ffn_rank)] = time.perf_counter()

    def complete(self, item: A2FWorkItem, *, compute_s: float = 0.0) -> F2ACompletion:
        ffn = int(item.ffn_rank)
        self.credit.release(ffn)
        with self._lock:
            self._inflight.pop(item.task.key(), None)
            self.stats.tasks_completed += 1
            self.stats.tokens_completed += max(0, int(item.num_tokens))
            if ffn < len(self.stats.ffn_busy_s) and compute_s > 0:
                self.stats.ffn_busy_s[ffn] += float(compute_s)
        return F2ACompletion(
            task=item.task,
            ffn_rank=ffn,
            mb_id=item.mb_id,
            t_done=time.perf_counter(),
        )

    def summary(self) -> str:
        s = self.stats
        util = ", ".join(
            f"f{j}={s.ffn_busy_frac(j):.2f}" for j in range(self.topo.num_ffn)
        )
        return (
            f"attn={self.attn_rank} submitted={s.tasks_submitted} "
            f"done={s.tasks_completed} tokens={s.tokens_completed} "
            f"wall_s={s.wall_s:.3f} tok_s={s.tok_s:.1f} ffn_rtt_frac=[{util}] "
            f"inflight={self.credit.snapshot()}"
        )
