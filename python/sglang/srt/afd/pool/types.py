# SPDX-License-Identifier: Apache-2.0
"""AfPool task / topology types (MxN Attn↔FFN work pool)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class AfTaskId:
    """Identifies one A2F hop in the pool."""

    req_id: int
    layer_id: int
    attn_rank: int
    seq: int = 0  # monotonic per attn for debugging

    def key(self) -> Tuple[int, int, int, int]:
        return (self.attn_rank, self.req_id, self.layer_id, self.seq)


@dataclass
class A2FWorkItem:
    task: AfTaskId
    num_tokens: int
    mb_id: int = 0
    ffn_rank: int = -1  # filled by router
    t_submit: float = 0.0


@dataclass
class F2ACompletion:
    task: AfTaskId
    ffn_rank: int
    mb_id: int
    t_done: float = 0.0
    ok: bool = True
    error: Optional[str] = None


@dataclass
class PoolTopology:
    """Na x Nf cuda_ipc endpoint matrix."""

    num_attn: int
    num_ffn: int
    endpoint_dir: str
    max_inflight_per_ffn: int = 4
    route: str = "least_inflight"

    def endpoint(self, attn_rank: int, ffn_rank: int) -> str:
        return f"{self.endpoint_dir.rstrip('/')}/a{int(attn_rank)}_f{int(ffn_rank)}.sock"

    def endpoints_for_attn(self, attn_rank: int) -> Tuple[str, ...]:
        return tuple(self.endpoint(attn_rank, j) for j in range(self.num_ffn))

    def endpoints_for_ffn(self, ffn_rank: int) -> Tuple[str, ...]:
        return tuple(self.endpoint(i, ffn_rank) for i in range(self.num_attn))


@dataclass
class PoolStats:
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tokens_completed: int = 0
    route_picks: list = field(default_factory=list)  # ffn_rank per pick
    ffn_busy_s: list = field(default_factory=list)
    wall_start: float = 0.0
    wall_end: float = 0.0

    def ensure_ffn(self, num_ffn: int) -> None:
        while len(self.ffn_busy_s) < num_ffn:
            self.ffn_busy_s.append(0.0)

    @property
    def wall_s(self) -> float:
        if self.wall_end <= self.wall_start:
            return 0.0
        return self.wall_end - self.wall_start

    @property
    def tok_s(self) -> float:
        w = self.wall_s
        if w <= 0:
            return 0.0
        return float(self.tokens_completed) / w

    def ffn_busy_frac(self, ffn_rank: int) -> float:
        w = self.wall_s
        if w <= 0 or ffn_rank >= len(self.ffn_busy_s):
            return 0.0
        return min(1.0, float(self.ffn_busy_s[ffn_rank]) / w)

    def mean_ffn_busy_frac(self) -> float:
        if not self.ffn_busy_s:
            return 0.0
        w = self.wall_s
        if w <= 0:
            return 0.0
        return sum(min(1.0, b / w) for b in self.ffn_busy_s) / len(self.ffn_busy_s)
