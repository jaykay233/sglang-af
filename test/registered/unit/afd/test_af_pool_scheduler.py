# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AfPool credit + router (no GPU)."""

from __future__ import annotations

from sglang.srt.afd.pool.credit import FfnCreditWindow
from sglang.srt.afd.pool.router import AfRouter
from sglang.srt.afd.pool.scheduler import AfScheduler
from sglang.srt.afd.pool.types import PoolTopology


def test_credit_acquire_release():
    c = FfnCreditWindow(2, max_inflight=2)
    assert c.try_acquire(0)
    assert c.try_acquire(0)
    assert not c.try_acquire(0)
    c.release(0)
    assert c.try_acquire(0)
    assert c.inflight(0) == 2


def test_router_least_inflight():
    c = FfnCreditWindow(2, max_inflight=4)
    r = AfRouter(2, "least_inflight")
    assert r.pick(c) == 0
    c.try_acquire(0)
    assert r.pick(c) == 1


def test_scheduler_assign():
    topo = PoolTopology(
        num_attn=1,
        num_ffn=2,
        endpoint_dir="/tmp/afd_pool_test",
        max_inflight_per_ffn=2,
        route="least_inflight",
    )
    sch = AfScheduler(topo, attn_rank=0)
    item = sch.next_task(0, 3, 8)
    f0 = sch.assign_ffn(item)
    assert f0 in (0, 1)
    item2 = sch.next_task(1, 3, 8)
    f1 = sch.assign_ffn(item2)
    assert {f0, f1} == {0, 1} or f0 == f1  # second may differ
    sch.complete(item, compute_s=0.01)
    assert sch.stats.tasks_completed == 1
