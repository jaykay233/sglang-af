# SPDX-License-Identifier: Apache-2.0
"""Unit tests for decode farm queues, coalesce, soft-persistent, layer CG."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sglang.srt.afd.farm.attn_farm import AttnSendQueue
from sglang.srt.afd.farm.batch_slice import slice_sampling_info
from sglang.srt.afd.farm.env import apply_farm_env, farm_enabled
from sglang.srt.afd.farm.lpu_sim import PerLayerBatchQueue
from sglang.srt.afd.farm.scheduler import (
    FarmContinuousScheduler,
    context_stage_ready,
    plan_context_ranges,
)
from sglang.srt.afd.farm.soft_persistent import (
    maybe_weight_outer_tax,
    reset_soft_persistent_stats,
    soft_persistent_stats,
)
from sglang.srt.afd.farm.token_queue import (
    LayerReadyQueues,
    simulate_bwin_kpi,
    simulate_farm_occupancy,
    take_contiguous_run,
)
from sglang.srt.afd.layer_pipeline import afd_layer_pipeline_enabled
from sglang.srt.afd.pool.scheduler import AfScheduler
from sglang.srt.afd.pool.types import PoolTopology
from sglang.srt.environ import envs


class _FakeQueuedBatch:
    def __init__(self, layer_id: int, num_tokens: int):
        self.layer_id = layer_id
        self.num_tokens = num_tokens


def _queued(layer_id: int, num_tokens: int):
    return object(), _FakeQueuedBatch(layer_id, num_tokens)


def test_per_layer_queue_releases_immediately_and_bounds_group():
    q = PerLayerBatchQueue(
        target_tokens=40,
        max_wait_us=1000,
        max_hops_per_layer=8,
        max_global_hops=16,
    )
    q.push([_queued(3, 10), _queued(3, 10), _queued(3, 10)], now=1.0)
    ready = q.pop_ready(now=1.0001)
    assert len(ready) == 3
    assert q.global_hops == 0
    q.push([_queued(3, 10)], now=1.0001)
    q.push(
        [_queued(3, 10), _queued(3, 10), _queued(3, 10), _queued(3, 10)],
        now=1.0002,
    )
    ready = q.pop_ready(now=1.0003)
    assert len(ready) == 4  # target_tokens bounds one compute group only
    assert q.global_hops == 1
    ready = q.pop_ready(now=1.0004)
    assert len(ready) == 1
    assert q.empty()


def test_per_layer_queue_records_stale_head_without_waiting():
    q = PerLayerBatchQueue(
        target_tokens=128,
        max_wait_us=300,
        max_hops_per_layer=8,
        max_global_hops=16,
    )
    q.push([_queued(1, 16)], now=2.0)
    assert q.time_until_due_us(now=2.0) == 0.0
    stats: dict = {}
    ready = q.pop_ready(now=2.000301, stats=stats)
    assert len(ready) == 1
    assert ready[0][1].layer_id == 1
    assert q.empty()
    assert stats["queue_wait_us_max"] >= 300.0


def test_per_layer_queue_global_cap_does_not_gate_ready_layers():
    q = PerLayerBatchQueue(
        target_tokens=128,
        max_wait_us=1000,
        max_hops_per_layer=8,
        max_global_hops=2,
    )
    q.push([_queued(5, 8)], now=3.0)
    q.push([_queued(6, 8)], now=3.0001)
    ready = q.pop_ready(now=3.0002)
    # global_hops bounds how many queue entries the FFN loop may hold, not how
    # many ready layers may make forward progress in one drain.
    assert [batch.layer_id for _tr, batch in ready] == [5, 6]
    assert q.global_hops == 0
    assert q.empty()


def test_per_layer_queue_merges_same_layer_hops():
    q = PerLayerBatchQueue(
        target_tokens=4,
        max_wait_us=1000,
        max_hops_per_layer=4,
        max_global_hops=16,
    )
    q.push(
        [_queued(0, 1), _queued(0, 1), _queued(0, 1), _queued(0, 1)],
        now=4.0,
    )
    ready = q.pop_ready(now=4.0001)
    assert len(ready) == 4
    assert {batch.layer_id for _tr, batch in ready} == {0}


class _FakeAttnClient:
    def __init__(self):
        self.issued = {}
        self.capacity = 16

    def issue_capacity(self, *, num_tokens=0):
        return int(self.capacity)

    def try_issue_remote_ffn(
        self,
        *,
        req_id,
        layer_id,
        hidden,
        topk_ids=None,
        topk_weights=None,
    ):
        pid = len(self.issued) + 1
        self.issued[pid] = hidden.detach().clone()
        return pid

    def poll_remote_ffn(self, pending_id):
        return self.issued.get(int(pending_id))

    def wait_remote_ffn(self, pending_id):
        return self.issued[int(pending_id)]


def _enqueue_attn(
    queue,
    *,
    value,
    layer_id=3,
    owner=None,
    now=None,
    defer_flush=False,
):
    return queue.enqueue(
        req_id=value,
        layer_id=layer_id,
        hidden=torch.tensor([[float(value)]], dtype=torch.float32),
        topk_ids=torch.tensor([[value, value + 1]], dtype=torch.int32),
        topk_weights=torch.tensor([[1.0, 0.5]], dtype=torch.float32),
        layer_i=1,
        seq_idxs=[value],
        token_lo=value,
        token_hi=value + 1,
        residual=torch.tensor([[float(value)]], dtype=torch.float32),
        meta={},
        owner=owner,
        now=now,
        defer_flush=defer_flush,
    )


def test_attn_send_queue_groups_same_layer_and_splits_output():
    client = _FakeAttnClient()
    queue = AttnSendQueue(
        client,
        target_tokens=4,
        max_wait_us=100000,
        max_hops=4,
    )
    hops = [
        _enqueue_attn(queue, value=i, defer_flush=True) for i in range(4)
    ]
    assert len(client.issued) == 0
    assert queue.pending()
    assert queue.flush_due() == 1
    assert len(client.issued) == 1
    assert torch.equal(
        client.issued[1],
        torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
    )
    stats = queue.stats()
    assert stats["groups"] == 1
    assert stats["hops"] == 4
    assert stats["tokens"] == 4
    for value, hop in enumerate(hops):
        out = queue.poll_group(hop.send_group)
        assert out is not None
        piece = hop.send_group.slice_for(hop)
        assert torch.equal(piece, torch.tensor([[float(value)]]))


def test_attn_send_queue_flushes_oldest_layer_first():
    client = _FakeAttnClient()
    client.capacity = 1
    queue = AttnSendQueue(
        client,
        target_tokens=8,
        max_wait_us=200,
        max_hops=4,
    )
    _enqueue_attn(queue, value=0, layer_id=2, now=1.0, defer_flush=True)
    _enqueue_attn(queue, value=1, layer_id=4, now=1.0001, defer_flush=True)
    assert len(client.issued) == 0
    assert queue.flush_due(now=1.0002) == 1
    assert len(client.issued) == 1
    assert queue.pending()
    assert queue.flush_due(now=1.0003) == 1
    assert len(client.issued) == 2
    assert not queue.pending()


def test_attn_send_queue_separates_active_and_blocked_wait():
    client = _FakeAttnClient()
    client.capacity = 0
    queue = AttnSendQueue(
        client,
        target_tokens=8,
        max_wait_us=0,
        max_hops=4,
    )
    hop = _enqueue_attn(queue, value=0, now=1.0)
    assert hop.send_group is None
    assert queue.flush_due(now=1.001) == 0
    client.capacity = 1
    assert queue.flush_due(now=1.002) == 1
    stats = queue.stats()
    assert stats["groups"] == 1
    assert stats["blocked_us_sum"] >= 1000.0
    assert stats["capacity_empty"] >= 1
    assert stats["flush_reason_hist"]["ready"] == 1


def test_attn_send_queue_merges_owners_at_same_layer():
    client = _FakeAttnClient()
    client.capacity = 0
    queue = AttnSendQueue(
        client,
        target_tokens=4,
        max_wait_us=0,
        max_hops=4,
    )
    a = _enqueue_attn(queue, value=1, layer_id=5, owner=(1, 0, 7))
    b = _enqueue_attn(queue, value=2, layer_id=5, owner=(2, 0, 7))
    assert a.owner == (1, 0, 7)
    assert b.owner == (2, 0, 7)

    client.capacity = 16
    assert queue.flush_due() == 1
    assert len(client.issued) == 1
    stats = queue.stats()
    assert stats["groups"] == 1
    assert stats["hops"] == 2
    assert stats["multi_owner_groups"] == 1
    assert stats["unique_owners_hist"][2] == 1
    assert a.send_group is b.send_group
    for hop, expected in ((a, 1.0), (b, 2.0)):
        assert queue.poll_group(hop.send_group) is not None
        piece = hop.send_group.slice_for(hop)
        assert torch.equal(piece, torch.tensor([[expected]]))


def test_attn_send_queue_tracks_active_layer_peaks():
    client = _FakeAttnClient()
    client.capacity = 4
    queue = AttnSendQueue(
        client,
        target_tokens=2,
        max_wait_us=0,
        max_hops=4,
    )
    _enqueue_attn(
        queue,
        value=1,
        layer_id=0,
        owner=(1, 0, 7),
        defer_flush=True,
    )
    _enqueue_attn(
        queue,
        value=2,
        layer_id=0,
        owner=(2, 0, 7),
        defer_flush=True,
    )
    _enqueue_attn(
        queue,
        value=3,
        layer_id=1,
        owner=(1, 0, 7),
        defer_flush=True,
    )

    assert queue.flush_due() == 2
    stats = queue.stats()
    assert stats["active_groups_peak"] == 2
    assert stats["active_layers_peak"] == 2
    assert stats["active_tokens_peak"] == 3
    assert stats["multi_owner_groups"] == 1


def test_contiguous_run():
    assert take_contiguous_run([0, 1, 2, 5, 6], 16) == [0, 1, 2]
    assert take_contiguous_run([4, 5, 6, 7], 2) == [4, 5]
    assert take_contiguous_run([], 8) == []


def test_plan_context_ranges_splits_across_contexts():
    assert plan_context_ranges(32, b_step=16, num_contexts=2) == (
        (0, 16),
        (16, 32),
    )
    assert plan_context_ranges(8, b_step=16, num_contexts=2) == ((0, 4), (4, 8))
    assert plan_context_ranges(24, b_step=16, num_contexts=2) == (
        (0, 12),
        (12, 24),
    )
    assert plan_context_ranges(0, b_step=16, num_contexts=2) == ()


def test_context_stage_gate_batches_same_stage_and_staggers_next_stage():
    assert context_stage_ready(
        next_context_idx=1,
        contexts_per_stage=2,
        context_stagger_layers=1,
        active_depths=[0],
        num_layers=4,
        max_inflight=4,
    )
    assert not context_stage_ready(
        next_context_idx=2,
        contexts_per_stage=2,
        context_stagger_layers=1,
        active_depths=[0],
        num_layers=4,
        max_inflight=4,
    )
    assert context_stage_ready(
        next_context_idx=2,
        contexts_per_stage=2,
        context_stagger_layers=1,
        active_depths=[1],
        num_layers=4,
        max_inflight=4,
    )


def test_ready_queue_pick_and_sticky():
    q = LayerReadyQueues(4, b_win_k=2)
    q.enqueue(0, range(8))
    q.enqueue(2, [10, 11])
    a = q.pick(4)
    assert a is not None
    li, idxs = a
    assert li == 0
    assert idxs == [0, 1, 2, 3]
    assert q.last_win_index == 0
    b = q.pick(4)
    assert b is not None
    assert b[0] == 0  # sticky
    assert b[1] == [4, 5, 6, 7]
    assert q.last_win_index == 1
    assert q.bwin.layer_switches == 1
    assert q.bwin.picks == 2


def test_ready_queue_skips_capped_layer_to_advance_wavefront():
    q = LayerReadyQueues(4, b_win_k=8)
    q.enqueue(0, range(16))
    q.enqueue(2, [10, 11])
    a = q.pick(4)
    assert a is not None and a[0] == 0
    # Simulate layer 0 already having its only allowed in-flight window.
    b = q.pick(4, skip_layers={0})
    assert b is not None and b[0] == 2
    assert b[1] == [10, 11]
    assert q.last_win_index == 0


def test_ready_queue_returns_none_when_all_candidates_capped():
    q = LayerReadyQueues(2, b_win_k=8)
    q.enqueue(0, range(16))
    q.enqueue(1, [10, 11])
    assert q.pick(4, skip_layers={0, 1}) is None
    # The queues are untouched and remain available after a completion.
    assert q.depth(0) == 16
    assert q.depth(1) == 2


def test_ready_queue_skips_capped_sticky_layer_when_quota_expires():
    q = LayerReadyQueues(4, b_win_k=1)
    q.enqueue(0, range(16))
    q.enqueue(2, [10, 11])
    assert q.pick(4)[0] == 0

    b = q.pick(4, skip_layers={0})
    assert b is not None and b[0] == 2
    assert b[1] == [10, 11]


def test_coalesce_packs_multiple_b_steps():
    q = LayerReadyQueues(2, b_win_k=8, coalesce_k=4)
    q.enqueue(0, range(32))
    a = q.pick(8)
    assert a is not None
    li, idxs = a
    assert li == 0
    assert len(idxs) == 32  # 4 * B_step
    assert idxs == list(range(32))
    assert q.last_micro_windows == 4
    assert q.bwin.amortize_factor == 4.0
    assert q.sticky_left == 4  # started at 8, consumed 4 micros


def test_coalesce_snaps_to_b_step_multiple():
    q = LayerReadyQueues(1, b_win_k=4, coalesce_k=4)
    q.enqueue(0, range(20))  # not multiple of 8 beyond 16
    a = q.pick(8)
    assert a is not None
    assert len(a[1]) == 16  # snapped from 20 → 16


def test_bwin_expires_switches_to_other_layer():
    q = LayerReadyQueues(3, b_win_k=1)
    q.enqueue(0, [0, 1, 2, 3])
    q.enqueue(1, [10, 11])
    a = q.pick(2)
    assert a is not None and a[0] == 0
    assert q.last_win_index == 0
    # K=1 expired; layer 1 has work → must switch
    b = q.pick(2)
    assert b is not None and b[0] == 1
    assert q.bwin.layer_switches == 2


def test_continuous_scheduler_parallel_contexts_are_isolated():
    sched = FarmContinuousScheduler(3, b_win_k=8)
    sched.begin_batch([0], ctx_id=1, mb_id=0, step_id=100)
    sched.begin_batch([1], ctx_id=2, mb_id=1, step_id=100)

    a0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    b0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert a0 is not None and b0 is not None
    assert a0.layer_i == b0.layer_i == 0
    assert a0.seq_idxs == (0,)
    assert b0.seq_idxs == (1,)
    assert a0.keys != b0.keys
    assert sched.commit(a0.ticket_id)
    assert sched.commit(b0.ticket_id)

    assert sched.complete(a0.ticket_id) == 0
    a1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert a1 is not None and a1.layer_i == 1
    assert sched.queues.running_depth(0) == 1

    assert sched.complete(b0.ticket_id) == 0
    b1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert b1 is not None and b1.layer_i == 1


def test_continuous_scheduler_does_not_skip_layers_for_one_context():
    sched = FarmContinuousScheduler(3, b_win_k=8)
    sched.begin_batch([0], ctx_id=1, step_id=1)

    t0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t0 is not None and t0.layer_i == 0
    assert sched.commit(t0.ticket_id)

    assert (
        sched.reserve(
            1,
            max_inflight=4,
            max_inflight_per_layer=4,
            ctx_id=1,
        )
        is None
    )
    assert sched.complete(t0.ticket_id) == 0
    t1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t1 is not None and t1.layer_i == 1
    assert sched.queues.depth(2) == 0


def test_continuous_scheduler_reports_context_wavefront_depth():
    sched = FarmContinuousScheduler(3, b_win_k=8)
    sched.begin_batch([0], ctx_id=1, step_id=1)
    assert sched.context_deepest_layer(1) == 0
    assert not sched.context_output_ready(1)

    t0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t0 is not None and t0.layer_i == 0
    assert sched.commit(t0.ticket_id)
    assert sched.context_deepest_layer(1) == 0
    assert sched.complete(t0.ticket_id) == 0
    assert sched.context_deepest_layer(1) == 1

    t1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t1 is not None and t1.layer_i == 1
    assert sched.commit(t1.ticket_id)
    assert sched.context_deepest_layer(1) == 1
    assert sched.complete(t1.ticket_id) == 0
    assert sched.context_deepest_layer(1) == 2

    t2 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t2 is not None and t2.layer_i == 2
    assert sched.commit(t2.ticket_id)
    assert sched.complete(t2.ticket_id) == 1
    assert sched.context_output_ready(1)
    assert sched.context_deepest_layer(1) == -1


def test_continuous_scheduler_can_overlap_staggered_contexts():
    sched = FarmContinuousScheduler(3, b_win_k=8)
    sched.begin_batch([0], ctx_id=1, step_id=1)

    t0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert t0 is not None and sched.commit(t0.ticket_id)
    assert sched.complete(t0.ticket_id) == 0
    assert sched.context_deepest_layer(1) == 1

    # The next context is injected only after context 1 reached layer 1.
    sched.begin_batch([1], ctx_id=2, step_id=1)
    a1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    b0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert a1 is not None and a1.layer_i == 1
    assert b0 is not None and b0.layer_i == 0
    assert {a1.layer_i, b0.layer_i} == {0, 1}


def test_finish_context_does_not_close_other_context():
    sched = FarmContinuousScheduler(2, b_win_k=8)
    sched.begin_batch([0], ctx_id=1, step_id=1)
    sched.begin_batch([1], ctx_id=2, step_id=1)

    a0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert a0 is not None and sched.commit(a0.ticket_id)
    assert sched.complete(a0.ticket_id) == 0
    a1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert a1 is not None and sched.commit(a1.ticket_id)
    assert sched.complete(a1.ticket_id) == 1
    assert sched.mark_context_sampled(1) == 1
    assert sched.finish_context(1)
    assert sched.context_ids == (2,)

    b0 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert b0 is not None and sched.commit(b0.ticket_id)
    assert sched.complete(b0.ticket_id) == 0
    b1 = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert b1 is not None and sched.commit(b1.ticket_id)
    assert sched.complete(b1.ticket_id) == 1
    assert sched.mark_context_sampled(2) == 1
    assert sched.finish_context(2)
    assert sched.idle()


def test_occupancy_farm_beats_lockstep():
    lock = simulate_farm_occupancy(
        n_tokens=32,
        n_layers=8,
        b_step=32,
        b_win_k=1,
        max_inflight=1,
        attn_ticks=1,
        ffn_ticks=3,
    )
    farm = simulate_farm_occupancy(
        n_tokens=32,
        n_layers=8,
        b_step=8,
        b_win_k=4,
        max_inflight=4,
        attn_ticks=1,
        ffn_ticks=3,
    )
    assert lock.finished == 32
    assert farm.finished == 32
    assert farm.mean_layers_busy > lock.mean_layers_busy
    assert farm.peak_layers_busy >= 2
    assert lock.peak_layers_busy == 1


def test_sweep_k_larger_k_fewer_switches():
    k1 = simulate_bwin_kpi(n_layers=8, n_per_layer=64, b_step=8, b_win_k=1, max_picks=64)
    k32 = simulate_bwin_kpi(
        n_layers=8, n_per_layer=64, b_step=8, b_win_k=32, max_picks=64
    )
    assert int(k32["layer_switches"]) < int(k1["layer_switches"])
    assert float(k32["mean_win_len"]) > float(k1["mean_win_len"])


def test_sweep_coalesce_raises_amortize_factor():
    c1 = simulate_bwin_kpi(
        n_layers=4, n_per_layer=64, b_step=8, b_win_k=8, coalesce_k=1, max_picks=32
    )
    c4 = simulate_bwin_kpi(
        n_layers=4, n_per_layer=64, b_step=8, b_win_k=8, coalesce_k=4, max_picks=32
    )
    assert float(c4["amortize_factor"]) > float(c1["amortize_factor"])
    assert int(c4["picks"]) < int(c1["picks"])


def test_apply_farm_env_disables_layer_pipe():
    prev = {
        "farm": envs.SGLANG_AFD_FARM.get(),
        "pipe": envs.SGLANG_AFD_LAYER_PIPELINE.get(),
        "mb": envs.SGLANG_AFD_NUM_MB.get(),
        "gather": envs.SGLANG_AFD_FFN_GATHER_US.get(),
        "inf": envs.SGLANG_AFD_FARM_MAX_INFLIGHT.get(),
        "lpu": envs.SGLANG_AFD_FARM_LPU_GATHER_US.get(),
        "gather_max": envs.SGLANG_AFD_FFN_GATHER_MAX.get(),
    }
    try:
        envs.SGLANG_AFD_FARM.set(True)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
        envs.SGLANG_AFD_NUM_MB.set(2)
        envs.SGLANG_AFD_FFN_GATHER_US.set(0)
        envs.SGLANG_AFD_FARM_MAX_INFLIGHT.set(4)
        envs.SGLANG_AFD_FARM_LPU_GATHER_US.set(50)
        assert farm_enabled()
        assert apply_farm_env()
        assert not bool(envs.SGLANG_AFD_LAYER_PIPELINE.get())
        assert int(envs.SGLANG_AFD_NUM_MB.get()) >= 4
        assert int(envs.SGLANG_AFD_FFN_GATHER_US.get()) == 50
        assert not afd_layer_pipeline_enabled()
    finally:
        envs.SGLANG_AFD_FARM.set(prev["farm"])
        envs.SGLANG_AFD_LAYER_PIPELINE.set(prev["pipe"])
        envs.SGLANG_AFD_NUM_MB.set(prev["mb"])
        envs.SGLANG_AFD_FFN_GATHER_US.set(prev["gather"])
        envs.SGLANG_AFD_FARM_MAX_INFLIGHT.set(prev["inf"])
        envs.SGLANG_AFD_FARM_LPU_GATHER_US.set(prev["lpu"])
        envs.SGLANG_AFD_FFN_GATHER_MAX.set(prev["gather_max"])


def test_true_overlap_does_not_reenable_layer_pipe_when_farm():
    from sglang.srt.afd.layer_pipeline import apply_true_overlap_env

    prev = {
        "farm": envs.SGLANG_AFD_FARM.get(),
        "pipe": envs.SGLANG_AFD_LAYER_PIPELINE.get(),
        "overlap": envs.SGLANG_AFD_TRUE_OVERLAP.get(),
        "ing": envs.SGLANG_AFD_IN_GRAPH_WAIT.get(),
        "mb": envs.SGLANG_AFD_NUM_MB.get(),
        "gather": envs.SGLANG_AFD_FFN_GATHER_US.get(),
    }
    try:
        envs.SGLANG_AFD_FARM.set(True)
        envs.SGLANG_AFD_TRUE_OVERLAP.set(True)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(True)
        apply_true_overlap_env()
        assert not bool(envs.SGLANG_AFD_LAYER_PIPELINE.get())
        assert not bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        assert not afd_layer_pipeline_enabled()
    finally:
        envs.SGLANG_AFD_FARM.set(prev["farm"])
        envs.SGLANG_AFD_LAYER_PIPELINE.set(prev["pipe"])
        envs.SGLANG_AFD_TRUE_OVERLAP.set(prev["overlap"])
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(prev["ing"])
        envs.SGLANG_AFD_NUM_MB.set(prev["mb"])
        envs.SGLANG_AFD_FFN_GATHER_US.set(prev["gather"])


def test_try_assign_ffn_smooth_send():
    topo = PoolTopology(
        num_attn=1,
        num_ffn=1,
        endpoint_dir="/tmp/afd_farm_test",
        max_inflight_per_ffn=2,
        route="least_inflight",
    )
    sch = AfScheduler(topo, attn_rank=0)
    a = sch.next_task(0, 1, 8)
    b = sch.next_task(1, 1, 8)
    c = sch.next_task(2, 1, 8)
    assert sch.try_assign_ffn(a) == 0
    assert sch.try_assign_ffn(b) == 0
    assert sch.try_assign_ffn(c) is None
    sch.abort_assign(b)
    assert sch.try_assign_ffn(c) == 0


def test_soft_persistent_taxes_first_window_only():
    prev = {
        "sp": envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.get(),
        "us": envs.SGLANG_AFD_FARM_WEIGHT_TAX_US.get(),
        "bytes": envs.SGLANG_AFD_FARM_WEIGHT_TAX_BYTES.get(),
    }
    try:
        envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.set(True)
        envs.SGLANG_AFD_FARM_WEIGHT_TAX_US.set(200.0)
        envs.SGLANG_AFD_FARM_WEIGHT_TAX_BYTES.set(0)
        reset_soft_persistent_stats()
        c0 = maybe_weight_outer_tax(layer_id=3, win_index=0)
        c1 = maybe_weight_outer_tax(layer_id=3, win_index=1)
        c2 = maybe_weight_outer_tax(layer_id=3, win_index=2)
        assert c0 >= 150.0
        assert c1 == 0.0
        assert c2 == 0.0
        st = soft_persistent_stats()
        assert st["tax_calls"] == 1
        assert st["skip_calls"] == 2
    finally:
        envs.SGLANG_AFD_FARM_SOFT_PERSISTENT.set(prev["sp"])
        envs.SGLANG_AFD_FARM_WEIGHT_TAX_US.set(prev["us"])
        envs.SGLANG_AFD_FARM_WEIGHT_TAX_BYTES.set(prev["bytes"])
        reset_soft_persistent_stats()


def test_layer_cg_graphable_callable():
    import torch

    if not torch.cuda.is_available():
        return
    from sglang.srt.afd.farm.layer_cuda_graph import capture_graphable_callable

    w = torch.randn(64, 64, device="cuda", dtype=torch.float16)

    def fn(x):
        return x @ w

    x0 = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    g, static_in, static_out = capture_graphable_callable(fn, x0)
    x1 = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    static_in.copy_(x1)
    g.replay()
    expect = fn(x1)
    assert torch.allclose(static_out, expect, atol=1e-2, rtol=1e-2)


def test_weight_outer_linear_matches_mm():
    import torch

    from sglang.srt.afd.farm.persistent_linear import (
        reset_persistent_linear_stats,
        weight_outer_linear,
    )
    from sglang.srt.environ import envs

    prev = envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.get()
    try:
        envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(True)
        reset_persistent_linear_stats()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        b_step, num_mb, k, n = 8, 4, 64, 96
        x = torch.randn(num_mb * b_step, k, device=device, dtype=dtype)
        w = torch.randn(n, k, device=device, dtype=dtype)
        y = weight_outer_linear(x, w, b_step=b_step)
        ref = x @ w.T
        assert y.shape == ref.shape
        assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2)
    finally:
        envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(prev)
        reset_persistent_linear_stats()


def test_try_persistent_requires_multi_mb():
    import torch

    from sglang.srt.afd.farm.persistent_linear import (
        reset_persistent_linear_stats,
        try_persistent_linear,
    )
    from sglang.srt.environ import envs

    prev = envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.get()
    try:
        envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(True)
        reset_persistent_linear_stats()
        x = torch.randn(8, 32)
        w = torch.randn(16, 32)
        assert try_persistent_linear(x, w, b_step=8) is None  # only 1 mb
        y = try_persistent_linear(
            torch.randn(16, 32), w, b_step=8
        )
        assert y is not None and y.shape == (16, 16)
    finally:
        envs.SGLANG_AFD_FARM_PERSISTENT_LINEAR.set(prev)
        reset_persistent_linear_stats()


def test_spin_wait_session_matches_mm():
    import torch

    from sglang.srt.afd.farm.spin_wait_linear import SpinWaitLinearSession

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    b_step, num_mb, k, n = 8, 4, 64, 96
    w = torch.randn(n, k, device=device, dtype=dtype)
    xs = torch.randn(num_mb, b_step, k, device=device, dtype=dtype)
    sess = SpinWaitLinearSession(weight=w, b_step=b_step)
    y = sess.run_many(xs)
    ref = xs.view(num_mb * b_step, k) @ w.T
    assert y.shape == ref.shape
    assert torch.allclose(y.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_bench_farm_amortize_cpu_import():
    from sglang.srt.afd.farm.bench_farm_amortize import sweep_cpu_coalesce

    rows = sweep_cpu_coalesce(b_step=8, coalesce_ks=[1, 4])
    assert rows[0]["amortize_factor"] == 1.0
    assert rows[1]["amortize_factor"] >= 2.0


@dataclass
class _SamplingInfoStub:
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    sampling_seed: object = None
    acc_additive_penalties: object = None
    acc_scaling_penalties: object = None
    logit_bias: object = None
    rids_int: object = None
    bootstrap_room_ids_int: object = None
    grammars: object = None
    grammar_mask: object = None
    custom_params: object = None
    custom_logit_processor: object = None
    has_custom_logit_processor: bool = False
    return_sampling_masks: object = None
    penalizer_orchestrator: object = None


def test_slice_sampling_info_slices_all_row_state():
    processor = object()
    custom_mask = torch.tensor([False, False, True, True])
    parent = _SamplingInfoStub(
        temperatures=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        top_ps=torch.tensor([1.0, 0.8, 0.9, 1.0]),
        top_ks=torch.tensor([4, 1, 1, 4]),
        min_ps=torch.tensor([0.0, 0.1, 0.2, 0.0]),
        sampling_seed=torch.tensor([10, 11, 12, 13]),
        acc_additive_penalties=torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
        acc_scaling_penalties=torch.tensor([[1.0], [0.9], [0.8], [1.0]]),
        logit_bias=torch.arange(12, dtype=torch.float32).view(4, 3),
        rids_int=torch.tensor([20, 21, 22, 23]),
        bootstrap_room_ids_int=torch.tensor([30, 31, 32, 33]),
        grammars=["g0", "g1", "g2", "g3"],
        grammar_mask=object(),
        custom_params=[{"row": 0}, {"row": 1}, {"row": 2}, {"row": 3}],
        custom_logit_processor={7: (processor, custom_mask)},
        has_custom_logit_processor=True,
        return_sampling_masks=[False, False, True, False],
    )

    child = slice_sampling_info(parent, seq_lo=1, seq_hi=3)

    assert child is not parent
    assert torch.equal(child.temperatures, parent.temperatures[1:3])
    assert torch.equal(child.top_ps, parent.top_ps[1:3])
    assert torch.equal(child.top_ks, parent.top_ks[1:3])
    assert torch.equal(child.min_ps, parent.min_ps[1:3])
    assert torch.equal(child.sampling_seed, parent.sampling_seed[1:3])
    assert torch.equal(
        child.acc_additive_penalties, parent.acc_additive_penalties[1:3]
    )
    assert torch.equal(
        child.acc_scaling_penalties, parent.acc_scaling_penalties[1:3]
    )
    assert torch.equal(child.logit_bias, parent.logit_bias[1:3])
    assert torch.equal(child.rids_int, parent.rids_int[1:3])
    assert torch.equal(
        child.bootstrap_room_ids_int, parent.bootstrap_room_ids_int[1:3]
    )
    assert child.grammars == ["g1", "g2"]
    assert child.custom_params == [{"row": 1}, {"row": 2}]
    assert child.return_sampling_masks == [False, True]
    assert child.grammar_mask is None
    assert child.penalizer_orchestrator is None
    assert child.has_custom_logit_processor
    assert child.custom_logit_processor[7][0] is processor
    assert torch.equal(child.custom_logit_processor[7][1], custom_mask[1:3])
    assert child.is_all_greedy
    assert child.need_top_p_sampling
    assert child.need_top_k_sampling
    assert child.need_min_p_sampling
    assert parent.grammars == ["g0", "g1", "g2", "g3"]


def test_final_layer_windows_sample_before_context_finishes():
    sched = FarmContinuousScheduler(2, b_win_k=8)
    sched.begin_batch([0, 1], ctx_id=1, step_id=1)

    for _ in range(2):
        ticket = sched.reserve(
            1,
            max_inflight=8,
            max_inflight_per_layer=8,
            ctx_id=1,
        )
        assert ticket is not None and ticket.layer_i == 0
        assert sched.commit(ticket.ticket_id)
        assert sched.complete(ticket.ticket_id) == 0

    first_last = sched.reserve(
        1,
        max_inflight=8,
        max_inflight_per_layer=8,
        ctx_id=1,
    )
    assert first_last is not None and first_last.layer_i == 1
    assert sched.commit(first_last.ticket_id)
    assert sched.complete(first_last.ticket_id) == 1
    assert sched.mark_context_sampled(1, 1) == 1
    assert not sched.finish_context(1)

    second_last = sched.reserve(
        1,
        max_inflight=8,
        max_inflight_per_layer=8,
        ctx_id=1,
    )
    assert second_last is not None and second_last.layer_i == 1
    assert sched.commit(second_last.ticket_id)
    assert sched.complete(second_last.ticket_id) == 1
    assert sched.mark_context_sampled(1, 1) == 1
    assert sched.finish_context(1)


def test_output_ready_context_stops_while_peer_can_continue():
    sched = FarmContinuousScheduler(1)
    sched.begin_batch([0], ctx_id=1, step_id=1)
    sched.begin_batch([1], ctx_id=2, step_id=1)

    first = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert first is not None and sched.commit(first.ticket_id)
    assert sched.complete(first.ticket_id) == 1
    assert (
        sched.reserve(
            1,
            max_inflight=4,
            max_inflight_per_layer=4,
            ctx_id=1,
        )
        is None
    )

    peer = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=2,
    )
    assert peer is not None and peer.layer_i == 0
    assert sched.commit(peer.ticket_id)


def test_sequence_waits_for_sampling_before_next_token():
    sched = FarmContinuousScheduler(1)
    first_ctx = sched.begin_batch([0], ctx_id=1, step_id=1)
    ticket = sched.reserve(
        1,
        max_inflight=4,
        max_inflight_per_layer=4,
        ctx_id=1,
    )
    assert ticket is not None and sched.commit(ticket.ticket_id)
    assert sched.complete(ticket.ticket_id) == 1

    with pytest.raises(RuntimeError, match="must be sampled"):
        sched.begin_batch([0])

    assert sched.mark_context_sampled(1) == 1
    assert sched.finish_context(1)
    next_ctx = sched.begin_batch([0])
    assert next_ctx != first_ctx
