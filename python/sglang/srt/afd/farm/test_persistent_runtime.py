# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the persistent AFD farm runtime (no GPU required).

Covers the invariants the cross-forward design depends on:

* a context survives a "forward boundary" and keeps its own tensors,
* ``zero-ready`` leaves every context live,
* the same sequence cannot start token N+1 before token N is sampled,
* ready / deferred row mapping is exact,
* only ready rows advance their seq_len and KV slot.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

from sglang.srt.afd.farm.persistent_runtime import (
    PersistentFarmRuntime,
    get_persistent_runtime,
    snapshot_forward_batch,
    snapshot_sampling_info,
)
from sglang.srt.afd.farm.scheduler import FarmContinuousScheduler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fb(**overrides):
    """Minimal ForwardBatch stand-in with the fields the snapshot touches."""
    base = dict(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([5, 6]),
        out_cache_loc=torch.tensor([7, 8], dtype=torch.int32),
        req_pool_indices=torch.tensor([10, 11]),
        seq_lens=torch.tensor([3, 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 4], dtype=torch.int32),
        rids=["a", "b"],
        num_token_non_padded=torch.tensor(2, dtype=torch.int32),
        sampling_info=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _sampling_info():
    return types.SimpleNamespace(
        temperatures=torch.tensor([1.0, 1.0]),
        top_ps=torch.tensor([1.0, 1.0]),
        top_ks=torch.tensor([-1, -1]),
        min_ps=torch.tensor([0.0, 0.0]),
        sampling_seed=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        rids_int=None,
        bootstrap_room_ids_int=None,
        grammar_mask=None,
    )


# ---------------------------------------------------------------------------
# persistent runtime
# ---------------------------------------------------------------------------
def test_snapshot_detaches_parent_views():
    parent_ids = torch.tensor([1, 2])
    parent_req = torch.tensor([10, 11])
    fb = _fb(input_ids=parent_ids[:1], req_pool_indices=parent_req[:1])
    sampling = _sampling_info()
    fb.sampling_info = sampling
    # Keep the pre-snapshot (parent) tensor handles: the snapshot replaces the
    # fields in place, so the objects to mutate are these originals.
    parent_temps = sampling.temperatures

    snapshot_forward_batch(fb)

    # Mutating the parent buffers must not be visible through the snapshot.
    parent_ids.fill_(99)
    parent_req.fill_(99)
    parent_temps.fill_(9.0)

    assert int(fb.input_ids.item()) == 1
    assert int(fb.req_pool_indices.item()) == 10
    assert float(fb.sampling_info.temperatures[0]) == 1.0


def test_snapshot_sampling_info_leaves_none_untouched():
    sampling = _sampling_info()
    snapshot_sampling_info(sampling)
    assert sampling.temperatures.tolist() == [1.0, 1.0]


def test_context_outlives_forward_and_zero_ready_keeps_it_live():
    runtime = PersistentFarmRuntime()
    hidden = torch.randn(1, 4)
    residual = torch.randn(1, 4)
    positions = torch.tensor([0])

    ctx = runtime.adopt(
        req_pool_idx=42,
        hidden=hidden,
        residual=residual,
        positions=positions,
        child_fb=_fb(),
    )
    runtime.bind_ctx_id(42, 7)

    # A "forward boundary": the caller's buffers are overwritten.
    hidden.fill_(float("nan"))
    assert torch.isfinite(ctx.hidden).all()

    # Zero-ready: nothing sampled, so the context must still be live.
    assert runtime.deferred_req_pool_indices() == [42]
    assert len(runtime) == 1
    assert runtime.get_by_ctx(7) is ctx

    # Second adoption of the same sequence must fail (dependency guard).
    with pytest.raises(RuntimeError, match="already exists"):
        runtime.adopt(
            req_pool_idx=42,
            hidden=torch.randn(1, 4),
            residual=None,
            positions=torch.tensor([1]),
            child_fb=_fb(),
        )


def test_drop_clears_ctx_lookup():
    runtime = PersistentFarmRuntime()
    runtime.adopt(
        req_pool_idx=1,
        hidden=torch.randn(1, 2),
        residual=None,
        positions=torch.tensor([0]),
        child_fb=_fb(),
    )
    runtime.bind_ctx_id(1, 3)
    assert runtime.get_by_ctx(3) is not None
    dropped = runtime.drop(1)
    assert dropped is not None
    assert runtime.get_by_ctx(3) is None
    assert len(runtime) == 0
    assert runtime.deferred_req_pool_indices() == []


def test_multiple_contexts_are_independent():
    runtime = PersistentFarmRuntime()
    for key in (5, 6, 7):
        runtime.adopt(
            req_pool_idx=key,
            hidden=torch.arange(2, dtype=torch.float32).view(1, 2),
            residual=None,
            positions=torch.tensor([0]),
            child_fb=_fb(),
        )
        runtime.bind_ctx_id(key, key * 10)
    assert runtime.deferred_req_pool_indices() == [5, 6, 7]
    assert runtime.get_by_ctx(60).req_pool_idx == 6


# ---------------------------------------------------------------------------
# grouped contexts (SGLANG_AFD_FARM_PERSISTENT_GROUPS)
# ---------------------------------------------------------------------------
def test_group_members_share_one_context():
    """Every member of a group resolves to, and defers through, one context."""
    runtime = PersistentFarmRuntime()
    ctx = runtime.adopt(
        req_pool_idx=10,
        req_pool_idxs=[10, 11, 12],
        hidden=torch.randn(3, 4),
        residual=None,
        positions=torch.tensor([0, 1, 2]),
        child_fb=_fb(),
    )
    runtime.bind_ctx_id(10, 5)

    assert runtime.get(11) is ctx
    assert runtime.get(12) is ctx
    assert ctx.members == [10, 11, 12]
    assert ctx.n_tokens == 3
    # The whole group is deferred, not just the primary key.
    assert runtime.deferred_req_pool_indices() == [10, 11, 12]
    assert len(runtime) == 1

    # A member already in a group cannot be adopted again.
    with pytest.raises(RuntimeError, match="already exists"):
        runtime.adopt(
            req_pool_idx=11,
            hidden=torch.randn(1, 4),
            residual=None,
            positions=torch.tensor([0]),
            child_fb=_fb(),
        )

    # Dropping by any member releases the whole group.
    assert runtime.drop(12) is ctx
    assert runtime.deferred_req_pool_indices() == []
    assert len(runtime) == 0
    assert runtime.get_by_ctx(5) is None


def test_ready_mask_covers_every_group_member(monkeypatch):
    from sglang.srt.afd.farm.persistent_runtime import (
        farm_ready_row_mask,
        reset_persistent_runtime,
    )

    monkeypatch.setenv("SGLANG_AFD_FARM_PERSISTENT", "1")
    reset_persistent_runtime()
    runtime = get_persistent_runtime()
    # A three-member group is mid-flight; row 13 is a fresh (ready) token.
    runtime.adopt(
        req_pool_idx=10,
        req_pool_idxs=[10, 11, 12],
        hidden=torch.zeros(3, 4),
        residual=None,
        positions=torch.tensor([0, 1, 2]),
        child_fb=_fb(),
    )

    mask = farm_ready_row_mask(torch.tensor([10, 11, 12, 13]))
    assert mask.tolist() == [False, False, False, True]

    reset_persistent_runtime()


def test_queue_idxs_are_dense_and_never_split_a_group():
    """A group's scheduler indices must be consecutive or the queue splits it."""
    runtime = PersistentFarmRuntime()
    first = runtime.alloc_queue_idxs(4)
    second = runtime.alloc_queue_idxs(1)
    assert first == [0, 1, 2, 3]
    assert second == [4]
    # No overlap, and each block is internally consecutive.
    assert not set(first) & set(second)

    sched = FarmContinuousScheduler(3)
    ctx_id = sched.begin_batch(first, mb_id=0)
    ticket = sched.reserve(4, max_inflight=8, max_inflight_per_layer=8)
    assert ticket is not None and ticket.ctx_id == ctx_id
    # One ticket must carry the whole block.
    assert len(ticket.keys) == 4


def test_scheduler_advances_group_with_one_ticket():
    """One ticket must carry all of a group's keys, so the group stays put."""
    sched = FarmContinuousScheduler(3)
    ctx_id = sched.begin_batch([1, 2, 3], mb_id=0)

    ticket = sched.reserve(3, max_inflight=8, max_inflight_per_layer=8)
    assert ticket is not None and ticket.ctx_id == ctx_id
    assert len(ticket.keys) == 3
    assert sched.commit(ticket.ticket_id)

    # All three keys advanced together; finishing the last layer returns 3.
    assert sched.complete(ticket.ticket_id) == 0
    deeper = sched.reserve(3, max_inflight=8, max_inflight_per_layer=8)
    assert deeper is not None and len(deeper.keys) == 3
    sched.commit(deeper.ticket_id)
    assert sched.complete(deeper.ticket_id) == 0
    last = sched.reserve(3, max_inflight=8, max_inflight_per_layer=8)
    assert last is not None and len(last.keys) == 3
    sched.commit(last.ticket_id)
    assert sched.complete(last.ticket_id) == 3
    sched.abort_batch()


# ---------------------------------------------------------------------------
# scheduler dependency guard
# ---------------------------------------------------------------------------
def test_scheduler_refuses_second_token_before_sampling():
    sched = FarmContinuousScheduler(4)
    ctx_id = sched.begin_batch([101], mb_id=0)

    # Same sequence cannot start its next token while the first is unsampled.
    with pytest.raises(RuntimeError, match="must be sampled"):
        sched.begin_batch([101], mb_id=0)

    # Drive the single sequence to the end, then sample and close it.
    while not sched.context_output_ready(ctx_id):
        ticket = sched.reserve(
            1, max_inflight=4, max_inflight_per_layer=1, ctx_id=ctx_id
        )
        assert ticket is not None
        assert sched.commit(ticket.ticket_id)
        sched.complete(ticket.ticket_id)

    assert sched.mark_context_sampled(ctx_id, 1) == 1
    assert sched.finish_context(ctx_id)

    # Now the next token may enter.
    next_ctx = sched.begin_batch([101], mb_id=0)
    assert next_ctx != ctx_id
    sched.abort_batch()


def test_scheduler_contexts_advance_independently():
    sched = FarmContinuousScheduler(4)
    a = sched.begin_batch([1], mb_id=0)
    b = sched.begin_batch([2], mb_id=0)
    assert {a, b} <= set(sched.context_ids)

    # Push only context A one layer forward.
    ticket = sched.reserve(1, max_inflight=4, max_inflight_per_layer=1, ctx_id=a)
    assert ticket is not None and ticket.ctx_id == a
    sched.commit(ticket.ticket_id)
    sched.complete(ticket.ticket_id)

    assert sched.context_deepest_layer(a) >= 1
    assert sched.context_deepest_layer(b) == 0
    sched.abort_batch()


# ---------------------------------------------------------------------------
# ready / deferred row mapping
# ---------------------------------------------------------------------------
def test_ready_row_map_and_mask():
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        _farm_ready_mask,
        _farm_ready_row_map,
    )

    batch = types.SimpleNamespace(
        req_pool_indices=torch.tensor([10, 11, 12]),
        reqs=[object(), object(), object()],
    )
    ready = torch.tensor([12, 10])

    mapping = _farm_ready_row_map(batch, ready)
    # ready[0]=12 -> row 2, ready[1]=10 -> row 0
    assert mapping == {2: 0, 0: 1}

    mask = _farm_ready_mask(batch, ready)
    assert mask.tolist() == [True, False, True]


def test_ready_row_map_rejects_non_subset():
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        _farm_ready_row_map,
    )

    batch = types.SimpleNamespace(
        req_pool_indices=torch.tensor([10, 11]),
        reqs=[object(), object()],
    )
    with pytest.raises(RuntimeError, match="not a subset"):
        _farm_ready_row_map(batch, torch.tensor([99]))


# ---------------------------------------------------------------------------
# ready-only seq_lens
# ---------------------------------------------------------------------------
def test_advance_ready_seq_lens_only_touches_ready_rows():
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = types.SimpleNamespace(
        seq_lens=torch.tensor([3, 4, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 4, 5], dtype=torch.int32),
        orig_seq_lens=torch.tensor([3, 4, 5], dtype=torch.int32),
        enable_overlap=False,
    )
    mask = torch.tensor([True, False, True])
    ScheduleBatch._advance_ready_seq_lens(batch, mask)

    assert batch.seq_lens.tolist() == [4, 4, 6]
    assert batch.seq_lens_cpu.tolist() == [4, 4, 6]
    assert batch.orig_seq_lens.tolist() == [4, 4, 6]


def test_ready_mask_from_live_runtime_contexts(monkeypatch):
    """The live farm context set is the deferral authority, not batch state."""
    from sglang.srt.afd.farm.persistent_runtime import (
        farm_ready_row_mask,
        reset_persistent_runtime,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    monkeypatch.setenv("SGLANG_AFD_FARM_PERSISTENT", "1")
    reset_persistent_runtime()
    runtime = get_persistent_runtime()
    # 11 is mid-flight; 10 and 12 have already been sampled (context dropped).
    runtime.adopt(
        req_pool_idx=11,
        hidden=torch.zeros(1, 4),
        residual=None,
        positions=torch.tensor([3]),
        child_fb=_fb(),
    )

    batch = types.SimpleNamespace(req_pool_indices=torch.tensor([10, 11, 12]))
    mask = ScheduleBatch._afd_farm_ready_mask(batch)
    assert mask.tolist() == [True, False, True]

    # A batch that is entirely ready keeps the cheap all-ready path.
    batch_all_ready = types.SimpleNamespace(req_pool_indices=torch.tensor([10, 12]))
    assert ScheduleBatch._afd_farm_ready_mask(batch_all_ready) is None

    reset_persistent_runtime()


# ---------------------------------------------------------------------------
# ready-only KV allocation
# ---------------------------------------------------------------------------
class _StubReqToTokenPool:
    def __init__(self, n_slots=64):
        self.writes = []

    def write(self, index, value):
        self.writes.append((index, value.tolist()))


class _StubTreeCache:
    page_size = 1

    def __init__(self):
        self._next = 1


def _patch_alloc(monkeypatch):
    """Make the non-paged alloc path runnable without server args / a GPU."""
    import sglang.srt.mem_cache.common as common

    def _alloc_token_slots(tree_cache, num_tokens, backup_state=False):
        counter = _patch_alloc.counter
        _patch_alloc.counter += num_tokens
        return torch.arange(
            counter + 1, counter + 1 + num_tokens, dtype=torch.int32
        )

    _patch_alloc.counter = 0
    monkeypatch.setattr(common, "_alloc_page_size", lambda batch: 1)
    monkeypatch.setattr(common, "alloc_token_slots", _alloc_token_slots)


def _decode_batch(pool, n_rows=3):
    return types.SimpleNamespace(
        seq_lens=torch.tensor([2] * n_rows, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2] * n_rows, dtype=torch.int32),
        req_pool_indices=torch.arange(n_rows, dtype=torch.int32),
        req_to_token_pool=pool,
        tree_cache=_StubTreeCache(),
        token_to_kv_pool_allocator=types.SimpleNamespace(),
        model_config=types.SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        out_cache_loc_dsv4=None,
        maybe_evict_swa=lambda: None,
    )


def test_masked_alloc_writes_only_ready_rows(monkeypatch):
    from sglang.srt.mem_cache.common import alloc_for_decode

    _patch_alloc(monkeypatch)
    pool = _StubReqToTokenPool()
    batch = _decode_batch(pool)

    out = alloc_for_decode(batch, 1, ready_mask=torch.tensor([True, False, True]))

    assert out.shape[0] == 3
    # Deferred row keeps the dummy slot and never touches the KV table.
    assert int(out[1].item()) == 0
    assert batch.out_cache_loc_is_dummy is True
    written_rows = [int(v) for idx, _ in pool.writes for v in idx[0].tolist()]
    assert 1 not in written_rows
    assert sorted(written_rows) == [0, 2]


def test_masked_alloc_zero_ready_is_a_noop(monkeypatch):
    from sglang.srt.mem_cache.common import alloc_for_decode

    _patch_alloc(monkeypatch)
    pool = _StubReqToTokenPool()
    batch = _decode_batch(pool, n_rows=2)

    out = alloc_for_decode(batch, 1, ready_mask=torch.tensor([False, False]))

    assert out.tolist() == [0, 0]
    assert pool.writes == []
    assert batch.out_cache_loc_is_dummy is True


def test_unmasked_alloc_is_unchanged_shape(monkeypatch):
    from sglang.srt.mem_cache.common import alloc_for_decode

    _patch_alloc(monkeypatch)
    pool = _StubReqToTokenPool()
    batch = _decode_batch(pool, n_rows=2)
    out = alloc_for_decode(batch, 1)
    assert out.shape[0] == 2
    assert not hasattr(batch, "out_cache_loc_is_dummy")
    assert sorted(
        int(v) for idx, _ in pool.writes for v in idx[0].tolist()
    ) == [0, 1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
