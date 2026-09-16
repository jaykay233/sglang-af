# SPDX-License-Identifier: Apache-2.0
"""One-shot farm window-slice cache (progress.md §19.6 item 1).

A decode window walks all 27 layers, and its child ``ForwardBatch`` depends only
on the window and the parent batch — never on the layer. Building it per hop was
11.9% of farm time, so it is built once per (window, forward) and reused.

The tests stub ``TboForwardBatchPreparer`` so they exercise only the caching
contract: one build per window, one shared child object, and a ``sampling_info``
that is attached on the final hop and can never leak into an intermediate one.
"""

from __future__ import annotations

import types

import pytest
import torch

from sglang.srt.afd.farm import batch_slice


class _StubChild(types.SimpleNamespace):
    """Stand-in for the ForwardBatch that ``filter_batch`` returns."""


def _parent(n_seq: int):
    return types.SimpleNamespace(
        num_token_non_padded=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        next_token_logits_buffer=None,
        return_hidden_states_before_norm=False,
        batch_size=n_seq,
        sampling_info=None,
    )


def _inputs(n_seq: int):
    hidden = torch.zeros(n_seq, 8)
    positions = torch.zeros(n_seq, dtype=torch.long)
    return hidden, positions


@pytest.fixture
def stub_filter_batch(monkeypatch):
    """Replace filter_batch with a recorder; returns the list of built windows."""
    builds = []

    class _StubPreparer:
        @staticmethod
        def filter_batch(
            batch,
            *,
            start_token_index,
            end_token_index,
            start_seq_index,
            end_seq_index,
            out_num_token_non_padded,
        ):
            builds.append((start_seq_index, end_seq_index))
            return _StubChild()

    import sglang.srt.batch_overlap.two_batch_overlap as tbo

    monkeypatch.setattr(tbo, "TboForwardBatchPreparer", _StubPreparer)
    return builds


def _slice(seq_idxs, parent, hidden, positions, **kw):
    return batch_slice.slice_decode_window(
        hidden_states=hidden,
        residual=None,
        positions=positions,
        forward_batch=parent,
        seq_idxs=seq_idxs,
        token_num_per_seq=1,
        **kw,
    )


def test_window_child_is_built_once_and_shared_across_layers(stub_filter_batch):
    """27 hops of one window must build the child once, not 27 times."""
    parent = _parent(4)
    hidden, positions = _inputs(4)
    cache: dict = {}

    first = _slice([0, 1], parent, hidden, positions, child_cache=cache)
    # 26 further hops of the same window would reuse this child.
    for _ in range(26):
        again = _slice([0, 1], parent, hidden, positions, child_cache=cache)
        assert again.forward_batch is first.forward_batch

    assert stub_filter_batch == [(0, 2)]
    assert len(cache) == 1


def test_distinct_windows_get_distinct_children(stub_filter_batch):
    parent = _parent(4)
    hidden, positions = _inputs(4)
    cache: dict = {}

    a = _slice([0, 1], parent, hidden, positions, child_cache=cache)
    b = _slice([2, 3], parent, hidden, positions, child_cache=cache)

    assert a.forward_batch is not b.forward_batch
    assert stub_filter_batch == [(0, 2), (2, 4)]


def test_no_cache_still_builds_per_call(stub_filter_batch):
    """The escape hatch (`child_cache=None`) must keep the old behaviour."""
    parent = _parent(2)
    hidden, positions = _inputs(2)

    a = _slice([0, 1], parent, hidden, positions)
    b = _slice([0, 1], parent, hidden, positions)

    assert a.forward_batch is not b.forward_batch
    assert stub_filter_batch == [(0, 2), (0, 2)]


def test_intermediate_hop_cannot_inherit_sampling_info(stub_filter_batch):
    """A reused child must not carry a stale sampler payload into layer < 26."""
    parent = _parent(2)
    hidden, positions = _inputs(2)
    cache: dict = {}

    final = _slice([0, 1], parent, hidden, positions, with_sampling_info=False,
                   child_cache=cache)
    final.forward_batch.sampling_info = object()
    final.forward_batch.temperature = object()
    final.forward_batch.top_p = object()

    mid = _slice([0, 1], parent, hidden, positions, with_sampling_info=False,
                 child_cache=cache)

    assert mid.forward_batch is final.forward_batch
    assert mid.forward_batch.sampling_info is None
    assert mid.forward_batch.temperature is None
    assert mid.forward_batch.top_p is None


def test_tensor_views_are_resent_sliced_not_cached(stub_filter_batch):
    """residual can be reallocated mid-forward, so views must be per layer."""
    parent = _parent(2)
    hidden, positions = _inputs(2)
    cache: dict = {}

    _slice([0, 1], parent, hidden, positions, child_cache=cache)
    residual_a = torch.zeros(2, 8)
    sl_a = batch_slice.slice_decode_window(
        hidden_states=hidden,
        residual=residual_a,
        positions=positions,
        forward_batch=parent,
        seq_idxs=[0, 1],
        token_num_per_seq=1,
        with_sampling_info=False,
        child_cache=cache,
    )
    residual_b = torch.ones(2, 8)
    sl_b = batch_slice.slice_decode_window(
        hidden_states=hidden,
        residual=residual_b,
        positions=positions,
        forward_batch=parent,
        seq_idxs=[0, 1],
        token_num_per_seq=1,
        with_sampling_info=False,
        child_cache=cache,
    )

    # Same child (one build), but the residual view tracks the new tensor.
    assert sl_a.forward_batch is sl_b.forward_batch
    assert sl_a.residual is not sl_b.residual
    assert float(sl_b.residual.sum()) == 2 * 8 * 1.0
    assert stub_filter_batch == [(0, 2)]
