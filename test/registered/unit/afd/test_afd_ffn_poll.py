# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the NA1F FFN poll path (no GPU).

``CudaIpcAfdTransport.get_batch`` used to guard its scan with
``while time.time() < deadline:``. For ``timeout_s == 0`` that is immediately
false, so it returned ``[]`` **without scanning**. Callers use ``timeout_s=0``
to mean "one non-blocking pass" — the multi-Attn FFN worker draining several
links, and ``extra_gather`` — so both were silent no-ops. That also made the
FFN worker block on an idle link while another link already had a ready hop.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

from sglang.srt.afd.cuda_ipc_transport import CudaIpcAfdTransport
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.pool.ffn_worker import AfFfnWorker


class _FakeMailbox:
    def __init__(self, posted: int, meta=(-1, -1)):
        self._posted = int(posted)
        self._meta = meta

    def get_posted(self, mb: int) -> int:
        return self._posted

    def get_meta(self, mb: int):
        return self._meta


def _fake_ffn_transport(*, posted: int, num_mb: int = 1) -> CudaIpcAfdTransport:
    """FFN-role transport with the IPC parts stubbed out (never touches CUDA)."""
    tr = object.__new__(CudaIpcAfdTransport)
    tr._role = AfdMode.FFN
    tr._ready = threading.Event()
    tr._ready.set()
    tr._num_mb = num_mb
    tr._ffn_seen = [0] * num_mb
    tr._worker_rank = 0
    tr._pool = None
    tr._timeline = None
    tr._ffn_hot = False
    tr._a2f = [[torch.zeros(2, 4)] for _ in range(num_mb)]
    tr._f2a = [[torch.zeros(2, 4)] for _ in range(num_mb)]
    tr._a2f_ready = [
        SimpleNamespace(synchronize=lambda: None) for _ in range(num_mb)
    ]
    tr._mailbox = _FakeMailbox(posted)
    return tr


def test_zero_timeout_scans_once():
    """timeout_s=0 must return a posted hop instead of a no-op []."""
    tr = _fake_ffn_transport(posted=7)
    got = tr.get_batch(timeout_s=0.0)
    assert len(got) == 1
    assert got[0].handler == 7


def test_zero_timeout_does_not_repeat():
    """The scan marks slots seen, so a second pass is empty (no double serve)."""
    tr = _fake_ffn_transport(posted=7)
    assert len(tr.get_batch(timeout_s=0.0)) == 1
    assert tr.get_batch(timeout_s=0.0) == []


def test_nonblocking_ignores_long_timeout():
    """nonblocking=True returns after one pass even with a long timeout."""
    tr = _fake_ffn_transport(posted=-1)
    t0 = time.perf_counter()
    assert tr.get_batch(timeout_s=10.0, nonblocking=True) == []
    assert time.perf_counter() - t0 < 0.1


def test_zero_timeout_is_not_noop_on_idle_slot():
    """Regression: idle slot 0 must not hide a posted slot 1 in the same pass."""
    tr = _fake_ffn_transport(posted=0, num_mb=2)
    # Only slot 1 has work; slot 0 stays at 0 (== seen).
    tr._mailbox = SimpleNamespace(
        get_posted=lambda mb: 5 if mb == 1 else 0,
        get_meta=lambda mb: (-1, -1),
    )
    got = tr.get_batch(timeout_s=0.0)
    assert len(got) == 1
    assert got[0].handler == 5


# --------------------------------------------------------------------------
# Head-of-line blocking in the NA1F FFN serve loop.
# --------------------------------------------------------------------------


class _IdleLink:
    """Models an idle Attn peer: blocks for the whole timeout, returns nothing."""

    def get_batch(self, timeout_s: float = 1.0, *, nonblocking: bool = False):
        if not nonblocking and timeout_s > 0:
            time.sleep(timeout_s)
        return []


class _OneShotLink:
    """Models a peer with exactly one ready hop."""

    def __init__(self, batch):
        self._batch = batch

    def get_batch(self, timeout_s: float = 1.0, *, nonblocking: bool = False):
        batch, self._batch = self._batch, None
        return [] if batch is None else [batch]


def _fake_worker(transports, *, drain_all: bool):
    worker = object.__new__(AfFfnWorker)
    worker._ffn_rank = 0
    worker._transports = list(transports)
    worker._link_ready = [threading.Event() for _ in transports]
    for ev in worker._link_ready:
        ev.set()
    worker._drain_all = drain_all
    return worker


def test_drain_all_does_not_starve_ready_link_behind_idle_link():
    """A ready hop on link 1 must not wait behind an idle link 0."""
    batch = object()
    timeout = 0.002
    worker = _fake_worker([_IdleLink(), _OneShotLink(batch)], drain_all=True)
    t0 = time.perf_counter()
    got = worker._poll_ready(2, timeout)
    dt = time.perf_counter() - t0
    assert [b for _tr, b in got] == [batch]
    assert dt < timeout / 2, f"ready hop was delayed {dt * 1e3:.3f}ms by an idle link"


def test_legacy_poll_delays_ready_link():
    """Documents the pre-fix behaviour, i.e. what the drain-all path removes."""
    batch = object()
    timeout = 0.002
    worker = _fake_worker([_IdleLink(), _OneShotLink(batch)], drain_all=False)
    t0 = time.perf_counter()
    got = worker._poll_ready(2, timeout)
    dt = time.perf_counter() - t0
    assert [b for _tr, b in got] == [batch]
    assert dt >= timeout * 0.8, f"expected legacy to block ~{timeout * 1e3}ms, got {dt * 1e3:.3f}ms"
