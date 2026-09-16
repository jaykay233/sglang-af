# SPDX-License-Identifier: Apache-2.0
"""Per-round-trip AFD timeline (cross-process via POSIX shm).

Phases (host ``time.perf_counter()``, seconds):
  attn_post   — Attn published mailbox / issued push_pull
  ffn_seen    — FFN poll observed the post
  ffn_compute — FFN finished compute (before respond copy)
  ffn_respond — FFN published done
  attn_done   — Attn wait returned

Enable with ``SGLANG_AFD_TIMELINE=1``. Summarize via
``python -m sglang.srt.afd.timeline`` or ``dump_summary()``.
"""

from __future__ import annotations

import logging
import os
import struct
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional, Sequence, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_RING = 4096
# slot: handle i64, layer i32, mb i32, 6x float64 timestamps
_SLOT = struct.Struct("<qii6d")
assert _SLOT.size == 8 + 4 + 4 + 6 * 8
_HDR = struct.Struct("<ii")  # nslots, write_idx
_SHM_BYTES = _HDR.size + _RING * _SLOT.size


def timeline_enabled() -> bool:
    return bool(envs.SGLANG_AFD_TIMELINE.get())


def _shm_name(endpoint: Optional[str] = None) -> str:
    ep = (endpoint or envs.SGLANG_AFD_IPC_ENDPOINT.get() or "/tmp/afd_cuda_ipc.sock").strip()
    base = os.path.basename(ep).replace(".", "_")
    return f"afd_tl_{base}"[:200]


@dataclass
class RtSample:
    handle: int
    layer: int
    mb: int
    attn_post: float
    ffn_seen: float
    ffn_compute: float
    ffn_respond: float
    attn_wait_enter: float
    attn_done: float

    def gaps_us(self) -> dict:
        """Return phase gaps in microseconds (None if missing)."""

        def g(a: float, b: float) -> Optional[float]:
            if a <= 0.0 or b <= 0.0 or b < a:
                return None
            us = (b - a) * 1e6
            # Drop corrupt / cross-slot tears (host clocks + shm races).
            if us > 100_000.0:
                return None
            return us

        return {
            "post_to_ffn_us": g(self.attn_post, self.ffn_seen),
            "ffn_compute_us": g(self.ffn_seen, self.ffn_compute),
            "ffn_to_respond_us": g(self.ffn_compute, self.ffn_respond),
            # Split the old respond_to_attn: how much of it was the Attn side
            # simply not being back in the wait yet (doing other contexts' work,
            # i.e. real overlap) vs actually parked in the spin loop.
            "respond_to_wait_enter_us": g(self.ffn_respond, self.attn_wait_enter),
            "wait_enter_to_done_us": g(self.attn_wait_enter, self.attn_done),
            "wait_enter_from_post_us": g(self.attn_post, self.attn_wait_enter),
            "respond_to_attn_us": g(self.ffn_respond, self.attn_done),
            "total_rt_us": g(self.attn_post, self.attn_done),
            # Fixed-ish: poll wake + respond publish (excludes FFN kernels)
            "fixed_excl_compute_us": (
                None
                if g(self.attn_post, self.ffn_seen) is None
                or g(self.ffn_compute, self.ffn_respond) is None
                or g(self.ffn_respond, self.attn_done) is None
                else (
                    g(self.attn_post, self.ffn_seen)
                    + g(self.ffn_compute, self.ffn_respond)
                    + g(self.ffn_respond, self.attn_done)
                )
            ),
        }


class AfdRtTimeline:
    """Ring buffer of RT timestamps shared by Attn + FFN processes."""

    def __init__(self, shm: shared_memory.SharedMemory, *, create: bool):
        self.shm = shm
        self._lock = threading.Lock()
        self._create = create
        if create:
            _HDR.pack_into(shm.buf, 0, _RING, 0)
            # zero slots
            for i in range(_RING):
                _SLOT.pack_into(
                    shm.buf, _HDR.size + i * _SLOT.size, 0, -1, -1, 0, 0, 0, 0, 0, 0
                )

    @classmethod
    def create(cls, endpoint: Optional[str] = None) -> "AfdRtTimeline":
        name = _shm_name(endpoint)
        try:
            old = shared_memory.SharedMemory(name=name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        shm = shared_memory.SharedMemory(name=name, create=True, size=_SHM_BYTES)
        return cls(shm, create=True)

    @classmethod
    def attach(cls, endpoint: Optional[str] = None) -> "AfdRtTimeline":
        name = _shm_name(endpoint)
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm, create=False)

    @classmethod
    def try_open(cls, *, create: bool, endpoint: Optional[str] = None) -> Optional["AfdRtTimeline"]:
        if not timeline_enabled():
            return None
        try:
            return cls.create(endpoint) if create else cls.attach(endpoint)
        except Exception as e:
            logger.warning("AFD timeline shm open failed create=%s: %s", create, e)
            return None

    def close(self, *, unlink: bool = False) -> None:
        try:
            self.shm.close()
        except Exception:
            pass
        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass

    def _slot_offset(self, idx: int) -> int:
        return _HDR.size + (idx % _RING) * _SLOT.size

    def _find_handle(self, handle: int) -> Optional[int]:
        """Scan recent slots for handle; return slot index or None."""
        n, write = _HDR.unpack_from(self.shm.buf, 0)
        # search last 64 only — hot path must stay cheap under lock
        span = min(64, _RING, max(write, 1))
        for k in range(span):
            idx = (write - 1 - k) % _RING
            hid, _, _, *_ = _SLOT.unpack_from(self.shm.buf, self._slot_offset(idx))
            if hid == handle:
                return idx
        return None

    def begin(self, handle: int, *, layer: int = -1, mb: int = -1, t_post: float) -> None:
        """Attn: allocate a slot when posting. Never blocks peers long."""
        # Best-effort: skip if lock busy (measurement-only).
        if not self._lock.acquire(blocking=False):
            return
        try:
            n, write = _HDR.unpack_from(self.shm.buf, 0)
            idx = write % _RING
            _SLOT.pack_into(
                self.shm.buf,
                self._slot_offset(idx),
                int(handle),
                int(layer),
                int(mb),
                float(t_post),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            _HDR.pack_into(self.shm.buf, 0, n, (write + 1) % (1 << 30))
        finally:
            self._lock.release()

    def _patch(self, handle: int, field: str, value: float, *, layer: int = -1) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            idx = self._find_handle(handle)
            if idx is None:
                return
            hid, ly, mb, t0, t1, t2, t3, t4, t5 = _SLOT.unpack_from(
                self.shm.buf, self._slot_offset(idx)
            )
            if layer >= 0 and ly < 0:
                ly = layer
            if field == "ffn_seen":
                t1 = value
            elif field == "ffn_compute":
                t2 = value
            elif field == "ffn_respond":
                t3 = value
            elif field == "attn_wait_enter":
                t4 = value
            elif field == "attn_done":
                t5 = value
            else:
                return
            _SLOT.pack_into(
                self.shm.buf,
                self._slot_offset(idx),
                hid,
                ly,
                mb,
                t0,
                t1,
                t2,
                t3,
                t4,
                t5,
            )
        finally:
            self._lock.release()

    def mark_ffn_seen(self, handle: int, t: float, *, layer: int = -1) -> None:
        self._patch(handle, "ffn_seen", t, layer=layer)

    def mark_ffn_compute(self, handle: int, t: float) -> None:
        self._patch(handle, "ffn_compute", t)

    def mark_ffn_respond(self, handle: int, t: float) -> None:
        self._patch(handle, "ffn_respond", t)

    def mark_attn_wait_enter(self, handle: int, t: float) -> None:
        self._patch(handle, "attn_wait_enter", t)

    def mark_attn_done(self, handle: int, t: float) -> None:
        self._patch(handle, "attn_done", t)

    def get_attn_post(self, handle: int) -> float:
        """Return attn_post timestamp for handle, or 0 if missing."""
        if not self._lock.acquire(blocking=False):
            return 0.0
        try:
            idx = self._find_handle(int(handle))
            if idx is None:
                return 0.0
            _hid, _ly, _mb, t0, *_rest = _SLOT.unpack_from(
                self.shm.buf, self._slot_offset(idx)
            )
            return float(t0)
        finally:
            self._lock.release()

    def samples(self, *, complete_only: bool = True) -> List[RtSample]:
        n, write = _HDR.unpack_from(self.shm.buf, 0)
        out: List[RtSample] = []
        # dump up to RING most recent
        count = min(_RING, write) if write < _RING else _RING
        start = 0 if write < _RING else (write % _RING)
        for k in range(count):
            idx = (start + k) % _RING if write >= _RING else k
            hid, ly, mb, t0, t1, t2, t3, t4, t5 = _SLOT.unpack_from(
                self.shm.buf, self._slot_offset(idx)
            )
            if hid <= 0 or t0 <= 0:
                continue
            if complete_only and (t1 <= 0 or t2 <= 0 or t3 <= 0 or t4 <= 0 or t5 <= 0):
                continue
            out.append(
                RtSample(hid, ly, mb, t0, t1, t2, t3, t4, t5)
            )
        return out


def _percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def summarize_samples(samples: Sequence[RtSample]) -> str:
    if not samples:
        return "AFD_RT_TIMELINE n=0"
    keys = [
        "post_to_ffn_us",
        "ffn_compute_us",
        "ffn_to_respond_us",
        "respond_to_wait_enter_us",
        "wait_enter_to_done_us",
        "wait_enter_from_post_us",
        "respond_to_attn_us",
        "fixed_excl_compute_us",
        "total_rt_us",
    ]
    buckets = {k: [] for k in keys}
    for s in samples:
        g = s.gaps_us()
        for k in keys:
            v = g.get(k)
            if v is not None:
                buckets[k].append(v)
    lines = [f"AFD_RT_TIMELINE n={len(samples)}"]
    for k in keys:
        xs = buckets[k]
        if not xs:
            lines.append(f"  {k}: n=0")
            continue
        lines.append(
            f"  {k}: n={len(xs)} p50={_percentile(xs,50):.0f} "
            f"p90={_percentile(xs,90):.0f} mean={sum(xs)/len(xs):.0f} us"
        )
    # fraction of total
    tot = buckets["total_rt_us"]
    fix = buckets["fixed_excl_compute_us"]
    comp = buckets["ffn_compute_us"]
    if tot and fix and comp:
        mt, mf, mc = sum(tot) / len(tot), sum(fix) / len(fix), sum(comp) / len(comp)
        lines.append(
            f"  share: fixed={100*mf/mt:.0f}% compute={100*mc/mt:.0f}% "
            f"(mean total={mt:.0f}us)"
        )
        lines.append(
            f"  verdict: {'MERGE_HANDSHAKE' if mf > mc else 'POOL_OR_CUT_COMPUTE_WAIT'} "
            f"(fixed_mean={mf:.0f}us compute_mean={mc:.0f}us)"
        )
    return "\n".join(lines)


def dump_summary_from_env() -> str:
    tl = AfdRtTimeline.try_open(create=False)
    if tl is None:
        return "AFD_RT_TIMELINE unavailable"
    try:
        return summarize_samples(tl.samples(complete_only=True))
    finally:
        tl.close(unlink=False)


def main() -> None:
    print(dump_summary_from_env())


if __name__ == "__main__":
    main()
