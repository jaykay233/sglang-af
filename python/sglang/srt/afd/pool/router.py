# SPDX-License-Identifier: Apache-2.0
"""MxN FFN rank selection."""

from __future__ import annotations

import itertools
import threading
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.afd.pool.credit import FfnCreditWindow


class AfRouter:
    """Pick an FFN with remaining credit."""

    def __init__(self, num_ffn: int, policy: str = "least_inflight"):
        self.num_ffn = max(1, int(num_ffn))
        p = (policy or "least_inflight").strip().lower()
        if p in ("rr", "round_robin", "round-robin"):
            self.policy = "rr"
        else:
            self.policy = "least_inflight"
        self._rr = itertools.cycle(range(self.num_ffn))
        self._lock = threading.Lock()

    def pick(
        self,
        credit: "FfnCreditWindow",
        *,
        prefer: Optional[Sequence[int]] = None,
    ) -> int:
        """Return ffn_rank that can accept work (best-effort; may still be full)."""
        candidates = list(prefer) if prefer else list(range(self.num_ffn))
        if not candidates:
            candidates = list(range(self.num_ffn))

        if self.policy == "rr":
            with self._lock:
                for _ in range(self.num_ffn):
                    r = next(self._rr)
                    if r in candidates and credit.window_remaining(r) > 0:
                        return int(r)
                # All full: still advance RR and return next preferred.
                return int(next(self._rr) % self.num_ffn)

        # least_inflight among those with remaining credit; else least inflight.
        best = candidates[0]
        best_inf = credit.inflight(best)
        best_rem = credit.window_remaining(best)
        for r in candidates[1:]:
            rem = credit.window_remaining(r)
            inf = credit.inflight(r)
            if rem > 0 and (best_rem <= 0 or inf < best_inf):
                best, best_inf, best_rem = r, inf, rem
            elif rem <= 0 and best_rem <= 0 and inf < best_inf:
                best, best_inf, best_rem = r, inf, rem
        return int(best)
