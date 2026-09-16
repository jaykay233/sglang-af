# SPDX-License-Identifier: Apache-2.0
"""Per-FFN inflight credit (simplified HBU-style window)."""

from __future__ import annotations

import threading
from typing import List


class FfnCreditWindow:
    """Tracks granted / in-flight A2F hops per FFN rank."""

    def __init__(self, num_ffn: int, max_inflight: int):
        self.num_ffn = max(1, int(num_ffn))
        self.max_inflight = max(1, int(max_inflight))
        self._inflight: List[int] = [0] * self.num_ffn
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def inflight(self, ffn_rank: int) -> int:
        with self._lock:
            return int(self._inflight[int(ffn_rank)])

    def window_remaining(self, ffn_rank: int) -> int:
        with self._lock:
            r = int(ffn_rank)
            return max(0, self.max_inflight - self._inflight[r])

    def try_acquire(self, ffn_rank: int) -> bool:
        with self._lock:
            r = int(ffn_rank)
            if self._inflight[r] >= self.max_inflight:
                return False
            self._inflight[r] += 1
            return True

    def acquire(self, ffn_rank: int, timeout_s: float = 30.0) -> bool:
        deadline = None
        import time

        if timeout_s is not None and timeout_s >= 0:
            deadline = time.monotonic() + float(timeout_s)
        with self._cv:
            r = int(ffn_rank)
            while self._inflight[r] >= self.max_inflight:
                if deadline is None:
                    self._cv.wait()
                else:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        return False
                    self._cv.wait(timeout=remain)
            self._inflight[r] += 1
            return True

    def release(self, ffn_rank: int) -> None:
        with self._cv:
            r = int(ffn_rank)
            if self._inflight[r] > 0:
                self._inflight[r] -= 1
            self._cv.notify_all()

    def snapshot(self) -> List[int]:
        with self._lock:
            return list(self._inflight)
