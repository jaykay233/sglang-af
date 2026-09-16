# SPDX-License-Identifier: Apache-2.0
"""FFN-side AfPool worker: poll multiple Attn cuda_ipc links."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Sequence

import torch

from sglang.srt.afd.buffers import AfdBufferPool, AfdBufferPoolConfig
from sglang.srt.afd.cuda_ipc_transport import CudaIpcAfdTransport
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.pool.types import PoolTopology
from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.afd.transport import FfnComputeFn
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class AfFfnWorker:
    """One FFN rank serving Na Attn peers.

    Handshake for each Attn link runs in a helper thread (accept can block).
    After links are up, a **single** CUDA serve loop round-robins get_batch so
    multi-Attn never shares one GPU compute across Python threads.
    """

    def __init__(
        self,
        topo: PoolTopology,
        *,
        ffn_rank: int,
        hidden_size: int,
        compute: FfnComputeFn,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_mb: int = 2,
        max_num_token: int = 256,
        moe_topk: int = 0,
    ):
        self.topo = topo
        self.ffn_rank = int(ffn_rank)
        self.compute = compute
        self.device = device
        self._stop = threading.Event()
        self._accept_threads: List[threading.Thread] = []
        self._serve_thread: Optional[threading.Thread] = None
        self._transports: List[Optional[CudaIpcAfdTransport]] = [None] * topo.num_attn
        self._pools: List[Optional[AfdBufferPool]] = [None] * topo.num_attn
        self.tasks_done = 0
        self.busy_s = 0.0
        self._stats_lock = threading.Lock()
        self._link_ready = [threading.Event() for _ in range(topo.num_attn)]
        self._num_mb = num_mb
        self._max_num_token = max_num_token
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._moe_topk = moe_topk
        self._lpu_stats: dict = {}
        self._lpu_stats_calls = 0
        self._skip_extra_gather = _env_flag(
            "SGLANG_AFD_FARM_SKIP_EXTRA_GATHER", default=False
        )
        self._drain_all = bool(envs.SGLANG_AFD_FFN_POLL_DRAIN_ALL.get())
        self._ffn_queue = None
        if envs.SGLANG_AFD_FFN_QUEUE_ENABLE.get():
            from sglang.srt.afd.farm.lpu_sim import PerLayerBatchQueue

            self._ffn_queue = PerLayerBatchQueue(
                target_tokens=envs.SGLANG_AFD_FFN_QUEUE_TARGET_TOKENS.get(),
                max_wait_us=envs.SGLANG_AFD_FFN_QUEUE_MAX_WAIT_US.get(),
                max_hops_per_layer=envs.SGLANG_AFD_FFN_QUEUE_MAX_HOPS.get(),
                max_global_hops=envs.SGLANG_AFD_FFN_QUEUE_MAX_GLOBAL_HOPS.get(),
            )
        self._lpu_stats_every = max(
            0, int(os.environ.get("SGLANG_AFD_FARM_LPU_STATS_EVERY", "0") or 0)
        )
        # FFN-side utilisation self-report (see utilization()). Off by default.
        self._util_path = (envs.SGLANG_AFD_POOL_UTIL_FILE.get() or "").strip()
        self._util_last = 0.0
        self._serve_t0 = time.perf_counter()

        self._stop.clear()
        for i in range(topo.num_attn):
            th = threading.Thread(
                target=self._accept_link,
                args=(i,),
                name=f"af-pool-ffn{self.ffn_rank}-accept-a{i}",
                daemon=True,
            )
            th.start()
            self._accept_threads.append(th)

        # Block until link0 is ready so Attn0 can connect/proceed.
        if not self._link_ready[0].wait(timeout=300):
            raise TimeoutError(
                f"AfPool FFN rank={self.ffn_rank} timed out waiting for attn=0"
            )
        logger.info(
            "AfPool FFN rank=%s link0 ready; remaining attns=%s connecting in bg",
            self.ffn_rank,
            topo.num_attn - 1,
        )

        self._serve_thread = threading.Thread(
            target=self._serve_loop,
            name=f"af-pool-ffn{self.ffn_rank}-serve",
            daemon=True,
        )
        self._serve_thread.start()

    def _accept_link(self, attn_rank: int) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                pass
        ep = self.topo.endpoint(attn_rank, self.ffn_rank)
        pool = AfdBufferPool(
            AfdBufferPoolConfig(
                num_mb=self._num_mb,
                max_num_token=self._max_num_token,
                hidden_size=self._hidden_size,
                dtype=self._dtype,
                device=self.device,
                moe_topk=self._moe_topk,
                worker_rank=attn_rank,
            )
        )
        tr = CudaIpcAfdTransport(endpoint=ep)
        tr.init(AfdMode.FFN, worker_rank=attn_rank)
        tr.register_buffers(pool)
        self._transports[attn_rank] = tr
        self._pools[attn_rank] = pool
        self._link_ready[attn_rank].set()
        logger.info(
            "AfPool FFN rank=%s link attn=%s endpoint=%s ready",
            self.ffn_rank,
            attn_rank,
            ep,
        )

    def _poll_once(self, n: int):
        """One non-blocking scan of every Attn link.

        Must never block on a single link: with Na Attn peers, blocking on an
        idle link i delays a hop that is already ready on link j, which
        serialises the pool and caps 2A1F below the sum of its parts.
        """
        items = []
        for i in range(n):
            if not self._link_ready[i].is_set():
                continue
            tr = self._transports[i]
            if tr is None:
                continue
            try:
                batches = tr.get_batch(timeout_s=0.0, nonblocking=True)
            except Exception as e:
                logger.exception(
                    "AfPool FFN=%s attn=%s get_batch failed: %s",
                    self.ffn_rank,
                    i,
                    e,
                )
                continue
            for batch in batches:
                items.append((tr, batch))
        return items

    def _poll_legacy(self, n: int, timeout_s: float):
        """Pre-fix scan: blocking get_batch per link, in link order."""
        items = []
        for i in range(n):
            if not self._link_ready[i].is_set():
                continue
            tr = self._transports[i]
            if tr is None:
                continue
            try:
                batches = tr.get_batch(timeout_s=timeout_s)
            except Exception as e:
                logger.exception(
                    "AfPool FFN=%s attn=%s get_batch failed: %s",
                    self.ffn_rank,
                    i,
                    e,
                )
                continue
            for batch in batches:
                items.append((tr, batch))
        return items

    def _poll_ready(self, n: int, timeout_s: float):
        """Drain all links, then park once and drain again if nothing was ready.

        The park is a single shared wait rather than one per link, so idle cost
        is O(1) in num_attn instead of O(num_attn) and no link can starve
        another.
        """
        if not self._drain_all:
            return self._poll_legacy(n, timeout_s)
        items = self._poll_once(n)
        if items or timeout_s <= 0:
            return items
        deadline = time.perf_counter() + timeout_s
        while not items and time.perf_counter() < deadline:
            self._park_idle()
            items = self._poll_once(n)
        return items

    def _park_idle(self) -> None:
        # Park on one link so an enabled eventfd still wakes us (off by
        # default); otherwise this is a GIL-yielding yield, matching the
        # transport's own idle path. Cost is O(1) in num_attn either way.
        tr = self._transports[0] if self._transports else None
        if tr is not None:
            try:
                tr.park_idle()
                return
            except Exception:
                pass
        time.sleep(0)

    def utilization(self):
        """FFN-side serve utilisation: ``(busy_s, elapsed_s, frac, tasks)``.

        The honest numerator. ``busy_s`` accumulates serve-loop wall time,
        which includes the post-serve ``torch.cuda.synchronize()`` when
        ``SGLANG_AFD_POOL_SERVE_SYNC`` is on, so it measures time actually
        spent serving hops. Contrast the attn-side round trip, which sums
        overlapping in-flight hops and is clamped to 1.0 — under any
        pipelining it reads 1.000 regardless of real headroom.
        """
        with self._stats_lock:
            busy = float(self.busy_s)
            tasks = int(self.tasks_done)
        elapsed = max(0.0, time.perf_counter() - self._serve_t0)
        frac = min(1.0, busy / elapsed) if elapsed > 0 else 0.0
        return busy, elapsed, frac, tasks

    def _publish_util(self) -> None:
        if not self._util_path:
            return
        busy, elapsed, frac, tasks = self.utilization()
        try:
            with open(self._util_path, "w", encoding="utf-8") as f:
                f.write(f"{busy:.6f},{elapsed:.6f},{frac:.6f},{tasks}\n")
        except Exception:
            pass

    def _serve_loop(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.cuda.set_device(self.device)
            except Exception:
                pass
        from sglang.srt.afd.farm.lpu_sim import dispatch_lpu_batches, extra_gather

        n = self.topo.num_attn
        self._serve_t0 = time.perf_counter()
        while not self._stop.is_set():
            if self._ffn_queue is not None:
                # Never age a ready layer behind an empty polling timeout. If
                # work is already queued, drain it now; only an empty queue
                # waits for new transport arrivals.
                wait_s = 0.0 if not self._ffn_queue.empty() else 0.002
                collected = self._poll_ready(n, wait_s)
                if collected:
                    self._ffn_queue.push(collected)
                collected = self._ffn_queue.pop_ready(
                    force=self._stop.is_set(), stats=self._lpu_stats
                )
                if not collected:
                    continue
            else:
                collected = self._poll_ready(n, 0.002)
                if not collected:
                    time.sleep(0.0005)
                    continue
                if not self._skip_extra_gather:
                    collected = extra_gather(
                        lambda: self._poll_ready(n, 0.0), collected
                    )
            t0 = time.perf_counter()
            try:
                served = dispatch_lpu_batches(
                    self.compute, collected, self._lpu_stats
                )
                self._lpu_stats_calls += 1
                if (
                    torch.cuda.is_available()
                    and served
                    and envs.SGLANG_AFD_POOL_SERVE_SYNC.get()
                ):
                    torch.cuda.synchronize()
            except Exception as e:
                logger.exception(
                    "AfPool FFN=%s LPU dispatch failed n=%s: %s",
                    self.ffn_rank,
                    len(collected),
                    e,
                )
                continue
            dt = time.perf_counter() - t0
            try:
                from sglang.srt.afd.detail_profile import (
                    profile_detail_enabled,
                    record_us,
                )

                if profile_detail_enabled():
                    record_us("serve_us", dt * 1e6)
            except Exception:
                pass
            with self._stats_lock:
                self.tasks_done += max(1, served if served else len(collected))
                self.busy_s += dt
            if self._util_path:
                _now = time.perf_counter()
                if _now - self._util_last >= 0.25:
                    self._util_last = _now
                    self._publish_util()
            if (
                self._lpu_stats_every > 0
                and self._lpu_stats_calls % self._lpu_stats_every == 0
            ):
                if self._ffn_queue is None:
                    logger.info(
                        "AfPool FFN rank=%s LPU stats calls=%s batches=%s tokens=%s "
                        "groups=%s singleton=%s multi=%s fused=%s/%s "
                        "unfused=%s/%s max_group=%s group_hist=%s "
                        "skip_extra_gather=%s",
                        self.ffn_rank,
                        self._lpu_stats.get("calls"),
                        self._lpu_stats.get("batches"),
                        self._lpu_stats.get("tokens"),
                        self._lpu_stats.get("layer_groups"),
                        self._lpu_stats.get("singleton_groups", 0),
                        self._lpu_stats.get("multi_groups", 0),
                        self._lpu_stats.get("fused_groups", 0),
                        self._lpu_stats.get("fused_batches", 0),
                        self._lpu_stats.get("unfused_groups", 0),
                        self._lpu_stats.get("unfused_batches", 0),
                        self._lpu_stats.get("max_group", 0),
                        self._lpu_stats.get("group_hist", {}),
                        self._skip_extra_gather,
                    )
                else:
                    queue_groups = max(1, int(self._lpu_stats.get("queue_groups", 0)))
                    queue_hops = int(self._lpu_stats.get("queue_hops", 0))
                    logger.info(
                        "AfPool FFN rank=%s LPU stats calls=%s batches=%s tokens=%s "
                        "groups=%s singleton=%s multi=%s fused=%s/%s "
                        "unfused=%s/%s max_group=%s group_hist=%s "
                        "queue_depth=%s/%s queue_groups=%s queue_hops=%s "
                        "queue_tokens=%s queue_wait_mean_us=%.1f "
                        "queue_wait_max_us=%.1f queue_group_hist=%s",
                        self.ffn_rank,
                        self._lpu_stats.get("calls"),
                        self._lpu_stats.get("batches"),
                        self._lpu_stats.get("tokens"),
                        self._lpu_stats.get("layer_groups"),
                        self._lpu_stats.get("singleton_groups", 0),
                        self._lpu_stats.get("multi_groups", 0),
                        self._lpu_stats.get("fused_groups", 0),
                        self._lpu_stats.get("fused_batches", 0),
                        self._lpu_stats.get("unfused_groups", 0),
                        self._lpu_stats.get("unfused_batches", 0),
                        self._lpu_stats.get("max_group", 0),
                        self._lpu_stats.get("group_hist", {}),
                        self._ffn_queue.global_hops,
                        self._ffn_queue.max_global_hops,
                        queue_groups,
                        queue_hops,
                        self._lpu_stats.get("queue_tokens", 0),
                        float(self._lpu_stats.get("queue_wait_us_sum", 0.0))
                        / queue_hops,
                        self._lpu_stats.get("queue_wait_us_max", 0.0),
                        self._lpu_stats.get("queue_group_hist", {}),
                    )

    def close(self) -> None:
        self._stop.set()
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=2.0)
        for t in self._accept_threads:
            t.join(timeout=1.0)
        for tr in self._transports:
            if tr is None:
                continue
            try:
                tr.close()
            except Exception:
                pass

    def start_background(self) -> None:
        logger.info(
            "AfPool FFN rank=%s single-thread serve active links_ready=%s",
            self.ffn_rank,
            sum(1 for e in self._link_ready if e.is_set()),
        )

    def run_until_stopped(self) -> None:
        self.start_background()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        finally:
            self.close()


def make_identity_compute(ffn_burn_us: float = 0.0) -> FfnComputeFn:
    def _burn(us: float) -> None:
        if us <= 0:
            return
        deadline = time.perf_counter() + us * 1e-6
        while time.perf_counter() < deadline:
            pass

    def compute(batch: AfdServerBatch) -> Sequence[torch.Tensor]:
        _burn(ffn_burn_us)
        t = batch.num_tokens
        return [batch.hidden[:t].clone()]

    return compute


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "on", "yes")
