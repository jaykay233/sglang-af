# SPDX-License-Identifier: Apache-2.0
"""Fine-grained AFD RT sub-phase stats (post_to_ffn + FFN compute).

Enable with ``SGLANG_AFD_PROFILE_DETAIL=1``. Aggregates host/CUDA timings and
dumps a summary every ``SGLANG_AFD_PROFILE_DETAIL_EVERY`` samples (default 512)
and on ``dump_summary()`` / process exit via atexit when registered.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from typing import Dict, List, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_KEYS = (
    "a2f_sync_us",  # FFN: event.synchronize waiting for Attn A2F
    "post_poll_us",  # FFN: attn_post → mailbox hit (needs timeline post ts)
    "compute_wall_us",  # FFN: host wall around compute()
    "compute_cuda_us",  # FFN: wall + cuda synchronize (kernel+graph)
    "ffn_prep_us",  # FFN: topk/tensor preparation before expert kernels
    "ffn_shared_us",  # FFN: shared-expert projection
    "ffn_routed_us",  # FFN: routed experts wall (dispatch + core + combine)
    "ffn_routed_cpu_us",  # FFN: routed experts process CPU time
    "ffn_routed_thr_us",  # FFN: routed experts calling-thread CPU time
    "ffn_fuse_us",  # FFN: routed scaling + shared-output fusion
    "ffn_post_us",  # FFN: TP all-reduce / shared-expert post-add
    "respond_us",  # FFN: copy F2A + set_done
    "attn_post_issue_us",  # Attn: record_event+set_posted (+optional copy)
    "attn_prep_us",  # Attn: prepare_attn + last-layer capture
    "attn_core_us",  # Attn: self_attn (MLA)
    "attn_prepmlp_us",  # Attn: prepare_mlp
    "attn_route_us",  # Attn: gate + topk (scheme-A routing)
    "attn_gate_us",  # Attn: MoE gate projection
    "attn_topk_us",  # Attn: select_experts / grouped topk
    "mla_qk_norm_us",  # MLA: q_a_layernorm + kv_a_layernorm
    "mla_qb_proj_us",  # MLA: q_b_proj
    "mla_qproj_us",  # MLA: q_proj + kv_a_proj_with_mqa + kv_a_layernorm (no-lora path)
    "mla_qproj_q_us",  # MLA: q_proj alone (no-lora path)
    "mla_qproj_kv_us",  # MLA: kv_a_proj_with_mqa alone (no-lora path)
    "mla_knorm_us",  # MLA: kv_a_layernorm alone (no-lora path)
    "mla_split_pe_us",  # MLA: _split_q_nope_pe
    "mla_qnope_bmm_us",  # MLA: q_nope @ w_kc (absorbed q projection)
    "mla_attn_block_us",  # MLA: rope + kv-cache store + attn_mqa
    "mla_vbmm_us",  # MLA: attn_output @ w_vc + flatten
    "mla_oproj_us",  # MLA: o_proj
    "mla_prepare_us",  # MLA: whole forward_absorb_prepare (sanity bracket)
    "mla_core_us",  # MLA: whole forward_absorb_core (sanity bracket)
    "serve_us",  # FFN pool: dispatch_lpu_batches (compute + respond) wall
    "rtt_us",  # Attn pool: push_pull → wait returned (full round trip)
    # MoE routed internals (progress.md §13): attribute the ~1.6ms host cost.
    "moe_cfg_us",  # config resolve + moe_align_block_size
    "moe_cfgsel_us",  # just try_get_optimal_moe_config
    "moe_align_us",  # just moe_align_block_size
    "moe_alloc_us",  # intermediate/output buffer allocation
    "moe_k1_us",  # gate_up (up-projection) kernel launch
    "moe_act_us",  # activation (silu_and_mul etc.)
    "moe_k2_us",  # down-projection kernel launch
    "moe_disp_us",  # FusedMoE.dispatcher.dispatch
    "moe_core_us",  # quant_method.apply (wraps the kernel sequence)
    "moe_apply_us",  # quant_method.apply only
    "moe_fx_us",  # fused_experts wrapper + inplace custom-op dispatch + impl
    "moe_comb_us",  # dispatcher.combine + contiguous
)


class _Agg:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._xs: Dict[str, List[float]] = {k: [] for k in _KEYS}
        self._n = 0
        self._every = 512
        self._out = ""
        self._registered = False

    def configure(self) -> None:
        self._every = max(32, int(os.environ.get("SGLANG_AFD_PROFILE_DETAIL_EVERY", "512")))
        self._out = os.environ.get("SGLANG_AFD_PROFILE_DETAIL_OUT", "")
        if not self._registered:
            atexit.register(self.dump)
            self._registered = True

    def enabled(self) -> bool:
        try:
            return bool(envs.SGLANG_AFD_PROFILE_DETAIL.get())
        except Exception:
            return os.environ.get("SGLANG_AFD_PROFILE_DETAIL", "0") in (
                "1",
                "true",
                "True",
            )

    def record(self, name: str, us: float) -> None:
        if name not in self._xs or us < 0 or us > 500_000:
            return
        with self._lock:
            self._xs[name].append(float(us))
            self._n += 1
            n = self._n
        if n > 0 and n % self._every == 0:
            self.dump()

    def dump(self) -> str:
        with self._lock:
            snap = {k: list(v) for k, v in self._xs.items()}
        lines = [f"AFD_DETAIL_PROFILE n_records={sum(len(v) for v in snap.values())}"]
        for k in _KEYS:
            xs = snap.get(k) or []
            if not xs:
                lines.append(f"  {k}: n=0")
                continue
            ys = sorted(xs)

            def pct(p: float) -> float:
                i = (len(ys) - 1) * (p / 100.0)
                f = int(i)
                c = min(f + 1, len(ys) - 1)
                return ys[f] + (ys[c] - ys[f]) * (i - f)

            lines.append(
                f"  {k}: n={len(xs)} p50={pct(50):.0f} p90={pct(90):.0f} "
                f"mean={sum(xs)/len(xs):.0f} us"
            )
        # Interpret post_to_ffn / compute split when both present.
        sync = snap.get("a2f_sync_us") or []
        poll = snap.get("post_poll_us") or []
        cwall = snap.get("compute_wall_us") or []
        ccuda = snap.get("compute_cuda_us") or []
        if sync and poll:
            ms, mp = sum(sync) / len(sync), sum(poll) / len(poll)
            lines.append(
                f"  post_to_ffn_split: poll_mean={mp:.0f}us sync_mean={ms:.0f}us "
                f"({100*mp/(mp+ms):.0f}% poll / {100*ms/(mp+ms):.0f}% a2f_event_sync)"
            )
        if cwall and ccuda:
            mw, mc = sum(cwall) / len(cwall), sum(ccuda) / len(ccuda)
            overhead = max(0.0, mc - mw)
            lines.append(
                f"  ffn_compute_split: wall_mean={mw:.0f}us cuda_sync_mean={mc:.0f}us "
                f"launch_overhead≈{overhead:.0f}us "
                f"(cuda includes kernels queued in wall)"
            )
            if mc > 0:
                lines.append(
                    f"  verdict: {'SYNC_BOUND_post_to_ffn' if (sync and sum(sync)/len(sync) > mc * 0.8) else 'COMPUTE_OR_LAUNCH'}"
                )
        text = "\n".join(lines)
        logger.info("\n%s", text)
        if self._out:
            try:
                with open(self._out, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
            except Exception:
                pass
        return text


_AGG = _Agg()


def profile_detail_enabled() -> bool:
    if not _AGG.enabled():
        return False
    _AGG.configure()
    return True


def record_us(name: str, us: float) -> None:
    if profile_detail_enabled():
        _AGG.record(name, us)


def record_span(name: str, t0: float, t1: Optional[float] = None) -> float:
    t1 = time.perf_counter() if t1 is None else t1
    us = (t1 - t0) * 1e6
    record_us(name, us)
    return t1


def dump_detail_summary() -> str:
    if not _AGG.enabled():
        return "AFD_DETAIL_PROFILE disabled"
    _AGG.configure()
    return _AGG.dump()
