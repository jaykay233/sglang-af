# SPDX-License-Identifier: Apache-2.0
"""Intra-step decode farm: tokens at different layers, B_step windows.

P0: 1A + 1 LPU-sim via existing AFD hops (cuda_ipc mailbox poll).
P1: AfPool credit + sticky same-layer send so FFN can gather.
P2: B_win sticky + coalesce K×B_step (true stock-kernel weight amortize).
P3: per-layer CUDAGraph replay (launch amortize; soft tax optional).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.afd.farm.attn_farm import (
    FarmHop,
    flush_hop_queues,
    poll_hop,
    try_issue_hop,
    wait_hop,
)
from sglang.srt.afd.farm.batch_slice import (
    decode_token_num_per_seq,
    scatter_rows,
    slice_decode_window,
)
from sglang.srt.afd.farm.env import (
    farm_b_step,
    farm_b_win_k,
    farm_coalesce_k,
    farm_contexts_per_stage,
    farm_context_stagger_layers,
    farm_layer_burst,
    farm_max_inflight,
    farm_max_inflight_per_layer,
    farm_num_contexts,
    farm_persistent_enabled,
    farm_persistent_groups,
    farm_persistent_poll_us,
    farm_sched,
    farm_wait_slice_us,
)
from sglang.srt.afd.farm.layer_cuda_graph import run_layer_forward_pre_ffn
from sglang.srt.afd.farm.scheduler import (
    context_stage_ready,
    get_continuous_scheduler,
    plan_context_ranges,
)
from sglang.srt.afd.farm.soft_persistent import maybe_weight_outer_tax
from sglang.srt.afd.remote_policy import afd_should_remote_ffn
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
except Exception:  # pragma: no cover

    def eager_on_graph(_enable: bool):
        def _deco(fn):
            return fn

        return _deco

logger = logging.getLogger(__name__)

_farm_forwards = 0
_farm_step_seq = 0
_occ_sum: List[int] = []
_occ_samples = 0
_peak_layers_busy = 0

# Phase wall-clock accumulators (CPU side). Gated by SGLANG_AFD_FARM_PHASE_TIMING.
_phase: dict = {}


def _phase_timing_enabled() -> bool:
    try:
        return bool(envs.SGLANG_AFD_FARM_PHASE_TIMING.get())
    except Exception:
        return False


def _note_phase(name: str, dt: float) -> None:
    acc = _phase.get(name)
    if acc is None:
        _phase[name] = [dt, 1]
    else:
        acc[0] += dt
        acc[1] += 1


def _log_phase(n_layers: int) -> None:
    if not _phase:
        return
    tot = _phase.get("loop_total", [0.0, 1])
    steps = max(1, tot[1])
    parts = []
    for k in (
        "pre",
        "issue",
        "drain_poll",
        "drain_block",
        "consume",
        "consume_remote",
    ):
        acc = _phase.get(k)
        if not acc:
            continue
        parts.append(f"{k}={acc[0] / acc[1] * 1e3:.2f}ms/call")
    cpu_sum = sum(
        _phase[k][0]
        for k in ("pre", "issue", "drain_poll", "drain_block", "consume")
        if k in _phase
    )
    logger.info(
        "AFD farm PHASE step=%s loop=%.1fms cpu_sum=%.1fms unattributed=%.1fms | %s",
        steps,
        tot[0] / steps * 1e3,
        cpu_sum / steps * 1e3,
        (tot[0] - cpu_sum) / steps * 1e3,
        " ".join(parts),
    )
    _phase.clear()


# --- Stage/concurrency stats: is the farm actually pipelining? -------------
# Gated by SGLANG_AFD_FARM_STAGE_STATS_EVERY (log every N decode forwards).
# Answers: how many A2F hops are in flight at once, do completions come back in
# layer order (lockstep) or interleaved (pipelined), and where does the host
# wall-clock go (attn pre / block-on-hop / consume).
_STAGE: dict = {}
_CAP_PREV: tuple = (0, 0)
_STAGE_KEYS = (
    "forwards",
    "hops",
    "issue_ok",
    "issue_nocredit",
    "pending_peak",
    "layers_peak",
    "span_peak",
    "ctxs_peak",
    "ctx_span_peak",
    "block_calls",
    "block_s",
    "wall_s",
    "pre_calls",
    "pre_s",
    "lat_sum",
    "lat_n",
    "order_inv",
    "issue_iter",
    "reserve_none",
    "hop_none",
    "layer_cap_blocks",
    "global_cap_blocks",
    "adopted_groups",
    "adopted_rows",
    "run_size_hist",
)


def _stage_enabled() -> bool:
    try:
        return int(os.environ.get("SGLANG_AFD_FARM_STAGE_STATS_EVERY", "0") or 0) > 0
    except Exception:
        return False


def _stage_every() -> int:
    try:
        return max(0, int(os.environ.get("SGLANG_AFD_FARM_STAGE_STATS_EVERY", "0") or 0))
    except Exception:
        return 0


def _stage_bump(key: str, value: float = 1.0) -> None:
    _STAGE[key] = _STAGE.get(key, 0.0) + float(value)


def _stage_peak(key: str, value: float) -> None:
    if value > _STAGE.get(key, 0.0):
        _STAGE[key] = float(value)


def _stage_reset() -> None:
    _STAGE.clear()
    for k in _STAGE_KEYS:
        _STAGE[k] = 0.0
    _STAGE["lat"] = []
    _STAGE["order_last"] = -1
    _STAGE["layer_min"] = -1
    _STAGE["layer_max"] = -1


def _stage_log(n_layers: int, n_seq: int) -> None:
    st = _STAGE
    fwd = max(1.0, st.get("forwards", 0.0))
    wall = max(1e-9, st.get("wall_s", 0.0))
    lat = sorted(st.get("lat", []))

    def _pct(p: float) -> float:
        if not lat:
            return 0.0
        i = min(len(lat) - 1, int(p * len(lat)))
        return lat[i] * 1e3

    avg_inflight = st.get("lat_sum", 0.0) / wall
    logger.info(
        "AFD farm STAGE forwards=%s rows=%s hops/fwd=%.1f issue_ok=%.1f "
        "issue_nocredit=%.1f reserve_none=%.1f hop_none=%.1f pending_peak=%s live_peak=%s layers_peak=%s/%s span_peak=%s "
        "ctxs_peak=%s ctx_span_peak=%s sched_peaks=%s/%s/%s "
        "grp=%.1f/%.1f peak=%s "
        "capblk/fwd=%.2f/%.2f "
        "avg_inflight=%.2f hop_lat_p50=%.2fms p90=%.2fms "
        "order_inv/fwd=%.1f | wall=%.1fms/head pre=%.2fms/call(%s) "
        "block=%.2fms/call(%s)",
        int(st.get("forwards", 0.0)),
        int(n_seq),
        st.get("hops", 0.0) / fwd,
        st.get("issue_ok", 0.0) / fwd,
        st.get("issue_nocredit", 0.0) / fwd,
        st.get("reserve_none", 0.0) / fwd,
        st.get("hop_none", 0.0) / fwd,
        int(st.get("pending_peak", 0.0)),
        int(st.get("live_peak", 0.0)),
        int(st.get("layers_peak", 0.0)),
        n_layers,
        int(st.get("span_peak", 0.0)),
        int(st.get("ctxs_peak", 0.0)),
        int(st.get("ctx_span_peak", 0.0)),
        int(st.get("run_peak", 0.0)),
        int(st.get("res_peak", 0.0)),
        int(st.get("sctx_peak", 0.0)),
        st.get("adopted_groups", 0.0) / fwd,
        st.get("adopted_rows", 0.0) / fwd,
        int(st.get("group_size_peak", 0.0)),
        st.get("layer_cap_blocks", 0.0) / fwd,
        st.get("global_cap_blocks", 0.0) / fwd,
        avg_inflight,
        _pct(0.5),
        _pct(0.9),
        st.get("order_inv", 0.0) / fwd,
        wall / fwd * 1e3,
        (st.get("pre_s", 0.0) / max(1.0, st.get("pre_calls", 0.0))) * 1e3,
        int(st.get("pre_calls", 0.0) / fwd),
        (st.get("block_s", 0.0) / max(1.0, st.get("block_calls", 0.0))) * 1e3,
        int(st.get("block_calls", 0.0) / fwd),
    )
    _stage_reset()


def _reset_occ(n_layers: int) -> None:
    global _occ_sum, _occ_samples, _peak_layers_busy
    if len(_occ_sum) != n_layers:
        _occ_sum = [0] * n_layers
        _occ_samples = 0
        _peak_layers_busy = 0


def _note_occ(ready: Sequence[int]) -> None:
    global _occ_samples, _peak_layers_busy
    n = len(ready)
    if len(_occ_sum) != n:
        _reset_occ(n)
    busy = 0
    for i in range(n):
        # ``occupancy`` already includes both waiting and running work.
        v = int(ready[i])
        _occ_sum[i] += v
        if v > 0:
            busy += 1
    _peak_layers_busy = max(_peak_layers_busy, busy)
    _occ_samples += 1


def _maybe_log_occ(n_layers: int) -> None:
    global _farm_forwards
    every = int(envs.SGLANG_AFD_FARM_LOG_EVERY.get() or 0)
    if every <= 0 or _farm_forwards % every != 0 or _occ_samples <= 0:
        return
    mean = [round(_occ_sum[i] / _occ_samples, 2) for i in range(n_layers)]
    logger.info(
        "AFD farm occupancy mean_tokens/layer=%s peak_layers_busy=%s samples=%s "
        "B_step=%s coalesce_k=%s layer_burst=%s sched=%s max_inflight=%s B_win_k=%s",
        mean,
        _peak_layers_busy,
        _occ_samples,
        farm_b_step(),
        farm_coalesce_k(),
        farm_layer_burst(),
        farm_sched(),
        farm_max_inflight(),
        farm_b_win_k(),
    )


def _mlp_out(raw) -> torch.Tensor:
    if isinstance(raw, tuple):
        return raw[0]
    return raw


def _consume_hop(
    hop: FarmHop,
    mlp_out: torch.Tensor,
    layers: Sequence[Any],
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    topk_store: Dict[Tuple[int, int, int], Dict[int, torch.Tensor]],
    scheduler,
    on_output_ready: Optional[Callable[[FarmHop, int], None]] = None,
) -> Tuple[Optional[torch.Tensor], int]:
    """post_ffn + scatter; enqueue next layer. Returns (residual, finished_delta)."""
    if not hop.sched_ticket:
        raise RuntimeError("farm hop is missing its scheduler ticket")
    layer = layers[hop.layer_i]
    hs, res, tk = layer.forward_post_ffn(mlp_out, hop.residual, hop.meta)
    scatter_rows(hidden_states, hs, hop.token_lo, hop.token_hi)
    if res is not None:
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        scatter_rows(residual, res, hop.token_lo, hop.token_hi)
    tps = max(1, (hop.token_hi - hop.token_lo) // max(1, len(hop.seq_idxs)))
    if tk is not None and tk.shape[0] == hop.token_hi - hop.token_lo:
        store = topk_store[hop.owner]
        for j, s in enumerate(hop.seq_idxs):
            lo = j * tps
            store[int(s)] = tk[lo : lo + tps]
    finished = scheduler.complete(hop.sched_ticket)
    hop.sched_ticket = 0
    if finished:
        if on_output_ready is not None:
            on_output_ready(hop, finished)
        scheduler.mark_context_sampled(hop.ctx_id, finished)
    return residual, finished


def _prev_topk(
    topk_store: Dict[Tuple[int, int, int], Dict[int, torch.Tensor]],
    owner: Tuple[int, int, int],
    seq_idxs: Sequence[int],
) -> Optional[torch.Tensor]:
    store = topk_store.get(tuple(int(x) for x in owner), {})
    chunks = [store.get(int(s)) for s in seq_idxs]
    if not chunks or any(chunk is None for chunk in chunks):
        return None
    return torch.cat(chunks, dim=0)


def _slice_logits_output(output: Any, row: int) -> Any:
    if output is None or not dataclasses.is_dataclass(output):
        return output
    updates: Dict[str, Any] = {}
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if isinstance(value, torch.Tensor) and value.shape and value.shape[0] > row:
            updates[field.name] = value[row : row + 1]
        elif isinstance(value, list) and len(value) > row:
            updates[field.name] = value[row : row + 1]
    return dataclasses.replace(output, **updates)


def _merge_logits_outputs(outputs: Sequence[Any]) -> Any:
    if not outputs:
        return None
    first = outputs[0]
    if first is None or not dataclasses.is_dataclass(first):
        return first

    updates: Dict[str, Any] = {}
    for field in dataclasses.fields(first):
        values = [getattr(output, field.name) for output in outputs]
        if all(value is None for value in values):
            updates[field.name] = None
            continue
        if all(isinstance(value, torch.Tensor) and value.shape for value in values):
            if all(value.shape[0] == 1 for value in values):
                updates[field.name] = torch.cat(values, dim=0)
                continue
        if all(isinstance(value, list) and len(value) == 1 for value in values):
            merged: List[Any] = []
            for value in values:
                merged.extend(value)
            updates[field.name] = merged
            continue

        # Non-row-aligned diagnostics are not safe to concatenate. The first
        # context still exposes them, while batch-shaped fields are merged.
        updates[field.name] = values[0]
    return dataclasses.replace(first, **updates)


def _row_of_logits(output: Any, row: int, n_rows: int) -> Any:
    """Extract one row from a multi-row ``LogitsProcessorOutput``.

    ``_merge_logits_outputs`` assumes one-row inputs (it only concatenates
    ``shape[0] == 1`` tensors). A grouped context produces ``n_rows`` rows at
    once, so each member needs its own single-row view first.
    """
    if output is None or not dataclasses.is_dataclass(output):
        return output
    updates: Dict[str, Any] = {}
    for field in dataclasses.fields(output):
        value = getattr(output, field.name)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            if int(value.shape[0]) == int(n_rows):
                updates[field.name] = value[row : row + 1]
                continue
        elif isinstance(value, list) and len(value) == int(n_rows):
            updates[field.name] = [value[row]]
            continue
        updates[field.name] = value
    return dataclasses.replace(output, **updates)


def _persistent_stage_snapshot(runtime, scheduler) -> None:
    """Record cross-forward overlap counters for the persistent path.

    ``layers_peak`` / ``pending_peak`` here are sampled across *forwards*, which
    is the point of the persistent runtime: the one-shot farm can only ever see
    the layers of a single forward.
    """
    if not _stage_enabled():
        return
    hops = [ctx.pending_hop for ctx in runtime.live_contexts() if ctx.pending_hop]
    _stage_peak("live_peak", float(len(runtime.live_contexts())))
    _stage_peak("pend_live_peak", float(len(hops) + len(scheduler._reserved)))
    _stage_peak("run_peak", float(len(scheduler._running)))
    _stage_peak("res_peak", float(len(scheduler._reserved)))
    _stage_peak("sctx_peak", float(len(scheduler._contexts)))
    global _CAP_PREV
    cur = (scheduler.stats.layer_cap_blocks, scheduler.stats.global_cap_blocks)
    _stage_bump("layer_cap_blocks", float(cur[0] - _CAP_PREV[0]))
    _stage_bump("global_cap_blocks", float(cur[1] - _CAP_PREV[1]))
    _CAP_PREV = cur
    _stage_peak("pending_peak", float(len(hops)))
    layers = {int(hop.layer_i) for hop in hops}
    _stage_peak("layers_peak", float(len(layers)))
    if layers:
        _stage_peak("span_peak", float(max(layers) - min(layers)))
    by_ctx: Dict[int, set] = defaultdict(set)
    for hop in hops:
        by_ctx[int(hop.ctx_id)].add(int(hop.layer_i))
    _stage_peak("ctxs_peak", float(len(by_ctx)))
    span = 0
    ctx_ids = list(by_ctx)
    for i, left in enumerate(ctx_ids):
        for right in ctx_ids[i + 1 :]:
            for ll in by_ctx[left]:
                for rl in by_ctx[right]:
                    span = max(span, abs(ll - rl))
    _stage_peak("ctx_span_peak", float(span))
    try:
        _note_occ(scheduler.occupancy())
    except Exception:
        pass


def _persistent_poll_and_advance(
    *,
    runtime,
    scheduler,
    layer_list,
    context_sampler,
    final_norm,
    sampled: List[Tuple[int, torch.Tensor, Any]],
    block: bool = False,
) -> int:
    """Non-blocking poll of every live hop; advance contexts that completed.

    Returns the number of hops consumed.
    """
    consumed = 0
    for ctx in runtime.live_contexts():
        hop = ctx.pending_hop
        if hop is None:
            continue
        out = wait_hop(hop) if block else poll_hop(hop)
        if out is None:
            continue
        consumed += _persistent_advance(
            ctx=ctx,
            hop=hop,
            mlp_out=_mlp_out(out),
            runtime=runtime,
            scheduler=scheduler,
            layer_list=layer_list,
            context_sampler=context_sampler,
            final_norm=final_norm,
            sampled=sampled,
        )
    return consumed


def _persistent_advance(
    *,
    ctx,
    hop: FarmHop,
    mlp_out: torch.Tensor,
    runtime,
    scheduler,
    layer_list,
    context_sampler,
    final_norm,
    sampled: List[Tuple[int, torch.Tensor, Any]],
) -> int:
    """post_ffn on the context's owned tensors, then layer-advance or sample.

    This is the split the design calls for: ``post_ffn + scheduler.complete``
    completes one layer; sampling and context close are a separate stage that
    only runs once the final layer is done.
    """
    if not hop.sched_ticket:
        raise RuntimeError("farm hop is missing its scheduler ticket")
    ticket_id = hop.sched_ticket
    layer = layer_list[hop.layer_i]
    hs, res, tk = layer.forward_post_ffn(mlp_out, hop.residual, hop.meta)
    ctx.hidden = hs
    if res is not None:
        # Mirror the one-shot scatter: a None residual means the pre-FFN
        # residual carries forward unchanged.
        ctx.residual = res
    ctx.prev_topk = tk
    ctx.current_layer = int(hop.layer_i) + 1
    hop.sched_ticket = 0
    ctx.pending_hop = None

    finished = scheduler.complete(ticket_id)
    if not finished:
        return 0

    # Final layer done: sample on the context's own hidden/residual.
    if context_sampler is not None:
        hidden = ctx.hidden
        residual = ctx.residual
        if final_norm is not None:
            if residual is None:
                hidden = final_norm(hidden)
            else:
                hidden, _ = final_norm(hidden, residual)
            sample_residual = None
        else:
            sample_residual = residual
        logits_output, next_token_ids = context_sampler(
            hidden, sample_residual, ctx.child_fb
        )
        if int(next_token_ids.shape[0]) != finished:
            raise RuntimeError(
                "AFD farm sampler row count does not match ready tokens: "
                f"rows={int(next_token_ids.shape[0])} finished={finished}"
            )
        ctx.note_sample(next_token_ids, logits_output)
        # The group's rows finish together but the result protocol is keyed by
        # req_pool_idx, so emit one entry per member. ``_merge_logits_outputs``
        # later concatenates the single-row slices back into one output whose
        # order matches the ready list.
        members = ctx.members
        if len(members) != int(next_token_ids.shape[0]):
            raise RuntimeError(
                "AFD farm group/sampler row mismatch: "
                f"members={len(members)} rows={int(next_token_ids.shape[0])}"
            )
        for row, member in enumerate(members):
            sampled.append(
                (
                    member,
                    next_token_ids[row : row + 1],
                    _row_of_logits(logits_output, row, len(members)),
                )
            )

    scheduler.mark_context_sampled(ctx.ctx_id, finished)
    scheduler.finish_context(ctx.ctx_id)
    runtime.drop(ctx.req_pool_idx)
    return int(finished)


def _run_persistent(
    *,
    layer_list: Sequence[Any],
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
    zero_allocator,
    gemm_output_zero_allocator,
    llama_4_scaling: Optional[torch.Tensor],
    layers_to_capture: Optional[Sequence[int]],
    aux_hidden_states: Optional[List[torch.Tensor]],
    context_sampler,
    final_norm,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Cross-forward farm drive.

    Adopts rows that have no live context, advances every context that is
    already in flight, issues as much new work as transport credit allows, and
    returns the subset sampled *this* forward. Never waits for the whole batch:
    a forward may return zero tokens and the rest stay live in the runtime.
    """
    from sglang.srt.afd.farm.persistent_runtime import get_persistent_runtime

    runtime = get_persistent_runtime()
    runtime.note_forward()
    stage_on = _stage_enabled()
    _t_stage_0 = time.perf_counter() if stage_on else 0.0

    n_layers = len(layer_list)
    tps = decode_token_num_per_seq(forward_batch)
    if tps is None or n_layers <= 0:
        raise RuntimeError(
            "AFD farm persistent mode requires a uniform decode batch "
            f"(tps={tps}, n_layers={n_layers})"
        )

    b_step = farm_b_step()
    coalesce_k = farm_coalesce_k()
    layer_burst = farm_layer_burst()
    max_inf = farm_max_inflight()
    per_layer_cap = farm_max_inflight_per_layer()
    poll_us = farm_persistent_poll_us()
    n_groups_cfg = farm_persistent_groups()

    scheduler = get_continuous_scheduler(
        n_layers,
        device_key=str(hidden_states.device),
        b_win_k=farm_b_win_k(),
        coalesce_k=coalesce_k,
        layer_burst=layer_burst,
        sched=farm_sched(),
    )

    req_pool_indices = forward_batch.req_pool_indices
    if not isinstance(req_pool_indices, torch.Tensor):
        raise RuntimeError("AFD farm persistent mode requires tensor req_pool_indices")
    row_keys = [int(v) for v in req_pool_indices.to("cpu").tolist()]
    in_batch = set(row_keys)

    # 1. Drop contexts whose request left the running batch (abort/finish).
    for key in list(runtime.deferred_req_pool_indices()):
        if key in in_batch:
            continue
        ctx = runtime.get(key)
        if ctx is not None and ctx.ctx_id:
            scheduler.abort_batch(ctx.ctx_id)
        runtime.drop(key)
        logger.debug("AFD farm persistent dropped orphan context req_pool_idx=%s", key)

    # 2. ADOPT the rows with no live context: these are tokens sampled in an
    #    earlier forward whose next token only now enters the farm.
    #
    #    Admission is by *contiguous run of pending rows* capped at
    #    ``group_size``, so every pending row is placed this forward (rows that
    #    were already marked ready must produce a token or their KV/seq_len
    #    advance would silently drift). Runs break at live rows, because a live
    #    row's slot in the parent batch holds placeholder hidden states.
    #
    #    Grouping is what keeps an A2F hop fat: the FFN charges a fixed host
    #    dispatch per call, so `rows` one-token calls lose to `rows/group_size`
    #    fat ones. ``G=1`` -> one fat chain; one row per group -> the thin
    #    extreme.
    n_groups = n_groups_cfg
    group_size = 1 if n_groups <= 0 else max(1, -(-len(row_keys) // n_groups))
    pending_rows: List[int] = [
        seq_idx for seq_idx, key in enumerate(row_keys) if key not in runtime
    ]
    runs: List[List[int]] = []
    for seq_idx in pending_rows:
        if runs and runs[-1][-1] == seq_idx - 1 and len(runs[-1]) < group_size:
            runs[-1].append(seq_idx)
        else:
            runs.append([seq_idx])
    if stage_on:
        _stage_bump("adopted_groups", float(len(runs)))
        _stage_bump("adopted_rows", float(len(pending_rows)))
        _stage_peak("group_size_peak", float(max((len(r) for r in runs), default=0)))
        _stage_peak("hops_per_call_peak", float(max((len(r) for r in runs), default=0)))

    for run in runs:
        key = row_keys[run[0]]
        members = [row_keys[r] for r in run]
        sl = slice_decode_window(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            forward_batch=forward_batch,
            seq_idxs=run,
            token_num_per_seq=tps,
        )
        if sl is None:
            raise RuntimeError(
                f"AFD farm persistent could not slice rows {run} for "
                f"req_pool_idx={key}"
            )
        # The scheduler key must be stable across forwards: use req_pool_idx,
        # not the batch row (rows are reused and shift when reqs leave).
        # The *queue* index is a separate dense block: the farm identifies a run
        # by consecutive indices, and pool indices are sparse.
        queue_idxs = runtime.alloc_queue_idxs(len(members))
        ctx_id = scheduler.begin_batch(queue_idxs, mb_id=0)
        ctx = runtime.adopt(
            req_pool_idx=key,
            req_pool_idxs=members,
            hidden=sl.hidden_states,
            residual=sl.residual,
            positions=sl.positions,
            child_fb=sl.forward_batch,
        )
        runtime.bind_ctx_id(key, ctx_id)
        ctx.current_layer = 0

    sampled: List[Tuple[int, torch.Tensor, Any]] = []

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    def _capture_for(layer):
        if (
            layers_to_capture is not None
            and aux_hidden_states is not None
            and layer.layer_id in layers_to_capture
        ):
            return aux_hidden_states
        return None

    def _issue_round() -> int:
        """Reserve + run pre-FFN + issue hops until credit or work runs out."""
        issued = 0
        queued: List[Tuple[Any, FarmHop]] = []
        # A ticket must cover *every* member of the context it is advancing: a
        # context owns one set of tensors, so a partial ticket would leave the
        # remaining rows to advance independently and overwrite the context's
        # hidden with a different row count. Reserve at least the largest live
        # group so `take_contiguous_run` can never truncate one.
        reserve_step = max(
            [group_size, 1] + [ctx.n_tokens for ctx in runtime.live_contexts()]
        )
        # A rolled-back ticket returns its keys to the ready queue, so a
        # transport that keeps refusing credit would spin forever here. Bail
        # out after a few consecutive refusals and retry next forward.
        stall = 0
        while len(runtime.pending_hops()) + len(queued) < max_inf:
            if stage_on:
                _stage_bump("issue_iter", 1.0)
            ticket = scheduler.reserve(
                reserve_step,
                max_inflight=max_inf,
                max_inflight_per_layer=per_layer_cap,
            )
            if ticket is None:
                if stage_on:
                    _stage_bump("reserve_none", 1.0)
                break
            ctx = runtime.get_by_ctx(ticket.ctx_id)
            if ctx is None:
                scheduler.rollback(ticket.ticket_id)
                raise RuntimeError(
                    f"AFD farm ticket {ticket.ticket_id} has no live context "
                    f"{ticket.ctx_id}"
                )
            layer = layer_list[ticket.layer_i]
            maybe_weight_outer_tax(
                layer_id=int(layer.layer_id),
                win_index=int(ticket.win_index),
                device=hidden_states.device,
            )
            hidden_a2f, res_s, topk_ids, topk_weights, meta = run_layer_forward_pre_ffn(
                layer,
                positions=ctx.positions,
                hidden_states=ctx.hidden,
                forward_batch=ctx.child_fb,
                residual=ctx.residual,
                zero_allocator=zero_allocator,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=ctx.prev_topk,
                captured_last_layer_outputs=_capture_for(layer),
            )
            # The context owns its residual: no scatter, no shared batch buffer.
            ctx.residual = res_s

            is_moe = isinstance(getattr(layer, "mlp", None), DeepseekV2MoE)
            remote = afd_should_remote_ffn(int(layer.layer_id), is_moe=is_moe)
            if not remote:
                mlp_out = layer.mlp(
                    hidden_a2f,
                    ctx.child_fb,
                    meta["should_allreduce_fusion"],
                    meta["use_reduce_scatter"],
                    meta["gemm_output_zero_allocator"],
                )
                hop = FarmHop(
                    layer_i=ticket.layer_i,
                    seq_idxs=list(ctx.members),
                    token_lo=0,
                    token_hi=ctx.n_tokens,
                    residual=ctx.residual,
                    meta=meta,
                    batch_id=ticket.ctx_id,
                    sched_ticket=ticket.ticket_id,
                    ctx_id=ticket.ctx_id,
                    mb_id=ticket.mb_id,
                    step_id=ticket.step_id,
                )
                scheduler.commit(ticket.ticket_id)
                _persistent_advance(
                    ctx=ctx,
                    hop=hop,
                    mlp_out=mlp_out,
                    runtime=runtime,
                    scheduler=scheduler,
                    layer_list=layer_list,
                    context_sampler=context_sampler,
                    final_norm=final_norm,
                    sampled=sampled,
                )
                issued += 1
                if stage_on:
                    _stage_bump("hops", 1.0)
                stall = 0
                continue

            hop = try_issue_hop(
                layer_id=int(layer.layer_id),
                hidden=hidden_a2f,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                layer_i=ticket.layer_i,
                seq_idxs=list(ctx.members),
                token_lo=0,
                token_hi=ctx.n_tokens,
                residual=ctx.residual,
                meta=meta,
                req_id=ticket.ticket_id,
                owner=(ticket.ctx_id, ticket.mb_id, ticket.step_id),
                defer_flush=True,
            )
            if hop is None:
                if stage_on:
                    _stage_bump("hop_none", 1.0)
                scheduler.rollback(ticket.ticket_id)
                stall += 1
                if stall >= 4:
                    break
                continue
            stall = 0
            hop.batch_id = ticket.ctx_id
            hop.sched_ticket = ticket.ticket_id
            queued.append((ctx, hop))

        if queued:
            flush_hop_queues()
            for ctx, hop in queued:
                if not (
                    hop.send_group is not None
                    or hop.pool_pid is not None
                    or hop.classic is not None
                ):
                    if hop.send_queue is not None:
                        hop.send_queue.cancel(hop)
                    scheduler.rollback(hop.sched_ticket)
                    hop.sched_ticket = 0
                    continue
                if not scheduler.commit(hop.sched_ticket):
                    raise RuntimeError(
                        f"AFD farm persistent lost ticket {hop.sched_ticket}"
                    )
                ctx.pending_hop = hop
                issued += 1
                if stage_on:
                    _stage_bump("hops", 1.0)
        return issued

    _persistent_poll_and_advance(
        runtime=runtime,
        scheduler=scheduler,
        layer_list=layer_list,
        context_sampler=context_sampler,
        final_norm=final_norm,
        sampled=sampled,
    )
    _issue_round()
    _persistent_poll_and_advance(
        runtime=runtime,
        scheduler=scheduler,
        layer_list=layer_list,
        context_sampler=context_sampler,
        final_norm=final_norm,
        sampled=sampled,
    )

    if poll_us > 0 and runtime.pending_hops():
        deadline = time.perf_counter() + poll_us * 1e-6
        while time.perf_counter() < deadline:
            if (
                _persistent_poll_and_advance(
                    runtime=runtime,
                    scheduler=scheduler,
                    layer_list=layer_list,
                    context_sampler=context_sampler,
                    final_norm=final_norm,
                    sampled=sampled,
                )
                > 0
            ):
                break
            time.sleep(0.00005)

    # 3. Result protocol: only rows sampled this forward are ready.
    forward_batch._afd_farm_handled = True
    if sampled:
        order = {key: i for i, key in enumerate(row_keys)}
        sorted_sampled = sorted(
            sampled, key=lambda item: order.get(item[0], len(row_keys))
        )
        merged_logits = _merge_logits_outputs(
            [logits for _, _, logits in sorted_sampled]
        )
        merged_next_ids = torch.cat(
            [tokens for _, tokens, _ in sorted_sampled], dim=0
        )
        forward_batch._afd_farm_logits_output = merged_logits
        forward_batch._afd_farm_next_token_ids = merged_next_ids
        ready_keys = [key for key, _, _ in sorted_sampled]
    else:
        forward_batch._afd_farm_logits_output = None
        forward_batch._afd_farm_next_token_ids = None
        ready_keys = []
    forward_batch._afd_farm_ready_req_pool_indices = ready_keys
    forward_batch._afd_farm_deferred_req_pool_indices = (
        runtime.deferred_req_pool_indices()
    )

    if stage_on:
        global _farm_forwards
        _persistent_stage_snapshot(runtime, scheduler)
        _stage_bump("forwards", 1.0)
        _stage_bump("wall_s", time.perf_counter() - _t_stage_0)
        every = _stage_every()
        if every > 0 and int(_STAGE.get("forwards", 0.0)) % every == 0:
            _stage_log(n_layers, len(row_keys))
        _farm_forwards += 1

    return hidden_states, residual, None


@eager_on_graph(True)
def run_farm_layers(
    layers: Sequence[Any],
    *,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
    zero_allocator,
    gemm_output_zero_allocator=None,
    llama_4_scaling: Optional[torch.Tensor] = None,
    layers_to_capture: Optional[Sequence[int]] = None,
    aux_hidden_states: Optional[List[torch.Tensor]] = None,
    context_sampler: Optional[
        Callable[
            [torch.Tensor, Optional[torch.Tensor], ForwardBatch],
            Tuple[Any, torch.Tensor],
        ]
    ] = None,
    final_norm: Optional[Callable[..., Any]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Farm decode over ``layers``. Falls back to sequential on split failure."""

    def _sequential():
        topk_indices = None
        hs, res = hidden_states, residual
        for layer in layers:
            capture = None
            if (
                layers_to_capture is not None
                and aux_hidden_states is not None
                and layer.layer_id in layers_to_capture
            ):
                capture = aux_hidden_states
            hs, res, topk_indices = layer(
                positions,
                hs,
                forward_batch,
                res,
                zero_allocator,
                gemm_output_zero_allocator,
                llama_4_scaling,
                prev_topk_indices=topk_indices,
                captured_last_layer_outputs=capture,
            )
        return hs, res, topk_indices

    tps = decode_token_num_per_seq(forward_batch)
    if tps is None:
        return _sequential()

    layer_list = list(layers)
    n_layers = len(layer_list)
    if n_layers <= 0:
        return hidden_states, residual, None

    if farm_persistent_enabled():
        return _run_persistent(
            layer_list=layer_list,
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            residual=residual,
            zero_allocator=zero_allocator,
            gemm_output_zero_allocator=gemm_output_zero_allocator,
            llama_4_scaling=llama_4_scaling,
            layers_to_capture=layers_to_capture,
            aux_hidden_states=aux_hidden_states,
            context_sampler=context_sampler,
            final_norm=final_norm,
        )

    n_seq = int(forward_batch.batch_size)
    b_step = min(farm_b_step(), n_seq)
    coalesce_k = farm_coalesce_k()
    layer_burst = farm_layer_burst()
    max_inf = farm_max_inflight()
    per_layer_cap = farm_max_inflight_per_layer()
    num_contexts = farm_num_contexts()
    contexts_per_stage = farm_contexts_per_stage()
    context_stagger_layers = farm_context_stagger_layers()
    scheduler = get_continuous_scheduler(
        n_layers,
        device_key=str(hidden_states.device),
        b_win_k=farm_b_win_k(),
        coalesce_k=coalesce_k,
        layer_burst=layer_burst,
        sched=farm_sched(),
    )
    if not scheduler.idle():
        logger.warning(
            "AFD farm aborting stale scheduler batch=%s running=%s",
            scheduler.batch_id,
            scheduler.running_count,
        )
        scheduler.abort_batch()
    global _farm_step_seq
    _farm_step_seq += 1
    step_id = _farm_step_seq
    context_ranges = plan_context_ranges(
        n_seq,
        b_step=b_step,
        num_contexts=num_contexts,
    )
    if not context_ranges:
        return _sequential()
    context_ids: List[int] = []
    next_context_idx = 0

    def _start_next_context() -> bool:
        nonlocal next_context_idx
        if next_context_idx >= len(context_ranges):
            return False

        active_depths = [
            scheduler.context_deepest_layer(cid)
            for cid in context_ids
            if not scheduler.context_output_ready(cid)
        ]
        if not context_stage_ready(
            next_context_idx=next_context_idx,
            contexts_per_stage=contexts_per_stage,
            context_stagger_layers=context_stagger_layers,
            active_depths=active_depths,
            num_layers=n_layers,
            max_inflight=max_inf,
        ):
            return False

        ctx_idx = next_context_idx
        seq_lo, seq_hi = context_ranges[ctx_idx]
        ctx_id = ctx_idx + 1
        scheduler.begin_batch(
            range(seq_lo, seq_hi),
            ctx_id=ctx_id,
            mb_id=ctx_idx,
            step_id=step_id,
        )
        context_ids.append(ctx_id)
        next_context_idx += 1
        return True

    _start_next_context()
    queues = scheduler.queues
    topk_store: Dict[Tuple[int, int, int], Dict[int, torch.Tensor]] = defaultdict(dict)
    # One child ForwardBatch per (seq_lo, seq_hi) window for this forward. A
    # window walks all 27 layers, so 26 of the per-hop builds are redundant
    # (§19.6). Cleared implicitly: this dict is local to the forward.
    _slice_cache: Dict[Tuple[int, int], ForwardBatch] = {}
    _slice_cache_on = bool(envs.SGLANG_AFD_FARM_SLICE_CACHE.get())
    pending: List[FarmHop] = []
    finished = scheduler.finished_tokens
    sampled_logits_by_seq: Dict[int, Any] = {}
    sampled_next_by_seq: Dict[int, torch.Tensor] = {}

    def _on_output_ready(hop: FarmHop, token_count: int) -> None:
        if context_sampler is None:
            return
        child_fb = hop.meta["forward_batch"]
        parent_seq_idxs = tuple(
            int(x) for x in getattr(child_fb, "_afd_farm_parent_seq_idxs", ())
        )
        if len(parent_seq_idxs) != int(token_count):
            raise RuntimeError(
                "AFD farm child sequence metadata does not match sampled rows: "
                f"seqs={parent_seq_idxs} rows={token_count}"
            )
        hidden_slice = hidden_states[hop.token_lo : hop.token_hi]
        residual_slice = (
            None if residual is None else residual[hop.token_lo : hop.token_hi]
        )
        if final_norm is not None:
            if residual_slice is None:
                hidden_slice = final_norm(hidden_slice)
            else:
                hidden_slice, _ = final_norm(hidden_slice, residual_slice)
        logits_output, next_token_ids = context_sampler(
            hidden_slice,
            None if final_norm is not None else residual_slice,
            child_fb,
        )
        if len(parent_seq_idxs) != int(next_token_ids.shape[0]):
            raise RuntimeError(
                "AFD farm sampler returned the wrong row count: "
                f"seqs={len(parent_seq_idxs)} ids={tuple(next_token_ids.shape)}"
            )
        for row, seq_idx in enumerate(parent_seq_idxs):
            sampled_logits_by_seq[seq_idx] = _slice_logits_output(
                logits_output, row
            )
            sampled_next_by_seq[seq_idx] = next_token_ids[row : row + 1]

    def _fallback():
        scheduler.abort_batch()
        return _sequential()

    _reset_occ(n_layers)
    stage_on = _stage_enabled()
    if stage_on and _STAGE.get("forwards", 0.0) == 0.0:
        _stage_reset()
    t_stage_0 = time.perf_counter() if stage_on else 0.0

    def _st_note_completion(hop: FarmHop, done_ts: float) -> None:
        if not stage_on:
            return
        _stage_bump("lat_sum", done_ts - hop.t_issue)
        _stage_bump("lat_n", 1.0)
        lat = _STAGE["lat"]
        if len(lat) < 200000:
            lat.append(done_ts - hop.t_issue)
        last = _STAGE.get("order_last", -1)
        if last >= 0 and hop.layer_i < last:
            _stage_bump("order_inv", 1.0)
        _STAGE["order_last"] = hop.layer_i

    def _st_note_issue(hop: FarmHop) -> None:
        if not stage_on:
            return
        hop.t_issue = time.perf_counter()
        _stage_bump("hops", 1.0)
        _stage_peak("pending_peak", len(pending) + 1)
        layers = {h.layer_i for h in pending}
        layers.add(hop.layer_i)
        _stage_peak("layers_peak", len(layers))
        _stage_peak("span_peak", max(layers) - min(layers))
        by_ctx: Dict[int, set[int]] = defaultdict(set)
        for pending_hop in (*pending, hop):
            by_ctx[int(pending_hop.ctx_id)].add(int(pending_hop.layer_i))
        _stage_peak("ctxs_peak", len(by_ctx))
        ctx_span = 0
        ctx_ids = list(by_ctx)
        for i, left in enumerate(ctx_ids):
            for right in ctx_ids[i + 1 :]:
                for left_layer in by_ctx[left]:
                    for right_layer in by_ctx[right]:
                        ctx_span = max(ctx_span, abs(left_layer - right_layer))
        _stage_peak("ctx_span_peak", ctx_span)

    try:
        from sglang.srt.afd.farm.persistent_linear import (
            install_persistent_linear_on_layers,
        )

        install_persistent_linear_on_layers(layer_list)
    except Exception:
        pass

    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    def _drain(block: bool) -> int:
        nonlocal residual, finished
        reaped = 0
        still: List[FarmHop] = []
        for hop in pending:
            _t_poll = time.perf_counter() if _phase_on else 0.0
            out = wait_hop(hop) if block and hop is pending[0] else poll_hop(hop)
            if _phase_on:
                _note_phase("drain_block" if block else "drain_poll",
                            time.perf_counter() - _t_poll)
            if out is None:
                still.append(hop)
                continue
            _t_c = time.perf_counter() if _phase_on else 0.0
            residual, d = _consume_hop(
                hop,
                _mlp_out(out),
                layer_list,
                hidden_states,
                residual,
                topk_store,
                scheduler,
                on_output_ready=_on_output_ready,
            )
            if _phase_on:
                _note_phase("consume", time.perf_counter() - _t_c)
            _start_next_context()
            _st_note_completion(hop, time.perf_counter())
            finished += d
            reaped += 1
        pending[:] = still
        return reaped

    def _block_one(force_block: bool = False) -> int:
        """Reap one completed hop.

        During an active issue round the caller should not wait, so the normal
        path polls. Once the scheduler has no issueable work, however, a
        bounded 250us slice only burns the safety counter while a normal FFN
        compute is still running. ``force_block`` waits that completion and
        releases its microbatch slot before the next scheduling round.
        """
        nonlocal residual, finished
        reaped = _drain(block=False)
        if reaped > 0 or not pending:
            return reaped

        wait_slice_us = farm_wait_slice_us()
        if force_block or wait_slice_us <= 0:
            # Keep the old fully blocking path available for A/B testing.
            hop = pending.pop(0)
            _t_b = time.perf_counter() if (_phase_on or stage_on) else 0.0
            out = wait_hop(hop)
            if _phase_on:
                _note_phase("drain_block", time.perf_counter() - _t_b)
            if stage_on:
                _stage_bump("block_calls", 1.0)
                _stage_bump("block_s", time.perf_counter() - _t_b)
            _t_c = time.perf_counter() if _phase_on else 0.0
            residual, d = _consume_hop(
                hop,
                _mlp_out(out),
                layer_list,
                hidden_states,
                residual,
                topk_store,
                scheduler,
                on_output_ready=_on_output_ready,
            )
            if _phase_on:
                _note_phase("consume", time.perf_counter() - _t_c)
            _start_next_context()
            _st_note_completion(hop, time.perf_counter())
            finished += d
            return reaped + 1

        _t_b = time.perf_counter() if (_phase_on or stage_on) else 0.0
        deadline = time.perf_counter() + wait_slice_us * 1e-6
        while pending and time.perf_counter() < deadline:
            reaped = _drain(block=False)
            if reaped > 0:
                break
            time.sleep(0.00005)
        if _phase_on:
            _note_phase("drain_block", time.perf_counter() - _t_b)
        if stage_on:
            _stage_bump("block_calls", 1.0)
            _stage_bump("block_s", time.perf_counter() - _t_b)
        return reaped

    def _hop_issued(hop: FarmHop) -> bool:
        return (
            hop.send_group is not None
            or hop.pool_pid is not None
            or hop.classic is not None
        )

    def _issue_round() -> Tuple[int, int]:
        """Fill every available in-flight/cap slot, then flush once.

        ``waiting`` is only popped while the scheduler can reserve a slot.
        ``mark_running`` is delayed until the send queue confirms that A2F
        credit was acquired. Work rejected for lack of credit is cancelled
        from the outbound queue and returned to ``waiting``.
        """
        nonlocal residual, finished
        _start_next_context()
        _drain(block=False)
        _start_next_context()
        queued_remote: List[FarmHop] = []
        issued = 0

        while len(pending) + len(queued_remote) < max_inf:
            _start_next_context()
            ticket = scheduler.reserve(
                b_step,
                max_inflight=max_inf,
                max_inflight_per_layer=per_layer_cap,
            )
            if ticket is None:
                break

            li = ticket.layer_i
            seq_idxs = list(ticket.seq_idxs)
            owner = (ticket.ctx_id, ticket.mb_id, ticket.step_id)
            win_index = ticket.win_index
            sl = slice_decode_window(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                forward_batch=forward_batch,
                seq_idxs=seq_idxs,
                token_num_per_seq=tps,
                with_sampling_info=(
                    li == n_layers - 1
                    or bool(envs.SGLANG_AFD_FARM_MID_SAMPLING_INFO.get())
                ),
                child_cache=_slice_cache if _slice_cache_on else None,
            )
            if sl is None:
                scheduler.rollback(ticket.ticket_id)
                if finished == 0 and not pending:
                    logger.warning("AFD farm slice failed; sequential fallback")
                    return _fallback()
                raise RuntimeError("AFD farm slice failed mid-flight")

            layer = layer_list[li]
            maybe_weight_outer_tax(
                layer_id=int(layer.layer_id),
                win_index=win_index,
                device=hidden_states.device,
            )
            capture = None
            if (
                layers_to_capture is not None
                and aux_hidden_states is not None
                and layer.layer_id in layers_to_capture
            ):
                capture = aux_hidden_states
            _t_pre = time.perf_counter() if (_phase_on or stage_on) else 0.0
            hidden_a2f, res_s, topk_ids, topk_weights, meta = run_layer_forward_pre_ffn(
                layer,
                positions=sl.positions,
                hidden_states=sl.hidden_states,
                forward_batch=sl.forward_batch,
                residual=sl.residual,
                zero_allocator=zero_allocator,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=_prev_topk(topk_store, owner, seq_idxs),
                captured_last_layer_outputs=capture,
            )
            if _phase_on:
                _note_phase("pre", time.perf_counter() - _t_pre)
            if stage_on:
                _stage_bump("pre_calls", 1.0)
                _stage_bump("pre_s", time.perf_counter() - _t_pre)
            if residual is None and res_s is not None:
                residual = torch.zeros_like(hidden_states)
                scatter_rows(residual, res_s, sl.token_lo, sl.token_hi)
            elif res_s is not None and residual is not None:
                if res_s.data_ptr() != residual[sl.token_lo : sl.token_hi].data_ptr():
                    scatter_rows(residual, res_s, sl.token_lo, sl.token_hi)

            is_moe = isinstance(getattr(layer, "mlp", None), DeepseekV2MoE)
            remote = afd_should_remote_ffn(int(layer.layer_id), is_moe=is_moe)
            if not remote:
                fb = meta["forward_batch"]
                mlp_out = layer.mlp(
                    hidden_a2f,
                    fb,
                    meta["should_allreduce_fusion"],
                    meta["use_reduce_scatter"],
                    meta["gemm_output_zero_allocator"],
                )
                dummy = FarmHop(
                    layer_i=li,
                    seq_idxs=list(seq_idxs),
                    token_lo=sl.token_lo,
                    token_hi=sl.token_hi,
                    residual=(
                        res_s
                        if residual is None
                        else residual[sl.token_lo : sl.token_hi]
                    ),
                    meta=meta,
                    batch_id=ticket.ctx_id,
                    sched_ticket=ticket.ticket_id,
                    ctx_id=ticket.ctx_id,
                    mb_id=ticket.mb_id,
                    step_id=ticket.step_id,
                )
                scheduler.commit(ticket.ticket_id)
                residual, d = _consume_hop(
                    dummy,
                    mlp_out,
                    layer_list,
                    hidden_states,
                    residual,
                    topk_store,
                    scheduler,
                    on_output_ready=_on_output_ready,
                )
                _start_next_context()
                finished += d
                continue

            _t_issue = time.perf_counter() if _phase_on else 0.0
            hop = try_issue_hop(
                layer_id=int(layer.layer_id),
                hidden=hidden_a2f,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                layer_i=li,
                seq_idxs=list(seq_idxs),
                token_lo=sl.token_lo,
                token_hi=sl.token_hi,
                residual=(
                    res_s if residual is None else residual[sl.token_lo : sl.token_hi]
                ),
                meta=meta,
                req_id=ticket.ticket_id,
                owner=owner,
                defer_flush=True,
            )
            if _phase_on:
                _note_phase("issue", time.perf_counter() - _t_issue)
            if hop is None:
                if stage_on:
                    _stage_bump("issue_nocredit", 1.0)
                scheduler.rollback(ticket.ticket_id)
                continue
            hop.batch_id = ticket.ctx_id
            hop.sched_ticket = ticket.ticket_id
            queued_remote.append(hop)

        if queued_remote:
            flush_hop_queues()
            for hop in queued_remote:
                if not _hop_issued(hop):
                    if hop.send_queue is not None:
                        hop.send_queue.cancel(hop)
                    scheduler.rollback(hop.sched_ticket)
                    hop.sched_ticket = 0
                    if stage_on:
                        _stage_bump("issue_nocredit", 1.0)
                    continue
                if not scheduler.commit(hop.sched_ticket):
                    raise RuntimeError(
                        f"AFD farm lost scheduler ticket {hop.sched_ticket}"
                    )
                if stage_on:
                    _stage_bump("issue_ok", 1.0)
                _st_note_issue(hop)
                pending.append(hop)
                issued += 1

        return issued, len(queued_remote)

    try:
        safety = 0
        limit = max(n_seq * n_layers * 8, 64)
        _phase_on = _phase_timing_enabled()
        _t_loop_start = time.perf_counter() if _phase_on else 0.0
        while finished < n_seq:
            _t_iter = time.perf_counter() if _phase_on else 0.0
            safety += 1
            if safety > limit:
                logger.warning(
                    "AFD farm safety stop finished=%s/%s pending=%s",
                    finished,
                    n_seq,
                    len(pending),
                )
                if finished == 0 and not pending:
                    return _fallback()
                raise RuntimeError("AFD farm safety stop mid-flight")

            issued, _ = _issue_round()
            _note_occ(queues.occupancy())

            if finished >= n_seq:
                break
            if issued > 0:
                continue
            if _start_next_context():
                continue
            if pending:
                # No layer can be reserved (commonly because all ready layers
                # hit per-layer caps), so complete the oldest hop and rescan.
                _block_one(force_block=True)
                continue
            if queues.waiting_empty():
                if finished == 0:
                    logger.warning(
                        "AFD farm idle with unfinished seqs finished=%s/%s",
                        finished,
                        n_seq,
                    )
                    return _fallback()
                raise RuntimeError(f"AFD farm lost tokens finished={finished}/{n_seq}")
            if finished == 0:
                logger.warning(
                    "AFD farm could not issue with empty inflight; sequential"
                )
                return _fallback()
            raise RuntimeError("AFD farm issue failed with empty inflight")
    except Exception as e:
        if pending:
            try:
                _drain(block=True)
            except Exception:
                logger.exception("AFD farm failed to drain pending hops during fallback")
        logger.exception("AFD farm failed (%s); sequential fallback", e)
        return _fallback()

    while pending:
        _block_one()

    if context_sampler is not None:
        missing = [seq_idx for seq_idx in range(n_seq) if seq_idx not in sampled_next_by_seq]
        if missing:
            raise RuntimeError(f"AFD farm missing sampled sequence rows: {missing}")
        merged_logits = _merge_logits_outputs(
            [sampled_logits_by_seq[seq_idx] for seq_idx in range(n_seq)]
        )
        merged_next_ids = torch.cat(
            [sampled_next_by_seq[seq_idx] for seq_idx in range(n_seq)],
            dim=0,
        )
        forward_batch._afd_farm_logits_output = merged_logits
        forward_batch._afd_farm_next_token_ids = merged_next_ids

    if not scheduler.finish_batch():
        raise RuntimeError(
            f"AFD farm contexts {context_ids} did not close cleanly "
            f"finished={scheduler.finished_tokens}/{scheduler.expected_tokens}"
        )

    global _farm_forwards
    _farm_forwards += 1
    if _phase_timing_enabled() and _t_loop_start > 0.0:
        _note_phase("loop_total", time.perf_counter() - _t_loop_start)
        if _farm_forwards % 8 == 0:
            _log_phase(n_layers)
    _maybe_log_occ(n_layers)
    if stage_on:
        _stage_bump("forwards", 1.0)
        _stage_bump("wall_s", time.perf_counter() - t_stage_0)
        _stage_every_n = _stage_every()
        if _stage_every_n > 0 and int(_STAGE.get("forwards", 0.0)) % _stage_every_n == 0:
            _stage_log(n_layers, n_seq)
    try:
        bwin = queues.bwin.as_dict()
        every = int(envs.SGLANG_AFD_FARM_LOG_EVERY.get() or 0)
        if every > 0 and _farm_forwards % every == 0:
            logger.info(
                "AFD farm B_win switches=%s mean_win_len=%.2f picks=%s "
                "amortize_factor=%.2f mean_tok/launch=%.1f coalesce_k=%s "
                "layer_burst=%s sched=%s",
                bwin.get("layer_switches"),
                bwin.get("mean_win_len"),
                bwin.get("picks"),
                bwin.get("amortize_factor"),
                bwin.get("mean_tokens_per_launch"),
                coalesce_k,
                layer_burst,
                farm_sched(),
            )
    except Exception:
        pass

    last_topk = None
    by_seq: Dict[int, torch.Tensor] = {}
    for store in topk_store.values():
        by_seq.update(store)
    if len(by_seq) == n_seq and all(s in by_seq for s in range(n_seq)):
        last_topk = torch.cat([by_seq[s] for s in range(n_seq)], dim=0)
    return hidden_states, residual, last_topk
