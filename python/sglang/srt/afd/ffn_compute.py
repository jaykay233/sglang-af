# SPDX-License-Identifier: Apache-2.0
"""Build FFN compute callbacks that dispatch into loaded SGLang layers.

Merge-mode: capture the entire interior layer group as ONE CUDA graph
(metadata init once outside; all layers share Triton cuda_graph_* buffers).
FFN alt_stream is disabled so MLA stays single-stream and capturable.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.afd.routing_scheme import is_scheme_b
from sglang.srt.afd.transport import FfnComputeFn

logger = logging.getLogger(__name__)

# Key = (start_layer, end_layer, num_tokens)
_GROUP_GRAPHS: Dict[Tuple[int, int, int], torch.cuda.CUDAGraph] = {}
_GROUP_STATIC: Dict[Tuple[int, int, int], Dict[str, torch.Tensor]] = {}
_GROUP_CAPTURED: set = set()
_GROUP_STREAM: Optional[torch.cuda.Stream] = None
_GRAPH_POOL = None
_ATTN_CG_READY = False


def _get_stream(device: torch.device) -> torch.cuda.Stream:
    global _GROUP_STREAM
    if _GROUP_STREAM is None:
        _GROUP_STREAM = torch.cuda.Stream(device=device)
    return _GROUP_STREAM


def _get_pool():
    global _GRAPH_POOL
    if _GRAPH_POOL is None:
        _GRAPH_POOL = torch.cuda.graph_pool_handle()
    return _GRAPH_POOL


def _ensure_attn_cg_state(attn_backend, *, max_bs: int, max_tokens: int) -> None:
    global _ATTN_CG_READY
    if _ATTN_CG_READY or attn_backend is None:
        return
    if getattr(attn_backend, "cuda_graph_kv_indices", None) is not None:
        _ATTN_CG_READY = True
        return
    if not hasattr(attn_backend, "init_cuda_graph_state"):
        return
    attn_backend.init_cuda_graph_state(max_bs, max_tokens)
    _ATTN_CG_READY = True
    logger.info(
        "AFD FFN attn cuda_graph_state ready max_bs=%d max_tokens=%d",
        max_bs, max_tokens,
    )


def _replay_group_graph(
    key: Tuple[int, int, int],
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    static = _GROUP_STATIC[key]
    static["hidden"].copy_(hidden_states)
    static["residual"].copy_(residual)
    static["seq_lens"].copy_(forward_batch.seq_lens)
    static["req_pool"].copy_(forward_batch.req_pool_indices)
    static["out_cache"].copy_(forward_batch.out_cache_loc)
    static["positions"].copy_(positions)
    static["input_ids"].copy_(forward_batch.input_ids)
    bs = int(forward_batch.batch_size)
    static["fb"].batch_size = bs
    static["fb"].seq_lens_sum = int(forward_batch.seq_lens_sum)
    static["fb"].seq_lens_cpu = static["seq_lens"][:bs].detach().to("cpu")

    ab = static.get("attn_backend")
    if ab is not None and hasattr(ab, "init_forward_metadata_out_graph"):
        ab.init_forward_metadata_out_graph(static["fb"], in_capture=False)

    _GROUP_GRAPHS[key].replay()
    hidden_states.copy_(static["out"])
    residual.copy_(static["res_out"])
    return hidden_states, residual


def _capture_group_graph(
    key: Tuple[int, int, int],
    layers: Sequence[nn.Module],
    start_layer: int,
    end: int,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    residual: torch.Tensor,
    zero_allocator,
    attn_backend,
) -> None:
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.utils.common import BumpAllocator

    device = hidden_states.device
    dtype = hidden_states.dtype
    H = hidden_states.shape[-1]
    T = hidden_states.shape[0]
    bs = int(forward_batch.batch_size)
    n_interior = end - (start_layer + 1)

    s_hidden = torch.zeros(T, H, dtype=dtype, device=device)
    s_residual = torch.zeros(T, H, dtype=dtype, device=device)
    s_out = torch.zeros(T, H, dtype=dtype, device=device)
    s_res_out = torch.zeros(T, H, dtype=dtype, device=device)
    s_positions = torch.zeros(T, dtype=torch.int64, device=device)

    s_seq_lens = torch.zeros_like(forward_batch.seq_lens)
    s_req_pool = torch.zeros_like(forward_batch.req_pool_indices)
    s_out_cache = torch.zeros_like(forward_batch.out_cache_loc)
    s_input_ids = torch.zeros_like(forward_batch.input_ids)

    s_fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=s_input_ids,
        req_pool_indices=s_req_pool,
        seq_lens=s_seq_lens,
        out_cache_loc=s_out_cache,
        seq_lens_sum=forward_batch.seq_lens_sum,
        seq_lens_cpu=forward_batch.seq_lens[:bs].detach().to("cpu"),
    )
    s_fb.positions = s_positions

    s_za = BumpAllocator(
        buffer_size=max(64, n_interior * 8, zero_allocator._buffer.numel()),
        dtype=zero_allocator._buffer.dtype,
        device=device,
    )

    snap_h = hidden_states.detach().clone()
    snap_r = residual.detach().clone()
    snap_pos = positions.detach().clone()
    snap_seq = forward_batch.seq_lens.detach().clone()
    snap_req = forward_batch.req_pool_indices.detach().clone()
    snap_outc = forward_batch.out_cache_loc.detach().clone()
    snap_ids = forward_batch.input_ids.detach().clone()

    # Warmup/capture writes KV at out_cache_loc; save & restore so we don't
    # corrupt the real cache (would produce garbage tokens).
    kv_backup = None
    try:
        from sglang.srt.afd import merge_kv as _mkv
        locs = snap_outc.to(dtype=torch.long)
        locs = locs[locs > 0]
        if locs.numel() > 0 and _mkv._local_kv:
            uniq = locs.unique()
            kv_backup = {
                lid: buf.index_select(0, uniq).clone()
                for lid, buf in _mkv._local_kv.items()
            }
            kv_backup_idx = uniq
        else:
            kv_backup_idx = None
    except Exception:
        kv_backup = None
        kv_backup_idx = None

    def _restore_kv():
        if kv_backup is None or kv_backup_idx is None:
            return
        try:
            from sglang.srt.afd import merge_kv as _mkv
            for lid, rows in kv_backup.items():
                buf = _mkv._local_kv.get(lid)
                if buf is not None:
                    buf.index_copy_(0, kv_backup_idx, rows)
        except Exception:
            pass

    def _load_inputs():
        s_hidden.copy_(snap_h)
        s_residual.copy_(snap_r)
        s_positions.copy_(snap_pos)
        s_seq_lens.copy_(snap_seq)
        s_req_pool.copy_(snap_req)
        s_out_cache.copy_(snap_outc)
        s_input_ids.copy_(snap_ids)
        s_fb.seq_lens_cpu = s_seq_lens[:bs].detach().to("cpu")
        s_za._pointer = 0

    def _run_body():
        hs = s_hidden
        rs = s_residual
        for lid in range(start_layer + 1, end):
            hs, rs, _ = layers[lid](s_positions, hs, s_fb, rs, s_za)
        s_out.copy_(hs)
        s_res_out.copy_(rs)

    stream = _get_stream(device)
    pool = _get_pool()
    default = torch.cuda.current_stream(device)
    default.wait_stream(stream)

    try:
        with model_capture_mode():
            with torch.cuda.device(device), torch.cuda.stream(stream):
                if attn_backend is not None and hasattr(
                    attn_backend, "init_forward_metadata_out_graph"
                ):
                    _load_inputs()
                    attn_backend.init_forward_metadata_out_graph(s_fb, in_capture=True)

                for _ in range(2):
                    _load_inputs()
                    _run_body()
                    stream.synchronize()
                    hook = getattr(attn_backend, "on_after_cuda_graph_warmup", None)
                    if hook is not None:
                        hook()
                    stream.synchronize()

                _load_inputs()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=stream):
                    if attn_backend is not None and hasattr(
                        attn_backend, "init_forward_metadata_in_graph"
                    ):
                        attn_backend.init_forward_metadata_in_graph(s_fb)
                    _run_body()
                stream.synchronize()
    finally:
        _restore_kv()

    default.wait_stream(stream)

    _GROUP_GRAPHS[key] = g
    _GROUP_STATIC[key] = {
        "hidden": s_hidden, "residual": s_residual,
        "out": s_out, "res_out": s_res_out,
        "seq_lens": s_seq_lens, "req_pool": s_req_pool,
        "out_cache": s_out_cache, "positions": s_positions,
        "input_ids": s_input_ids,
        "fb": s_fb, "attn_backend": attn_backend,
    }
    _GROUP_CAPTURED.add(key)
    logger.info(
        "AFD FFN group CG captured layers=%d-%d T=%d bs=%d",
        start_layer + 1, end - 1, T, bs,
    )


def run_layer_ffn(
    mlp: nn.Module,
    hidden: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from sglang.srt.afd.moe_bridge import ffn_apply_experts
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

    if isinstance(mlp, DeepseekV2MoE):
        if is_scheme_b():
            return mlp(hidden)
        if topk_ids is None or topk_weights is None:
            raise RuntimeError("AFD MoE batch missing topk/topk_weights")
        return ffn_apply_experts(mlp, hidden, topk_ids, topk_weights)
    if isinstance(mlp, DeepseekV2MLP):
        return mlp(hidden)
    if callable(mlp):
        try:
            return mlp(hidden)
        except TypeError:
            return mlp(hidden, None)
    raise TypeError(f"Unsupported MLP type for AFD FFN: {type(mlp)}")


def run_layer_group(
    layers: Sequence[nn.Module],
    *,
    start_layer: int,
    merge_k: int,
    hidden: torch.Tensor,
    residual: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    forward_batch=None,
    zero_allocator=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    n = len(layers)
    if start_layer < 0 or start_layer >= n:
        raise IndexError(f"AFD start_layer={start_layer}")
    k = max(1, int(merge_k))
    end = min(start_layer + k, n)

    layer0 = layers[start_layer]
    y = run_layer_ffn(layer0.mlp, hidden, topk_ids=topk_ids, topk_weights=topk_weights)

    if k <= 1 or end <= start_layer + 1:
        return y, residual

    if forward_batch is None or positions is None or residual is None:
        raise RuntimeError("AFD LAYER_MERGE_K>1 missing forward_batch/positions/residual")

    hidden_states, residual = layer0.layer_communicator.postprocess_layer(
        y, residual, forward_batch
    )

    if zero_allocator is None:
        from sglang.srt.utils.common import BumpAllocator
        zero_allocator = BumpAllocator(
            buffer_size=max(64, (end - start_layer) * 8),
            dtype=torch.float32, device=hidden_states.device,
        )

    from contextlib import nullcontext
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    attn_backend = None
    try:
        from sglang.srt.afd.merge_kv import get_ffn_attn_backend
        attn_backend = get_ffn_attn_backend()
    except Exception:
        attn_backend = None
    ctx = (
        forward_context(ForwardContext(attn_backend=attn_backend))
        if attn_backend is not None else nullcontext()
    )

    T = int(hidden_states.shape[0])
    try:
        from sglang.srt.environ import envs as _afd_envs
        cfg_max = max(1, int(_afd_envs.SGLANG_AFD_MAX_NUM_TOKEN.get() or 8))
    except Exception:
        cfg_max = 8
    _ensure_attn_cg_state(
        attn_backend,
        max_bs=max(cfg_max, int(forward_batch.batch_size), T),
        max_tokens=max(cfg_max, T),
    )

    # Diagnostics for merge correctness (first few calls).
    _n = int(getattr(run_layer_group, "_afd_diag_n", 0))
    if _n < 3:
        run_layer_group._afd_diag_n = _n + 1  # type: ignore[attr-defined]
        sa = getattr(layers[start_layer + 1], "self_attn", None)
        logger.info(
            "AFD FFN interior diag#%d layer=%d attn=%s "
            "fb.bs=%s seq0=%s out_cache0=%s pos0=%s T=%s",
            _n,
            start_layer + 1,
            type(sa).__name__ if sa is not None else None,
            forward_batch.batch_size,
            int(forward_batch.seq_lens[0].item()) if forward_batch.seq_lens.numel() else -1,
            int(forward_batch.out_cache_loc[0].item()) if forward_batch.out_cache_loc.numel() else -1,
            int(positions[0].item()) if positions is not None and positions.numel() else -1,
            T,
        )

    key = (start_layer, end, T)
    use_cg = True
    try:
        from sglang.srt.environ import envs as _afd_envs
        use_cg = bool(_afd_envs.SGLANG_AFD_FFN_CUDA_GRAPH.get())
    except Exception:
        pass

    with ctx:
        if use_cg:
            if key not in _GROUP_CAPTURED:
                try:
                    _capture_group_graph(
                        key, layers, start_layer, end,
                        hidden_states, positions, forward_batch, residual,
                        zero_allocator, attn_backend,
                    )
                except Exception as e:
                    logger.warning(
                        "AFD FFN group CG capture failed: %s; using eager", e,
                    )
                    for lid in range(start_layer + 1, end):
                        hidden_states, residual, _ = layers[lid](
                            positions, hidden_states, forward_batch, residual, zero_allocator,
                        )
                    return hidden_states, residual

            return _replay_group_graph(
                key, hidden_states, residual, positions, forward_batch,
            )

        for lid in range(start_layer + 1, end):
            hidden_states, residual, _ = layers[lid](
                positions, hidden_states, forward_batch, residual, zero_allocator,
            )
        return hidden_states, residual


def run_same_layer_fused(
    mlp: nn.Module, batches: Sequence[AfdServerBatch],
) -> List[List[torch.Tensor]]:
    if not batches:
        return []
    try:
        dtype = next(mlp.parameters()).dtype
    except StopIteration:
        dtype = torch.bfloat16
    for b in batches:
        b.compute_dtype = dtype

    if len(batches) == 1:
        b = batches[0]
        t = b.num_tokens
        hidden = b.hidden[:t]
        y = run_layer_ffn(mlp, hidden,
            topk_ids=None if b.topk_ids is None else b.topk_ids[:t],
            topk_weights=None if b.topk_weights is None else b.topk_weights[:t])
        out_rows = b.hidden_wire.shape[0]
        out = torch.empty(out_rows, hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
        out[:t] = y.to(hidden.dtype)
        if t < out_rows:
            out[t:].zero_()
        return [[out]]

    ts = [int(b.num_tokens) for b in batches]
    hiddens = [b.hidden[:t] for b, t in zip(batches, ts)]
    cat_h = torch.cat(hiddens, dim=0)
    topk_ids = topk_weights = None
    if batches[0].topk_ids is not None:
        topk_ids = torch.cat([b.topk_ids[:t] for b, t in zip(batches, ts)], dim=0)
        topk_weights = torch.cat([b.topk_weights[:t] for b, t in zip(batches, ts)], dim=0)

    y = run_layer_ffn(mlp, cat_h, topk_ids=topk_ids, topk_weights=topk_weights)
    y = y.to(hiddens[0].dtype)
    outs: List[List[torch.Tensor]] = []
    off = 0
    for b, t in zip(batches, ts):
        out_rows = b.hidden_wire.shape[0]
        out = torch.empty(out_rows, y.shape[-1], dtype=y.dtype, device=y.device)
        out[:t] = y[off:off + t]
        if t < out_rows:
            out[t:].zero_()
        outs.append([out])
        off += t
    return outs


def _pack_group_outputs(
    batch: AfdServerBatch, y: torch.Tensor, residual: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    t = batch.num_tokens
    out_rows = batch.hidden_wire.shape[0]
    out = torch.empty(out_rows, y.shape[-1], dtype=y.dtype, device=y.device)
    out[:t] = y[:t].to(out.dtype)
    if t < out_rows:
        out[t:].zero_()
    if residual is None:
        return [out]
    res_out = torch.empty(out_rows, residual.shape[-1], dtype=residual.dtype, device=residual.device)
    res_out[:t] = residual[:t]
    if t < out_rows:
        res_out[t:].zero_()
    return [out, res_out]


def make_model_ffn_compute(model: nn.Module) -> FfnComputeFn:
    from sglang.srt.afd.moe_bridge import get_decoder_layers
    from sglang.srt.afd.remote_policy import layer_merge_k

    layers = get_decoder_layers(model)
    try:
        n_layers = len(layers)
    except TypeError:
        raise AttributeError("model layers is not sized")
    if n_layers <= 0:
        raise AttributeError("model has zero decoder layers")

    def compute(batch: AfdServerBatch):
        layer_id = batch.layer_id
        if layer_id < 0 or layer_id >= len(layers):
            raise IndexError(f"AFD layer_id={layer_id}")
        try:
            dtype = next(layers[layer_id].mlp.parameters()).dtype
        except StopIteration:
            dtype = torch.bfloat16
        batch.compute_dtype = dtype

        merge_k = int(batch.merge_k) if batch.merge_k else 1
        env_k = layer_merge_k()
        if env_k > 1 and merge_k <= 1:
            merge_k = env_k
        t = batch.num_tokens
        hidden = batch.hidden[:t]
        topk_ids = None if batch.topk_ids is None else batch.topk_ids[:t]
        topk_weights = None if batch.topk_weights is None else batch.topk_weights[:t]

        if merge_k <= 1:
            fused = run_same_layer_fused(layers[layer_id].mlp, [batch])
            return fused[0]

        ffn_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        hidden = hidden.to(ffn_dev, copy=True)
        if topk_ids is not None:
            topk_ids = topk_ids.to(ffn_dev, copy=True)
        if topk_weights is not None:
            topk_weights = topk_weights.to(ffn_dev, copy=True)

        residual = None
        if batch.residual is not None:
            residual = batch.residual[:t].to(dtype=hidden.dtype, device=ffn_dev, copy=True)
        positions = None
        if batch.positions is not None:
            positions = batch.positions[:t].to(device=ffn_dev, copy=True)
        else:
            from sglang.srt.afd.merge_kv import _meta as _mkv_meta
            if _mkv_meta is not None and "positions" in _mkv_meta:
                positions = _mkv_meta["positions"][:t].to(device=ffn_dev, copy=True)

        forward_batch = None
        zero_allocator = None
        try:
            from sglang.srt.afd.merge_kv import get_merge_forward_batch
            forward_batch, zero_allocator = get_merge_forward_batch(
                num_tokens=t, positions=positions)
        except Exception as e:
            logger.warning("AFD merge ForwardBatch unavailable: %s", e)

        if forward_batch is None or positions is None or residual is None:
            logger.warning("AFD merge_kv not ready, falling back to MLP-only")
            y = run_layer_ffn(layers[layer_id].mlp, hidden, topk_ids=topk_ids, topk_weights=topk_weights)
            return _pack_group_outputs(batch, y, None)

        y, residual_out = run_layer_group(
            layers, start_layer=layer_id, merge_k=merge_k,
            hidden=hidden, residual=residual, positions=positions,
            topk_ids=topk_ids, topk_weights=topk_weights,
            forward_batch=forward_batch, zero_allocator=zero_allocator,
        )
        return _pack_group_outputs(batch, y, residual_out)

    compute._afd_layers = layers
    return compute


def try_compute_same_layer_group(
    compute: FfnComputeFn, batches: Sequence[AfdServerBatch]
) -> Optional[List[List[torch.Tensor]]]:
    if len(batches) <= 1:
        return None
    if any(int(b.merge_k or 1) > 1 for b in batches):
        return None
    layers = getattr(compute, "_afd_layers", None)
    if layers is None:
        return None
    layer_id = batches[0].layer_id
    if any(b.layer_id != layer_id for b in batches):
        return None
    if layer_id < 0 or layer_id >= len(layers):
        return None
    return run_same_layer_fused(layers[layer_id].mlp, batches)


def make_identity_ffn_compute() -> FfnComputeFn:
    def compute(batch: AfdServerBatch):
        t = batch.num_tokens
        h = batch.hidden
        out_rows = batch.hidden_wire.shape[0]
        out = torch.empty(out_rows, h.shape[-1], dtype=h.dtype, device=h.device)
        out[:t] = h[:t]
        if t < out_rows:
            out[t:].zero_()
        return [out]
    return compute
