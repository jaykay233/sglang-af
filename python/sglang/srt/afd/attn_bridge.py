# SPDX-License-Identifier: Apache-2.0
"""Attn-side bridge: replace local MLP with remote FFN when AFD is active."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.srt.afd.pipeline import afd_pipeline_enabled, remote_ffn_pipelined
from sglang.srt.afd.runtime import get_afd_runtime, is_afd_attn
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

if TYPE_CHECKING:
    from sglang.srt.afd.pipeline import AfdPendingTransfer

RemoteFfnResult = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

# Stable scratch for breakable-CG merge breaks (avoid mempool weak-ref reuse).
_scratch: dict = {}


def _stable_slice(src: torch.Tensor, n: int, key: str) -> torch.Tensor:
    """Copy ``src[:n]`` into a process-stable buffer (not CUDAGraph mempool)."""
    global _scratch
    buf = _scratch.get(key)
    if (
        buf is None
        or buf.device != src.device
        or buf.dtype != src.dtype
        or buf.shape[0] < n
        or buf.shape[1:] != src.shape[1:]
    ):
        shape = (max(n, int(src.shape[0])),) + tuple(src.shape[1:])
        buf = torch.empty(shape, dtype=src.dtype, device=src.device)
        _scratch[key] = buf
    buf[:n].copy_(src[:n])
    return buf[:n]


def _require_runtime():
    rt = get_afd_runtime()
    if rt is None:
        raise RuntimeError(
            "SGLANG_AFD_MODE=attn but AFD runtime is not initialized. "
            "Call init_afd_runtime / maybe_init_afd_from_model_runner."
        )
    return rt


def afd_in_graph_wait_enabled() -> bool:
    """GPU write_flag/wait_flag inside decode CUDA Graph (no eager break)."""
    return bool(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())


def _remote_ffn_impl(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    merge_k: int = 1,
    residual: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    forward_batch=None,
) -> RemoteFfnResult:
    raw_t = int(hidden_states.shape[0])
    if merge_k > 1:
        try:
            from sglang.srt.afd.merge_kv import (
                publish_merge_forward_meta,
                _live_publish_fb,
            )
            # Prefer live FB set by load_batch / eager forward — never trust
            # the capture-time ForwardBatch closed over by breakable CG.
            live = _live_publish_fb
            if live is not None:
                raw_t = int(getattr(live, "raw_num_token", None) or live.batch_size or raw_t)
                raw_t = max(1, min(raw_t, int(hidden_states.shape[0])))
            if forward_batch is not None and positions is not None:
                try:
                    forward_batch.positions = positions
                except Exception:
                    pass
            publish_merge_forward_meta(None)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "AFD publish_merge_forward_meta failed: %s", e
            )

    # Decode CG pads to capture-bs; FFN must only see real tokens or pad
    # slots (req_pool=0, out_cache=0) corrupt interior KV (retok ~153).
    #
    # Breakable CG weak-refs mid-graph tensors (esp. residual) that live in the
    # CUDAGraph mempool. Copy into stable scratch before IPC.
    if merge_k > 1:
        hs = _stable_slice(hidden_states, raw_t, "hs")
        rs = _stable_slice(residual, raw_t, "rs") if residual is not None else None
        pos = _stable_slice(positions, raw_t, "pos") if positions is not None else None
        tk_ids = _stable_slice(topk_ids, raw_t, "tk") if topk_ids is not None else None
        tk_w = _stable_slice(topk_weights, raw_t, "tw") if topk_weights is not None else None
    else:
        hs = hidden_states[:raw_t]
        rs = residual[:raw_t] if residual is not None else None
        pos = positions[:raw_t] if positions is not None else None
        tk_ids = topk_ids[:raw_t] if topk_ids is not None else None
        tk_w = topk_weights[:raw_t] if topk_weights is not None else None

    # AfPool MxN path (SGLANG_AFD_POOL=1): route via credit scheduler.
    try:
        from sglang.srt.afd.pool import get_af_attn_client, pool_enabled

        client = get_af_attn_client() if pool_enabled() else None
    except Exception:
        client = None
    if client is not None:
        req_id = 0
        if forward_batch is not None:
            req_id = int(getattr(forward_batch, "bid", None) or 0)
        out = client.remote_ffn(
            req_id=req_id,
            layer_id=int(layer_id),
            hidden=hs,
            topk_ids=tk_ids,
            topk_weights=tk_w,
            merge_k=merge_k,
            residual=rs,
            positions=pos,
        )
        if merge_k > 1 and isinstance(out, tuple):
            out_h, out_r = out
            hidden_states[:raw_t].copy_(out_h)
            if residual is not None:
                residual[:raw_t].copy_(out_r)
            return hidden_states, residual
        if isinstance(out, torch.Tensor):
            hidden_states[:raw_t].copy_(out)
            return hidden_states
        return out

    rt = _require_runtime()
    if (
        afd_pipeline_enabled()
        and not afd_in_graph_wait_enabled()
        and merge_k <= 1
    ):
        return remote_ffn_pipelined(
            layer_id=layer_id,
            hidden=hs,
            topk_ids=tk_ids,
            topk_weights=tk_w,
            runtime=rt,
        )
    out = rt.remote_ffn(
        layer_id=layer_id,
        hidden=hs,
        topk_ids=tk_ids,
        topk_weights=tk_w,
        merge_k=merge_k,
        residual=rs,
        positions=pos,
    )
    if merge_k > 1 and isinstance(out, tuple):
        out_h, out_r = out
        hidden_states[:raw_t].copy_(out_h)
        if residual is not None:
            residual[:raw_t].copy_(out_r)
        return hidden_states, residual
    if isinstance(out, torch.Tensor):
        hidden_states[:raw_t].copy_(out)
        return hidden_states
    return out


@eager_on_graph(True)
def _afd_remote_ffn_graph_break(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    merge_k: int = 1,
    residual: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    forward_batch=None,
) -> RemoteFfnResult:
    """Runs outside CUDA Graph segments (breakable backend)."""
    return _remote_ffn_impl(
        layer_id=layer_id,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        merge_k=merge_k,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
    )


@eager_on_graph(True)
def issue_remote_ffn(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    mb_id: int = 0,
    merge_k: int = 1,
    residual: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> "AfdPendingTransfer":
    """Issue A2F without waiting (P7 layer pipeline / async). Graph break."""
    rt = _require_runtime()
    return rt.remote_ffn_async(
        layer_id=layer_id,
        hidden=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        mb_id=mb_id,
        merge_k=merge_k,
        residual=residual,
        positions=positions,
    )


@eager_on_graph(True)
def wait_remote_ffn(
    pending: "AfdPendingTransfer", *, clone: bool = True
) -> RemoteFfnResult:
    """Wait for a previously issued remote FFN. Graph break."""
    rt = _require_runtime()
    return rt.wait_remote_ffn(pending, clone=clone)


def maybe_remote_ffn(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    local_mlp_out: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    is_moe: Optional[bool] = None,
    merge_k: int = 1,
    residual: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    forward_batch=None,
) -> Optional[RemoteFfnResult]:
    """If running as AFD Attn worker, send ``hidden_states`` and return FFN out.

    Returns ``None`` when AFD Attn is not active or this layer keeps local FFN
    (caller should run local MLP) — e.g. ``REMOTE_MOE_ONLY`` dense layers.

    When ``merge_k>1``, returns ``(mlp_out, residual)``.

    Decode uses TRUE_OVERLAP (breakable CG + dual-mb issue/wait). Prefill and
    unsplit batches use per-layer ``@eager_on_graph`` break + push_pull/wait.
    """
    del local_mlp_out
    if not is_afd_attn():
        return None
    from sglang.srt.afd.remote_policy import afd_should_remote_ffn

    if is_moe is None:
        is_moe = topk_ids is not None
    if not afd_should_remote_ffn(int(layer_id), is_moe=is_moe):
        return None
    # IN_GRAPH_WAIT is incompatible with required TRUE_OVERLAP; always breakable.
    return _afd_remote_ffn_graph_break(
        layer_id=layer_id,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        merge_k=merge_k,
        residual=residual,
        positions=positions,
        forward_batch=forward_batch,
    )
