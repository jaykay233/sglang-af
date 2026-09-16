# SPDX-License-Identifier: Apache-2.0
"""P3: per-layer CUDAGraph replay for farm Attn (launch amortize).

Captures ``layer.forward_pre_ffn`` once per ``(layer_id, n_tokens)`` with:
  - static ``positions`` / ``hidden`` / ``residual`` buffers
  - a graph-owned ``ForwardBatch`` whose tensor storages stay fixed; each
    replay ``copy_``s index tensors from the live slice FB into that skeleton

Combines with P2 coalesce (larger M) for real HBM weight amortize. This module
alone mainly cuts launch / CPU dispatch — not weight-tile-outer persistence.

FlashInfer / MLA often refuses capture. Those keys are blacklisted → eager.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_FB_TENSOR_FIELDS = (
    "input_ids",
    "req_pool_indices",
    "seq_lens",
    "out_cache_loc",
    "positions",
    "orig_seq_lens",
    "seq_lens_cpu",
    "num_token_non_padded",
    "global_num_tokens_gpu",
    "global_num_tokens_for_logprob_gpu",
)

_stats = {
    "eager": 0,
    "capture_ok": 0,
    "capture_fail": 0,
    "replay": 0,
}


def layer_cg_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FARM_LAYER_CG.get())


def layer_cg_stats() -> dict:
    return dict(_stats)


def reset_layer_cg_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.shape != src.shape:
        if dst.numel() < src.numel():
            return False
        dst.view(-1)[: src.numel()].copy_(src.view(-1))
        return True
    dst.copy_(src)
    return True


def _clone_fb_skeleton(fb: Any) -> Any:
    sk = copy.copy(fb)
    for name in _FB_TENSOR_FIELDS:
        t = getattr(fb, name, None)
        if torch.is_tensor(t):
            setattr(sk, name, t.clone())
    return sk


def _sync_fb(dst: Any, src: Any) -> bool:
    for name in _FB_TENSOR_FIELDS:
        a = getattr(dst, name, None)
        b = getattr(src, name, None)
        if a is None or b is None:
            continue
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            continue
        if not _copy_into(a, b):
            return False
    try:
        dst.batch_size = src.batch_size
        dst.seq_lens_sum = src.seq_lens_sum
    except Exception:
        pass
    return True


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    pos: torch.Tensor
    hidden: torch.Tensor
    residual: Optional[torch.Tensor]
    fb: Any
    out_hidden: torch.Tensor
    out_residual: Optional[torch.Tensor]
    out_topk_ids: Optional[torch.Tensor]
    out_topk_weights: Optional[torch.Tensor]
    meta_template: Dict[str, Any]


@dataclass
class FarmLayerCudaGraphCache:
    entries: Dict[Tuple[int, int], _GraphEntry] = field(default_factory=dict)
    blacklist: set = field(default_factory=set)

    def run_forward_pre_ffn(
        self,
        layer: Any,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
        residual: Optional[torch.Tensor],
        zero_allocator: Any,
        gemm_output_zero_allocator: Any = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        captured_last_layer_outputs: Optional[Any] = None,
    ):
        def _eager():
            _stats["eager"] += 1
            return layer.forward_pre_ffn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=prev_topk_indices,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )

        if not layer_cg_enabled():
            return _eager()
        if not str(hidden_states.device).startswith("cuda"):
            return _eager()
        try:
            if torch.cuda.is_current_stream_capturing():
                return _eager()
        except Exception:
            return _eager()
        if captured_last_layer_outputs is not None:
            return _eager()
        # prev_topk varies by window; keep eager to avoid stale static ptrs.
        if prev_topk_indices is not None:
            return _eager()

        layer_id = int(getattr(layer, "layer_id", id(layer)) or 0)
        n_tok = int(hidden_states.shape[0])
        key = (layer_id, n_tok)
        if key in self.blacklist:
            return _eager()

        entry = self.entries.get(key)
        if entry is not None:
            return self._replay(
                entry,
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                eager=_eager,
            )

        try:
            warm = layer.forward_pre_ffn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                zero_allocator=zero_allocator,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                llama_4_scaling=llama_4_scaling,
                prev_topk_indices=None,
                captured_last_layer_outputs=None,
            )
            torch.cuda.synchronize()
            wh, wr, w_tid, w_tw, w_meta = warm

            pos_s = positions.clone()
            hid_s = hidden_states.clone()
            res_s = residual.clone() if residual is not None else None
            fb_s = _clone_fb_skeleton(forward_batch)
            if not _sync_fb(fb_s, forward_batch):
                raise RuntimeError("FB skeleton sync failed at capture")

            out_hid = torch.empty_like(wh)
            out_res = torch.empty_like(wr) if wr is not None else None
            out_topk_ids = torch.empty_like(w_tid) if w_tid is not None else None
            out_topk_weights = torch.empty_like(w_tw) if w_tw is not None else None

            g = torch.cuda.CUDAGraph()
            meta_cap = None
            with torch.cuda.graph(g):
                h, r, topk_ids, topk_weights, meta = layer.forward_pre_ffn(
                    positions=pos_s,
                    hidden_states=hid_s,
                    forward_batch=fb_s,
                    residual=res_s,
                    zero_allocator=zero_allocator,
                    gemm_output_zero_allocator=gemm_output_zero_allocator,
                    llama_4_scaling=llama_4_scaling,
                    prev_topk_indices=None,
                    captured_last_layer_outputs=None,
                )
                out_hid.copy_(h)
                if out_res is not None and r is not None:
                    out_res.copy_(r)
                if out_topk_ids is not None and topk_ids is not None:
                    out_topk_ids.copy_(topk_ids)
                if out_topk_weights is not None and topk_weights is not None:
                    out_topk_weights.copy_(topk_weights)
                meta_cap = meta

            meta_template = {
                k: v
                for k, v in (meta_cap or {}).items()
                if k
                not in (
                    "forward_batch",
                    "hidden_for_mlp",
                    "hidden_states_orig",
                    "gemm_output_zero_allocator",
                    "topk_indices",
                )
            }
            self.entries[key] = _GraphEntry(
                graph=g,
                pos=pos_s,
                hidden=hid_s,
                residual=res_s,
                fb=fb_s,
                out_hidden=out_hid,
                out_residual=out_res,
                out_topk_ids=out_topk_ids,
                out_topk_weights=out_topk_weights,
                meta_template=meta_template,
            )
            _stats["capture_ok"] += 1
            return warm
        except Exception as e:
            self.blacklist.add(key)
            _stats["capture_fail"] += 1
            logger.info(
                "AFD farm layer CG capture failed layer=%s n_tok=%s (%s); eager",
                layer_id,
                n_tok,
                e,
            )
            return _eager()

    def _replay(self, entry: _GraphEntry, *, eager, **kwargs):
        positions = kwargs["positions"]
        hidden_states = kwargs["hidden_states"]
        residual = kwargs["residual"]
        forward_batch = kwargs["forward_batch"]
        try:
            if not _copy_into(entry.pos, positions):
                raise RuntimeError("pos shape mismatch")
            if not _copy_into(entry.hidden, hidden_states):
                raise RuntimeError("hidden shape mismatch")
            if entry.residual is not None and residual is not None:
                if not _copy_into(entry.residual, residual):
                    raise RuntimeError("residual shape mismatch")
            if not _sync_fb(entry.fb, forward_batch):
                raise RuntimeError("FB sync failed")
            entry.graph.replay()
            _stats["replay"] += 1
            meta = dict(entry.meta_template)
            meta["forward_batch"] = entry.fb
            meta["gemm_output_zero_allocator"] = kwargs.get(
                "gemm_output_zero_allocator"
            )
            meta["hidden_for_mlp"] = entry.out_hidden
            meta["hidden_states_orig"] = entry.hidden
            res_out = entry.out_residual if entry.out_residual is not None else residual
            return (
                entry.out_hidden,
                res_out,
                entry.out_topk_ids,
                entry.out_topk_weights,
                meta,
            )
        except Exception as e:
            for k, v in list(self.entries.items()):
                if v is entry:
                    self.blacklist.add(k)
                    self.entries.pop(k, None)
                    break
            _stats["capture_fail"] += 1
            logger.info("AFD farm layer CG replay failed (%s); eager", e)
            return eager()


_default_cache = FarmLayerCudaGraphCache()


def run_layer_forward_pre_ffn(layer: Any, **kwargs):
    return _default_cache.run_forward_pre_ffn(layer, **kwargs)


def reset_layer_cg_cache() -> None:
    _default_cache.entries.clear()
    _default_cache.blacklist.clear()
    reset_layer_cg_stats()


def capture_graphable_callable(
    fn: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
) -> Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
    """Test helper: capture ``out = fn(static_in)``."""
    static_in = sample.clone()
    static_out = torch.empty_like(static_in)
    static_out.copy_(fn(static_in))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out.copy_(fn(static_in))
    return g, static_in, static_out
