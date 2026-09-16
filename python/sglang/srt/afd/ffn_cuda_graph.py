# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph for AFD FFN compute (MLP / MoE experts only).

Model-level decode CUDA Graph cannot run on the FFN worker: ``self_attn`` is an
``AfdMissingModule`` stub. Instead we capture per-(layer, token-bucket) graphs
around the same kernels used by ``make_model_ffn_compute``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from sglang.srt.afd.ffn_compute import make_model_ffn_compute, run_layer_ffn
from sglang.srt.afd.moe_bridge import get_decoder_layers
from sglang.srt.afd.protocol import AfdServerBatch
from sglang.srt.afd.transport import FfnComputeFn
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_GraphKey = Tuple[int, int, int]  # (layer_id, bucket_tokens, mb_id)


def afd_ffn_cuda_graph_enabled() -> bool:
    return bool(envs.SGLANG_AFD_FFN_CUDA_GRAPH.get())


def default_ffn_cg_buckets(max_num_token: int) -> List[int]:
    """Token buckets aligned with common decode CG sizes."""
    max_n = max(1, int(max_num_token))
    candidates = [
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
    ]
    out = [b for b in candidates if b <= max_n]
    if not out or out[-1] != max_n:
        out.append(max_n)
    return out


def pick_ffn_cg_bucket(num_tokens: int, buckets: Sequence[int]) -> int:
    if num_tokens <= 0:
        return buckets[0]
    for b in buckets:
        if b >= num_tokens:
            return b
    return buckets[-1]


class CudaGraphFfnCompute:
    """Lazy per-(layer, bs) CUDA Graph wrapper over model FFN compute."""

    def __init__(
        self,
        model: nn.Module,
        *,
        max_num_token: int,
        device: torch.device,
        dtype: torch.dtype,
        hidden_size: Optional[int] = None,
        buckets: Optional[Sequence[int]] = None,
        enabled: bool = True,
    ):
        self._model = model
        self._layers = get_decoder_layers(model)
        self._afd_layers = self._layers  # used by same-layer gather fuse
        self._eager: FfnComputeFn = make_model_ffn_compute(model)
        self._max_num_token = int(max_num_token)
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self._dtype = dtype
        self._hidden = int(hidden_size) if hidden_size else None
        self._buckets = list(buckets or default_ffn_cg_buckets(self._max_num_token))
        self._enabled = bool(enabled) and self._device.type == "cuda"
        self._graphs: Dict[_GraphKey, torch.cuda.CUDAGraph] = {}
        self._static_in: Dict[_GraphKey, torch.Tensor] = {}
        self._static_out: Dict[_GraphKey, torch.Tensor] = {}
        self._static_topk_ids: Dict[_GraphKey, torch.Tensor] = {}
        self._static_topk_weights: Dict[_GraphKey, torch.Tensor] = {}
        self._out_scratch: Dict[_GraphKey, torch.Tensor] = {}
        self._capture_failed: set[_GraphKey] = set()
        # Experimental: MoE kernels capture/replay directly on IPC buffers.
        self._shared: Dict[int, Dict[str, torch.Tensor]] = {}
        # Stable: IPC A2F/F2A used only as memcpy endpoints (MoE on private static).
        self._ipc_io: Dict[int, Dict[str, torch.Tensor]] = {}
        # Per-mb streams used when SGLANG_AFD_FFN_PARALLEL_MB=1.
        self._streams: Dict[int, torch.cuda.Stream] = {}
        self._stream = (
            torch.cuda.Stream(device=self._device) if self._enabled else None
        )
        if self._enabled and self._stream is not None:
            self._streams[0] = self._stream

    def _parallel_mb_enabled(self) -> bool:
        return bool(envs.SGLANG_AFD_FFN_PARALLEL_MB.get())

    def _stream_for_mb(self, mb: int) -> Optional[torch.cuda.Stream]:
        """Per-mb capture/replay stream when PARALLEL_MB; else None (default stream)."""
        if not self._enabled or not self._parallel_mb_enabled():
            return None
        mb_i = int(mb)
        st = self._streams.get(mb_i)
        if st is None:
            st = torch.cuda.Stream(device=self._device)
            self._streams[mb_i] = st
            if mb_i == 0:
                self._stream = st
        return st

    def bind_shared_slot(
        self,
        mb: int,
        *,
        hidden: torch.Tensor,
        mlp_out: torch.Tensor,
        topk_ids: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Bind FFN-local IPC buffers for the CUDA-graph IO path.

        Default (stable): MoE runs on private static buffers; the graph ends with
        ``F2A[:bs].copy_(static_out)``. IPC storage is only a memcpy destination
        (still a local ``cudaMalloc`` on the FFN exporter), not MoE workspace.

        ``SGLANG_AFD_FFN_SHARED_IO=1``: capture MoE directly against IPC views
        (historically ``cudaErrorIllegalAddress`` with Triton MoE — experimental).
        """
        slot = {
            "hidden": hidden,
            "mlp_out": mlp_out,
            **({"topk_ids": topk_ids} if topk_ids is not None else {}),
            **({"topk_weights": topk_weights} if topk_weights is not None else {}),
        }
        mb_i = int(mb)
        if bool(envs.SGLANG_AFD_FFN_SHARED_IO.get()):
            self._shared[mb_i] = slot
            self._ipc_io.pop(mb_i, None)
            logger.warning(
                "AFD FFN SHARED_IO=1: capturing MoE on IPC buffers (mb=%s)", mb_i
            )
        else:
            self._ipc_io[mb_i] = slot
            self._shared.pop(mb_i, None)
            logger.info(
                "AFD FFN IPC staging bound mb=%s (private MoE + in-graph F2A copy)",
                mb_i,
            )
        self._drop_cached_graphs()

    def _drop_cached_graphs(self) -> None:
        for k in list(self._graphs.keys()):
            self._graphs.pop(k, None)
            self._static_in.pop(k, None)
            self._static_out.pop(k, None)
            self._static_topk_ids.pop(k, None)
            self._static_topk_weights.pop(k, None)
            self._out_scratch.pop(k, None)
            self._capture_failed.discard(k)


    @property
    def enabled(self) -> bool:
        return self._enabled

    def __call__(self, batch: AfdServerBatch):
        if not self._enabled:
            return self._eager(batch)
        t = batch.num_tokens
        if t <= 0:
            return self._eager(batch)
        try:
            return self._call_graph(batch)
        except Exception:
            logger.exception(
                "AFD FFN CUDA graph path failed layer=%s tokens=%s; falling back to eager",
                batch.layer_id,
                t,
            )
            return self._eager(batch)

    def capture_buckets(
        self,
        *,
        layer_ids: Optional[Sequence[int]] = None,
        buckets: Optional[Sequence[int]] = None,
    ) -> int:
        """Eagerly capture graphs (optional warm-up). Returns #graphs captured."""
        if not self._enabled:
            return 0
        layers = (
            list(layer_ids)
            if layer_ids is not None
            else list(range(len(self._layers)))
        )
        bs_list = list(buckets) if buckets is not None else self._buckets
        # Warm against every bound IPC mb (usually 1); else mb=0 private-only.
        mbs = sorted(set(self._ipc_io) | set(self._shared)) or [0]
        n = 0
        for lid in layers:
            for bs in bs_list:
                for mb in mbs:
                    if self._ensure_graph(lid, bs, mb=mb):
                        n += 1
        logger.info(
            "AFD FFN CUDA graph warm-up done: captured=%s layers=%s buckets=%s mbs=%s",
            n,
            len(layers),
            bs_list,
            mbs,
        )
        return n

    def _call_graph(self, batch: AfdServerBatch):
        from contextlib import nullcontext

        layer_id = batch.layer_id
        t = batch.num_tokens
        bs = pick_ffn_cg_bucket(t, self._buckets)
        mb = int(getattr(batch, "_mb_id", 0) or 0)
        key: _GraphKey = (layer_id, bs, mb)
        shared = self._shared.get(mb)
        ipc = self._ipc_io.get(mb)
        if key in self._capture_failed:
            return self._eager(batch)
        if not self._ensure_graph(layer_id, bs, mb=mb):
            return self._eager(batch)

        # Replay on the capture stream (required). Per-mb streams only when PARALLEL_MB.
        st = self._stream_for_mb(mb)
        ctx = torch.cuda.stream(st) if st is not None else nullcontext()
        cur = torch.cuda.current_stream(self._device)

        if shared is not None:
            # Experimental: MoE on IPC views.
            with ctx:
                if t < bs:
                    shared["hidden"][t:bs].zero_()
                    if "topk_ids" in shared:
                        shared["topk_ids"][t:bs].zero_()
                        shared["topk_weights"][t:bs].zero_()
                self._graphs[key].replay()
            if st is not None:
                cur.wait_stream(st)
            out = shared["mlp_out"]
            batch._f2a_bufs = [out]  # type: ignore[attr-defined]
            return [out]

        hidden = batch.hidden
        static_in = self._static_in[key]
        topk_ids = batch.topk_ids
        topk_weights = batch.topk_weights
        if key in self._static_topk_ids and (
            topk_ids is None or topk_weights is None
        ):
            return self._eager(batch)
        with ctx:
            static_in[:t].copy_(hidden[:t], non_blocking=True)
            if t < bs:
                static_in[t:].zero_()

            if key in self._static_topk_ids:
                self._static_topk_ids[key][:t].copy_(topk_ids[:t], non_blocking=True)
                self._static_topk_weights[key][:t].copy_(
                    topk_weights[:t], non_blocking=True
                )
                if t < bs:
                    self._static_topk_ids[key][t:].zero_()
                    self._static_topk_weights[key][t:].zero_()

            self._graphs[key].replay()

            y = self._static_out[key]
            f2a_bufs = getattr(batch, "_f2a_bufs", None)
            # In-graph F2A copy already filled IPC when staging was bound at capture.
            if ipc is not None and f2a_bufs is not None and len(f2a_bufs) == 1:
                if f2a_bufs[0].data_ptr() == ipc["mlp_out"].data_ptr():
                    if st is not None:
                        cur.wait_stream(st)
                    return [f2a_bufs[0]]
            if (
                isinstance(f2a_bufs, (list, tuple))
                and len(f2a_bufs) == 1
                and f2a_bufs[0].shape[-1] == y.shape[-1]
                and f2a_bufs[0].dtype == y.dtype
                and f2a_bufs[0].device == y.device
            ):
                n = min(f2a_bufs[0].shape[0], t)
                f2a_bufs[0][:n].copy_(y[:n], non_blocking=True)
                if st is not None:
                    cur.wait_stream(st)
                return [f2a_bufs[0]]

            out_rows = batch.hidden_wire.shape[0]
            scratch = self._out_scratch.get(key)
            if (
                scratch is None
                or scratch.shape[0] != out_rows
                or scratch.shape[1] != y.shape[-1]
                or scratch.dtype != y.dtype
            ):
                scratch = torch.empty(
                    out_rows, y.shape[-1], dtype=y.dtype, device=y.device
                )
                self._out_scratch[key] = scratch
            scratch[:t].copy_(y[:t], non_blocking=True)
            if st is not None:
                cur.wait_stream(st)
            return [scratch]

    def _ensure_graph(self, layer_id: int, bs: int, mb: int = 0) -> bool:
        key: _GraphKey = (layer_id, bs, mb)
        if key in self._graphs:
            return True
        if key in self._capture_failed:
            return False
        try:
            self._capture(layer_id, bs, mb=mb)
            return True
        except Exception as e:
            self._capture_failed.add(key)
            logger.warning(
                "AFD FFN CUDA graph capture failed layer=%s bs=%s mb=%s: %s",
                layer_id,
                bs,
                mb,
                e,
            )
            return False

    def _capture(self, layer_id: int, bs: int, mb: int = 0) -> None:
        if layer_id < 0 or layer_id >= len(self._layers):
            raise IndexError(f"layer_id={layer_id} out of range")
        mlp = self._layers[layer_id].mlp
        key: _GraphKey = (layer_id, bs, mb)
        device = self._device
        dtype = self._dtype
        try:
            dtype = next(mlp.parameters()).dtype
        except StopIteration:
            pass

        # Infer whether this layer needs topk from scheme-A MoE path.
        needs_topk = False
        topk_k = 0
        from sglang.srt.afd.routing_scheme import is_scheme_b
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        if isinstance(mlp, DeepseekV2MoE) and not is_scheme_b():
            needs_topk = True
            cfg = getattr(mlp, "config", None)
            topk_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
            if topk_k <= 0:
                topk_k = int(getattr(mlp, "top_k", 0) or 0)
            if topk_k <= 0:
                raise RuntimeError(
                    f"MoE layer {layer_id} needs topk width for FFN CUDA graph"
                )

        shared = self._shared.get(mb)
        ipc = None if shared is not None else self._ipc_io.get(mb)
        mode = "shared_moe" if shared is not None else (
            "private+f2a" if ipc is not None else "private"
        )

        if shared is not None:
            hid = shared["hidden"]
            out_buf = shared["mlp_out"]
            if hid.shape[0] < bs or out_buf.shape[0] < bs:
                raise RuntimeError(
                    f"shared IPC rows {hid.shape[0]} < bucket bs={bs}"
                )
            static_in = hid[:bs]
            static_out = out_buf[:bs]
            static_ids = None
            static_w = None
            if needs_topk:
                if "topk_ids" not in shared or "topk_weights" not in shared:
                    raise RuntimeError("shared IPC missing topk buffers for MoE")
                static_ids = shared["topk_ids"][:bs]
                static_w = shared["topk_weights"][:bs]
            f2a_dst = None
        else:
            static_in = torch.zeros(bs, self._hidden_size(), dtype=dtype, device=device)
            static_out = torch.empty_like(static_in)
            static_ids = None
            static_w = None
            if needs_topk:
                static_ids = torch.zeros(bs, topk_k, dtype=torch.int32, device=device)
                static_w = torch.zeros(bs, topk_k, dtype=torch.float32, device=device)
            f2a_dst = None
            if ipc is not None:
                if ipc["mlp_out"].shape[0] < bs:
                    raise RuntimeError(
                        f"IPC F2A rows {ipc['mlp_out'].shape[0]} < bucket bs={bs}"
                    )
                f2a_dst = ipc["mlp_out"][:bs]

        def _run():
            y = run_layer_ffn(
                mlp,
                static_in,
                topk_ids=static_ids,
                topk_weights=static_w,
            )
            static_out.copy_(y)
            if f2a_dst is not None:
                f2a_dst.copy_(static_out)

        assert self._stream is not None or self._streams or self._enabled
        cap_stream = self._stream_for_mb(mb)
        if cap_stream is None:
            # Non-parallel: capture/replay on the caller's current stream.
            with torch.cuda.device(device):
                _run()
                torch.cuda.current_stream(device).synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _run()
                torch.cuda.current_stream(device).synchronize()
        else:
            torch.cuda.current_stream(device).wait_stream(cap_stream)
            with torch.cuda.device(device), torch.cuda.stream(cap_stream):
                # Warmup outside graph.
                _run()
                cap_stream.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=cap_stream):
                    _run()
                cap_stream.synchronize()

        self._graphs[key] = g
        self._static_in[key] = static_in
        self._static_out[key] = static_out
        if needs_topk:
            assert static_ids is not None and static_w is not None
            self._static_topk_ids[key] = static_ids
            self._static_topk_weights[key] = static_w
        logger.info(
            "AFD FFN CUDA graph captured layer=%s bs=%s mb=%s topk=%s mode=%s",
            layer_id,
            bs,
            mb,
            needs_topk,
            mode,
        )

    def _hidden_size(self) -> int:
        if self._hidden is not None:
            return self._hidden
        for layer in self._layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if hasattr(mlp, "gate_proj") and hasattr(mlp.gate_proj, "weight"):
                return int(mlp.gate_proj.weight.shape[1])
            cfg = getattr(mlp, "config", None)
            if cfg is not None and hasattr(cfg, "hidden_size"):
                return int(cfg.hidden_size)
            try:
                p = next(mlp.parameters())
                if p.dim() >= 2:
                    return int(p.shape[-1])
            except StopIteration:
                continue
        raise RuntimeError("Cannot infer hidden size for FFN CUDA graph")


def maybe_wrap_ffn_cuda_graph(
    model: nn.Module,
    *,
    max_num_token: int,
    device,
    dtype: torch.dtype,
    hidden_size: Optional[int] = None,
) -> FfnComputeFn:
    """Return graph-wrapped FFN compute when enabled; else plain eager compute.

    Layer-merge still uses eager for interior Attn, but the wrapper keeps a
    graphed single-layer MLP path for ``merge_k==1`` traffic and as a fast
    path for the group-start MLP when shapes match.
    """
    eager = make_model_ffn_compute(model)
    if not afd_ffn_cuda_graph_enabled():
        return eager
    merge_on = False
    try:
        from sglang.srt.afd.remote_policy import layer_merge_k

        merge_on = layer_merge_k() > 1
    except Exception:
        pass
    if merge_on:
        logger.info(
            "AFD FFN CUDA graph skipped for merge mode (group-start MLP is "
            "only 1/K layers; CG stream ops risk corrupting scheduler context)"
        )
        return eager
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type != "cuda":
        logger.info("AFD FFN CUDA graph skipped (device=%s)", dev)
        return eager
    runner = CudaGraphFfnCompute(
        model,
        max_num_token=max_num_token,
        device=dev,
        dtype=dtype,
        hidden_size=hidden_size,
        enabled=True,
    )
    return runner
