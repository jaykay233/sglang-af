# SPDX-License-Identifier: Apache-2.0
"""Interior-layer KV for AFD layer-merge (same-host cuda_ipc).

Pool sizes match via --max-total-tokens (no remap needed).
Layout:
  * FFN keeps LOCAL kv_buffer (MLA reads local HBM, fast).
  * Decode IPC views are P2P copy source.
  * req_to_token = Decode IPC view (shared, read-only, same indices).
  * Each decode step: bulk copy new KV rows Decode→FFN local.
  * ForwardBatch metadata cloned to FFN GPU.
"""

from __future__ import annotations

import logging
import os
import pickle
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.afd.mode import AfdMode, get_afd_mode, get_afd_transport_name
from sglang.srt.afd.remote_policy import afd_is_merge_interior, layer_merge_k
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_HDR = struct.Struct("!Q")

_merge_ready = False
_merge_lock = threading.Lock()
_remote_kv: Dict[int, torch.Tensor] = {}
_local_kv: Dict[int, torch.Tensor] = {}
_req_to_token: Optional[torch.Tensor] = None
_meta: Optional[Dict[str, torch.Tensor]] = None
_ffn_model_runner = None
_attn_model_runner = None
_max_bs: int = 64
_max_tokens: int = 256
_last_synced: Dict[int, int] = {}
_md_cache_key: Optional[Tuple[int, int]] = None
_kv_zero_copy: bool = False


def merge_kv_enabled() -> bool:
    return layer_merge_k() > 1 and get_afd_transport_name() == "cuda_ipc"


def _endpoint() -> str:
    base = (envs.SGLANG_AFD_IPC_ENDPOINT.get() or "/tmp/afd_cuda_ipc.sock").strip()
    return base + ".merge_kv"


def _send_msg(conn: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(_HDR.pack(len(payload)) + payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("AFD merge_kv socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> Any:
    (length,) = _HDR.unpack(_recv_exact(conn, _HDR.size))
    return pickle.loads(_recv_exact(conn, length))


def _share_tensor(t: torch.Tensor) -> Dict[str, Any]:
    t = t.contiguous()
    storage = t.untyped_storage()
    (device, handle, sz, off, rc_h, rc_o, ev_h, ev_s) = storage._share_cuda_()
    return {
        "dtype": str(t.dtype).replace("torch.", ""),
        "size": tuple(t.size()), "stride": tuple(t.stride()),
        "storage_offset": int(t.storage_offset()),
        "device": device, "handle": handle,
        "storage_size_bytes": sz, "storage_offset_bytes": off,
        "ref_counter_handle": rc_h, "ref_counter_offset": rc_o,
        "event_handle": ev_h, "event_sync_required": ev_s,
    }


def _open_tensor(meta: Dict[str, Any], device: torch.device) -> torch.Tensor:
    dtype = getattr(torch, meta["dtype"])
    storage = torch.UntypedStorage._new_shared_cuda(
        meta["device"], meta["handle"], meta["storage_size_bytes"],
        meta["storage_offset_bytes"], meta["ref_counter_handle"],
        meta["ref_counter_offset"], meta["event_handle"],
        meta["event_sync_required"],
    )
    t = torch.empty(0, dtype=dtype, device=device)
    t.set_(storage, meta["storage_offset"], torch.Size(meta["size"]))
    if tuple(t.stride()) != tuple(meta["stride"]):
        t = torch.as_strided(t, meta["size"], meta["stride"], meta["storage_offset"])
    return t


def _interior_layer_ids(num_layers: int) -> List[int]:
    return [i for i in range(num_layers) if afd_is_merge_interior(i)]


def _rebind_ffn_attn_backend(model_runner, pool, req_to_token: torch.Tensor) -> None:
    attn = getattr(model_runner, "attn_backend", None)
    if attn is None:
        return
    if hasattr(attn, "req_to_token"):
        attn.req_to_token = req_to_token
    # Keep pool object in sync — some paths read pool.req_to_token directly.
    rtp = getattr(attn, "req_to_token_pool", None) or getattr(
        model_runner, "req_to_token_pool", None
    )
    if rtp is not None and hasattr(rtp, "req_to_token"):
        rtp.req_to_token = req_to_token
    if hasattr(attn, "token_to_kv_pool"):
        attn.token_to_kv_pool = pool
    if hasattr(model_runner, "req_to_token_pool") and model_runner.req_to_token_pool is not None:
        model_runner.req_to_token_pool.req_to_token = req_to_token


def ensure_ffn_attn_bound() -> None:
    if not _merge_ready or _ffn_model_runner is None or _req_to_token is None:
        return
    pool = getattr(_ffn_model_runner, "token_to_kv_pool", None)
    if pool is None:
        return
    _rebind_ffn_attn_backend(_ffn_model_runner, pool, _req_to_token)


def get_ffn_attn_backend():
    if _ffn_model_runner is None:
        return None
    ensure_ffn_attn_bound()
    return getattr(_ffn_model_runner, "attn_backend", None)


def maybe_init_merge_kv_from_model_runner(model_runner) -> None:
    global _merge_ready, _remote_kv, _local_kv, _req_to_token, _meta
    global _ffn_model_runner, _attn_model_runner, _max_bs, _max_tokens

    if not merge_kv_enabled():
        return
    mode = get_afd_mode()
    if mode == AfdMode.NULL:
        return

    with _merge_lock:
        if _merge_ready and (
            (mode == AfdMode.ATTN and _attn_model_runner is not None)
            or (mode == AfdMode.FFN and _ffn_model_runner is not None)
        ):
            return

    _max_bs = max(1, int(envs.SGLANG_AFD_MAX_NUM_TOKEN.get() or 64))
    try:
        cfg = getattr(model_runner.server_args, "cuda_graph_config", None)
        if cfg is not None and getattr(cfg, "decode", None) is not None:
            _max_bs = max(_max_bs, int(cfg.decode.max_bs or _max_bs))
    except Exception:
        pass
    _max_tokens = max(_max_bs, int(envs.SGLANG_AFD_MAX_NUM_TOKEN.get() or _max_bs))

    if mode == AfdMode.ATTN:
        _attn_export(model_runner)
    elif mode == AfdMode.FFN:
        _ffn_import(model_runner)


def _attn_export(model_runner) -> None:
    global _merge_ready, _remote_kv, _req_to_token, _meta, _attn_model_runner

    pool = getattr(model_runner, "token_to_kv_pool", None)
    rtp = getattr(model_runner, "req_to_token_pool", None)
    if pool is None or not hasattr(pool, "kv_buffer"):
        logger.warning("AFD merge_kv Attn: no MLA kv_buffer; skip share")
        return
    if rtp is None or not hasattr(rtp, "req_to_token"):
        logger.warning("AFD merge_kv Attn: no req_to_token; skip share")
        return

    num_layers = len(pool.kv_buffer)
    start = int(getattr(pool, "start_layer", 0) or 0)
    interiors = _interior_layer_ids(start + num_layers)
    layer_meta = {}
    for lid in interiors:
        idx = lid - start
        if 0 <= idx < num_layers:
            layer_meta[lid] = _share_tensor(pool.kv_buffer[idx])
            _remote_kv[lid] = pool.kv_buffer[idx]

    device = pool.kv_buffer[0].device
    meta_bufs = {
        "batch_size": torch.zeros(1, dtype=torch.int32, device=device),
        "seq_lens": torch.zeros(_max_bs, dtype=torch.int32, device=device),
        "req_pool_indices": torch.zeros(_max_bs, dtype=torch.int32, device=device),
        "out_cache_loc": torch.zeros(_max_tokens, dtype=torch.int32, device=device),
        "input_ids": torch.zeros(_max_tokens, dtype=torch.int32, device=device),
        "positions": torch.zeros(_max_tokens, dtype=torch.int64, device=device),
        "seq_lens_sum": torch.zeros(1, dtype=torch.int32, device=device),
    }
    _meta = meta_bufs
    _req_to_token = rtp.req_to_token
    _attn_model_runner = model_runner

    payload = {
        "layers": layer_meta,
        "req_to_token": _share_tensor(rtp.req_to_token),
        "meta": {k: _share_tensor(v) for k, v in meta_bufs.items()},
        "max_bs": _max_bs, "max_tokens": _max_tokens,
    }

    path = _endpoint()
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(1)
    sock.settimeout(180.0)
    logger.info("AFD merge_kv Attn waiting for FFN on %s (interiors=%s)", path, interiors)
    conn, _ = sock.accept()
    conn.settimeout(120.0)
    _send_msg(conn, payload)
    ack = _recv_msg(conn)
    conn.close()
    sock.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    if not isinstance(ack, dict) or ack.get("status") != "ok":
        raise RuntimeError(f"AFD merge_kv bad FFN ack: {ack!r}")
    with _merge_lock:
        _merge_ready = True
    logger.info("AFD merge_kv Attn exported layers=%s", interiors)


def _ffn_import(model_runner) -> None:
    global _merge_ready, _remote_kv, _local_kv, _req_to_token, _meta
    global _ffn_model_runner, _max_bs, _max_tokens, _last_synced

    path = _endpoint()
    deadline = time.time() + 180.0
    conn = None
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(path)
            conn = s
            break
        except OSError as e:
            last_err = e
            time.sleep(0.2)
    if conn is None:
        raise RuntimeError(f"AFD merge_kv FFN connect failed {path}: {last_err}")

    payload = _recv_msg(conn)
    device = torch.device(f"cuda:{model_runner.gpu_id}")
    try:
        torch.cuda.set_device(device)
    except Exception:
        pass

    _remote_kv = {int(lid): _open_tensor(meta, device) for lid, meta in payload["layers"].items()}
    req_to_token = _open_tensor(payload["req_to_token"], device)
    meta_bufs = {k: _open_tensor(v, device) for k, v in payload["meta"].items()}
    _max_bs = int(payload.get("max_bs", _max_bs))
    _max_tokens = int(payload.get("max_tokens", _max_tokens))

    pool = getattr(model_runner, "token_to_kv_pool", None)
    if pool is None or not hasattr(pool, "kv_buffer"):
        raise RuntimeError("FFN has no local MLA kv_buffer for merge_kv")
    start = int(getattr(pool, "start_layer", 0) or 0)

    # Prefer zero-copy: FFN MLA reads/writes Decode's interior KV via IPC.
    # Local-mirror + prefix sync is the fallback when disabled.
    global _kv_zero_copy
    _kv_zero_copy = bool(int(os.environ.get("SGLANG_AFD_KV_ZERO_COPY", "1")))
    for lid in _remote_kv:
        idx = int(lid) - start
        if idx < 0 or idx >= len(pool.kv_buffer):
            raise IndexError(f"merge_kv layer {lid} out of FFN pool range")
        if pool.kv_buffer[idx].shape != _remote_kv[lid].shape:
            raise RuntimeError(
                f"AFD merge_kv pool size mismatch layer={lid} "
                f"FFN={tuple(pool.kv_buffer[idx].shape)} "
                f"Decode={tuple(_remote_kv[lid].shape)}. "
                f"Use --max-total-tokens {_remote_kv[lid].shape[0]} to match."
            )
        if _kv_zero_copy:
            pool.kv_buffer[idx] = _remote_kv[lid]
            _local_kv[lid] = _remote_kv[lid]
        else:
            _local_kv[lid] = pool.kv_buffer[idx]

    # Live IPC view — Decode updates this every step; must NOT clone.
    _req_to_token = req_to_token
    _meta = meta_bufs
    _ffn_model_runner = model_runner
    _last_synced = {}
    _rebind_ffn_attn_backend(model_runner, pool, _req_to_token)
    _send_msg(conn, {"status": "ok"})
    conn.close()
    with _merge_lock:
        _merge_ready = True
    logger.info(
        "AFD merge_kv FFN %s KV layers=%s shape=%s",
        "zero-copy" if _kv_zero_copy else "local-mirror",
        sorted(_local_kv.keys()), tuple(next(iter(_local_kv.values())).shape),
    )


def sync_interior_kv_prefix() -> int:
    """Copy Decode→FFN prefix only; never the current decode token slot.

    Interior attention runs on FFN, so Decode's KV at out_cache_loc is stale.
    Copy ``[have, seq_len-1)`` from Decode, then let FFN write the newest row.
    """
    global _last_synced
    if _kv_zero_copy:
        return 0
    if not _merge_ready or _meta is None or not _local_kv or not _remote_kv:
        return 0
    if _req_to_token is None:
        return 0

    bs = int(_meta["batch_size"].item())
    if bs <= 0:
        return 0

    # Pull lens/ids to CPU once (small bs).
    seq_lens = _meta["seq_lens"][:bs].detach().cpu().tolist()
    req_ids = _meta["req_pool_indices"][:bs].detach().cpu().tolist()

    all_locs: List[torch.Tensor] = []
    for i in range(bs):
        rid = int(req_ids[i])
        sl = int(seq_lens[i])
        if sl <= 1:
            continue
        if rid < 0 or rid >= _req_to_token.shape[0]:
            continue
        if sl > _req_to_token.shape[1]:
            continue

        have = _last_synced.get(rid, 0)
        # Req_pool slot reuse: seq_len can shrink — force full prefix resync.
        if sl < have:
            have = 0
        # Copy only newly needed prefix rows from Decode. FFN owns writes for
        # prior decode tokens on interior layers; resetting have=0 every step
        # overwrote those with Decode's never-updated (stale) interior KV and
        # produced garbage (retok ~153/1329).
        copy_end = sl - 1
        if copy_end > have:
            locs = _req_to_token[rid, have:copy_end].to(dtype=torch.long, copy=True)
            all_locs.append(locs)
        _last_synced[rid] = sl

    if not all_locs:
        return 0

    flat = torch.cat(all_locs)
    flat = flat[flat > 0]
    if flat.numel() == 0:
        return 0

    row_t = flat.unique()
    for lid, dst in _local_kv.items():
        src = _remote_kv.get(lid)
        if src is None:
            continue
        dst.index_copy_(0, row_t, src.index_select(0, row_t))

    return int(row_t.numel())


# Live FB / padded buffer view set by DecodeCudaGraphRunner.load_batch (or
# eager model.forward). Breakable CG must NOT publish the capture-time FB:
# its Python ints (batch_size, seq_lens_sum) stay frozen at capture dummies.
_live_publish_fb: Any = None


def set_live_publish_forward_batch(forward_batch) -> None:
    global _live_publish_fb
    _live_publish_fb = forward_batch


def publish_merge_forward_meta(forward_batch=None) -> None:
    if not _merge_ready or _meta is None or get_afd_mode() != AfdMode.ATTN:
        return
    fb = forward_batch if forward_batch is not None else _live_publish_fb
    if fb is None:
        return
    bs = int(fb.batch_size)
    sl = fb.seq_lens
    if sl is None or sl.numel() == 0:
        return
    t = int(fb.input_ids.shape[0]) if getattr(fb, "input_ids", None) is not None else bs
    # Always derive sum from the tensor — FB.seq_lens_sum is a stale Python
    # int under breakable CUDA-graph replay.
    # Avoid .item() / D2H while a CUDAGraph is being captured — that sync
    # invalidates the capture stream.
    capturing = bool(torch.cuda.is_current_stream_capturing())
    if capturing:
        # Device-only write; no D2H.
        _meta["seq_lens_sum"].copy_(
            sl[:bs].to(dtype=_meta["seq_lens_sum"].dtype).sum().reshape(1)
        )
    else:
        _meta["seq_lens_sum"].fill_(int(sl[:bs].sum().item()))
    _meta["batch_size"].fill_(bs)
    _meta["seq_lens"][:bs].copy_(sl[:bs].to(dtype=_meta["seq_lens"].dtype))
    rpi = fb.req_pool_indices
    _meta["req_pool_indices"][:bs].copy_(rpi[:bs].to(dtype=_meta["req_pool_indices"].dtype))
    ocl = fb.out_cache_loc
    n_loc = min(t, ocl.numel(), _meta["out_cache_loc"].numel())
    _meta["out_cache_loc"][:n_loc].copy_(ocl[:n_loc].to(dtype=_meta["out_cache_loc"].dtype))
    if getattr(fb, "input_ids", None) is not None:
        n_id = min(t, fb.input_ids.numel(), _meta["input_ids"].numel())
        _meta["input_ids"][:n_id].copy_(fb.input_ids[:n_id].to(dtype=_meta["input_ids"].dtype))
    pos = getattr(fb, "positions", None)
    if pos is not None:
        n_p = min(t, pos.numel(), _meta["positions"].numel())
        _meta["positions"][:n_p].copy_(pos[:n_p].to(dtype=_meta["positions"].dtype))
    if capturing:
        return
    nlog = int(getattr(publish_merge_forward_meta, "_nlog", 0))
    if nlog < 3:
        publish_merge_forward_meta._nlog = nlog + 1  # type: ignore[attr-defined]
        logger.info(
            "AFD merge_kv publish meta#%d bs=%s seq0=%s out0=%s pos0=%s",
            nlog,
            bs,
            int(sl[0].item()),
            int(ocl[0].item()) if ocl is not None and ocl.numel() else -1,
            int(pos[0].item()) if pos is not None and pos.numel() else -1,
        )


def get_merge_forward_batch(
    *, num_tokens: int, positions: Optional[torch.Tensor] = None
) -> Tuple[Any, Any]:
    global _md_cache_key
    if not _merge_ready or _meta is None or _ffn_model_runner is None:
        raise RuntimeError("AFD merge_kv not ready on FFN")

    ensure_ffn_attn_bound()
    sync_interior_kv_prefix()

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.utils import BumpAllocator

    bs = int(_meta["batch_size"].item())
    if bs <= 0:
        bs = max(1, num_tokens)
    t = max(1, int(num_tokens))
    ffn_dev = _ffn_model_runner.device

    seq_lens = _meta["seq_lens"][:bs].to(ffn_dev, copy=True)
    req_pool_indices = _meta["req_pool_indices"][:bs].to(ffn_dev, copy=True)
    out_cache_loc = _meta["out_cache_loc"][:t].to(ffn_dev, copy=True)
    input_ids = _meta["input_ids"][:t].to(ffn_dev, copy=True)
    pos = positions if positions is not None else _meta["positions"][:t].to(ffn_dev, copy=True)
    seq_lens_sum = int(seq_lens.sum().item())
    max_seq = int(seq_lens.max().item()) if seq_lens.numel() else 0

    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens.detach().to("cpu"),
    )
    fb.positions = pos[:t]

    attn = getattr(_ffn_model_runner, "attn_backend", None)
    # Rebuild metadata every step — kv page tables must track out_cache_loc.
    if attn is not None and hasattr(attn, "init_forward_metadata"):
        try:
            attn.init_forward_metadata(fb)
            _md_cache_key = (bs, max_seq)
        except Exception as e:
            logger.warning("AFD merge_kv init_forward_metadata failed: %s", e)
            _md_cache_key = None

    zero_allocator = BumpAllocator(buffer_size=64, dtype=torch.float32, device=ffn_dev)
    return fb, zero_allocator


def is_merge_kv_ready() -> bool:
    return _merge_ready