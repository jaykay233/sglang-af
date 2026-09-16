# SPDX-License-Identifier: Apache-2.0
"""Simulate MoE EP dispatch/combine + optional MoE compute tax on Lite.

Purpose
-------
Fair AF vs PD/full comparisons without loading V3: pure decode on TP=1 normally
has **zero** cross-device MoE traffic, while AF always pays A2F/F2A. Large DeepSeek
serving pays DeepEP-style all-to-all and much heavier expert GEMMs. This module
injects an **equivalent traffic (+ optional compute) tax** into each
``DeepseekV2MoE.forward``.

Volume model (per MoE layer, both directions)::

    V_bytes ≈ 2 * N_tok * topk * hidden * elem_size * (1 - 1/ep_size) * bytes_scale

``SGLANG_FAKE_V3_PROFILE=v3_proxy`` sizes topk/hidden/ep/compute like DeepSeek-V3
while still running DeepSeek-V2-Lite weights (see ``apply_v3_proxy_profile``).

Modes
-----
- ``delay``: CUDA-sync + host sleep for T = lat + V/bw
- ``copy``:  two device memcpy of size V/2 (HBM traffic proxy)
- ``nvlink``: copy to peer GPU and back when visible

Enable::

    export SGLANG_FAKE_EP_COMM=1
    export SGLANG_FAKE_V3_PROFILE=v3_proxy   # or set SIZE/TOPK/HIDDEN/COMPUTE manually
    export SGLANG_FAKE_EP_FULL_ONLY=1        # default: only tax AFD mode=null for *EP*
    # MoE compute scale still applies on AF FFN (experts live there).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_bufs: dict = {}
_w_cache: dict = {}
_logged = False
_profile_applied = False
_stats = {"calls": 0, "bytes": 0, "us": 0.0, "compute_us": 0.0}

# DeepSeek-V3-ish vs V2-Lite MoE (approx; for proxy sizing only).
_V3 = dict(hidden=7168, topk=8, ep=64, intermediate=2048, moe_layers=58)
_LITE = dict(hidden=2048, topk=6, ep=8, intermediate=1408, moe_layers=26)


def apply_v3_proxy_profile(force: bool = False) -> Optional[dict]:
    """Apply env defaults that make Lite MoE tax look like V3 decode intensity."""
    global _profile_applied
    from sglang.srt.environ import envs

    name = str(envs.SGLANG_FAKE_V3_PROFILE.get() or "").strip().lower()
    if name not in ("v3", "v3_proxy", "deepseek_v3"):
        return None
    if _profile_applied and not force:
        return {"profile": name, "already": True}

    # Per-layer payload ~ (topk * H * remote_frac)_v3 / _lite
    remote_v3 = 1.0 - 1.0 / _V3["ep"]
    remote_lite = 1.0 - 1.0 / _LITE["ep"]
    per_layer_comm = (
        (_V3["topk"] / _LITE["topk"])
        * (_V3["hidden"] / _LITE["hidden"])
        * (remote_v3 / remote_lite)
    )
    # Fold deeper MoE stack into per-layer tax so Lite's fewer layers still hurt.
    layer_factor = _V3["moe_layers"] / _LITE["moe_layers"]
    # FLOPs ~ topk * intermediate * hidden
    flop_v3 = _V3["topk"] * _V3["intermediate"] * _V3["hidden"]
    flop_lite = _LITE["topk"] * _LITE["intermediate"] * _LITE["hidden"]
    compute_scale = (flop_v3 / flop_lite) * layer_factor

    envs.SGLANG_FAKE_EP_SIZE.set(_V3["ep"])
    envs.SGLANG_FAKE_EP_TOPK.set(_V3["topk"])
    envs.SGLANG_FAKE_EP_HIDDEN.set(_V3["hidden"])
    # Dims already V3-sized; multiply by layer_factor only.
    envs.SGLANG_FAKE_EP_BYTES_SCALE.set(float(layer_factor))
    envs.SGLANG_FAKE_MOE_COMPUTE_SCALE.set(float(compute_scale))
    if not envs.SGLANG_FAKE_EP_MODE.get():
        envs.SGLANG_FAKE_EP_MODE.set("copy")
    _profile_applied = True
    meta = {
        "profile": name,
        "ep_size": _V3["ep"],
        "topk": _V3["topk"],
        "hidden": _V3["hidden"],
        "bytes_scale": layer_factor,
        "per_layer_comm_vs_lite": per_layer_comm,
        "compute_scale": compute_scale,
        "layer_factor": layer_factor,
    }
    logger.warning("FAKE_V3_PROFILE applied: %s", meta)
    return meta


def fake_ep_enabled() -> bool:
    from sglang.srt.environ import envs

    apply_v3_proxy_profile()
    if not bool(envs.SGLANG_FAKE_EP_COMM.get()):
        return False
    if bool(envs.SGLANG_FAKE_EP_FULL_ONLY.get()):
        try:
            from sglang.srt.afd.mode import AfdMode, get_afd_mode

            if get_afd_mode() != AfdMode.NULL:
                return False
        except Exception:
            pass
    return True


def fake_moe_compute_enabled() -> bool:
    from sglang.srt.environ import envs

    apply_v3_proxy_profile()
    return float(envs.SGLANG_FAKE_MOE_COMPUTE_SCALE.get() or 0.0) > 0.0


def ep_comm_bytes(
    num_tokens: int,
    *,
    topk: int,
    hidden: int,
    elem_size: int,
    ep_size: int,
    bytes_scale: float = 1.0,
) -> int:
    """Dispatch+combine payload bytes (remote fraction only)."""
    if num_tokens <= 0 or topk <= 0 or hidden <= 0:
        return 0
    ep = max(1, int(ep_size))
    remote_frac = 0.0 if ep <= 1 else (1.0 - 1.0 / ep)
    one_way = int(num_tokens * topk * hidden * elem_size * remote_frac)
    return int(2 * one_way * max(0.0, float(bytes_scale)))


def ep_comm_seconds(nbytes: int, *, bw_gbs: float, lat_us: float) -> float:
    if nbytes <= 0:
        return 0.0
    bw = max(1e-3, float(bw_gbs)) * (1 << 30)
    return (float(lat_us) * 1e-6) + (nbytes / bw)


def _cached_buf(key: Tuple, n_bytes: int, device: torch.device) -> torch.Tensor:
    n_elem = max(1, (n_bytes + 1) // 2)  # float16 elements
    cached = _bufs.get(key)
    if cached is None or cached.numel() < n_elem or cached.device != device:
        cached = torch.empty(n_elem, dtype=torch.float16, device=device)
        _bufs[key] = cached
    return cached


def _do_copy_traffic(nbytes: int, device: torch.device) -> None:
    """Two device copies totaling ~nbytes (dispatch + combine proxy)."""
    half = max(2, nbytes // 2)
    a = _cached_buf(("a", device), half, device)
    b = _cached_buf(("b", device), half, device)
    n = max(1, half // 2)
    b[:n].copy_(a[:n])
    a[:n].copy_(b[:n])


def _do_nvlink_traffic(nbytes: int, src: torch.device, peer_idx: int) -> None:
    if peer_idx < 0 or not torch.cuda.is_available():
        _do_copy_traffic(nbytes, src)
        return
    if peer_idx >= torch.cuda.device_count():
        _do_copy_traffic(nbytes, src)
        return
    half = max(2, nbytes // 2)
    n = max(1, half // 2)
    local = _cached_buf(("nv_local", src), half, src)
    with torch.cuda.device(peer_idx):
        remote = _cached_buf(
            ("nv_remote", torch.device("cuda", peer_idx)), half, torch.device("cuda", peer_idx)
        )
        remote[:n].copy_(local[:n])
        local[:n].copy_(remote[:n])


def _is_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _effective_topk_hidden(topk: int, hidden_size: int) -> Tuple[int, int]:
    from sglang.srt.environ import envs

    t = int(envs.SGLANG_FAKE_EP_TOPK.get() or 0)
    h = int(envs.SGLANG_FAKE_EP_HIDDEN.get() or 0)
    return (t if t > 0 else int(topk), h if h > 0 else int(hidden_size))


def maybe_simulate_moe_compute(
    hidden_states: torch.Tensor,
    *,
    hidden_size: int,
) -> None:
    """Burn extra GEMM FLOPs to approximate larger MoE experts (V3 vs Lite).

    Uses repeated ``[T,H]@[H,H]`` (not a fat ``[H, scale*H]``) so peak memory
    stays ~H² even when ``compute_scale`` is large — important under ~40GB free.
    """
    if not fake_moe_compute_enabled():
        return
    if hidden_states is None or hidden_states.numel() == 0:
        return

    from sglang.srt.environ import envs

    scale = float(envs.SGLANG_FAKE_MOE_COMPUTE_SCALE.get() or 0.0)
    if scale <= 0:
        return

    device = hidden_states.device
    dtype = hidden_states.dtype
    h = int(hidden_size)
    n_iters = max(1, int(round(scale)))
    key = (device, dtype, h)
    w = _w_cache.get(key)
    if w is None or w.device != device or w.dtype != dtype or w.shape != (h, h):
        w = torch.zeros(h, h, device=device, dtype=dtype)
        _w_cache[key] = w

    t0 = time.perf_counter()
    x = hidden_states.reshape(-1, h)
    for _ in range(n_iters):
        x = torch.mm(x, w)
    _stats["compute_us"] += (time.perf_counter() - t0) * 1e6


def maybe_simulate_ep_comm(
    hidden_states: torch.Tensor,
    *,
    topk: int,
    hidden_size: int,
) -> None:
    """Call once per MoE layer forward when fake EP and/or compute tax is on."""
    apply_v3_proxy_profile()

    # Compute tax: PD decode + AF FFN (both execute MoE.forward).
    maybe_simulate_moe_compute(hidden_states, hidden_size=hidden_size)

    if not fake_ep_enabled():
        return
    if hidden_states is None or hidden_states.numel() == 0:
        return

    from sglang.srt.environ import envs

    global _logged
    num_tokens = int(hidden_states.shape[0])
    topk_eff, hidden_eff = _effective_topk_hidden(topk, hidden_size)
    ep_size = max(1, int(envs.SGLANG_FAKE_EP_SIZE.get() or 1))
    bw = float(envs.SGLANG_FAKE_EP_BW_GBS.get() or 150.0)
    lat_us = float(envs.SGLANG_FAKE_EP_LAT_US.get() or 10.0)
    mode = str(envs.SGLANG_FAKE_EP_MODE.get() or "copy").lower()
    bytes_scale = float(envs.SGLANG_FAKE_EP_BYTES_SCALE.get() or 1.0)
    elem = int(hidden_states.element_size())
    nbytes = ep_comm_bytes(
        num_tokens,
        topk=topk_eff,
        hidden=hidden_eff,
        elem_size=elem,
        ep_size=ep_size,
        bytes_scale=bytes_scale,
    )
    if nbytes <= 0:
        return

    if not _logged:
        logger.warning(
            "FAKE_EP_COMM enabled mode=%s ep_size=%s topk=%s hidden=%s "
            "bytes_scale=%.3f bw=%.1fGB/s lat=%.1fus compute_scale=%.2f FULL_ONLY=%s",
            mode,
            ep_size,
            topk_eff,
            hidden_eff,
            bytes_scale,
            bw,
            lat_us,
            float(envs.SGLANG_FAKE_MOE_COMPUTE_SCALE.get() or 0.0),
            envs.SGLANG_FAKE_EP_FULL_ONLY.get(),
        )
        _logged = True

    device = hidden_states.device
    capturing = _is_capturing(device)
    t0 = time.perf_counter()

    if capturing or mode == "copy":
        _do_copy_traffic(nbytes, device)
        if device.type == "cuda" and not capturing:
            torch.cuda.synchronize(device)
    elif mode == "nvlink":
        peer = int(envs.SGLANG_FAKE_EP_PEER_GPU.get() or -1)
        _do_nvlink_traffic(nbytes, device, peer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    else:  # delay
        torch.cuda.synchronize(device)
        dt = ep_comm_seconds(nbytes, bw_gbs=bw, lat_us=lat_us)
        if dt > 0:
            time.sleep(dt)

    dt_us = (time.perf_counter() - t0) * 1e6
    _stats["calls"] += 1
    _stats["bytes"] += nbytes
    _stats["us"] += dt_us


def fake_ep_stats() -> dict:
    return dict(_stats)


def reset_fake_ep_stats() -> None:
    _stats["calls"] = 0
    _stats["bytes"] = 0
    _stats["us"] = 0.0
    _stats["compute_us"] = 0.0


def v3_proxy_meta() -> dict:
    """Return sizing math used by ``v3_proxy`` (for bench reports)."""
    remote_v3 = 1.0 - 1.0 / _V3["ep"]
    remote_lite = 1.0 - 1.0 / _LITE["ep"]
    per_layer_comm = (
        (_V3["topk"] / _LITE["topk"])
        * (_V3["hidden"] / _LITE["hidden"])
        * (remote_v3 / remote_lite)
    )
    layer_factor = _V3["moe_layers"] / _LITE["moe_layers"]
    flop_v3 = _V3["topk"] * _V3["intermediate"] * _V3["hidden"]
    flop_lite = _LITE["topk"] * _LITE["intermediate"] * _LITE["hidden"]
    return {
        "v3": dict(_V3),
        "lite": dict(_LITE),
        "per_layer_comm_ratio": per_layer_comm,
        "layer_factor": layer_factor,
        "bytes_scale_applied": layer_factor,
        "compute_scale": (flop_v3 / flop_lite) * layer_factor,
        "note": (
            "Lite weights; EP volume uses V3 topk/hidden/ep; "
            "bytes_scale folds deeper MoE stack; compute_scale burns GEMM on MoE.forward. "
            "EP tax FULL_ONLY (PD decode only); compute tax also on AF FFN."
        ),
    }
