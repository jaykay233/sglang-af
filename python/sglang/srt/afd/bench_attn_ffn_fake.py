# SPDX-License-Identifier: Apache-2.0
"""Stripped Attn+FFN microbench (no full model) over real cuda_ipc.

Measures whether A2F/F2A sync + dual-mb overlap still has headroom by burning
configurable Attn/FFN module times around the real mailbox hop.

Roles::

    # one-shot launcher (spawns FFN child, runs Attn sweeps)
    python -m sglang.srt.afd.bench_attn_ffn_fake \\
        --attn-gpu 6 --ffn-gpu 7 --layers 26 --tokens 8

    # or manual two-process:
    SGLANG_AFD_IPC_ENDPOINT=/tmp/afd_fake.sock python -m ... --role ffn --ffn-us 400
    SGLANG_AFD_IPC_ENDPOINT=/tmp/afd_fake.sock python -m ... --role attn --attn-us 400
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.runtime import init_afd_runtime, shutdown_afd_runtime
from sglang.srt.afd.transport import AfdServerBatch
from sglang.srt.environ import envs


def _burn_us(us: float) -> None:
    """Host-timed burn so module latency is deterministic (not fused into GPU)."""
    if us <= 0:
        return
    deadline = time.perf_counter() + us * 1e-6
    while time.perf_counter() < deadline:
        pass


def _pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _configure_common(
    *,
    endpoint: str,
    num_mb: int,
    max_token: int,
    hidden: int,
) -> None:
    envs.SGLANG_AFD_TRANSPORT.set("cuda_ipc")
    envs.SGLANG_AFD_IPC_ENDPOINT.set(endpoint)
    envs.SGLANG_AFD_NUM_MB.set(int(num_mb))
    envs.SGLANG_AFD_MAX_NUM_TOKEN.set(int(max_token))
    envs.SGLANG_AFD_PIPELINE.set(False)
    envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
    envs.SGLANG_AFD_TRUE_OVERLAP.set(False)
    envs.SGLANG_AFD_USE_WAIT_FLAG.set(False)
    envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
    envs.SGLANG_AFD_TIMELINE.set(False)
    os.environ["SGLANG_AFD_EVENTFD_WAKE"] = "0"
    del hidden  # reserved for future wire checks


def _make_ffn_compute(ffn_us: float):
    def ffn(batch: AfdServerBatch):
        _burn_us(ffn_us)
        t = batch.num_tokens
        # Identity through F2A (Attn only checks round-trip latency).
        return [batch.hidden[:t].clone()]

    return ffn


def _run_ffn_role(*, endpoint: str, num_mb: int, max_token: int, hidden: int, ffn_us: float) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_ipc fake bench requires CUDA")
    torch.cuda.set_device(0)  # already remapped by CUDA_VISIBLE_DEVICES
    _configure_common(
        endpoint=endpoint, num_mb=num_mb, max_token=max_token, hidden=hidden
    )
    envs.SGLANG_AFD_MODE.set("ffn")
    shutdown_afd_runtime()
    rt = init_afd_runtime(
        hidden_size=hidden,
        mode=AfdMode.FFN,
        transport_name="cuda_ipc",
        device="cuda",
        dtype=torch.bfloat16,
        ffn_compute=_make_ffn_compute(ffn_us),
        moe_topk=0,
    )
    assert rt is not None
    from sglang.srt.afd.bootstrap import _start_ffn_poll_loop

    _start_ffn_poll_loop(rt, _make_ffn_compute(ffn_us))
    print(
        f"[ffn] ready endpoint={endpoint} mb={num_mb} ffn_us={ffn_us}",
        flush=True,
    )
    # Stay alive until killed by launcher.
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_afd_runtime()
    return 0


@dataclass
class SweepResult:
    mode: str
    attn_us: float
    ffn_us: float
    layers: int
    tokens: int
    num_mb: int
    wall_ms_p50: float
    wall_ms_p90: float
    hop_us_p50: float
    hop_us_p90: float
    per_layer_ms: float
    sync_tax_us: float  # hop - ffn (approx fixed excl compute)


def _run_layers_pd(
    *,
    layers: int,
    attn_us: float,
    ffn_us: float,
) -> Tuple[float, List[float]]:
    """Colocated PD: same host burns, no remote hop (Attn then FFN on one GPU)."""
    hops: List[float] = []
    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    for _li in range(layers):
        _burn_us(attn_us)
        t0 = time.perf_counter()
        _burn_us(ffn_us)
        hops.append((time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t_wall0) * 1e3
    return wall_ms, hops


def _run_layers_sequential(
    rt,
    *,
    layers: int,
    tokens: int,
    hidden: int,
    attn_us: float,
) -> Tuple[float, List[float]]:
    """mb=1: Attn → issue+wait → next layer (no overlap)."""
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    hops: List[float] = []
    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    for li in range(layers):
        _burn_us(attn_us)
        t0 = time.perf_counter()
        out = rt.remote_ffn(layer_id=li, hidden=x, mb_id=0)
        hops.append((time.perf_counter() - t0) * 1e6)
        x = out
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t_wall0) * 1e3
    return wall_ms, hops


def _run_layers_dual_mb(
    rt,
    *,
    layers: int,
    tokens: int,
    hidden: int,
    attn_us: float,
) -> Tuple[float, List[float]]:
    """mb=2 TRUE_OVERLAP-style stagger on two token halves."""
    assert tokens >= 2 and rt.num_mb >= 2
    mid = tokens // 2
    xs = [
        torch.randn(mid, hidden, device="cuda", dtype=torch.bfloat16),
        torch.randn(tokens - mid, hidden, device="cuda", dtype=torch.bfloat16),
    ]
    pending = [None, None]
    hops: List[float] = []
    torch.cuda.synchronize()
    t_wall0 = time.perf_counter()
    for li in range(layers):
        for mb in (0, 1):
            if pending[mb] is not None:
                t0 = time.perf_counter()
                out = rt.wait_remote_ffn(pending[mb], clone=False)
                hops.append((time.perf_counter() - t0) * 1e6)
                xs[mb] = out
                pending[mb] = None
            _burn_us(attn_us)
            pending[mb] = rt.remote_ffn_async(
                layer_id=li, hidden=xs[mb], mb_id=mb
            )
    for mb in (0, 1):
        if pending[mb] is not None:
            t0 = time.perf_counter()
            xs[mb] = rt.wait_remote_ffn(pending[mb], clone=False)
            hops.append((time.perf_counter() - t0) * 1e6)
            pending[mb] = None
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t_wall0) * 1e3
    return wall_ms, hops


def _run_pd_sweep(
    *,
    layers: int,
    attn_us_list: Sequence[float],
    ffn_us: float,
    rounds: int,
    warmup: int,
) -> List[SweepResult]:
    """No IPC: colocated Attn+FFN burns (PD lower bound for this fake)."""
    if not torch.cuda.is_available():
        raise RuntimeError("PD fake bench wants CUDA for sync parity")
    torch.cuda.set_device(0)
    results: List[SweepResult] = []
    for attn_us in attn_us_list:
        walls: List[float] = []
        hops_all: List[float] = []
        for r in range(warmup + rounds):
            wall, hops = _run_layers_pd(
                layers=layers, attn_us=float(attn_us), ffn_us=ffn_us
            )
            if r >= warmup:
                walls.append(wall)
                hops_all.extend(hops)
        results.append(
            SweepResult(
                mode="pd",
                attn_us=float(attn_us),
                ffn_us=float(ffn_us),
                layers=layers,
                tokens=0,
                num_mb=1,
                wall_ms_p50=statistics.median(walls),
                wall_ms_p90=_pct(walls, 90),
                hop_us_p50=statistics.median(hops_all),
                hop_us_p90=_pct(hops_all, 90),
                per_layer_ms=statistics.median(walls) / max(layers, 1),
                sync_tax_us=statistics.median(hops_all) - ffn_us,
            )
        )
    return results


def _run_attn_sweep(
    *,
    endpoint: str,
    num_mb: int,
    max_token: int,
    hidden: int,
    layers: int,
    tokens: int,
    attn_us_list: Sequence[float],
    ffn_us: float,
    rounds: int,
    warmup: int,
    modes: Sequence[str],
) -> List[SweepResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_ipc fake bench requires CUDA")
    torch.cuda.set_device(0)
    _configure_common(
        endpoint=endpoint, num_mb=num_mb, max_token=max_token, hidden=hidden
    )
    envs.SGLANG_AFD_MODE.set("attn")
    # FFN already started with its ffn_us; Attn only varies attn_us / schedule.
    del ffn_us
    shutdown_afd_runtime()
    # Wait until FFN is listening.
    deadline = time.time() + 120.0
    rt = None
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            rt = init_afd_runtime(
                hidden_size=hidden,
                mode=AfdMode.ATTN,
                transport_name="cuda_ipc",
                device="cuda",
                dtype=torch.bfloat16,
                moe_topk=0,
            )
            break
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    if rt is None:
        raise RuntimeError(f"Attn failed to connect to FFN: {last_err}")
    print(f"[attn] connected endpoint={endpoint} mb={rt.num_mb}", flush=True)

    results: List[SweepResult] = []
    try:
        for mode in modes:
            use_dual = mode == "dual_mb"
            if use_dual and rt.num_mb < 2:
                print(f"[attn] skip dual_mb (num_mb={rt.num_mb})")
                continue
            for attn_us in attn_us_list:
                walls: List[float] = []
                hops_all: List[float] = []
                for r in range(warmup + rounds):
                    if use_dual:
                        wall, hops = _run_layers_dual_mb(
                            rt,
                            layers=layers,
                            tokens=tokens,
                            hidden=hidden,
                            attn_us=attn_us,
                        )
                    else:
                        wall, hops = _run_layers_sequential(
                            rt,
                            layers=layers,
                            tokens=tokens,
                            hidden=hidden,
                            attn_us=attn_us,
                        )
                    if r >= warmup:
                        walls.append(wall)
                        hops_all.extend(hops)
                # ffn_us unknown here precisely; caller fills sync_tax with known ffn.
                results.append(
                    SweepResult(
                        mode=mode,
                        attn_us=float(attn_us),
                        ffn_us=float("nan"),
                        layers=layers,
                        tokens=tokens,
                        num_mb=int(rt.num_mb),
                        wall_ms_p50=statistics.median(walls),
                        wall_ms_p90=_pct(walls, 90),
                        hop_us_p50=statistics.median(hops_all),
                        hop_us_p90=_pct(hops_all, 90),
                        per_layer_ms=statistics.median(walls) / max(layers, 1),
                        sync_tax_us=float("nan"),
                    )
                )
    finally:
        shutdown_afd_runtime()
    return results


def _print_table(rows: Sequence[SweepResult], ffn_us: float) -> None:
    print()
    print(
        f"{'mode':<10} {'attn_us':>8} {'ffn_us':>8} {'hop_p50':>10} {'hop_p90':>10} "
        f"{'sync_tax':>10} {'wall_p50':>10} {'ms/layer':>10}"
    )
    print("-" * 88)
    for r in rows:
        tax = r.hop_us_p50 - ffn_us
        print(
            f"{r.mode:<10} {r.attn_us:8.0f} {ffn_us:8.0f} {r.hop_us_p50:10.1f} "
            f"{r.hop_us_p90:10.1f} {tax:10.1f} {r.wall_ms_p50:10.2f} {r.per_layer_ms:10.3f}"
        )
    print()
    print(
        "sync_tax ≈ hop_p50 - ffn_us (fixed sync excl FFN burn). "
        "dual_mb wall should drop when attn≈ffn and both ≫ sync_tax."
    )


def _launcher(args: argparse.Namespace) -> int:
    remote_modes = [m for m in args.modes if m in ("seq", "dual_mb")]
    want_pd = "pd" in args.modes

    if want_pd:
        env_pd = os.environ.copy()
        env_pd["CUDA_VISIBLE_DEVICES"] = str(args.attn_gpu)
        pd_cmd = [
            sys.executable,
            "-m",
            "sglang.srt.afd.bench_attn_ffn_fake",
            "--role",
            "pd",
            "--layers",
            str(args.layers),
            "--ffn-us",
            str(args.ffn_us),
            "--rounds",
            str(args.rounds),
            "--warmup",
            str(args.warmup),
            "--attn-us",
            ",".join(str(x) for x in args.attn_us_list),
        ]
        print(f"[launcher] PD gpu={args.attn_gpu} cmd={' '.join(pd_cmd)}", flush=True)
        rc = subprocess.call(pd_cmd, env=env_pd)
        if rc != 0:
            return int(rc)
        if not remote_modes:
            return 0

    if not remote_modes:
        return 0

    endpoint = args.endpoint
    try:
        os.unlink(endpoint)
    except FileNotFoundError:
        pass

    env_ffn = os.environ.copy()
    env_ffn["CUDA_VISIBLE_DEVICES"] = str(args.ffn_gpu)
    env_ffn["SGLANG_AFD_IPC_ENDPOINT"] = endpoint
    env_ffn["SGLANG_AFD_EVENTFD_WAKE"] = "0"
    ffn_cmd = [
        sys.executable,
        "-m",
        "sglang.srt.afd.bench_attn_ffn_fake",
        "--role",
        "ffn",
        "--endpoint",
        endpoint,
        "--num-mb",
        str(args.num_mb),
        "--max-token",
        str(args.max_token),
        "--hidden",
        str(args.hidden),
        "--ffn-us",
        str(args.ffn_us),
    ]
    print(f"[launcher] FFN gpu={args.ffn_gpu} cmd={' '.join(ffn_cmd)}", flush=True)
    ffn_proc = subprocess.Popen(
        ffn_cmd,
        env=env_ffn,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Drain FFN logs in background. FFN blocks on accept() before "[ffn] ready",
    # so we must start Attn once the UDS listen socket exists (not post-accept).
    def _drain_ffn() -> None:
        assert ffn_proc.stdout is not None
        try:
            for line in ffn_proc.stdout:
                print(line.rstrip(), flush=True)
        except Exception:
            pass

    drain_th = threading.Thread(target=_drain_ffn, name="ffn-drain", daemon=True)
    drain_th.start()

    t0 = time.time()
    while time.time() - t0 < 120:
        if ffn_proc.poll() is not None:
            raise RuntimeError(f"FFN exited early code={ffn_proc.returncode}")
        if os.path.exists(endpoint):
            # Give listen() a tick after bind creates the path.
            time.sleep(0.05)
            break
        time.sleep(0.05)
    else:
        ffn_proc.kill()
        raise RuntimeError("FFN listen socket timeout")
    print(f"[launcher] FFN listening on {endpoint}", flush=True)

    env_attn = os.environ.copy()
    env_attn["CUDA_VISIBLE_DEVICES"] = str(args.attn_gpu)
    env_attn["SGLANG_AFD_IPC_ENDPOINT"] = endpoint
    env_attn["SGLANG_AFD_EVENTFD_WAKE"] = "0"
    attn_cmd = [
        sys.executable,
        "-m",
        "sglang.srt.afd.bench_attn_ffn_fake",
        "--role",
        "attn",
        "--endpoint",
        endpoint,
        "--num-mb",
        str(args.num_mb),
        "--max-token",
        str(args.max_token),
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--tokens",
        str(args.tokens),
        "--ffn-us",
        str(args.ffn_us),
        "--rounds",
        str(args.rounds),
        "--warmup",
        str(args.warmup),
        "--attn-us",
        ",".join(str(x) for x in args.attn_us_list),
        "--modes",
        ",".join(remote_modes),
    ]
    print(f"[launcher] Attn gpu={args.attn_gpu} cmd={' '.join(attn_cmd)}", flush=True)
    try:
        rc = subprocess.call(attn_cmd, env=env_attn)
    finally:
        ffn_proc.terminate()
        try:
            ffn_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffn_proc.kill()
        try:
            os.unlink(endpoint)
        except FileNotFoundError:
            pass
    return int(rc)


def _emit_rows(rows: List[SweepResult], ffn_us: float, attn_us_list: Sequence[float]) -> None:
    for r in rows:
        r.ffn_us = ffn_us
        r.sync_tax_us = r.hop_us_p50 - ffn_us
    _print_table(rows, ffn_us)

    pd_rows = [r for r in rows if r.mode == "pd"]
    seq = [r for r in rows if r.mode == "seq"]
    dual = [r for r in rows if r.mode == "dual_mb"]
    target = min(attn_us_list, key=lambda a: abs(a - ffn_us))

    if pd_rows and seq:
        p = next((r for r in pd_rows if r.attn_us == target), pd_rows[-1])
        s = next((r for r in seq if r.attn_us == target), seq[-1])
        gap = s.wall_ms_p50 - p.wall_ms_p50
        print(
            f"COMPARE @ attn_us≈{target:.0f} ffn_us={ffn_us:.0f}: "
            f"pd_wall={p.wall_ms_p50:.2f}ms seq_afd={s.wall_ms_p50:.2f}ms "
            f"gap={gap:.2f}ms ({gap / max(p.wall_ms_p50, 1e-9) * 100:.0f}% over PD)"
        )
    if dual and seq:
        d = next((r for r in dual if r.attn_us == target), dual[-1])
        s = next((r for r in seq if r.attn_us == target), seq[-1])
        speedup = s.wall_ms_p50 / d.wall_ms_p50 if d.wall_ms_p50 > 0 else 0.0
        print(
            f"VERDICT @ attn_us≈{target:.0f} ffn_us={ffn_us:.0f}: "
            f"seq_wall={s.wall_ms_p50:.2f}ms dual_wall={d.wall_ms_p50:.2f}ms "
            f"speedup={speedup:.2f}x sync_tax≈{d.sync_tax_us:.0f}µs/hop"
        )
        if speedup < 1.15:
            print(
                "  → dual_mb speedup weak at this burn: compute ≪ RTT; "
                "same story as Lite serving."
            )
        elif speedup >= 1.3:
            print(
                "  → dual_mb helps once Attn/FFN are long enough; "
                "serving Lite is short-module limited, not missing MxN."
            )
    print("AFD_ATTN_FFN_FAKE_BENCH_OK")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AFD Attn+FFN stripped cuda_ipc fake bench")
    p.add_argument("--role", choices=("launcher", "ffn", "attn", "pd"), default="launcher")
    p.add_argument("--endpoint", default="/tmp/afd_attn_ffn_fake.sock")
    p.add_argument("--attn-gpu", type=int, default=6)
    p.add_argument("--ffn-gpu", type=int, default=7)
    p.add_argument("--num-mb", type=int, default=2)
    p.add_argument("--max-token", type=int, default=16)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--layers", type=int, default=26)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--ffn-us", type=float, default=400.0)
    p.add_argument(
        "--attn-us",
        type=str,
        default="50,200,400,800",
        help="Comma-separated Attn burn times (µs)",
    )
    p.add_argument("--modes", type=str, default="pd,seq,dual_mb")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args(argv)
    args.attn_us_list = [float(x) for x in args.attn_us.split(",") if x.strip()]
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.role == "launcher":
        return _launcher(args)
    if args.role == "ffn":
        return _run_ffn_role(
            endpoint=args.endpoint,
            num_mb=args.num_mb,
            max_token=args.max_token,
            hidden=args.hidden,
            ffn_us=args.ffn_us,
        )
    if args.role == "pd":
        rows = _run_pd_sweep(
            layers=args.layers,
            attn_us_list=args.attn_us_list,
            ffn_us=args.ffn_us,
            rounds=args.rounds,
            warmup=args.warmup,
        )
        _emit_rows(rows, args.ffn_us, args.attn_us_list)
        return 0
    # attn
    rows = _run_attn_sweep(
        endpoint=args.endpoint,
        num_mb=args.num_mb,
        max_token=max(args.max_token, args.tokens),
        hidden=args.hidden,
        layers=args.layers,
        tokens=args.tokens,
        attn_us_list=args.attn_us_list,
        ffn_us=args.ffn_us,
        rounds=args.rounds,
        warmup=args.warmup,
        modes=args.modes,
    )
    _emit_rows(rows, args.ffn_us, args.attn_us_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
