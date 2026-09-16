# SPDX-License-Identifier: Apache-2.0
"""AfPool MxN throughput bench over real cuda_ipc (no fake transport).

Compares 1A1F vs 1A2F / 2A2F under multi-request load. KPI: tok/s + FFN util.

Example::

    python -m sglang.srt.afd.bench_af_pool \\
        --num-attn 1 --num-ffn 2 --gpus 6,7 \\
        --reqs 8 --layers 26 --tokens 8 \\
        --attn-us 200 --ffn-us 400
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time
from typing import List, Optional, Sequence

import torch

from sglang.srt.afd.pool import (
    AfAttnClient,
    AfFfnWorker,
    apply_pool_env,
    make_identity_compute,
    shutdown_af_pool,
)
from sglang.srt.afd.pool.topology import load_topology_from_env
from sglang.srt.environ import envs


def _burn_us(us: float) -> None:
    if us <= 0:
        return
    deadline = time.perf_counter() + us * 1e-6
    while time.perf_counter() < deadline:
        pass


def _parse_gpus(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _child_env(
    *,
    role: str,
    local_rank: int,
    endpoint_dir: str,
    num_attn: int,
    num_ffn: int,
    gpu: int,
    max_inflight: int,
    num_mb: int,
    max_token: int,
) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_AFD_POOL"] = "1"
    env["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
    env["SGLANG_AFD_MODE"] = role
    env["SGLANG_AFD_POOL_NUM_ATTN"] = str(num_attn)
    env["SGLANG_AFD_POOL_NUM_FFN"] = str(num_ffn)
    env["SGLANG_AFD_POOL_ENDPOINT_DIR"] = endpoint_dir
    env["SGLANG_AFD_POOL_LOCAL_RANK"] = str(local_rank)
    env["SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN"] = str(max_inflight)
    env["SGLANG_AFD_NUM_MB"] = str(num_mb)
    env["SGLANG_AFD_MAX_NUM_TOKEN"] = str(max_token)
    env["SGLANG_AFD_EVENTFD_WAKE"] = "0"
    env["SGLANG_AFD_TRUE_OVERLAP"] = "0"
    env["SGLANG_AFD_LAYER_PIPELINE"] = "0"
    return env


def _run_ffn_child(
    *,
    local_rank: int,
    hidden: int,
    ffn_us: float,
    num_mb: int,
    max_token: int,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_af_pool requires CUDA")
    torch.cuda.set_device(0)
    topo = load_topology_from_env()
    worker = AfFfnWorker(
        topo,
        ffn_rank=local_rank,
        hidden_size=hidden,
        compute=make_identity_compute(ffn_us),
        device="cuda",
        dtype=torch.bfloat16,
        num_mb=num_mb,
        max_num_token=max_token,
    )
    worker.start_background()
    print(f"[ffn{local_rank}] ready links={topo.num_attn}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        shutdown_af_pool()
    return 0


def _run_attn_child(
    *,
    local_rank: int,
    hidden: int,
    attn_us: float,
    reqs: int,
    layers: int,
    tokens: int,
    num_mb: int,
    max_token: int,
    warmup: int,
    dual_mb: bool,
    result_path: str,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_af_pool requires CUDA")
    torch.cuda.set_device(0)
    # Wait for FFN listen sockets.
    time.sleep(2.0)
    topo = load_topology_from_env()
    client = AfAttnClient(
        topo,
        attn_rank=local_rank,
        hidden_size=hidden,
        device="cuda",
        dtype=torch.bfloat16,
        num_mb=num_mb,
        max_num_token=max_token,
    )
    print(
        f"[attn{local_rank}] ready ffn_links={topo.num_ffn} reqs={reqs} layers={layers}",
        flush=True,
    )
    try:
        for _ in range(max(0, warmup)):
            client.run_multi_req_layers(
                num_reqs=min(2, reqs),
                layers=min(2, layers),
                tokens=tokens,
                hidden_size=hidden,
                attn_burn_us=attn_us,
                dual_mb=dual_mb,
            )
        client.scheduler.stats = type(client.scheduler.stats)()
        client.scheduler.stats.ensure_ffn(topo.num_ffn)
        client.run_multi_req_layers(
            num_reqs=reqs,
            layers=layers,
            tokens=tokens,
            hidden_size=hidden,
            attn_burn_us=attn_us,
            dual_mb=dual_mb,
        )
        summary = client.scheduler.summary()
        s = client.scheduler.stats
        line = (
            f"attn_rank={local_rank} wall_s={s.wall_s:.4f} tok_s={s.tok_s:.1f} "
            f"tasks={s.tasks_completed} tokens={s.tokens_completed} "
            f"mean_ffn_util={s.mean_ffn_busy_frac():.3f} | {summary}"
        )
        print(line, flush=True)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(
                f"{s.wall_s},{s.tok_s},{s.tasks_completed},{s.tokens_completed},"
                f"{s.mean_ffn_busy_frac()}\n"
            )
    finally:
        client.close()
        shutdown_af_pool()
    return 0


def _launch_matrix(
    *,
    num_attn: int,
    num_ffn: int,
    gpus: Sequence[int],
    endpoint_dir: str,
    hidden: int,
    attn_us: float,
    ffn_us: float,
    reqs: int,
    layers: int,
    tokens: int,
    num_mb: int,
    max_token: int,
    max_inflight: int,
    warmup: int,
    dual_mb: bool,
) -> dict:
    if len(gpus) < num_attn + num_ffn:
        raise ValueError(
            f"need {num_attn}+{num_ffn} GPUs, got {list(gpus)}"
        )
    if os.path.isdir(endpoint_dir):
        shutil.rmtree(endpoint_dir, ignore_errors=True)
    apply_pool_env(
        num_attn=num_attn,
        num_ffn=num_ffn,
        endpoint_dir=endpoint_dir,
        max_inflight=max_inflight,
        enabled=True,
    )

    py = sys.executable
    mod = "sglang.srt.afd.bench_af_pool"
    procs: List[subprocess.Popen] = []
    result_paths: List[str] = []

    # FFN children first (listen).
    for j in range(num_ffn):
        gpu = gpus[num_attn + j]
        env = _child_env(
            role="ffn",
            local_rank=j,
            endpoint_dir=endpoint_dir,
            num_attn=num_attn,
            num_ffn=num_ffn,
            gpu=gpu,
            max_inflight=max_inflight,
            num_mb=num_mb,
            max_token=max_token,
        )
        cmd = [
            py,
            "-m",
            mod,
            "--role",
            "ffn",
            "--local-rank",
            str(j),
            "--hidden",
            str(hidden),
            "--ffn-us",
            str(ffn_us),
            "--num-mb",
            str(num_mb),
            "--max-token",
            str(max_token),
        ]
        procs.append(subprocess.Popen(cmd, env=env))

    time.sleep(3.0)

    attn_procs: List[subprocess.Popen] = []
    for i in range(num_attn):
        gpu = gpus[i]
        result_path = os.path.join(endpoint_dir, f"attn{i}_result.csv")
        result_paths.append(result_path)
        env = _child_env(
            role="attn",
            local_rank=i,
            endpoint_dir=endpoint_dir,
            num_attn=num_attn,
            num_ffn=num_ffn,
            gpu=gpu,
            max_inflight=max_inflight,
            num_mb=num_mb,
            max_token=max_token,
        )
        cmd = [
            py,
            "-m",
            mod,
            "--role",
            "attn",
            "--local-rank",
            str(i),
            "--hidden",
            str(hidden),
            "--attn-us",
            str(attn_us),
            "--reqs",
            str(reqs),
            "--layers",
            str(layers),
            "--tokens",
            str(tokens),
            "--num-mb",
            str(num_mb),
            "--max-token",
            str(max_token),
            "--warmup",
            str(warmup),
            "--result-path",
            result_path,
        ]
        if dual_mb:
            cmd.append("--dual-mb")
        p = subprocess.Popen(cmd, env=env)
        attn_procs.append(p)
        procs.append(p)

    rc = 0
    for p in attn_procs:
        r = p.wait()
        if r != 0:
            rc = r
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()

    walls, toks, utils = [], [], []
    for path in result_paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            parts = f.read().strip().split(",")
        if len(parts) >= 5:
            walls.append(float(parts[0]))
            toks.append(float(parts[1]))
            utils.append(float(parts[4]))

    # Aggregate: max wall across Attn (barrier-ish), sum tok/s approx.
    wall = max(walls) if walls else float("nan")
    tok_s = sum(toks) if toks else float("nan")
    util = statistics.mean(utils) if utils else float("nan")
    return {
        "num_attn": num_attn,
        "num_ffn": num_ffn,
        "wall_s": wall,
        "tok_s": tok_s,
        "mean_ffn_util": util,
        "rc": rc,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AfPool MxN cuda_ipc throughput bench")
    p.add_argument("--role", choices=["launch", "attn", "ffn"], default="launch")
    p.add_argument("--local-rank", type=int, default=0)
    p.add_argument("--num-attn", type=int, default=1)
    p.add_argument("--num-ffn", type=int, default=2)
    p.add_argument("--gpus", type=str, default="6,7", help="GPU ids, Attn first then FFN")
    p.add_argument("--endpoint-dir", type=str, default="/tmp/afd_pool_bench")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--attn-us", type=float, default=200.0)
    p.add_argument("--ffn-us", type=float, default=400.0)
    p.add_argument("--reqs", type=int, default=8)
    p.add_argument("--layers", type=int, default=26)
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--num-mb", type=int, default=2)
    p.add_argument("--max-token", type=int, default=64)
    p.add_argument("--max-inflight", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dual-mb", action="store_true", default=True)
    p.add_argument("--no-dual-mb", action="store_true")
    p.add_argument("--result-path", type=str, default="")
    p.add_argument(
        "--compare-1a1f",
        action="store_true",
        help="Also run 1A1F baseline with first two GPUs and print ratio",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    dual_mb = bool(args.dual_mb) and not bool(args.no_dual_mb)

    if args.role == "ffn":
        apply_pool_env(
            num_attn=int(envs.SGLANG_AFD_POOL_NUM_ATTN.get()),
            num_ffn=int(envs.SGLANG_AFD_POOL_NUM_FFN.get()),
            endpoint_dir=envs.SGLANG_AFD_POOL_ENDPOINT_DIR.get(),
            enabled=True,
        )
        return _run_ffn_child(
            local_rank=args.local_rank,
            hidden=args.hidden,
            ffn_us=args.ffn_us,
            num_mb=args.num_mb,
            max_token=args.max_token,
        )

    if args.role == "attn":
        apply_pool_env(
            num_attn=int(envs.SGLANG_AFD_POOL_NUM_ATTN.get()),
            num_ffn=int(envs.SGLANG_AFD_POOL_NUM_FFN.get()),
            endpoint_dir=envs.SGLANG_AFD_POOL_ENDPOINT_DIR.get(),
            enabled=True,
        )
        return _run_attn_child(
            local_rank=args.local_rank,
            hidden=args.hidden,
            attn_us=args.attn_us,
            reqs=args.reqs,
            layers=args.layers,
            tokens=args.tokens,
            num_mb=args.num_mb,
            max_token=args.max_token,
            warmup=args.warmup,
            dual_mb=dual_mb,
            result_path=args.result_path or "/tmp/afd_pool_attn_result.csv",
        )

    gpus = _parse_gpus(args.gpus)
    configs = [(args.num_attn, args.num_ffn)]
    if args.compare_1a1f and not (args.num_attn == 1 and args.num_ffn == 1):
        configs.insert(0, (1, 1))

    results = []
    for na, nf in configs:
        need = na + nf
        if len(gpus) < need:
            print(
                f"WARNING: need {need} distinct GPUs for {na}A{nf}F, got {list(gpus)}; "
                f"reusing (Attn/FFN may share a device — speedup will look weak)",
                flush=True,
            )
            g = list(gpus)
            while len(g) < need:
                g.append(g[len(g) % len(gpus)])
        else:
            g = list(gpus[:need])
        print(f"\n=== AfPool bench {na}A{nf}F gpus={g} ===", flush=True)
        r = _launch_matrix(
            num_attn=na,
            num_ffn=nf,
            gpus=g,
            endpoint_dir=f"{args.endpoint_dir}_{na}a{nf}f",
            hidden=args.hidden,
            attn_us=args.attn_us,
            ffn_us=args.ffn_us,
            reqs=args.reqs,
            layers=args.layers,
            tokens=args.tokens,
            num_mb=args.num_mb,
            max_token=args.max_token,
            max_inflight=args.max_inflight,
            warmup=args.warmup,
            dual_mb=dual_mb,
        )
        results.append(r)
        print(
            f"RESULT {na}A{nf}F wall_s={r['wall_s']:.4f} tok_s={r['tok_s']:.1f} "
            f"mean_ffn_util={r['mean_ffn_util']:.3f} rc={r['rc']}",
            flush=True,
        )

    if len(results) >= 2 and results[0]["tok_s"] > 0:
        ratio = results[-1]["tok_s"] / results[0]["tok_s"]
        print(
            f"\nSPEEDUP tok_s {results[-1]['num_attn']}A{results[-1]['num_ffn']}F "
            f"/ {results[0]['num_attn']}A{results[0]['num_ffn']}F = {ratio:.2f}x",
            flush=True,
        )
    return 0 if all(r.get("rc", 0) == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
