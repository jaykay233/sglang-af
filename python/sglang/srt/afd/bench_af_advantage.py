# SPDX-License-Identifier: Apache-2.0
"""Simulate when AF beats full multi-replica without DeepSeek-V3/V4.

Two experiments on Lite / synthetic AfPool:

  B) FFN wall — synthetic cuda_ipc AfPool, sweep ffn_us; compare 1A1F/1A2F/2A2F
  A) KV / mem wall — real servers; report KV tokens + high-conc tok/s for
     1×full vs 3×full vs 2A1F (same GPU count where applicable)

Example::

    source /root/.cuda/afd_env.sh
    python -m sglang.srt.afd.bench_af_advantage --out /tmp/afd_advantage
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL = os.environ.get("MODEL_PATH", "/data/share/models/DeepSeek-V2-Lite-Chat")
PY = sys.executable


def _run(cmd: List[str], *, env: Optional[dict] = None, cwd: Optional[str] = None) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=cwd)


def _kill_ports(*ports: int) -> None:
    for p in ports:
        subprocess.call(
            ["bash", "-c", f"fuser -k {p}/tcp >/dev/null 2>&1 || true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1.5)


def _pgid_kill(pids: List[int]) -> None:
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            subprocess.call(["pkill", "-9", "-P", str(pid)], stdout=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Experiment B: synthetic FFN wall
# ---------------------------------------------------------------------------


def exp_ffn_wall(out: Path, gpus: str) -> List[dict]:
    print("\n======== EXP-B: FFN wall (synthetic AfPool) ========", flush=True)
    rows: List[dict] = []
    # attn cheap, ffn increasingly expensive
    attn_us = 100.0
    for ffn_us in (100.0, 400.0, 800.0, 1600.0):
        for na, nf in ((1, 1), (1, 2), (2, 2)):
            need = na + nf
            gpu_list = [x for x in gpus.split(",") if x.strip()]
            if len(gpu_list) < need:
                print(f"skip {na}A{nf}F: need {need} GPUs", flush=True)
                continue
            use = ",".join(gpu_list[:need])
            ep = str(out / f"pool_a{na}f{nf}_ffn{int(ffn_us)}")
            cmd = [
                PY,
                "-m",
                "sglang.srt.afd.bench_af_pool",
                "--num-attn",
                str(na),
                "--num-ffn",
                str(nf),
                "--gpus",
                use,
                "--endpoint-dir",
                ep,
                "--attn-us",
                str(attn_us),
                "--ffn-us",
                str(ffn_us),
                "--reqs",
                "8",
                "--layers",
                "26",
                "--tokens",
                "8",
                "--max-inflight",
                "4",
                "--warmup",
                "1",
            ]
            if na == 1 and nf == 1:
                # baseline only
                pass
            # capture stdout for tok/s line
            print(f"\n--- ffn_us={ffn_us}  {na}A{nf}F gpus={use} ---", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            (out / f"ffn_wall_a{na}f{nf}_{int(ffn_us)}.log").write_text(text)
            # Parse last JSON-ish or "tok_s=" lines from bench_af_pool
            tok_s = _parse_float(text, r"tok_s[=:\s]+([0-9.]+)")
            wall = _parse_float(text, r"wall_s[=:\s]+([0-9.]+)")
            util = _parse_float(text, r"mean_ffn_util[=:\s]+([0-9.]+)")
            # Also try printed dict / summary table
            if tok_s != tok_s:  # nan
                tok_s = _parse_float(text, r"'tok_s':\s*([0-9.]+)")
                wall = _parse_float(text, r"'wall_s':\s*([0-9.]+)")
                util = _parse_float(text, r"'mean_ffn_util':\s*([0-9.]+)")
            row = {
                "exp": "ffn_wall",
                "ffn_us": ffn_us,
                "attn_us": attn_us,
                "num_attn": na,
                "num_ffn": nf,
                "gpus": use,
                "tok_s": tok_s,
                "wall_s": wall,
                "mean_ffn_util": util,
                "rc": proc.returncode,
            }
            rows.append(row)
            print(
                f"  → tok_s={tok_s:.1f} wall={wall:.2f}s util={util:.2f} rc={proc.returncode}",
                flush=True,
            )
    path = out / "ffn_wall.json"
    path.write_text(json.dumps(rows, indent=2))
    _print_ffn_table(rows)
    return rows


def _parse_float(text: str, pat: str) -> float:
    ms = re.findall(pat, text)
    if not ms:
        return float("nan")
    try:
        return float(ms[-1])
    except ValueError:
        return float("nan")


def _print_ffn_table(rows: List[dict]) -> None:
    print("\n--- FFN-wall summary (tok/s) ---", flush=True)
    ffn_vals = sorted({r["ffn_us"] for r in rows})
    tops = [(1, 1), (1, 2), (2, 2)]
    hdr = f"{'ffn_us':>8}" + "".join(f"{'a'+str(a)+'f'+str(f):>12}" for a, f in tops)
    print(hdr, flush=True)
    for fu in ffn_vals:
        line = f"{fu:8.0f}"
        base = None
        for a, f in tops:
            hit = next(
                (r for r in rows if r["ffn_us"] == fu and r["num_attn"] == a and r["num_ffn"] == f),
                None,
            )
            v = hit["tok_s"] if hit else float("nan")
            if a == 1 and f == 1:
                base = v
            if v == v:
                line += f"{v:12.1f}"
            else:
                line += f"{'—':>12}"
        # ratios vs 1A1F
        line += "   |"
        for a, f in tops:
            hit = next(
                (r for r in rows if r["ffn_us"] == fu and r["num_attn"] == a and r["num_ffn"] == f),
                None,
            )
            v = hit["tok_s"] if hit else float("nan")
            if base and base == base and v == v and base > 0:
                line += f"  {a}A{f}F={v/base:.2f}x"
        print(line, flush=True)


# ---------------------------------------------------------------------------
# Experiment A: KV / mem wall with real servers
# ---------------------------------------------------------------------------


def _start_holder(gpu: int, gib: float, log: Path) -> int:
    """Occupy `gib` GiB on one GPU so leftover mimics a small card."""
    code = f"""
import torch, time, os
os.environ['CUDA_VISIBLE_DEVICES']='{gpu}'
n = int({gib} * (1024**3) / 2)
t = torch.empty(n, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()
print('HOLD', {gib}, 'GiB on', {gpu}, 'numel', t.numel(), flush=True)
while True:
    time.sleep(3600)
"""
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen(
        [PY, "-c", code],
        stdout=f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    time.sleep(2)
    return p.pid


def _server_env_af(
    *,
    mode: str,
    gpu: int,
    num_attn: int,
    num_ffn: int,
    rank: int,
    ep: str,
    max_running: int,
) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_AFD_MODE"] = mode
    env["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
    env["SGLANG_AFD_POOL"] = "1"
    env["SGLANG_AFD_POOL_NUM_ATTN"] = str(num_attn)
    env["SGLANG_AFD_POOL_NUM_FFN"] = str(num_ffn)
    env["SGLANG_AFD_POOL_LOCAL_RANK"] = str(rank)
    env["SGLANG_AFD_POOL_ENDPOINT_DIR"] = ep
    env["SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN"] = str(max(8, max_running))
    env["SGLANG_AFD_POOL_ROUTE"] = "least_inflight"
    env["SGLANG_AFD_MODULE_STUBS"] = "1"
    env["SGLANG_AFD_RELEASE_UNUSED_PARAMS"] = "1"
    env["SGLANG_AFD_NUM_MB"] = "2"
    env["SGLANG_AFD_MAX_NUM_TOKEN"] = "512"
    env["SGLANG_AFD_FFN_CUDA_GRAPH"] = "1"
    env.pop("DMLC_ROLE", None)
    return env


def _server_env_full(gpu: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SGLANG_AFD_MODE"] = "null"
    env["SGLANG_AFD_POOL"] = "0"
    env.pop("DMLC_ROLE", None)
    return env


def _launch_server(
    *,
    env: dict,
    port: int,
    log: Path,
    max_running: int,
    max_total_tokens: int,
    mem_fraction: float,
    skip_warmup: bool,
) -> int:
    cmd = [
        PY,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_running),
        "--context-length",
        "4096",
        "--max-total-tokens",
        str(max_total_tokens),
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "breakable",
        "--cuda-graph-max-bs-decode",
        str(max_running),
        "--port",
        str(port),
        "--disaggregation-mode",
        "null",
    ]
    if skip_warmup:
        cmd.append("--skip-server-warmup")
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p.pid


def _wait_http(url: str, pid: int, timeout: int, log: Path) -> bool:
    for i in range(timeout):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"  process {pid} died; tail {log}:", flush=True)
            print(log.read_text()[-2000:] if log.is_file() else "", flush=True)
            return False
        r = subprocess.call(
            ["curl", "-sf", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if r == 0:
            print(f"  healthy {url} in {i}s", flush=True)
            return True
        time.sleep(1)
    print(f"  timeout {url}", flush=True)
    print(log.read_text()[-2000:] if log.is_file() else "", flush=True)
    return False


def _parse_kv(log: Path) -> Tuple[Optional[int], Optional[float]]:
    text = log.read_text() if log.is_file() else ""
    m = re.search(r"#tokens:\s*(\d+).*?KV size:\s*([0-9.]+)\s*GB", text)
    if m:
        return int(m.group(1)), float(m.group(2))
    m2 = re.search(r"max_total_num_tokens=(\d+)", text)
    toks = int(m2.group(1)) if m2 else None
    return toks, None


def _start_rr_lb(port: int, backends: List[str], log: Path) -> int:
    code = r"""
import itertools, sys
from aiohttp import ClientSession, TCPConnector, web
port=int(sys.argv[1]); backends=sys.argv[2:]
cycle=itertools.cycle(backends)
skip={"host","content-length","transfer-encoding","connection"}
session=None
async def on_start(app):
    global session
    session=ClientSession(connector=TCPConnector(limit=0))
async def on_stop(app):
    global session
    if session: await session.close()
async def health(_): return web.Response(text="ok")
async def proxy(request):
    url=f"{next(cycle)}{request.rel_url}"; body=await request.read()
    headers={k:v for k,v in request.headers.items() if k.lower() not in skip}
    async with session.request(request.method,url,headers=headers,data=body) as resp:
        oh={k:v for k,v in resp.headers.items() if k.lower() not in skip}
        out=web.StreamResponse(status=resp.status, headers=oh)
        await out.prepare(request)
        async for chunk in resp.content.iter_chunked(65536):
            await out.write(chunk)
        await out.write_eof(); return out
app=web.Application(); app.on_startup.append(on_start); app.on_cleanup.append(on_stop)
app.router.add_get("/health", health); app.router.add_route("*", "/{path:.*}", proxy)
web.run_app(app, host="127.0.0.1", port=port, print=lambda *_: None)
"""
    f = open(log, "w")
    p = subprocess.Popen(
        [PY, "-c", code, str(port), *backends],
        stdout=f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return p.pid


def _bench(base_url: str, out_json: Path, *, prompts: int, conc: int) -> dict:
    cmd = [
        PY,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        base_url,
        "--model",
        MODEL,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(prompts),
        "--random-input-len",
        "1",
        "--random-output-len",
        "64",
        "--random-range-ratio",
        "0.0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(conc),
        "--warmup-requests",
        "2",
        "--output-file",
        str(out_json),
        "--disable-tqdm",
    ]
    subprocess.call(cmd)
    if not out_json.is_file() or out_json.stat().st_size == 0:
        return {}
    text = out_json.read_text().strip()
    dec = json.JSONDecoder()
    objs = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        o, e = dec.raw_decode(text, i)
        objs.append(o)
        i = e
    return objs[-1] if objs else {}


def exp_kv_wall(
    out: Path,
    *,
    gpus: List[int],
    reserve_gib: float,
    prompts: int,
    conc: int,
    mem_fraction: float,
) -> List[dict]:
    print(
        f"\n======== EXP-A: KV/mem wall (reserve={reserve_gib}GiB/GPU) ========",
        flush=True,
    )
    assert len(gpus) >= 3, "need ≥3 GPUs"
    g0, g1, g2 = gpus[0], gpus[1], gpus[2]
    rows: List[dict] = []
    holders: List[int] = []
    base_port = 37000

    def cleanup(pids: List[int]) -> None:
        _pgid_kill(pids)
        _kill_ports(
            base_port,
            base_port + 1,
            base_port + 2,
            base_port + 10,
            base_port + 20,
            base_port + 21,
            base_port + 22,
            base_port + 30,
        )

    if reserve_gib > 0:
        for g in (g0, g1, g2):
            pid = _start_holder(g, reserve_gib, out / f"hold_g{g}.log")
            holders.append(pid)
            print(f"  holder gpu{g} pid={pid} {reserve_gib}GiB", flush=True)
        time.sleep(3)

    configs = [
        # tag, kind, ports/gpus
        ("full1", "full1"),
        ("full3", "full3"),
        ("a2f1", "a2f1"),
    ]

    for tag, kind in configs:
        print(f"\n--- {tag} ---", flush=True)
        pids: List[int] = list(holders)
        try:
            if kind == "full1":
                port = base_port + 20
                _kill_ports(port)
                env = _server_env_full(g0)
                log = out / f"{tag}_srv.log"
                pid = _launch_server(
                    env=env,
                    port=port,
                    log=log,
                    max_running=32,
                    max_total_tokens=200000,
                    mem_fraction=mem_fraction,
                    skip_warmup=False,
                )
                pids.append(pid)
                ok = _wait_http(f"http://127.0.0.1:{port}/health", pid, 360, log)
                kv_tok, kv_gb = _parse_kv(log)
                metrics: dict = {}
                if ok:
                    metrics = _bench(
                        f"http://127.0.0.1:{port}",
                        out / f"{tag}_bench.json",
                        prompts=prompts,
                        conc=min(conc, 32),
                    )
                rows.append(
                    {
                        "exp": "kv_wall",
                        "tag": tag,
                        "gpus": 1,
                        "ok": ok,
                        "kv_tokens": kv_tok,
                        "kv_gb": kv_gb,
                        "median_tpot_ms": metrics.get("median_tpot_ms"),
                        "output_throughput": metrics.get("output_throughput"),
                        "completed": metrics.get("completed"),
                        "reserve_gib": reserve_gib,
                    }
                )
            elif kind == "full3":
                ports = [base_port + 20, base_port + 21, base_port + 22]
                lb = base_port + 30
                _kill_ports(*ports, lb)
                logs = []
                for gi, port in zip((g0, g1, g2), ports):
                    env = _server_env_full(gi)
                    log = out / f"{tag}_g{gi}.log"
                    logs.append(log)
                    pid = _launch_server(
                        env=env,
                        port=port,
                        log=log,
                        max_running=32,
                        max_total_tokens=200000,
                        mem_fraction=mem_fraction,
                        skip_warmup=False,
                    )
                    pids.append(pid)
                oks = [
                    _wait_http(f"http://127.0.0.1:{p}/health", pid, 360, log)
                    for p, pid, log in zip(ports, pids[-3:], logs)
                ]
                kv_toks = []
                kv_gbs = []
                for log in logs:
                    t, g = _parse_kv(log)
                    if t:
                        kv_toks.append(t)
                    if g:
                        kv_gbs.append(g)
                metrics = {}
                if all(oks):
                    lb_pid = _start_rr_lb(
                        lb,
                        [f"http://127.0.0.1:{p}" for p in ports],
                        out / f"{tag}_lb.log",
                    )
                    pids.append(lb_pid)
                    time.sleep(1)
                    if _wait_http(f"http://127.0.0.1:{lb}/health", lb_pid, 30, out / f"{tag}_lb.log"):
                        metrics = _bench(
                            f"http://127.0.0.1:{lb}",
                            out / f"{tag}_bench.json",
                            prompts=prompts,
                            conc=conc,
                        )
                rows.append(
                    {
                        "exp": "kv_wall",
                        "tag": tag,
                        "gpus": 3,
                        "ok": all(oks),
                        "kv_tokens": sum(kv_toks) if kv_toks else None,
                        "kv_gb": sum(kv_gbs) if kv_gbs else None,
                        "median_tpot_ms": metrics.get("median_tpot_ms"),
                        "output_throughput": metrics.get("output_throughput"),
                        "completed": metrics.get("completed"),
                        "reserve_gib": reserve_gib,
                    }
                )
            else:  # a2f1
                fport, a0, a1, lb = base_port, base_port + 1, base_port + 2, base_port + 10
                _kill_ports(fport, a0, a1, lb)
                ep = str(out / "a2f1_socks")
                if os.path.isdir(ep):
                    shutil.rmtree(ep, ignore_errors=True)
                os.makedirs(ep, exist_ok=True)
                # FFN on g2, Attn on g0/g1
                flog = out / f"{tag}_ffn.log"
                fpid = _launch_server(
                    env=_server_env_af(
                        mode="ffn",
                        gpu=g2,
                        num_attn=2,
                        num_ffn=1,
                        rank=0,
                        ep=ep,
                        max_running=32,
                    ),
                    port=fport,
                    log=flog,
                    max_running=32,
                    max_total_tokens=50000,
                    mem_fraction=mem_fraction,
                    skip_warmup=True,
                )
                pids.append(fpid)
                # wait for weight load
                for _ in range(90):
                    if "Load weight end" in flog.read_text() if flog.is_file() else "":
                        break
                    if "AfPool FFN" in (flog.read_text() if flog.is_file() else ""):
                        break
                    time.sleep(2)
                a0log = out / f"{tag}_attn0.log"
                a1log = out / f"{tag}_attn1.log"
                p0 = _launch_server(
                    env=_server_env_af(
                        mode="attn",
                        gpu=g0,
                        num_attn=2,
                        num_ffn=1,
                        rank=0,
                        ep=ep,
                        max_running=32,
                    ),
                    port=a0,
                    log=a0log,
                    max_running=32,
                    max_total_tokens=200000,
                    mem_fraction=mem_fraction,
                    skip_warmup=False,
                )
                pids.append(p0)
                ok0 = _wait_http(f"http://127.0.0.1:{a0}/health", p0, 360, a0log)
                p1 = _launch_server(
                    env=_server_env_af(
                        mode="attn",
                        gpu=g1,
                        num_attn=2,
                        num_ffn=1,
                        rank=1,
                        ep=ep,
                        max_running=32,
                    ),
                    port=a1,
                    log=a1log,
                    max_running=32,
                    max_total_tokens=200000,
                    mem_fraction=mem_fraction,
                    skip_warmup=False,
                )
                pids.append(p1)
                ok1 = _wait_http(f"http://127.0.0.1:{a1}/health", p1, 360, a1log)
                t0, g0b = _parse_kv(a0log)
                t1, g1b = _parse_kv(a1log)
                metrics = {}
                if ok0 and ok1:
                    lb_pid = _start_rr_lb(
                        lb,
                        [f"http://127.0.0.1:{a0}", f"http://127.0.0.1:{a1}"],
                        out / f"{tag}_lb.log",
                    )
                    pids.append(lb_pid)
                    time.sleep(1)
                    if _wait_http(f"http://127.0.0.1:{lb}/health", lb_pid, 30, out / f"{tag}_lb.log"):
                        metrics = _bench(
                            f"http://127.0.0.1:{lb}",
                            out / f"{tag}_bench.json",
                            prompts=prompts,
                            conc=conc,
                        )
                rows.append(
                    {
                        "exp": "kv_wall",
                        "tag": tag,
                        "gpus": 3,
                        "ok": bool(ok0 and ok1),
                        "kv_tokens": (t0 or 0) + (t1 or 0) if (t0 or t1) else None,
                        "kv_gb": ((g0b or 0) + (g1b or 0)) or None,
                        "median_tpot_ms": metrics.get("median_tpot_ms"),
                        "output_throughput": metrics.get("output_throughput"),
                        "completed": metrics.get("completed"),
                        "reserve_gib": reserve_gib,
                    }
                )
        finally:
            # kill servers but keep holders for next config if shared
            srv = [p for p in pids if p not in holders]
            _pgid_kill(srv)
            _kill_ports(
                base_port,
                base_port + 1,
                base_port + 2,
                base_port + 10,
                base_port + 20,
                base_port + 21,
                base_port + 22,
                base_port + 30,
            )
            time.sleep(2)

    _pgid_kill(holders)
    path = out / f"kv_wall_reserve{int(reserve_gib)}.json"
    path.write_text(json.dumps(rows, indent=2))
    print("\n--- KV/mem-wall summary ---", flush=True)
    print(
        f"{'tag':<8} {'ok':>3} {'kv_tok':>10} {'kv_GB':>8} {'out_tps':>10} {'tpot':>8} {'done':>6}",
        flush=True,
    )
    for r in rows:
        kv_tok = r.get("kv_tokens")
        kv_gb = r.get("kv_gb")
        out_tps = r.get("output_throughput")
        tpot = r.get("median_tpot_ms")
        done = r.get("completed")
        kv_tok_s = str(kv_tok) if kv_tok is not None else "—"
        kv_gb_s = f"{kv_gb:.2f}" if kv_gb is not None else "—"
        out_s = f"{out_tps:.1f}" if out_tps is not None else "—"
        tpot_s = f"{tpot:.1f}" if tpot is not None else "—"
        done_s = str(done) if done is not None else "—"
        print(
            f"{r['tag']:<8} {str(r['ok']):>3} {kv_tok_s:>10} {kv_gb_s:>8} "
            f"{out_s:>10} {tpot_s:>8} {done_s:>6}",
            flush=True,
        )
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/afd_advantage")
    ap.add_argument("--gpus", default="5,6,7,4", help="Attn/full first, then FFN extras")
    ap.add_argument("--skip-ffn-wall", action="store_true")
    ap.add_argument("--skip-kv-wall", action="store_true")
    ap.add_argument("--reserve-gib", type=float, default=0.0, help="Extra occupancy per GPU")
    ap.add_argument("--reserve-gib-tight", type=float, default=12.0, help="Second KV pass")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.82)
    ap.add_argument("--only-ffn", action="store_true")
    ap.add_argument("--only-kv", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # Ensure clean ports
    _kill_ports(*range(37000, 37040))

    all_rows: List[dict] = []
    do_ffn = not args.skip_ffn_wall and not args.only_kv
    do_kv = not args.skip_kv_wall and not args.only_ffn

    if do_ffn:
        all_rows.extend(exp_ffn_wall(out, args.gpus))

    if do_kv:
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
        # Pass 1: ambient free (~40GB with ComfyUI)
        all_rows.extend(
            exp_kv_wall(
                out,
                gpus=gpu_ids,
                reserve_gib=float(args.reserve_gib),
                prompts=args.prompts,
                conc=args.conc,
                mem_fraction=args.mem_fraction,
            )
        )
        # Pass 2: tighten memory to mimic smaller cards
        if args.reserve_gib_tight > 0:
            all_rows.extend(
                exp_kv_wall(
                    out,
                    gpus=gpu_ids,
                    reserve_gib=float(args.reserve_gib_tight),
                    prompts=args.prompts,
                    conc=args.conc,
                    mem_fraction=args.mem_fraction,
                )
            )

    (out / "all_results.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nAF_ADVANTAGE_OK wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
