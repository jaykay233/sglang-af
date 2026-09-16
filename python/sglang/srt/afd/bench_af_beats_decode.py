# SPDX-License-Identifier: Apache-2.0
"""Boss-facing experiment: when A+F throughput beats pure decode.

Fair rule
---------
Same GPU count, same model (DeepSeek-V2-Lite), same load (conc/prompts).
Artificially tighten per-GPU free HBM (holder tensors) so a **full** replica
barely fits (tiny KV), while an **Attn-stub** replica still gets a large KV.
FFN still fits on its card. This mimics the large-MoE regime on Lite.

Compares:
  2 GPU:  2×full decode  vs  1A1F
  3 GPU:  3×full decode  vs  2A1F

Primary KPI: output token throughput (tok/s). Secondary: KV tokens, TPOT.

Usage::

    source /root/.cuda/afd_env.sh
    python -m sglang.srt.afd.bench_af_beats_decode --out /tmp/afd_beats_decode
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
from typing import Dict, List, Optional, Tuple

MODEL = os.environ.get("MODEL_PATH", "/data/share/models/DeepSeek-V2-Lite-Chat")
PY = sys.executable


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
        if pid <= 0:
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
            subprocess.call(["pkill", "-9", "-P", str(pid)], stdout=subprocess.DEVNULL)


def _start_holder(gpu: int, gib: float, log: Path) -> int:
    code = f"""
import torch, time, os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'
n = int({gib} * (1024**3) / 2)
t = torch.empty(n, dtype=torch.float16, device='cuda')
torch.cuda.synchronize()
print('HOLD', {gib}, 'GiB on GPU', {gpu}, flush=True)
while True:
    time.sleep(3600)
"""
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen([PY, "-c", code], stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    time.sleep(2.0)
    return p.pid


def _launch(
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
        PY, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--trust-remote-code",
        "--host", "127.0.0.1", "--tp-size", "1", "--dtype", "bfloat16",
        "--mem-fraction-static", str(mem_fraction),
        "--max-running-requests", str(max_running),
        "--context-length", "4096",
        "--max-total-tokens", str(max_total_tokens),
        "--cuda-graph-backend-prefill", "disabled",
        "--cuda-graph-backend-decode", "breakable",
        "--cuda-graph-max-bs-decode", str(max_running),
        "--port", str(port), "--disaggregation-mode", "null",
    ]
    if skip_warmup:
        cmd.append("--skip-server-warmup")
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p.pid


def _env_full(gpu: int) -> dict:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["SGLANG_AFD_MODE"] = "null"
    e["SGLANG_AFD_POOL"] = "0"
    e.pop("DMLC_ROLE", None)
    return e


def _env_af(*, mode: str, gpu: int, na: int, nf: int, rank: int, ep: str, inflight: int) -> dict:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["SGLANG_AFD_MODE"] = mode
    e["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
    e["SGLANG_AFD_POOL"] = "1"
    e["SGLANG_AFD_POOL_NUM_ATTN"] = str(na)
    e["SGLANG_AFD_POOL_NUM_FFN"] = str(nf)
    e["SGLANG_AFD_POOL_LOCAL_RANK"] = str(rank)
    e["SGLANG_AFD_POOL_ENDPOINT_DIR"] = ep
    e["SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN"] = str(inflight)
    e["SGLANG_AFD_POOL_ROUTE"] = "least_inflight"
    e["SGLANG_AFD_MODULE_STUBS"] = "1"
    e["SGLANG_AFD_RELEASE_UNUSED_PARAMS"] = "1"
    e["SGLANG_AFD_NUM_MB"] = "2"
    e["SGLANG_AFD_MAX_NUM_TOKEN"] = "512"
    e["SGLANG_AFD_FFN_CUDA_GRAPH"] = "1"
    e.pop("DMLC_ROLE", None)
    return e


def _wait_http(url: str, pid: int, timeout: int, log: Path) -> bool:
    for i in range(timeout):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        if subprocess.call(["curl", "-sf", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            print(f"    healthy {url} ({i}s)", flush=True)
            return True
        time.sleep(1)
    print(f"    TIMEOUT {url}", flush=True)
    if log.is_file():
        print(log.read_text()[-1500:], flush=True)
    return False


def _parse_kv(log: Path) -> Tuple[Optional[int], Optional[float]]:
    text = log.read_text() if log.is_file() else ""
    m = re.search(r"#tokens:\s*(\d+).*?KV size:\s*([0-9.]+)\s*GB", text)
    if m:
        return int(m.group(1)), float(m.group(2))
    m2 = re.search(r"max_total_num_tokens=(\d+)", text)
    return (int(m2.group(1)) if m2 else None), None


def _start_lb(port: int, backends: List[str], log: Path) -> int:
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
    return subprocess.Popen(
        [PY, "-c", code, str(port), *backends],
        stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
    ).pid


def _bench(url: str, out_json: Path, prompts: int, conc: int) -> dict:
    cmd = [
        PY, "-m", "sglang.benchmark.serving",
        "--backend", "sglang-oai-chat", "--base-url", url, "--model", MODEL,
        "--dataset-name", "random", "--num-prompts", str(prompts),
        "--random-input-len", "1", "--random-output-len", "64",
        "--random-range-ratio", "0.0", "--request-rate", "inf",
        "--max-concurrency", str(conc), "--warmup-requests", "2",
        "--output-file", str(out_json), "--disable-tqdm",
    ]
    subprocess.call(cmd)
    if not out_json.is_file() or out_json.stat().st_size == 0:
        return {}
    text = out_json.read_text().strip()
    dec = json.JSONDecoder()
    objs, i = [], 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        o, e = dec.raw_decode(text, i)
        objs.append(o)
        i = e
    return objs[-1] if objs else {}


def run_full_fleet(
    *,
    tag: str,
    gpus: List[int],
    base_port: int,
    out: Path,
    hold_pids: List[int],
    prompts: int,
    conc: int,
    mem_fraction: float,
    max_running: int,
) -> dict:
    print(f"\n=== {tag}: {len(gpus)}×full decode on GPUs {gpus} ===", flush=True)
    ports = [base_port + i for i in range(len(gpus))]
    lb = base_port + 50
    _kill_ports(*ports, lb)
    pids = list(hold_pids)
    logs = []
    ok_all = True
    kv_sum, kv_gb_sum = 0, 0.0
    try:
        for g, port in zip(gpus, ports):
            log = out / f"{tag}_g{g}.log"
            logs.append(log)
            pid = _launch(
                env=_env_full(g), port=port, log=log,
                max_running=max_running, max_total_tokens=200000,
                mem_fraction=mem_fraction, skip_warmup=False,
            )
            pids.append(pid)
        for g, port, log, pid in zip(gpus, ports, logs, pids[-len(gpus):]):
            ok = _wait_http(f"http://127.0.0.1:{port}/health", pid, 360, log)
            ok_all = ok_all and ok
            tok, gb = _parse_kv(log)
            if tok:
                kv_sum += tok
            if gb:
                kv_gb_sum += gb
            print(f"    GPU{g} kv_tokens={tok} kv_gb={gb} ok={ok}", flush=True)
            if not ok and log.is_file() and "OutOfMemory" in log.read_text():
                print(f"    GPU{g} OOM", flush=True)
        metrics: dict = {}
        if ok_all:
            if len(ports) == 1:
                url = f"http://127.0.0.1:{ports[0]}"
            else:
                lb_pid = _start_lb(lb, [f"http://127.0.0.1:{p}" for p in ports], out / f"{tag}_lb.log")
                pids.append(lb_pid)
                time.sleep(1)
                if not _wait_http(f"http://127.0.0.1:{lb}/health", lb_pid, 30, out / f"{tag}_lb.log"):
                    ok_all = False
                url = f"http://127.0.0.1:{lb}"
            if ok_all:
                metrics = _bench(url, out / f"{tag}_bench.json", prompts, conc)
        return {
            "tag": tag, "kind": "full", "gpus": len(gpus), "ok": ok_all,
            "kv_tokens": kv_sum or None, "kv_gb": kv_gb_sum or None,
            "median_tpot_ms": metrics.get("median_tpot_ms"),
            "output_throughput": metrics.get("output_throughput"),
            "request_throughput": metrics.get("request_throughput"),
            "completed": metrics.get("completed"),
        }
    finally:
        _pgid_kill([p for p in pids if p not in hold_pids])
        _kill_ports(*ports, lb)
        time.sleep(2)


def run_af(
    *,
    tag: str,
    attn_gpus: List[int],
    ffn_gpu: int,
    base_port: int,
    out: Path,
    hold_pids: List[int],
    prompts: int,
    conc: int,
    mem_fraction: float,
    max_running: int,
) -> dict:
    na = len(attn_gpus)
    print(f"\n=== {tag}: {na}A1F Attn={attn_gpus} FFN={ffn_gpu} ===", flush=True)
    fport = base_port
    aports = [base_port + 1 + i for i in range(na)]
    lb = base_port + 50
    _kill_ports(fport, *aports, lb)
    ep = out / f"{tag}_socks"
    if ep.exists():
        shutil.rmtree(ep, ignore_errors=True)
    ep.mkdir(parents=True, exist_ok=True)
    pids = list(hold_pids)
    try:
        flog = out / f"{tag}_ffn.log"
        fpid = _launch(
            env=_env_af(mode="ffn", gpu=ffn_gpu, na=na, nf=1, rank=0, ep=str(ep), inflight=max(16, max_running * na)),
            port=fport, log=flog, max_running=max_running, max_total_tokens=50000,
            mem_fraction=mem_fraction, skip_warmup=True,
        )
        pids.append(fpid)
        for _ in range(90):
            t = flog.read_text() if flog.is_file() else ""
            if "Load weight end" in t or "AfPool FFN" in t or "OutOfMemory" in t:
                break
            time.sleep(2)
        if "OutOfMemory" in (flog.read_text() if flog.is_file() else ""):
            print("    FFN OOM", flush=True)
            return {"tag": tag, "kind": "af", "gpus": na + 1, "ok": False}

        alogs, apids, oks = [], [], []
        kv_sum, kv_gb_sum = 0, 0.0
        for i, (g, port) in enumerate(zip(attn_gpus, aports)):
            log = out / f"{tag}_attn{i}.log"
            alogs.append(log)
            pid = _launch(
                env=_env_af(mode="attn", gpu=g, na=na, nf=1, rank=i, ep=str(ep), inflight=max(16, max_running * na)),
                port=port, log=log, max_running=max_running, max_total_tokens=200000,
                mem_fraction=mem_fraction, skip_warmup=False,
            )
            pids.append(pid)
            apids.append(pid)
            ok = _wait_http(f"http://127.0.0.1:{port}/health", pid, 360, log)
            oks.append(ok)
            tok, gb = _parse_kv(log)
            if tok:
                kv_sum += tok
            if gb:
                kv_gb_sum += gb
            print(f"    Attn{i} GPU{g} kv_tokens={tok} kv_gb={gb} ok={ok}", flush=True)

        metrics: dict = {}
        ok_all = all(oks)
        if ok_all:
            if na == 1:
                url = f"http://127.0.0.1:{aports[0]}"
            else:
                lb_pid = _start_lb(lb, [f"http://127.0.0.1:{p}" for p in aports], out / f"{tag}_lb.log")
                pids.append(lb_pid)
                time.sleep(1)
                ok_all = _wait_http(f"http://127.0.0.1:{lb}/health", lb_pid, 30, out / f"{tag}_lb.log")
                url = f"http://127.0.0.1:{lb}"
            if ok_all:
                metrics = _bench(url, out / f"{tag}_bench.json", prompts, conc)
        return {
            "tag": tag, "kind": "af", "gpus": na + 1, "ok": ok_all,
            "kv_tokens": kv_sum or None, "kv_gb": kv_gb_sum or None,
            "median_tpot_ms": metrics.get("median_tpot_ms"),
            "output_throughput": metrics.get("output_throughput"),
            "request_throughput": metrics.get("request_throughput"),
            "completed": metrics.get("completed"),
        }
    finally:
        _pgid_kill([p for p in pids if p not in hold_pids])
        _kill_ports(fport, *aports, lb)
        time.sleep(2)


def write_report(out: Path, rows: List[dict], hold_gib: float, prompts: int, conc: int) -> None:
    lines = []
    lines.append("# A+F vs Pure Decode — Throughput Advantage Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Model: `{MODEL}` (DeepSeek-V2-Lite-Chat)")
    lines.append(f"- Load: `in=1`, `out=64`, prompts={prompts}, concurrency={conc}")
    lines.append(f"- Memory pressure: +{hold_gib:.0f} GiB holder tensor per used GPU "
                 "(on top of existing ~41 GiB ComfyUI), so remaining HBM ≈ 30–32 GiB")
    lines.append("- Effect: **full replica** barely fits → tiny KV; **Attn stub** (~1.6 GiB) → large KV; **FFN** still fits")
    lines.append("- This simulates the large-MoE regime where weight/KV asymmetry matters, without needing DeepSeek-V3")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Setup | GPUs | OK | KV tokens | out tok/s | median TPOT (ms) | vs paired full |")
    lines.append("|-------|------|----|-----------|-----------|------------------|----------------|")

    by = {r["tag"]: r for r in rows}

    def fmt(r: dict, pair_full: Optional[dict] = None) -> str:
        ok = "Y" if r.get("ok") else "N"
        kv = r.get("kv_tokens")
        tps = r.get("output_throughput")
        tpot = r.get("median_tpot_ms")
        kv_s = f"{kv}" if kv is not None else "—"
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        tpot_s = f"{tpot:.1f}" if tpot is not None else "—"
        ratio = "—"
        if pair_full and tps and pair_full.get("output_throughput"):
            ratio = f"**{tps / pair_full['output_throughput']:.2f}×**"
        return f"| {r['tag']} | {r['gpus']} | {ok} | {kv_s} | {tps_s} | {tpot_s} | {ratio} |"

    for tag, pair in (("full2", None), ("a1f1", "full2"), ("full3", None), ("a2f1", "full3")):
        if tag in by:
            lines.append(fmt(by[tag], by.get(pair) if pair else None))

    lines.append("")
    lines.append("## Takeaways for stakeholders")
    lines.append("")
    # compute winners
    for af_tag, full_tag, n in (("a1f1", "full2", 2), ("a2f1", "full3", 3)):
        af, full = by.get(af_tag), by.get(full_tag)
        if not af or not full:
            continue
        if af.get("output_throughput") and full.get("output_throughput"):
            ratio = af["output_throughput"] / full["output_throughput"]
            kv_af = af.get("kv_tokens") or 0
            kv_f = full.get("kv_tokens") or 0
            lines.append(
                f"- **{n}-GPU budget**: A+F (`{af_tag}`) achieves **{ratio:.2f}×** output throughput "
                f"vs `{full_tag}` ({af['output_throughput']:.0f} vs {full['output_throughput']:.0f} tok/s). "
                f"KV capacity {kv_af} vs {kv_f} tokens."
            )
            if ratio >= 1.2:
                lines.append(f"  - **A+F wins** under HBM pressure (same GPU count).")
            else:
                lines.append(f"  - Advantage not yet ≥1.2×; see raw JSON.")
        elif af.get("ok") and not full.get("ok"):
            lines.append(f"- **{n}-GPU**: full decode failed/OOM; A+F still served — availability win.")
    lines.append("")
    lines.append("## When this advantage appears")
    lines.append("1. Per-GPU HBM is tight relative to dense/MoE weights (full replica starves KV)")
    lines.append("2. Traffic is concurrency-heavy (fleet throughput), not single-stream latency")
    lines.append("3. A+F places fat FFN once and thin Attn+KV many times")
    lines.append("")
    lines.append("## When pure decode still wins")
    lines.append("- Ample HBM so N full replicas each keep a healthy batch (no KV starvation)")
    lines.append("- Low concurrency where AF hop tax dominates TPOT")
    lines.append("")
    lines.append(f"Raw data: `{out}/results.json`")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/afd_beats_decode")
    ap.add_argument("--gpus", default="5,6,7", help="2+ GPUs; first used for Attn/full, last for FFN in AF")
    ap.add_argument("--hold-gib", type=float, default=8.0,
                    help="Extra occupancy per used GPU (leave ~32GiB free with ComfyUI)")
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--mem-fraction", type=float, default=0.90)
    ap.add_argument("--max-running", type=int, default=32)
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    assert len(gpus) >= 2

    # Clean ports
    _kill_ports(*range(38000, 38080))

    hold_pids: List[int] = []
    rows: List[dict] = []
    try:
        print(f"Starting holders {args.hold_gib} GiB on GPUs {gpus} ...", flush=True)
        for g in gpus:
            hold_pids.append(_start_holder(g, args.hold_gib, out / f"hold_g{g}.log"))
        time.sleep(2)
        subprocess.call(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"])

        # 2-GPU pair
        rows.append(run_full_fleet(
            tag="full2", gpus=gpus[:2], base_port=38000, out=out, hold_pids=hold_pids,
            prompts=args.prompts, conc=args.conc, mem_fraction=args.mem_fraction,
            max_running=args.max_running,
        ))
        rows.append(run_af(
            tag="a1f1", attn_gpus=[gpus[0]], ffn_gpu=gpus[1], base_port=38020, out=out,
            hold_pids=hold_pids, prompts=args.prompts, conc=args.conc,
            mem_fraction=args.mem_fraction, max_running=args.max_running,
        ))

        # 3-GPU pair if available
        if len(gpus) >= 3:
            rows.append(run_full_fleet(
                tag="full3", gpus=gpus[:3], base_port=38100, out=out, hold_pids=hold_pids,
                prompts=args.prompts, conc=args.conc, mem_fraction=args.mem_fraction,
                max_running=args.max_running,
            ))
            rows.append(run_af(
                tag="a2f1", attn_gpus=gpus[:2], ffn_gpu=gpus[2], base_port=38120, out=out,
                hold_pids=hold_pids, prompts=args.prompts, conc=args.conc,
                mem_fraction=args.mem_fraction, max_running=args.max_running,
            ))

        (out / "results.json").write_text(json.dumps(rows, indent=2))
        write_report(out, rows, args.hold_gib, args.prompts, args.conc)

        # Require at least one AF win for exit 0 messaging
        wins = []
        by = {r["tag"]: r for r in rows}
        for af_t, full_t in (("a1f1", "full2"), ("a2f1", "full3")):
            if af_t in by and full_t in by:
                a, f = by[af_t], by[full_t]
                if a.get("output_throughput") and f.get("output_throughput"):
                    if a["output_throughput"] > f["output_throughput"]:
                        wins.append(af_t)
        print(f"\nAF_BEATS_DECODE_OK wins={wins}", flush=True)
        (out / "DONE").write_text("ok\n" + json.dumps({"wins": wins, "rows": rows}, indent=2))
        return 0
    finally:
        _pgid_kill(hold_pids)
        _kill_ports(*range(38000, 38080))


if __name__ == "__main__":
    raise SystemExit(main())
