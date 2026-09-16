# SPDX-License-Identifier: Apache-2.0
"""Microbench + short serving compare: full vs full+fake-EP vs (optional) note AF hop.

Example::

    source /root/.cuda/afd_env.sh
    python -m sglang.srt.afd.bench_fake_ep_comm --micro
    python -m sglang.srt.afd.bench_fake_ep_comm --serve --prompts 32 --conc 16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from sglang.srt.afd.fake_ep_comm import (
    ep_comm_bytes,
    ep_comm_seconds,
    fake_ep_stats,
    maybe_simulate_ep_comm,
    reset_fake_ep_stats,
)
from sglang.srt.environ import envs


def _micro(tokens: int, layers: int, topk: int, hidden: int, ep: int, mode: str) -> dict:
    envs.SGLANG_FAKE_EP_COMM.set(True)
    envs.SGLANG_FAKE_EP_SIZE.set(ep)
    envs.SGLANG_FAKE_EP_MODE.set(mode)
    envs.SGLANG_FAKE_EP_FULL_ONLY.set(False)  # force on for micro
    envs.SGLANG_FAKE_EP_BW_GBS.set(150.0)
    envs.SGLANG_FAKE_EP_LAT_US.set(10.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    # warmup
    for _ in range(3):
        maybe_simulate_ep_comm(h, topk=topk, hidden_size=hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    reset_fake_ep_stats()
    t0 = time.perf_counter()
    for _ in range(layers):
        maybe_simulate_ep_comm(h, topk=topk, hidden_size=hidden)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    nbytes = ep_comm_bytes(tokens, topk=topk, hidden=hidden, elem_size=2, ep_size=ep)
    model_ms = ep_comm_seconds(nbytes, bw_gbs=150.0, lat_us=10.0) * layers * 1e3
    st = fake_ep_stats()
    return {
        "tokens": tokens,
        "layers": layers,
        "topk": topk,
        "hidden": hidden,
        "ep_size": ep,
        "mode": mode,
        "bytes_per_layer": nbytes,
        "model_ms_total": model_ms,
        "wall_ms_total": wall_ms,
        "stats": st,
    }


def _serve_once(*, tag: str, out: Path, fake_ep: bool, port: int, prompts: int, conc: int) -> dict:
    model = os.environ.get("MODEL_PATH", "/data/share/models/DeepSeek-V2-Lite-Chat")
    env = os.environ.copy()
    env["SGLANG_AFD_MODE"] = "null"
    env["SGLANG_AFD_POOL"] = "0"
    env["SGLANG_FAKE_EP_COMM"] = "1" if fake_ep else "0"
    if fake_ep:
        env.setdefault("SGLANG_FAKE_EP_SIZE", "8")
        env.setdefault("SGLANG_FAKE_EP_MODE", "copy")
        env.setdefault("SGLANG_FAKE_EP_FULL_ONLY", "1")
        env.setdefault("SGLANG_FAKE_EP_BW_GBS", "150")
        env.setdefault("SGLANG_FAKE_EP_LAT_US", "10")
    env["CUDA_VISIBLE_DEVICES"] = env.get("BENCH_GPU", "5")
    log = out / f"{tag}.log"
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model, "--trust-remote-code",
        "--host", "127.0.0.1", "--port", str(port),
        "--tp-size", "1", "--dtype", "bfloat16",
        "--mem-fraction-static", "0.82",
        "--max-running-requests", str(max(8, conc)),
        "--context-length", "4096", "--max-total-tokens", "50000",
        "--cuda-graph-backend-prefill", "disabled",
        "--cuda-graph-backend-decode", "breakable",
        "--cuda-graph-max-bs-decode", str(max(8, conc)),
        "--disaggregation-mode", "null",
    ]
    f = open(log, "w")
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    try:
        for i in range(300):
            if p.poll() is not None:
                raise RuntimeError(f"{tag} died early; see {log}")
            if subprocess.call(
                ["curl", "-sf", f"http://127.0.0.1:{port}/health"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) == 0:
                print(f"{tag} healthy {i}s", flush=True)
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"{tag} health timeout")
        bench_json = out / f"{tag}_bench.json"
        subprocess.check_call([
            sys.executable, "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", model, "--dataset-name", "random",
            "--num-prompts", str(prompts),
            "--random-input-len", "1", "--random-output-len", "64",
            "--random-range-ratio", "0.0", "--request-rate", "inf",
            "--max-concurrency", str(conc), "--warmup-requests", "2",
            "--output-file", str(bench_json), "--disable-tqdm",
        ])
        text = bench_json.read_text().strip()
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
        d = objs[-1]
        return {
            "tag": tag,
            "fake_ep": fake_ep,
            "median_tpot_ms": d.get("median_tpot_ms"),
            "output_throughput": d.get("output_throughput"),
            "completed": d.get("completed"),
        }
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        subprocess.call(["bash", "-c", f"fuser -k {port}/tcp >/dev/null 2>&1 || true"])
        time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--out", default="/tmp/afd_fake_ep")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--layers", type=int, default=26)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--mode", default="copy", choices=["delay", "copy", "nvlink"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.micro or not args.serve:
        rows = []
        for mode in ([args.mode] if args.micro else ["delay", "copy"]):
            for tok in (1, 8, 32, 128):
                r = _micro(tok, args.layers, topk=6, hidden=2048, ep=args.ep, mode=mode)
                rows.append(r)
                print(
                    f"micro mode={mode} tok={tok} bytes/layer={r['bytes_per_layer']} "
                    f"wall={r['wall_ms_total']:.2f}ms model={r['model_ms_total']:.2f}ms",
                    flush=True,
                )
        (out / "micro.json").write_text(json.dumps(rows, indent=2))

    if args.serve:
        results = []
        results.append(_serve_once(
            tag="full0", out=out, fake_ep=False, port=38200,
            prompts=args.prompts, conc=args.conc,
        ))
        results.append(_serve_once(
            tag="full_ep_sim", out=out, fake_ep=True, port=38201,
            prompts=args.prompts, conc=args.conc,
        ))
        (out / "serve.json").write_text(json.dumps(results, indent=2))
        print("\n=== full vs full+fake-EP ===", flush=True)
        for r in results:
            print(
                f"  {r['tag']}: tpot={r['median_tpot_ms']:.2f}ms "
                f"out_tps={r['output_throughput']:.1f} done={r['completed']}",
                flush=True,
            )
        if results[0]["output_throughput"] and results[1]["output_throughput"]:
            print(
                f"  ratio full_ep_sim/full0 = "
                f"{results[1]['output_throughput']/results[0]['output_throughput']:.3f}x",
                flush=True,
            )
        print("FAKE_EP_COMM_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
