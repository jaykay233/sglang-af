# SPDX-License-Identifier: Apache-2.0
"""Fair 3-GPU Prefill+Decode vs Prefill+AF under V3-proxy MoE tax (Lite weights).

Same total GPUs
---------------
- ``pd_2d``   : 1 Prefill + 2 Decode (fake V3 EP+compute on each Decode)
- ``pd_afd``  : 1 Prefill + 1 Attn + 1 FFN (cuda_ipc AF; EP tax off on AF;
               MoE compute tax still on FFN)

Why this is fairer than previous Lite benches
---------------------------------------------
PD decode normally has zero DeepEP hop on TP=1; AF always pays A2F/F2A.
``SGLANG_FAKE_V3_PROFILE=v3_proxy`` sizes EP volume + expert FLOPs like V3 while
keeping Lite resident (fits ~40GB free).

Example::

    source /root/.cuda/afd_env.sh
    python -m sglang.srt.afd.bench_v3_proxy_fair \\
        --out /tmp/afd_v3_proxy_fair --prompts 32 --conc 16
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from sglang.srt.afd.fake_ep_comm import v3_proxy_meta


def _load_bench_json(path: Path) -> dict:
    text = path.read_text().strip()
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
    if not objs:
        raise RuntimeError(f"empty bench json {path}")
    return objs[-1]


def _kill_pids(pids: List[int]) -> None:
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.killpg(p, signal.SIGTERM)
        except Exception:
            pass
    time.sleep(2)
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.killpg(p, signal.SIGKILL)
        except Exception:
            pass


def _fuser_ports(ports: List[int]) -> None:
    for port in ports:
        subprocess.call(
            ["bash", "-c", f"fuser -k {port}/tcp >/dev/null 2>&1 || true"]
        )


def _wait_http(url: str, name: str, pid: Optional[int], timeout: int = 400) -> None:
    for i in range(timeout):
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError as e:
                raise RuntimeError(f"{name} died early") from e
        if subprocess.call(
            ["curl", "-sf", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) == 0:
            print(f"{name} healthy after {i}s", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"{name} health timeout ({url})")


def _wait_log(path: Path, pattern: str, pid: int, timeout: int = 300) -> None:
    import re

    rx = re.compile(pattern)
    for i in range(timeout):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError(f"process died waiting for {pattern} in {path}")
        if path.exists() and rx.search(path.read_text(errors="ignore")):
            print(f"log hit {pattern!r} after {i}s", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"timeout waiting for {pattern} in {path}")


def _base_env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.pop("DMLC_ROLE", None)
    # Keep AF defaults clean unless overwritten.
    env.setdefault("SGLANG_AFD_POOL", "0")
    return env


def _v3_proxy_env(env: dict, *, enable_ep: bool, enable_compute: bool) -> dict:
    env = dict(env)
    env["SGLANG_FAKE_V3_PROFILE"] = "v3_proxy"
    env["SGLANG_FAKE_EP_COMM"] = "1" if enable_ep else "0"
    env["SGLANG_FAKE_EP_FULL_ONLY"] = "1"
    env["SGLANG_FAKE_EP_MODE"] = env.get("SGLANG_FAKE_EP_MODE", "copy")
    if not enable_compute:
        # Profile would set compute; clear after profile apply via explicit 0.
        env["SGLANG_FAKE_MOE_COMPUTE_SCALE"] = "0"
        env["SGLANG_FAKE_V3_PROFILE"] = ""
        if enable_ep:
            # Manual V3 EP dims without compute.
            env["SGLANG_FAKE_EP_SIZE"] = "64"
            env["SGLANG_FAKE_EP_TOPK"] = "8"
            env["SGLANG_FAKE_EP_HIDDEN"] = "7168"
            env["SGLANG_FAKE_EP_BYTES_SCALE"] = "2.23"
    return env


def _model_cmd(
    model: str,
    port: int,
    *,
    max_bs: int,
    extra: List[str],
    mem_fraction: float = 0.82,
    max_total_tokens: int = 50000,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        str(mem_fraction),
        "--max-running-requests",
        str(max_bs),
        "--context-length",
        "4096",
        "--max-total-tokens",
        str(max_total_tokens),
        "--cuda-graph-backend-prefill",
        "disabled",
        *extra,
    ]


def _pd_args(ib: str) -> List[str]:
    return [
        "--disaggregation-transfer-backend",
        "mooncake",
        "--disaggregation-ib-device",
        ib,
    ]


def _run_bench(
    *,
    base_url: str,
    model: str,
    out_json: Path,
    prompts: int,
    conc: int,
    in_len: int,
    out_len: int,
) -> dict:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang-oai-chat",
            "--base-url",
            base_url,
            "--model",
            model,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(prompts),
            "--random-input-len",
            str(in_len),
            "--random-output-len",
            str(out_len),
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
    )
    return _load_bench_json(out_json)


def run_pd_2d(
    *,
    out: Path,
    model: str,
    gpus: List[str],
    ports: dict,
    ib: str,
    bootstrap: int,
    max_bs: int,
    prompts: int,
    conc: int,
    in_len: int,
    out_len: int,
) -> dict:
    """1 Prefill + 2 Decode with V3-proxy tax on decode. Total 3 GPUs."""
    assert len(gpus) >= 3
    tag = "pd_2d"
    log = out / tag
    log.mkdir(parents=True, exist_ok=True)
    pids: List[int] = []
    pref_p, d0_p, d1_p, lb_p = ports["prefill"], ports["d0"], ports["d1"], ports["lb"]
    _fuser_ports([pref_p, d0_p, d1_p, lb_p, bootstrap])

    # Prefill: no MoE tax needed for fairness on decode path; keep off.
    env_p = _v3_proxy_env(_base_env(gpus[0]), enable_ep=False, enable_compute=False)
    env_p["SGLANG_AFD_MODE"] = "null"
    f_pref = open(log / "prefill.log", "w")
    p_pref = subprocess.Popen(
        _model_cmd(
            model,
            pref_p,
            max_bs=max_bs,
            extra=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_p,
        stdout=f_pref,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_pref.pid)

    dec_pids = []
    for i, (gpu, port) in enumerate([(gpus[1], d0_p), (gpus[2], d1_p)]):
        env_d = _v3_proxy_env(_base_env(gpu), enable_ep=True, enable_compute=True)
        env_d["SGLANG_AFD_MODE"] = "null"
        f = open(log / f"decode{i}.log", "w")
        p = subprocess.Popen(
            _model_cmd(
                model,
                port,
                max_bs=max_bs,
                extra=[
                    "--cuda-graph-backend-decode",
                    "full",
                    "--cuda-graph-max-bs-decode",
                    str(max_bs),
                    "--disaggregation-mode",
                    "decode",
                    "--disaggregation-bootstrap-port",
                    str(bootstrap),
                    *_pd_args(ib),
                ],
            ),
            env=env_d,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p.pid)
        dec_pids.append(p.pid)

    try:
        _wait_http(f"http://127.0.0.1:{pref_p}/health", "Prefill", p_pref.pid)
        _wait_http(f"http://127.0.0.1:{d0_p}/health", "Decode0", dec_pids[0], 500)
        _wait_http(f"http://127.0.0.1:{d1_p}/health", "Decode1", dec_pids[1], 500)

        f_lb = open(log / "router.log", "w")
        p_lb = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://127.0.0.1:{pref_p}",
                "--decode",
                f"http://127.0.0.1:{d0_p}",
                "--decode",
                f"http://127.0.0.1:{d1_p}",
                "--decode-policy",
                "round_robin",
                "--host",
                "127.0.0.1",
                "--port",
                str(lb_p),
            ],
            stdout=f_lb,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p_lb.pid)
        _wait_http(f"http://127.0.0.1:{lb_p}/health", "Router", p_lb.pid)

        bench = _run_bench(
            base_url=f"http://127.0.0.1:{lb_p}",
            model=model,
            out_json=log / "bench.json",
            prompts=prompts,
            conc=conc,
            in_len=in_len,
            out_len=out_len,
        )
        return {
            "tag": tag,
            "gpus": 3,
            "layout": "1P+2D",
            "fake_v3_proxy": True,
            "median_tpot_ms": bench.get("median_tpot_ms"),
            "median_ttft_ms": bench.get("median_ttft_ms"),
            "output_throughput": bench.get("output_throughput"),
            "completed": bench.get("completed"),
        }
    finally:
        _kill_pids(pids)
        _fuser_ports([pref_p, d0_p, d1_p, lb_p, bootstrap])
        time.sleep(3)


def run_pd_afd(
    *,
    out: Path,
    model: str,
    gpus: List[str],
    ports: dict,
    ib: str,
    bootstrap: int,
    max_bs: int,
    prompts: int,
    conc: int,
    in_len: int,
    out_len: int,
) -> dict:
    """1 Prefill + 1 Attn + 1 FFN. EP tax off; FFN still pays MoE compute scale."""
    assert len(gpus) >= 3
    tag = "pd_afd"
    log = out / tag
    log.mkdir(parents=True, exist_ok=True)
    pids: List[int] = []
    pref_p, attn_p, ffn_p, lb_p = (
        ports["prefill"],
        ports["attn"],
        ports["ffn"],
        ports["lb"],
    )
    sock = str(out / "afd_cuda_ipc.sock")
    _fuser_ports([pref_p, attn_p, ffn_p, lb_p, bootstrap])
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass

    # Prefill
    env_p = _v3_proxy_env(_base_env(gpus[0]), enable_ep=False, enable_compute=False)
    env_p["SGLANG_AFD_MODE"] = "null"
    f_pref = open(log / "prefill.log", "w")
    p_pref = subprocess.Popen(
        _model_cmd(
            model,
            pref_p,
            max_bs=max_bs,
            extra=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_p,
        stdout=f_pref,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_pref.pid)

    # FFN: compute tax ON (experts here), EP OFF (FULL_ONLY + not null... wait FFN mode
    # is ffn so EP skipped; compute still on via profile).
    env_f = _base_env(gpus[2])
    env_f["SGLANG_AFD_MODE"] = "ffn"
    env_f["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
    env_f["SGLANG_AFD_IPC_ENDPOINT"] = sock
    env_f["SGLANG_AFD_MODULE_STUBS"] = "1"
    env_f["SGLANG_AFD_RELEASE_UNUSED_PARAMS"] = "1"
    env_f["SGLANG_AFD_NUM_MB"] = "2"
    # Keep pad tight to max_bs — large MAX_NUM_TOKEN blows FFN CUDA-graph HBM.
    env_f["SGLANG_AFD_MAX_NUM_TOKEN"] = str(max_bs)
    env_f["SGLANG_AFD_FFN_CUDA_GRAPH"] = "1"
    env_f["SGLANG_FAKE_EP_COMM"] = "0"
    env_f["SGLANG_FAKE_V3_PROFILE"] = "v3_proxy"  # sets compute scale
    env_f["SGLANG_FAKE_EP_FULL_ONLY"] = "1"
    f_ffn = open(log / "ffn.log", "w")
    p_ffn = subprocess.Popen(
        _model_cmd(
            model,
            ffn_p,
            max_bs=max_bs,
            extra=["--disaggregation-mode", "null", "--skip-server-warmup"],
        ),
        env=env_f,
        stdout=f_ffn,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_ffn.pid)

    # Attn decode: no EP, no MoE compute (stubs / remote FFN)
    env_a = _base_env(gpus[1])
    env_a["SGLANG_AFD_MODE"] = "attn"
    env_a["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
    env_a["SGLANG_AFD_IPC_ENDPOINT"] = sock
    env_a["SGLANG_AFD_MODULE_STUBS"] = "1"
    env_a["SGLANG_AFD_NUM_MB"] = "2"
    env_a["SGLANG_AFD_MAX_NUM_TOKEN"] = str(max_bs)
    # Prefer stable bring-up over aggressive overlap under tight HBM.
    env_a["SGLANG_AFD_TRUE_OVERLAP"] = "0"
    env_a["SGLANG_AFD_LAYER_PIPELINE"] = "0"
    env_a["SGLANG_AFD_PIPELINE"] = "0"
    env_a["SGLANG_AFD_REMOTE_FROM_LAYER"] = "0"
    env_a["SGLANG_AFD_LAYER_MERGE_K"] = "1"
    env_a["SGLANG_FAKE_EP_COMM"] = "0"
    env_a["SGLANG_FAKE_MOE_COMPUTE_SCALE"] = "0"
    env_a["SGLANG_FAKE_V3_PROFILE"] = ""
    f_attn = open(log / "attn.log", "w")
    p_attn = subprocess.Popen(
        _model_cmd(
            model,
            attn_p,
            max_bs=max_bs,
            extra=[
                "--cuda-graph-backend-decode",
                "breakable",
                "--cuda-graph-max-bs-decode",
                str(max_bs),
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_a,
        stdout=f_attn,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_attn.pid)

    try:
        _wait_http(f"http://127.0.0.1:{pref_p}/health", "Prefill", p_pref.pid)
        _wait_log(log / "ffn.log", r"Load weight end|AFD cuda_ipc FFN waiting", p_ffn.pid)
        _wait_http(f"http://127.0.0.1:{attn_p}/health", "Attn", p_attn.pid, 500)

        f_lb = open(log / "router.log", "w")
        p_lb = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://127.0.0.1:{pref_p}",
                "--decode",
                f"http://127.0.0.1:{attn_p}",
                "--host",
                "127.0.0.1",
                "--port",
                str(lb_p),
            ],
            stdout=f_lb,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p_lb.pid)
        _wait_http(f"http://127.0.0.1:{lb_p}/health", "Router", p_lb.pid)

        bench = _run_bench(
            base_url=f"http://127.0.0.1:{lb_p}",
            model=model,
            out_json=log / "bench.json",
            prompts=prompts,
            conc=conc,
            in_len=in_len,
            out_len=out_len,
        )
        return {
            "tag": tag,
            "gpus": 3,
            "layout": "1P+1A+1F",
            "fake_v3_proxy": "ffn_compute_only",
            "median_tpot_ms": bench.get("median_tpot_ms"),
            "median_ttft_ms": bench.get("median_ttft_ms"),
            "output_throughput": bench.get("output_throughput"),
            "completed": bench.get("completed"),
        }
    finally:
        _kill_pids(pids)
        _fuser_ports([pref_p, attn_p, ffn_p, lb_p, bootstrap])
        time.sleep(3)


def run_pd_afd_2a_same(
    *,
    out: Path,
    model: str,
    gpus: List[str],
    ports: dict,
    ib: str,
    bootstrap: int,
    max_bs: int,
    prompts: int,
    conc: int,
    in_len: int,
    out_len: int,
) -> dict:
    """1 Prefill + 2 Attn on SAME GPU + 1 FFN (AfPool). Same 3 physical GPUs."""
    assert len(gpus) >= 3
    tag = "pd_afd_2a_same"
    log = out / tag
    log.mkdir(parents=True, exist_ok=True)
    ep = log / "socks"
    ep.mkdir(parents=True, exist_ok=True)
    pids: List[int] = []
    pref_p, a0_p, a1_p, ffn_p, lb_p = (
        ports["prefill"],
        ports["a0"],
        ports["a1"],
        ports["ffn"],
        ports["lb"],
    )
    _fuser_ports([pref_p, a0_p, a1_p, ffn_p, lb_p, bootstrap])

    # Dual Attn share one GPU (~40GB free with Comfy) → lower fraction + bs.
    attn_bs = max(4, max_bs // 2)
    attn_mem = 0.38

    env_p = _v3_proxy_env(_base_env(gpus[0]), enable_ep=False, enable_compute=False)
    env_p["SGLANG_AFD_MODE"] = "null"
    f_pref = open(log / "prefill.log", "w")
    p_pref = subprocess.Popen(
        _model_cmd(
            model,
            pref_p,
            max_bs=max_bs,
            extra=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_p,
        stdout=f_pref,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_pref.pid)

    def _pool_env(gpu: str, mode: str, rank: int) -> dict:
        e = _base_env(gpu)
        e["SGLANG_AFD_MODE"] = mode
        e["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
        e["SGLANG_AFD_POOL"] = "1"
        e["SGLANG_AFD_POOL_NUM_ATTN"] = "2"
        e["SGLANG_AFD_POOL_NUM_FFN"] = "1"
        e["SGLANG_AFD_POOL_LOCAL_RANK"] = str(rank)
        e["SGLANG_AFD_POOL_ENDPOINT_DIR"] = str(ep)
        e["SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN"] = "16"
        e["SGLANG_AFD_POOL_ROUTE"] = "least_inflight"
        e["SGLANG_AFD_MODULE_STUBS"] = "1"
        e["SGLANG_AFD_NUM_MB"] = "2"
        e["SGLANG_AFD_MAX_NUM_TOKEN"] = str(attn_bs)
        e["SGLANG_AFD_TRUE_OVERLAP"] = "0"
        e["SGLANG_AFD_LAYER_PIPELINE"] = "0"
        e["SGLANG_AFD_PIPELINE"] = "0"
        e["SGLANG_AFD_REMOTE_FROM_LAYER"] = "0"
        e["SGLANG_AFD_LAYER_MERGE_K"] = "1"
        e["SGLANG_FAKE_EP_COMM"] = "0"
        e["SGLANG_FAKE_MOE_COMPUTE_SCALE"] = "0"
        e["SGLANG_FAKE_V3_PROFILE"] = ""
        return e

    env_f = _pool_env(gpus[2], "ffn", 0)
    env_f["SGLANG_AFD_RELEASE_UNUSED_PARAMS"] = "1"
    env_f["SGLANG_AFD_FFN_CUDA_GRAPH"] = "1"
    env_f["SGLANG_FAKE_V3_PROFILE"] = "v3_proxy"
    env_f["SGLANG_FAKE_EP_FULL_ONLY"] = "1"
    f_ffn = open(log / "ffn.log", "w")
    p_ffn = subprocess.Popen(
        _model_cmd(
            model,
            ffn_p,
            max_bs=max_bs,
            mem_fraction=0.82,
            extra=["--disaggregation-mode", "null", "--skip-server-warmup"],
        ),
        env=env_f,
        stdout=f_ffn,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_ffn.pid)

    attn_pids = []
    for rank, port, name in ((0, a0_p, "attn0"), (1, a1_p, "attn1")):
        env_a = _pool_env(gpus[1], "attn", rank)
        f = open(log / f"{name}.log", "w")
        p = subprocess.Popen(
            _model_cmd(
                model,
                port,
                max_bs=attn_bs,
                mem_fraction=attn_mem,
                extra=[
                    "--cuda-graph-backend-decode",
                    "breakable",
                    "--cuda-graph-max-bs-decode",
                    str(attn_bs),
                    "--disaggregation-mode",
                    "decode",
                    "--disaggregation-bootstrap-port",
                    str(bootstrap),
                    *_pd_args(ib),
                ],
            ),
            env=env_a,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p.pid)
        attn_pids.append(p.pid)

    try:
        _wait_http(f"http://127.0.0.1:{pref_p}/health", "Prefill", p_pref.pid)
        _wait_log(
            log / "ffn.log",
            r"Load weight end|AFD cuda_ipc FFN waiting|AfPool",
            p_ffn.pid,
        )
        # Start second Attn after first begins connecting — both on same GPU.
        _wait_http(f"http://127.0.0.1:{a0_p}/health", "Attn0", attn_pids[0], 500)
        _wait_http(f"http://127.0.0.1:{a1_p}/health", "Attn1", attn_pids[1], 500)

        f_lb = open(log / "router.log", "w")
        p_lb = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://127.0.0.1:{pref_p}",
                "--decode",
                f"http://127.0.0.1:{a0_p}",
                "--decode",
                f"http://127.0.0.1:{a1_p}",
                "--decode-policy",
                "round_robin",
                "--host",
                "127.0.0.1",
                "--port",
                str(lb_p),
            ],
            stdout=f_lb,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p_lb.pid)
        _wait_http(f"http://127.0.0.1:{lb_p}/health", "Router", p_lb.pid)

        bench = _run_bench(
            base_url=f"http://127.0.0.1:{lb_p}",
            model=model,
            out_json=log / "bench.json",
            prompts=prompts,
            conc=conc,
            in_len=in_len,
            out_len=out_len,
        )
        return {
            "tag": tag,
            "gpus": 3,
            "layout": "1P+2A(same)+1F",
            "fake_v3_proxy": "ffn_compute_only",
            "attn_mem_fraction": attn_mem,
            "attn_max_bs": attn_bs,
            "median_tpot_ms": bench.get("median_tpot_ms"),
            "median_ttft_ms": bench.get("median_ttft_ms"),
            "output_throughput": bench.get("output_throughput"),
            "completed": bench.get("completed"),
        }
    finally:
        _kill_pids(pids)
        _fuser_ports([pref_p, a0_p, a1_p, ffn_p, lb_p, bootstrap])
        time.sleep(3)


def run_pd_afd_a1f2_same(
    *,
    out: Path,
    model: str,
    gpus: List[str],
    ports: dict,
    ib: str,
    bootstrap: int,
    max_bs: int,
    prompts: int,
    conc: int,
    in_len: int,
    out_len: int,
) -> dict:
    """1 Prefill + 1 Attn + 2 FFN on the SAME GPU (AfPool). Low FFN KV budget."""
    assert len(gpus) >= 3
    tag = "pd_afd_a1f2_same"
    log = out / tag
    log.mkdir(parents=True, exist_ok=True)
    ep = log / "socks"
    ep.mkdir(parents=True, exist_ok=True)
    pids: List[int] = []
    pref_p, attn_p, f0_p, f1_p, lb_p = (
        ports["prefill"],
        ports["attn"],
        ports["f0"],
        ports["f1"],
        ports["lb"],
    )
    _fuser_ports([pref_p, attn_p, f0_p, f1_p, lb_p, bootstrap])

    # FFN: no real KV need — shrink KV tokens; with a free GPU, mem can be higher
    # so two expert replicas fit (weights dominate, not KV).
    ffn_mem = float(os.environ.get("AFD_FFN_MEM_FRACTION", "0.45"))
    ffn_kv_tokens = int(os.environ.get("AFD_FFN_MAX_TOTAL_TOKENS", "2048"))
    attn_mem = float(os.environ.get("AFD_ATTN_MEM_FRACTION", "0.82"))

    env_p = _v3_proxy_env(_base_env(gpus[0]), enable_ep=False, enable_compute=False)
    env_p["SGLANG_AFD_MODE"] = "null"
    f_pref = open(log / "prefill.log", "w")
    p_pref = subprocess.Popen(
        _model_cmd(
            model,
            pref_p,
            max_bs=max_bs,
            extra=[
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_p,
        stdout=f_pref,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_pref.pid)

    def _pool_env(gpu: str, mode: str, rank: int) -> dict:
        e = _base_env(gpu)
        e["SGLANG_AFD_MODE"] = mode
        e["SGLANG_AFD_TRANSPORT"] = "cuda_ipc"
        e["SGLANG_AFD_POOL"] = "1"
        e["SGLANG_AFD_POOL_NUM_ATTN"] = "1"
        e["SGLANG_AFD_POOL_NUM_FFN"] = "2"
        e["SGLANG_AFD_POOL_LOCAL_RANK"] = str(rank)
        e["SGLANG_AFD_POOL_ENDPOINT_DIR"] = str(ep)
        e["SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN"] = "8"
        e["SGLANG_AFD_POOL_ROUTE"] = "least_inflight"
        e["SGLANG_AFD_MODULE_STUBS"] = "1"
        e["SGLANG_AFD_NUM_MB"] = "2"
        e["SGLANG_AFD_MAX_NUM_TOKEN"] = str(max_bs)
        e["SGLANG_AFD_TRUE_OVERLAP"] = "0"
        e["SGLANG_AFD_LAYER_PIPELINE"] = "0"
        e["SGLANG_AFD_PIPELINE"] = "0"
        e["SGLANG_AFD_REMOTE_FROM_LAYER"] = "0"
        e["SGLANG_AFD_LAYER_MERGE_K"] = "1"
        e["SGLANG_FAKE_EP_COMM"] = "0"
        e["SGLANG_FAKE_MOE_COMPUTE_SCALE"] = "0"
        e["SGLANG_FAKE_V3_PROFILE"] = ""
        return e

    # Two FFN processes on gpus[2]
    ffn_pids = []
    for rank, port, name in ((0, f0_p, "ffn0"), (1, f1_p, "ffn1")):
        env_f = _pool_env(gpus[2], "ffn", rank)
        env_f["SGLANG_AFD_RELEASE_UNUSED_PARAMS"] = "1"
        env_f["SGLANG_AFD_FFN_CUDA_GRAPH"] = "1"
        env_f["SGLANG_FAKE_V3_PROFILE"] = "v3_proxy"
        env_f["SGLANG_FAKE_EP_FULL_ONLY"] = "1"
        f = open(log / f"{name}.log", "w")
        p = subprocess.Popen(
            _model_cmd(
                model,
                port,
                max_bs=max_bs,
                mem_fraction=ffn_mem,
                max_total_tokens=ffn_kv_tokens,
                extra=["--disaggregation-mode", "null", "--skip-server-warmup"],
            ),
            env=env_f,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p.pid)
        ffn_pids.append(p.pid)

    env_a = _pool_env(gpus[1], "attn", 0)
    f_attn = open(log / "attn.log", "w")
    p_attn = subprocess.Popen(
        _model_cmd(
            model,
            attn_p,
            max_bs=max_bs,
            mem_fraction=attn_mem,
            extra=[
                "--cuda-graph-backend-decode",
                "breakable",
                "--cuda-graph-max-bs-decode",
                str(max_bs),
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                str(bootstrap),
                *_pd_args(ib),
            ],
        ),
        env=env_a,
        stdout=f_attn,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    pids.append(p_attn.pid)

    try:
        _wait_http(f"http://127.0.0.1:{pref_p}/health", "Prefill", p_pref.pid)
        for i, pid in enumerate(ffn_pids):
            _wait_log(
                log / f"ffn{i}.log",
                r"Load weight end|AFD cuda_ipc FFN waiting|AfPool",
                pid,
                timeout=400,
            )
        _wait_http(f"http://127.0.0.1:{attn_p}/health", "Attn", p_attn.pid, 500)

        f_lb = open(log / "router.log", "w")
        p_lb = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                f"http://127.0.0.1:{pref_p}",
                "--decode",
                f"http://127.0.0.1:{attn_p}",
                "--host",
                "127.0.0.1",
                "--port",
                str(lb_p),
            ],
            stdout=f_lb,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        pids.append(p_lb.pid)
        _wait_http(f"http://127.0.0.1:{lb_p}/health", "Router", p_lb.pid)

        bench = _run_bench(
            base_url=f"http://127.0.0.1:{lb_p}",
            model=model,
            out_json=log / "bench.json",
            prompts=prompts,
            conc=conc,
            in_len=in_len,
            out_len=out_len,
        )
        return {
            "tag": tag,
            "gpus": 3,
            "layout": "1P+1A+2F(same)",
            "fake_v3_proxy": "ffn_compute_only",
            "ffn_mem_fraction": ffn_mem,
            "ffn_max_total_tokens": ffn_kv_tokens,
            "attn_mem_fraction": attn_mem,
            "median_tpot_ms": bench.get("median_tpot_ms"),
            "median_ttft_ms": bench.get("median_ttft_ms"),
            "output_throughput": bench.get("output_throughput"),
            "completed": bench.get("completed"),
        }
    finally:
        _kill_pids(pids)
        _fuser_ports([pref_p, attn_p, f0_p, f1_p, lb_p, bootstrap])
        time.sleep(3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/afd_v3_proxy_fair")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "/data/share/models/DeepSeek-V2-Lite-Chat"))
    ap.add_argument("--gpus", default=os.environ.get("FAIR_GPUS", "5,6,7"), help="3 GPUs: P,D/A,D/F")
    ap.add_argument("--ib-device", default=os.environ.get("IB_DEVICE", "mlx5_1"))
    ap.add_argument("--base-port", type=int, default=38400)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--in-len", type=int, default=256)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--max-bs", type=int, default=16)
    ap.add_argument(
        "--modes",
        default="pd_2d,pd_afd",
        help="Comma list: pd_2d,pd_afd,pd_afd_2a_same,pd_afd_a1f2_same",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        # Keep prior runs; write into fresh subdir only if requested clean — wipe contents.
        pass
    out.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if len(gpus) < 3:
        print("Need 3 GPUs", file=sys.stderr)
        return 2

    meta = v3_proxy_meta()
    (out / "v3_proxy_meta.json").write_text(json.dumps(meta, indent=2))
    print("=== V3 proxy meta ===", flush=True)
    print(json.dumps(meta, indent=2), flush=True)

    # Kill stale servers only (do not match this driver / awk).
    me = str(os.getpid())
    subprocess.call(
        [
            "bash",
            "-c",
            (
                "ps -eo pid,cmd | awk -v me=%s "
                "'/sglang\\.launch_server|sglang_router\\.launch_router/ "
                "&& !/awk/ && $1!=me {print $1}' | xargs -r kill -9 2>/dev/null || true"
            )
            % me,
        ]
    )
    time.sleep(2)

    results: List[dict] = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if "pd_2d" in modes:
        print("\n========== RUN pd_2d (1P+2D, V3-proxy on D) ==========", flush=True)
        ports = {
            "prefill": args.base_port,
            "d0": args.base_port + 1,
            "d1": args.base_port + 2,
            "lb": args.base_port + 10,
        }
        r = run_pd_2d(
            out=out,
            model=args.model,
            gpus=gpus,
            ports=ports,
            ib=args.ib_device,
            bootstrap=args.base_port + 50,
            max_bs=args.max_bs,
            prompts=args.prompts,
            conc=args.conc,
            in_len=args.in_len,
            out_len=args.out_len,
        )
        results.append(r)
        print(r, flush=True)

    if "pd_afd" in modes:
        print("\n========== RUN pd_afd (1P+1A+1F) ==========", flush=True)
        ports = {
            "prefill": args.base_port + 100,
            "attn": args.base_port + 101,
            "ffn": args.base_port + 102,
            "lb": args.base_port + 110,
        }
        r = run_pd_afd(
            out=out,
            model=args.model,
            gpus=gpus,
            ports=ports,
            ib=args.ib_device,
            bootstrap=args.base_port + 150,
            max_bs=args.max_bs,
            prompts=args.prompts,
            conc=args.conc,
            in_len=args.in_len,
            out_len=args.out_len,
        )
        results.append(r)
        print(r, flush=True)

    if "pd_afd_2a_same" in modes:
        print("\n========== RUN pd_afd_2a_same (1P+2A same GPU+1F) ==========", flush=True)
        ports = {
            "prefill": args.base_port + 200,
            "a0": args.base_port + 201,
            "a1": args.base_port + 202,
            "ffn": args.base_port + 203,
            "lb": args.base_port + 210,
        }
        r = run_pd_afd_2a_same(
            out=out,
            model=args.model,
            gpus=gpus,
            ports=ports,
            ib=args.ib_device,
            bootstrap=args.base_port + 250,
            max_bs=args.max_bs,
            prompts=args.prompts,
            conc=args.conc,
            in_len=args.in_len,
            out_len=args.out_len,
        )
        results.append(r)
        print(r, flush=True)

    if "pd_afd_a1f2_same" in modes:
        print("\n========== RUN pd_afd_a1f2_same (1P+1A+2F same GPU) ==========", flush=True)
        ports = {
            "prefill": args.base_port + 300,
            "attn": args.base_port + 301,
            "f0": args.base_port + 302,
            "f1": args.base_port + 303,
            "lb": args.base_port + 310,
        }
        r = run_pd_afd_a1f2_same(
            out=out,
            model=args.model,
            gpus=gpus,
            ports=ports,
            ib=args.ib_device,
            bootstrap=args.base_port + 350,
            max_bs=args.max_bs,
            prompts=args.prompts,
            conc=args.conc,
            in_len=args.in_len,
            out_len=args.out_len,
        )
        results.append(r)
        print(r, flush=True)

    # Merge prior rows from compare_merged.json if present (e.g. only re-running 2A).
    prior = out / "compare_merged.json"
    if prior.exists() and results:
        try:
            old = json.loads(prior.read_text())
            by_old = {r["tag"]: r for r in old.get("results", [])}
            for r in results:
                by_old[r["tag"]] = r
            # Keep preferred order.
            order = ["pd_2d", "pd_afd", "pd_afd_2a_same", "pd_afd_a1f2_same"]
            merged = [by_old[t] for t in order if t in by_old]
            for t, r in by_old.items():
                if t not in order:
                    merged.append(r)
            results = merged
        except Exception as e:
            print(f"warn: could not merge prior: {e}", flush=True)

    (out / "compare.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    (out / "compare_merged.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print("\n========== FAIR COMPARE (same 3 GPUs) ==========", flush=True)
    print(f"{'tag':<16} {'layout':<18} {'tpot_ms':>10} {'ttft_ms':>10} {'out_tps':>10}", flush=True)
    by: Dict[str, dict] = {}
    for r in results:
        by[r["tag"]] = r
        print(
            f"{r['tag']:<16} {r['layout']:<18} {r['median_tpot_ms']:10.2f} "
            f"{r.get('median_ttft_ms') or 0:10.2f} {r['output_throughput']:10.1f}",
            flush=True,
        )
    if "pd_2d" in by:
        base = by["pd_2d"]["output_throughput"] or 1.0
        for t in ("pd_afd", "pd_afd_2a_same", "pd_afd_a1f2_same"):
            if t in by:
                print(
                    f"{t} / pd_2d tok/s = {by[t]['output_throughput']/base:.3f}x",
                    flush=True,
                )
    print("V3_PROXY_FAIR_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
