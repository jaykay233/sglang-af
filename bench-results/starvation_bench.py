#!/usr/bin/env python3
"""Starvation-stress benchmark for SGLang unified (non-PD) scheduling.

Phase A: start long-running victim decode streams.
Phase B: after victims get first token, flood short/medium prefills.
Measure victim ITL/TPOT during the flood (this is where decode starvation shows).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class StreamStats:
    role: str
    success: bool
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    itls_ms: List[float] = field(default_factory=list)
    e2e_ms: float = 0.0
    error: str = ""
    first_token_wall: float = 0.0
    end_wall: float = 0.0

    @property
    def tpot_ms(self) -> float:
        if len(self.itls_ms) == 0:
            return 0.0
        return statistics.mean(self.itls_ms)

    @property
    def p99_itl_ms(self) -> float:
        if not self.itls_ms:
            return 0.0
        xs = sorted(self.itls_ms)
        idx = min(len(xs) - 1, int(round(0.99 * (len(xs) - 1))))
        return xs[idx]


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[idx]


def summarize(name: str, rows: List[StreamStats]) -> Dict[str, Any]:
    ok = [r for r in rows if r.success and r.output_tokens > 0]
    itls = [x for r in ok for x in r.itls_ms]
    tpots = [r.tpot_ms for r in ok if r.itls_ms]
    ttfts = [r.ttft_ms for r in ok]
    return {
        "name": name,
        "n_ok": len(ok),
        "n_total": len(rows),
        "mean_ttft_ms": statistics.mean(ttfts) if ttfts else 0.0,
        "p99_ttft_ms": _pct(ttfts, 0.99) if ttfts else 0.0,
        "mean_tpot_ms": statistics.mean(tpots) if tpots else 0.0,
        "p99_tpot_ms": _pct(tpots, 0.99) if tpots else 0.0,
        "mean_itl_ms": statistics.mean(itls) if itls else 0.0,
        "p50_itl_ms": _pct(itls, 0.50) if itls else 0.0,
        "p99_itl_ms": _pct(itls, 0.99) if itls else 0.0,
        "max_itl_ms": max(itls) if itls else 0.0,
        "mean_output_tokens": statistics.mean([r.output_tokens for r in ok]) if ok else 0.0,
    }


async def stream_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    role: str,
    temperature: float = 0.0,
) -> StreamStats:
    stats = StreamStats(role=role, success=False)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url.rstrip('/')}/v1/completions"
    t0 = time.perf_counter()
    last = t0
    got_first = False
    text_chunks = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                stats.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return stats
            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                piece = choice.get("text") or ""
                usage = obj.get("usage") or {}
                if usage.get("prompt_tokens"):
                    stats.prompt_tokens = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    stats.output_tokens = usage["completion_tokens"]
                if not piece:
                    continue
                now = time.perf_counter()
                if not got_first:
                    stats.ttft_ms = (now - t0) * 1000
                    stats.first_token_wall = time.time()
                    got_first = True
                else:
                    stats.itls_ms.append((now - last) * 1000)
                last = now
                text_chunks += 1
                if stats.output_tokens == 0:
                    stats.output_tokens = text_chunks
        stats.success = got_first
        stats.e2e_ms = (time.perf_counter() - t0) * 1000
        stats.end_wall = time.time()
    except Exception as e:
        stats.error = str(e)
    return stats


def make_prompt(n_chars: int, seed: int) -> str:
    # Approximate token length via repeated ascii; tokenizer may differ slightly.
    unit = f"word{seed % 97} "
    return (unit * ((n_chars // len(unit)) + 1))[:n_chars]


async def run_starvation(args: argparse.Namespace) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    victim_prompt = make_prompt(args.victim_input_chars, 1)
    attacker_prompt = make_prompt(args.attacker_input_chars, 2)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup
        for _ in range(args.warmup):
            await stream_completion(
                session,
                args.base_url,
                args.model,
                make_prompt(64, 0),
                16,
                role="warmup",
            )

        # Flush cache if available
        try:
            async with session.post(f"{args.base_url.rstrip('/')}/flush_cache") as r:
                await r.text()
        except Exception:
            pass

        victim_tasks = [
            asyncio.create_task(
                stream_completion(
                    session,
                    args.base_url,
                    args.model,
                    victim_prompt,
                    args.victim_output_tokens,
                    role="victim",
                )
            )
            for _ in range(args.num_victims)
        ]

        # Wait until all victims have first token (or timeout)
        deadline = time.time() + args.victim_ttft_timeout_s
        while time.time() < deadline:
            dones = [t for t in victim_tasks if t.done()]
            # peek incomplete via running tasks count only; wait a bit
            await asyncio.sleep(0.05)
            # Check by querying partially - we can't peek StreamStats until done.
            # Instead: wait until first victim completes TTFT by polling shared event.
            break

        # Better: launch victims, poll with a wrapper
    # Re-implement with events for cleaner TTFT barrier
    return await _run_starvation_with_events(args)


async def _run_starvation_with_events(args: argparse.Namespace) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    victim_prompt = make_prompt(args.victim_input_chars, 1)
    attacker_prompt = make_prompt(args.attacker_input_chars, 2)

    first_token_events: List[asyncio.Event] = [
        asyncio.Event() for _ in range(args.num_victims)
    ]
    victims: List[StreamStats] = []
    attackers: List[StreamStats] = []
    flood_start = {"t": 0.0}
    flood_end = {"t": 0.0}

    async def victim_one(session: aiohttp.ClientSession, idx: int) -> StreamStats:
        stats = StreamStats(role="victim", success=False)
        payload = {
            "model": args.model,
            "prompt": victim_prompt,
            "max_tokens": args.victim_output_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
        url = f"{args.base_url.rstrip('/')}/v1/completions"
        t0 = time.perf_counter()
        last = t0
        got_first = False
        text_chunks = 0
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    stats.error = f"HTTP {resp.status}"
                    first_token_events[idx].set()
                    return stats
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    piece = (obj.get("choices") or [{}])[0].get("text") or ""
                    usage = obj.get("usage") or {}
                    if usage.get("completion_tokens"):
                        stats.output_tokens = usage["completion_tokens"]
                    if not piece:
                        continue
                    now = time.perf_counter()
                    if not got_first:
                        stats.ttft_ms = (now - t0) * 1000
                        stats.first_token_wall = time.time()
                        got_first = True
                        first_token_events[idx].set()
                    else:
                        # Only count ITLs during flood window if configured
                        itl = (now - last) * 1000
                        if flood_start["t"] <= time.time() <= flood_end["t"] or flood_end["t"] == 0:
                            # Before flood_end is set, still collecting; mark all post-first
                            # After flood starts, keep collecting until victim ends; we filter later.
                            stats.itls_ms.append(itl)
                        else:
                            stats.itls_ms.append(itl)
                    last = now
                    text_chunks += 1
                    if stats.output_tokens == 0:
                        stats.output_tokens = text_chunks
            stats.success = got_first
            stats.e2e_ms = (time.perf_counter() - t0) * 1000
            stats.end_wall = time.time()
        except Exception as e:
            stats.error = str(e)
            first_token_events[idx].set()
        return stats

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for _ in range(args.warmup):
            await stream_completion(
                session, args.base_url, args.model, make_prompt(64, 0), 16, "warmup"
            )
        try:
            async with session.post(f"{args.base_url.rstrip('/')}/flush_cache") as r:
                await r.text()
        except Exception:
            pass

        victim_tasks = [
            asyncio.create_task(victim_one(session, i)) for i in range(args.num_victims)
        ]

        # Wait for all victims' first tokens
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in first_token_events]),
            timeout=args.victim_ttft_timeout_s,
        )

        # Quiet baseline window (optional): measure ITL without attackers briefly
        quiet_itls: List[float] = []
        quiet_end = time.time() + args.quiet_seconds
        # We cannot easily split ITLs already appended; instead sleep quiet then start flood.
        await asyncio.sleep(args.quiet_seconds)

        flood_start["t"] = time.time()
        flood_end["t"] = flood_start["t"] + args.flood_seconds

        sem = asyncio.Semaphore(args.attacker_concurrency)

        async def attack_one() -> Optional[StreamStats]:
            async with sem:
                if time.time() > flood_end["t"]:
                    return None
                return await stream_completion(
                    session,
                    args.base_url,
                    args.model,
                    attacker_prompt,
                    args.attacker_output_tokens,
                    role="attacker",
                )

        attack_tasks = []
        while time.time() < flood_end["t"]:
            # Keep the attacker pool saturated
            attack_tasks = [t for t in attack_tasks if not t.done()]
            while len(attack_tasks) < args.attacker_concurrency and time.time() < flood_end["t"]:
                attack_tasks.append(asyncio.create_task(attack_one()))
            await asyncio.sleep(0.01)

        attack_results = await asyncio.gather(*attack_tasks)
        attackers = [r for r in attack_results if r is not None]

        victims = await asyncio.gather(*victim_tasks)

    # Split victim ITLs into quiet vs flood using wall timestamps is hard without per-itl stamps.
    # Re-run measurement approach: store timestamps. For this version, report overall victim ITL
    # after quiet+flood, which is dominated by flood under starvation.
    # Better: enrich victim_one to store (wall, itl) pairs.

    return {
        "config": {
            "label": args.label,
            "num_victims": args.num_victims,
            "victim_output_tokens": args.victim_output_tokens,
            "victim_input_chars": args.victim_input_chars,
            "attacker_output_tokens": args.attacker_output_tokens,
            "attacker_input_chars": args.attacker_input_chars,
            "attacker_concurrency": args.attacker_concurrency,
            "quiet_seconds": args.quiet_seconds,
            "flood_seconds": args.flood_seconds,
        },
        "victim": summarize("victim", victims),
        "attacker": summarize("attacker", attackers),
        "victims_raw": [asdict(v) for v in victims],
        "attackers_raw_n": len(attackers),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="/data/share/tmp-1/Qwen2.5-0.5B")
    p.add_argument("--label", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-victims", type=int, default=4)
    p.add_argument("--victim-output-tokens", type=int, default=1536)
    p.add_argument("--victim-input-chars", type=int, default=400)  # ~short/mid prompt
    p.add_argument("--attacker-output-tokens", type=int, default=32)
    p.add_argument("--attacker-input-chars", type=int, default=300)
    p.add_argument("--attacker-concurrency", type=int, default=48)
    p.add_argument("--quiet-seconds", type=float, default=1.0)
    p.add_argument("--flood-seconds", type=float, default=25.0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=600)
    p.add_argument("--victim-ttft-timeout-s", type=float, default=60)
    args = p.parse_args()

    # Improved version with per-ITL wall clock for flood filtering
    result = asyncio.run(run_starvation_v2(args))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"label": args.label, "victim": result["victim"], "attacker": result["attacker"]}, indent=2))


async def run_starvation_v2(args: argparse.Namespace) -> Dict[str, Any]:
    """Same as starvation, but each victim ITL is tagged with wall time for flood filtering."""
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    victim_prompt = make_prompt(args.victim_input_chars, 1)
    attacker_prompt = make_prompt(args.attacker_input_chars, 2)

    first_events = [asyncio.Event() for _ in range(args.num_victims)]
    flood_start = {"t": None}
    flood_end = {"t": None}

    async def victim_one(session: aiohttp.ClientSession, idx: int) -> Dict[str, Any]:
        out = {
            "success": False,
            "ttft_ms": 0.0,
            "itl_all_ms": [],
            "itl_flood_ms": [],
            "itl_quiet_ms": [],
            "output_tokens": 0,
            "error": "",
        }
        payload = {
            "model": args.model,
            "prompt": victim_prompt,
            "max_tokens": args.victim_output_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
        url = f"{args.base_url.rstrip('/')}/v1/completions"
        t0 = time.perf_counter()
        last = t0
        got_first = False
        chunks = 0
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    out["error"] = f"HTTP {resp.status}"
                    first_events[idx].set()
                    return out
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    piece = (obj.get("choices") or [{}])[0].get("text") or ""
                    usage = obj.get("usage") or {}
                    if usage.get("completion_tokens"):
                        out["output_tokens"] = usage["completion_tokens"]
                    if not piece:
                        continue
                    now = time.perf_counter()
                    wall = time.time()
                    if not got_first:
                        out["ttft_ms"] = (now - t0) * 1000
                        got_first = True
                        first_events[idx].set()
                    else:
                        itl = (now - last) * 1000
                        out["itl_all_ms"].append(itl)
                        fs, fe = flood_start["t"], flood_end["t"]
                        if fs is not None and fe is not None and fs <= wall <= fe:
                            out["itl_flood_ms"].append(itl)
                        elif fs is None or wall < fs:
                            out["itl_quiet_ms"].append(itl)
                    last = now
                    chunks += 1
                    if out["output_tokens"] == 0:
                        out["output_tokens"] = chunks
            out["success"] = got_first
        except Exception as e:
            out["error"] = str(e)
            first_events[idx].set()
        return out

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for _ in range(args.warmup):
            await stream_completion(
                session, args.base_url, args.model, make_prompt(64, 0), 16, "warmup"
            )
        try:
            async with session.post(f"{args.base_url.rstrip('/')}/flush_cache") as r:
                await r.text()
        except Exception:
            pass

        vtasks = [asyncio.create_task(victim_one(session, i)) for i in range(args.num_victims)]
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in first_events]),
            timeout=args.victim_ttft_timeout_s,
        )
        await asyncio.sleep(args.quiet_seconds)

        flood_start["t"] = time.time()
        flood_end["t"] = flood_start["t"] + args.flood_seconds
        sem = asyncio.Semaphore(args.attacker_concurrency)
        attackers: List[StreamStats] = []

        async def attack_one() -> Optional[StreamStats]:
            async with sem:
                if time.time() > flood_end["t"]:
                    return None
                return await stream_completion(
                    session,
                    args.base_url,
                    args.model,
                    attacker_prompt,
                    args.attacker_output_tokens,
                    "attacker",
                )

        atasks = []
        attackers: List[StreamStats] = []
        while time.time() < flood_end["t"]:
            still = []
            for t in atasks:
                if t.done():
                    r = t.result()
                    if r is not None:
                        attackers.append(r)
                else:
                    still.append(t)
            atasks = still
            while len(atasks) < args.attacker_concurrency and time.time() < flood_end["t"]:
                atasks.append(asyncio.create_task(attack_one()))
            await asyncio.sleep(0.01)
        ares = await asyncio.gather(*atasks)
        attackers.extend([x for x in ares if x is not None])
        victims = await asyncio.gather(*vtasks)

    def agg_itl(key: str) -> Dict[str, float]:
        xs = [x for v in victims if v.get("success") for x in v.get(key, [])]
        if not xs:
            return {"n": 0, "mean": 0, "p50": 0, "p99": 0, "max": 0}
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "p50": _pct(xs, 0.5),
            "p99": _pct(xs, 0.99),
            "max": max(xs),
        }

    victim_ttfts = [v["ttft_ms"] for v in victims if v.get("success")]
    return {
        "config": {
            "label": args.label,
            "num_victims": args.num_victims,
            "victim_output_tokens": args.victim_output_tokens,
            "attacker_output_tokens": args.attacker_output_tokens,
            "attacker_concurrency": args.attacker_concurrency,
            "quiet_seconds": args.quiet_seconds,
            "flood_seconds": args.flood_seconds,
        },
        "victim": {
            "n_ok": sum(1 for v in victims if v.get("success")),
            "mean_ttft_ms": statistics.mean(victim_ttfts) if victim_ttfts else 0,
            "itl_quiet": agg_itl("itl_quiet_ms"),
            "itl_flood": agg_itl("itl_flood_ms"),
            "itl_all": agg_itl("itl_all_ms"),
            "mean_output_tokens": statistics.mean(
                [v["output_tokens"] for v in victims if v.get("success")]
            )
            if any(v.get("success") for v in victims)
            else 0,
        },
        "attacker": summarize("attacker", attackers),
        "victims_detail": victims,
    }


if __name__ == "__main__":
    main()
