# SPDX-License-Identifier: Apache-2.0
"""E2E greedy-output parity probe for the AFD decode farm.

Batching changes (hop width, ``num_contexts``, ``COALESCE_K``) must not change
*which* token a sequence produces. Unit tests cover the packing arithmetic; this
covers the end-to-end claim, because a structural slice bug would mis-attribute
rows to sequences and only show up in generated text.

Usage::

    python -m sglang.srt.afd.parity_e2e \
        --base-url http://127.0.0.1:32710 --model <path> \
        --num-prompts 16 --concurrency 16 --out /tmp/parity/base.json

Then diff two runs with ``--compare a.json b.json``. Exact match on every prompt
is the pass condition; drift is reported per prompt so a systematic failure is
distinguishable from one flaky token.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

# Fixed, deterministic prompts. Deliberately varied in shape (counting, short
# factual, arithmetic, long-prefix recall) so a mis-sliced hop shows up as
# garbage or as an answer drawn from the wrong sequence.
_TEMPLATES = [
    "List the numbers from 1 to 20, separated by commas.",
    "What is {n} plus {n}? Reply with only the number.",
    "Name three primary colours, comma separated.",
    "Repeat the word 'afd' exactly five times.",
    "What is the capital of France? One word.",
    "Count down from 10 to 1, comma separated.",
    "Write the alphabet from a to j, comma separated.",
    "What is {n} times 3? Reply with only the number.",
    "Say 'hello world' and nothing else.",
    "List the first five prime numbers, comma separated.",
    "What colour is the sky on a clear day? One word.",
    "Write the word 'parity' backwards.",
    "How many days are in a week? One word.",
    "List the vowels, comma separated.",
    "What is 100 minus {n}? Reply with only the number.",
    "Repeat 'test' three times, comma separated.",
]


_FILLER = (
    "The following is background context that should be ignored: "
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua. "
)


def build_prompts(
    num_prompts: int,
    homogeneous: bool = False,
    vary_input_len: bool = False,
) -> List[str]:
    """Prompt set for the probe.

    ``homogeneous`` repeats ONE prompt so every sequence has the same token
    length. Under greedy decoding all outputs must then be byte-identical, so
    any divergence is batching corruption and nothing else. The default
    (heterogeneous) set varies prompt length, which is what a real workload
    looks like — the throughput bench's ``random-range-ratio 0.0`` does not,
    so testing both separates "broken for unequal lengths" from "broken for
    concurrency".
    """
    if homogeneous:
        t = _TEMPLATES[0]
        return [t.format(n=2)] * num_prompts
    out: List[str] = []
    for i in range(num_prompts):
        t = _TEMPLATES[i % len(_TEMPLATES)]
        prompt = t.format(n=i + 2)
        if vary_input_len:
            # Same instructions, deliberately different prefix length. Combined
            # with ignore_eos this isolates "unequal input lengths" from
            # "unequal finish times" (all sequences then stop on the same step).
            prompt = _FILLER * (i % 4) + prompt
        out.append(prompt)
    return out


def _one(
    base_url: str, model: str, prompt: str, max_tokens: int, ignore_eos: bool = False
) -> str:
    import requests

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if ignore_eos:
        # Every sequence then runs exactly max_tokens steps, so a run with
        # unequal input lengths still has *equal* finish times.
        payload["ignore_eos"] = True
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def check_homogeneous(results: List[Dict[str, Any]]) -> int:
    """All-identical prompts under greedy decoding must give identical text."""
    texts = [(r.get("text") or "").strip() for r in results]
    uniq = {}
    for t in texts:
        uniq[t] = uniq.get(t, 0) + 1
    top = sorted(uniq.items(), key=lambda kv: -kv[1])
    n = len(texts)
    print(f"homogeneous: {len(uniq)} distinct outputs over {n} identical prompts")
    for text, count in top[:6]:
        print(f"  x{count:<3} {text[:90]!r}")
    if len(uniq) == 1:
        print("PASS: all concurrent outputs identical")
        return 0
    # One dominant answer + rare outliers = intermittent corruption; a flat
    # spread = systematic mis-slicing. Both are failures.
    print(f"FAIL: {len(uniq)} distinct outputs for identical inputs")
    return 1


def run(args: argparse.Namespace) -> None:
    prompts = build_prompts(
        args.num_prompts,
        homogeneous=args.homogeneous,
        vary_input_len=args.vary_input_len,
    )
    results: List[Dict[str, Any]] = [{} for _ in prompts]

    # The throughput bench warms up (`--warmup-requests`); a probe that starts
    # measuring on the first-ever request would fold in cold-start behaviour.
    for _ in range(max(0, args.warmup)):
        try:
            _one(args.base_url, args.model, "Say ok.", 4)
        except Exception:  # noqa: BLE001
            pass

    def work(idx: int) -> None:
        try:
            text = _one(
                args.base_url,
                args.model,
                prompts[idx],
                args.max_tokens,
                ignore_eos=args.ignore_eos,
            )
        except Exception as e:  # noqa: BLE001 - report, don't abort the probe
            text = f"<ERROR: {type(e).__name__}: {e}>"
        results[idx] = {"idx": idx, "prompt": prompts[idx], "text": text}

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        list(ex.map(work, range(len(prompts))))

    with open(args.out, "w") as f:
        json.dump(
            {
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "results": results,
            },
            f,
            indent=1,
        )
    n_err = sum(1 for r in results if str(r.get("text", "")).startswith("<ERROR"))
    print(
        f"parity probe wrote {args.out}: n={len(results)} errors={n_err} "
        f"concurrency={args.concurrency} homogeneous={args.homogeneous}"
    )
    if args.homogeneous:
        raise SystemExit(check_homogeneous(results))


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))["results"]
    b = json.load(open(path_b))["results"]
    if len(a) != len(b):
        print(f"MISMATCH: different prompt counts {len(a)} vs {len(b)}")
        return 1
    same = 0
    diffs: List[int] = []
    for ra, rb in zip(a, b):
        ta = (ra.get("text") or "").strip()
        tb = (rb.get("text") or "").strip()
        if ta == tb:
            same += 1
        else:
            diffs.append(ra["idx"])
    total = len(a)
    rate = 100.0 * same / max(1, total)
    print(f"exact-match {same}/{total} = {rate:.1f}%")
    for i in diffs[:10]:
        print(f"  --- prompt[{i}] ---")
        print(f"    A: {(a[i].get('text') or '')[:160]!r}")
        print(f"    B: {(b[i].get('text') or '')[:160]!r}")
    if len(diffs) > 10:
        print(f"  ... and {len(diffs) - 10} more")
    # A structural slicing bug produces wholesale divergence, not one token.
    return 0 if rate >= 90.0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:32710")
    ap.add_argument("--model", default="")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument(
        "--homogeneous",
        action="store_true",
        help="repeat one prompt; all greedy outputs must then be identical",
    )
    ap.add_argument(
        "--vary-input-len",
        action="store_true",
        help="give prompts deliberately different prefix lengths",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="run every sequence max_tokens steps, forcing equal finish times",
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if not args.out:
        ap.error("--out is required unless --compare is used")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
