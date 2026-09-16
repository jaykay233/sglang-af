#!/usr/bin/env python3
"""Plot starvation ITL curves and ablation bar charts for interview evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/.cuda/sglang/bench-results")
ABL = ROOT / "ablation"
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

MODES = ["baseline", "mixed", "budget"]
LABELS = {
    "baseline": "Baseline (prefill-first)",
    "mixed": "Mixed chunk only",
    "budget": "Mixed + stall guard",
}
COLORS = {"baseline": "#c0392b", "mixed": "#d68910", "budget": "#1e8449"}


def starvation_path(mode: str) -> Path:
    heavy = ABL / f"starvation_{mode}_heavy.json"
    light = ABL / f"starvation_{mode}.json"
    return heavy if heavy.exists() else light


def load_starvation(mode: str) -> dict:
    with open(starvation_path(mode)) as f:
        return json.load(f)


def cdf(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) == 0:
        return xs, xs
    y = np.arange(1, len(xs) + 1) / len(xs)
    return xs, y


def plot_itl_cdf():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, phase, title in [
        (axes[0], "itl_quiet", "Quiet window (no prefill flood)"),
        (axes[1], "itl_flood", "Flood window (continuous prefill)"),
    ]:
        for mode in MODES:
            if not starvation_path(mode).exists():
                continue
            d = load_starvation(mode)
            xs = []
            for v in d.get("victims_detail", []):
                xs.extend(v.get(f"{phase}_ms", []))
            if not xs:
                continue
            x, y = cdf(xs)
            ax.plot(x, y, label=LABELS[mode], color=COLORS[mode], lw=2)
        ax.set_xlabel("ITL (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        if phase == "itl_flood":
            ax.set_xscale("log")
            ax.set_xlabel("ITL (ms, log)")
    axes[0].set_ylabel("CDF")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Victim decode ITL under continuous prefill pressure", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "starvation_itl_cdf.png", dpi=160)
    plt.close(fig)


def plot_flood_bars():
    means, p99s, maxs, modes_ok = [], [], [], []
    for mode in MODES:
        if not starvation_path(mode).exists():
            continue
        flood = load_starvation(mode)["victim"]["itl_flood"]
        means.append(flood["mean"])
        p99s.append(flood["p99"])
        maxs.append(flood["max"])
        modes_ok.append(mode)

    x = np.arange(len(modes_ok))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w, means, w, label="mean", color=[COLORS[m] for m in modes_ok], alpha=0.9)
    ax.bar(x, p99s, w, label="p99", color=[COLORS[m] for m in modes_ok], alpha=0.55, edgecolor="k")
    ax.bar(
        x + w,
        maxs,
        w,
        label="max",
        color=[COLORS[m] for m in modes_ok],
        alpha=0.3,
        edgecolor="k",
        hatch="//",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in modes_ok], rotation=8)
    ax.set_ylabel("Victim ITL during flood (ms)")
    ax.set_title("Ablation: decode starvation under prefill flood")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "starvation_flood_bars.png", dpi=160)
    plt.close(fig)


def plot_random_ttft_tradeoff():
    series = {
        "baseline": ROOT,
        "budget": Path("/root/.cuda/sglang/bench-results-budget"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, base in series.items():
        concs, ttfts, tpots = [], [], []
        for c in [1, 2, 4, 8, 16, 32, 64, 128]:
            p = base / f"result_c{c}.json"
            if not p.exists():
                continue
            j = json.load(open(p))
            if "mean_ttft_ms" not in j:
                continue
            concs.append(c)
            ttfts.append(j["mean_ttft_ms"])
            tpots.append(j["mean_tpot_ms"])
        if not concs:
            continue
        axes[0].plot(
            concs, ttfts, "o-", label=LABELS.get(mode, mode), color=COLORS.get(mode), lw=2
        )
        axes[1].plot(
            concs, tpots, "o-", label=LABELS.get(mode, mode), color=COLORS.get(mode), lw=2
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean TTFT (ms)")
    axes[1].set_ylabel("Mean TPOT (ms)")
    axes[0].set_title("Random load: TTFT cost of fairness")
    axes[1].set_title("Random load: TPOT")
    fig.tight_layout()
    fig.savefig(OUT / "random_ttft_tpot.png", dpi=160)
    plt.close(fig)


def write_summary_table():
    lines = [
        "# Starvation ablation numbers\n\n",
        "| Mode | file | quiet mean | flood mean | flood p99 | flood max |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ]
    for mode in MODES:
        p = starvation_path(mode)
        if not p.exists():
            continue
        d = load_starvation(mode)
        q, f = d["victim"]["itl_quiet"], d["victim"]["itl_flood"]
        lines.append(
            f"| {LABELS[mode]} | {p.name} | {q['mean']:.1f} | {f['mean']:.1f} | {f['p99']:.1f} | {f['max']:.1f} |\n"
        )
    text = "".join(lines)
    (OUT / "ABLATION_TABLE.md").write_text(text)
    print(text)


if __name__ == "__main__":
    plot_itl_cdf()
    plot_flood_bars()
    plot_random_ttft_tradeoff()
    write_summary_table()
    print("plots ->", OUT)
