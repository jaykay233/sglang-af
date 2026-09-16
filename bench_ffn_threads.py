#!/usr/bin/env python3
"""Per-thread CPU accounting inside the FFN process during live decode.

progress.md §15: process CPU exceeds the routed region's wall clock by ~1ms/hop
while the *calling thread* uses only ~0.45ms, so another thread in the process is
burning a core. This attributes it by name.
"""

import glob
import os
import sys
import time

OUT = "/tmp/afd_thr"
FFN_LOG = f"{OUT}/1a1f/ffn0.log"


def ffn_pids():
    """All pids holding ffn0.log open. The launcher parent plus the real worker
    both inherit the stdout redirect, so pick by thread count / CPU, not by fd
    ownership alone."""
    pids = set()
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == FFN_LOG:
                pids.add(int(fd.split("/")[2]))
        except OSError:
            continue
    return sorted(pids)


def thread_names(pid):
    names = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        tid = t.split("/")[-1]
        try:
            names[tid] = open(f"{t}/comm").read().strip()
        except OSError:
            names[tid] = "?"
    return names


def thread_ticks(pid):
    out = {}
    for t in glob.glob(f"/proc/{pid}/task/*"):
        tid = t.split("/")[-1]
        try:
            f = open(f"{t}/stat").read()
        except OSError:
            continue
        # comm may contain spaces/parens; split after the last ')'
        rest = f[f.rindex(")") + 2:].split()
        out[tid] = int(rest[11]) + int(rest[12])  # utime + stime
    return out


def sample_one(pid, names, secs):
    a = thread_ticks(pid)
    t0 = time.time()
    time.sleep(secs)
    t1 = time.time()
    b = thread_ticks(pid)
    dt = t1 - t0
    hz = os.sysconf("SC_CLK_TCK")
    rows = []
    for tid in sorted(set(a) & set(b)):
        d = b[tid] - a[tid]
        rows.append((d / hz / dt, tid, names.get(tid, "?")))
    return rows


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    pids = ffn_pids()
    if not pids:
        print("FFN pid not found (is the bench running?)")
        return
    print(f"ffn0.log held by pids={pids}")
    # Snapshot names, then measure each pid for a slice.
    targets = []
    for pid in pids:
        names = thread_names(pid)
        if len(names) > 1 or len(thread_ticks(pid)) > 1:
            targets.append((pid, names))
    if not targets:
        print("  (no multi-threaded holder; using all)")
        targets = [(p, thread_names(p)) for p in pids]

    per = max(1.0, secs / len(targets))
    for pid, names in targets:
        rows = sample_one(pid, names, per)
        total = sum(r[0] for r in rows)
        print(f"--- pid={pid} threads={len(rows)} total={total:.2f} cores ---")
        for cpu, tid, name in sorted(rows, reverse=True):
            if cpu > 0.002:
                print(f"  {cpu:6.2f} cores  tid={tid:>8}  {name}")


if __name__ == "__main__":
    main()
