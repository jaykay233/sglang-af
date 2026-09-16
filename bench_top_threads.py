#!/usr/bin/env python3
"""Top CPU-consuming threads, found by *thread name* (not cmdline).

The AFD workers are multiprocessing ``spawn_main`` children whose cmdline does
not contain ``launch_server``, so filtering by cmdline misses them (progress.md
§15).
"""

import glob
import os
import sys
import time

MATCH = sys.argv[2] if len(sys.argv) > 2 else ""


def all_threads():
    """tid -> (pid, comm) for every thread of every process we can read."""
    out = {}
    for t in glob.glob("/proc/[0-9]*/task/*"):
        tid = t.split("/")[-1]
        pid = t.split("/")[2]
        try:
            comm = open(f"{t}/comm").read().strip()
        except OSError:
            continue
        out[tid] = (pid, comm)
    return out


def ticks():
    out = {}
    for t in glob.glob("/proc/[0-9]*/task/*"):
        tid = t.split("/")[-1]
        try:
            f = open(f"{t}/stat").read()
        except OSError:
            continue
        rest = f[f.rindex(")") + 2:].split()
        out[tid] = int(rest[11]) + int(rest[12])
    return out


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    meta = all_threads()
    a = ticks()
    t0 = time.time()
    time.sleep(secs)
    b = ticks()
    dt = time.time() - t0
    hz = os.sysconf("SC_CLK_TCK")
    rows = []
    for tid in set(a) & set(b):
        d = b[tid] - a[tid]
        if d <= 0:
            continue
        pid, comm = meta.get(tid, ("?", "?"))
        if MATCH and MATCH not in comm and MATCH not in pid:
            continue
        rows.append((d / hz / dt, tid, pid, comm))
    rows.sort(reverse=True)
    print(f"{dt:.1f}s, {len(rows)} busy threads (filter={MATCH!r})")
    for cpu, tid, pid, comm in rows[:20]:
        print(f"  {cpu:6.2f} cores  pid={pid:>8} tid={tid:>8}  {comm}")
    print(f"  total busy {sum(r[0] for r in rows):.2f} cores")


if __name__ == "__main__":
    main()
