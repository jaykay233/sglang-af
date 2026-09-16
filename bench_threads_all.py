#!/usr/bin/env python3
"""Per-thread CPU census across all sglang processes during live AFD decode.

progress.md §15: inside the FFN process, process CPU (1705us/hop) exceeds both
the routed region's wall (1497us) and the calling thread's CPU (681us), so a
second thread burns ~1 core. This finds it by name across both sides.
"""

import glob
import os
import sys
import time

PATTERN = sys.argv[2] if len(sys.argv) > 2 else "launch_server"


def read_names(pids):
    names = {}
    for pid in pids:
        for t in glob.glob(f"/proc/{pid}/task/*"):
            tid = t.split("/")[-1]
            try:
                names[tid] = open(f"{t}/comm").read().strip()
            except OSError:
                pass
    return names


def ticks(pids):
    out = {}
    for pid in pids:
        for t in glob.glob(f"/proc/{pid}/task/*"):
            tid = t.split("/")[-1]
            try:
                f = open(f"{t}/stat").read()
            except OSError:
                continue
            rest = f[f.rindex(")") + 2:].split()
            out[tid] = int(rest[11]) + int(rest[12])
    return out


def pids_for(needle):
    found = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        pid = int(p.split("/")[2])
        if pid == os.getpid():
            continue
        try:
            cmd = open(p, "rb").read().decode(errors="replace").replace("\0", " ")
        except OSError:
            continue
        if needle in cmd:
            found.append(pid)
    return found


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    pids = pids_for(PATTERN)
    if not pids:
        print(f"no processes matching {PATTERN!r}")
        return
    names = read_names(pids)
    a = ticks(pids)
    t0 = time.time()
    time.sleep(secs)
    b = ticks(pids)
    dt = time.time() - t0
    hz = os.sysconf("SC_CLK_TCK")
    rows = []
    for tid in set(a) & set(b):
        d = b[tid] - a[tid]
        rows.append((d / hz / dt, tid, names.get(tid, "?")))
    rows.sort(reverse=True)
    print(f"{len(pids)} procs, {len(rows)} threads, {dt:.1f}s")
    shown = 0
    for cpu, tid, name in rows:
        if cpu < 0.05:
            break
        print(f"  {cpu:6.2f} cores  tid={tid:>8}  {name}")
        shown += 1
    print(f"  busiest {shown}; total {sum(r[0] for r in rows):.2f} cores")


if __name__ == "__main__":
    main()
