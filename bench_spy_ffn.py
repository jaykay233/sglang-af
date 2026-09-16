#!/usr/bin/env python3
"""Sample py-spy stacks of the FFN process during a live AFD decode.

progress.md §14.3: the FFN process is GIL-bound at ~1 core and
``ffn_routed_cpu_us`` exceeds the region's wall clock, so another thread in the
same process is competing. This finds which one.
"""

import glob
import os
import re
import subprocess
import sys
import time

OUT = "/tmp/afd_spy"
ENDPOINTS = f"{OUT}/1a1f_endpoints"


def ffn_pid():
    """The process that owns ffn0.log is the FFN process (compute + poll loop)."""
    needle = f"{OUT}/1a1f/ffn0.log"
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == needle:
                return int(fd.split("/")[2])
        except OSError:
            continue
    return None


def children(pid):
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        ).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def dump(pid, tag):
    try:
        r = subprocess.run(
            ["py-spy", "dump", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        txt = r.stdout + r.stderr
    except Exception as e:
        txt = f"dump failed: {e}"
    # Keep only thread headers + the python frames, drop C-lib noise.
    lines = [
        l
        for l in txt.splitlines()
        if re.search(r"Thread \d+ \(|^\s+\w+ \(|^\s+<module>|Python v", l)
    ]
    with open(f"{OUT}/stacks.txt", "a") as f:
        f.write(f"\n===== {tag} pid={pid} t={time.time():.1f} =====\n")
        f.write("\n".join(lines) + "\n")


def main():
    open(f"{OUT}/stacks.txt", "w").close()
    t_end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 90)
    seen = set()
    while time.time() < t_end:
        launcher = ffn_pid()
        if launcher:
            if launcher not in seen:
                print(f"FFN launcher pid={launcher}", flush=True)
                seen.add(launcher)
            dump(launcher, "FFN-launcher")
            for c in children(launcher):
                dump(c, "FFN-child")
        time.sleep(1.5)
    print("SAMPLING_DONE", flush=True)


if __name__ == "__main__":
    main()
