#!/usr/bin/env python3
"""Table of median FillBoundary time (ms) per variant and size from summary.txt."""

import statistics
import sys
from collections import defaultdict

times = defaultdict(list)
checkfail = defaultdict(int)
failed = []
labels, sizes = [], []

with open(sys.argv[1]) as f:
    for line in f:
        fields = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        if line.startswith("FAILED"):
            failed.append(line.strip())
            continue
        if not line.startswith("RESULT"):
            continue
        lab, size = fields["label"], fields["size"]
        if lab not in labels:
            labels.append(lab)
        if size not in sizes:
            sizes.append(size)
        times[lab, size].append(float(fields["ms_med"]))
        if fields.get("check", "ok") == "FAIL":
            checkfail[lab] += 1

print("Median of per-run medians, ms per FillBoundary (lower is better)")
print(f"{'variant':16s}" + "".join(f"{s:>12s}" for s in sizes) + f"{'check':>8s}")
for lab in labels:
    row = f"{lab:16s}"
    for s in sizes:
        v = times.get((lab, s))
        row += f"{statistics.median(v):12.4f}" if v else f"{'-':>12s}"
    chk = f"FAIL({checkfail[lab]})" if checkfail[lab] else "ok"
    print(row + f"{chk:>8s}")
for line in failed:
    print(line)
