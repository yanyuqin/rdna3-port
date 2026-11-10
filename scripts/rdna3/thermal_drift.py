#!/usr/bin/env python3
"""thermal_drift.py <outdir> -- did throughput fall as the card heated?

Splits polyglot100 into thirds and prints tok/s and p50 latency per third. A thermal
throttle shows up as a monotone drop in tok/s with a rise in per-token latency; a flat
profile rules it out. Also correlates per-item tok/s against the junction temperature
sampled at that item's end time.
"""
import json, os, sys, csv, statistics
d = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(d, "polyglot100.jsonl"))]
rows.sort(key=lambda r: r["sec"])  # keep file order; sec is per item, not a clock
# the jsonl is written in execution order, so file order == time order
for name, part in [("first 33", rows[:33]), ("middle 34", rows[33:67]), ("last 33", rows[67:])]:
    sec = sum(r["sec"] for r in part)
    tok = sum(r["ctok"] for r in part)
    print(f"{name}: tok/s {tok/sec:.1f}  ctok {tok}  mean {sec/len(part):.1f}s  "
          f"p50 {statistics.median([r['sec'] for r in part]):.1f}s  pass {int(sum(r['score'] for r in part))}/{len(part)}")

# map item end-time to nearest junction sample
tp = os.path.join(d, "temp.csv")
samples = [(int(r["epoch"]), float(r["junction_c"])) for r in csv.DictReader(open(tp)) if r["junction_c"]]
t0 = samples[0][0]
import bisect
eps = [s[0] for s in samples]
def j_at(ep):
    i = bisect.bisect_left(eps, ep)
    i = min(i, len(samples) - 1)
    return samples[i][1]
# reconstruct cumulative start: polyglot began at phase mark
start = None
for line in open(os.path.join(d, "phases.txt")):
    p = line.split()
    if p[-1].endswith("begin") and p[2] == "polyglot":
        start = int(p[0])
cur = start
pairs = []
for r in rows:
    cur += r["sec"]
    pairs.append((r["ctok"] / r["sec"], j_at(cur)))
lo = [v for v, t in pairs if t < 75]
hi = [v for v, t in pairs if t >= 77]
print(f"\nitems at junction <75C: n={len(lo)} tok/s mean {statistics.mean(lo):.1f}" if lo else "\nno low-temp items")
if hi:
    print(f"items at junction >=77C: n={len(hi)} tok/s mean {statistics.mean(hi):.1f}")
