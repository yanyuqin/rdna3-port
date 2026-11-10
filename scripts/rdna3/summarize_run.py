#!/usr/bin/env python3
"""summarize_run.py <outdir> -- scores, speeds and thermal stats for a run_all.sh run.

Reads eval100.jsonl / ruler_qa.jsonl / polyglot100.jsonl / temp.csv / phases.txt and
writes summary.json + summary.md into the same dir. Speed is reported two ways:
  decode tok/s = sum(ctok)/sum(sec)  (end-to-end, prefill included -> conservative)
  p50/p90 latency
Thermal stats are split per phase using the epoch markers in phases.txt, because a
mean over the whole run hides the polyglot ramp.
"""
import json, os, sys, csv, statistics

d = sys.argv[1]
J = lambda n: [json.loads(l) for l in open(os.path.join(d, n))] if os.path.exists(os.path.join(d, n)) else []

def speed(rows):
    ok = [r for r in rows if "error" not in r]
    sec = sum(r.get("sec", 0) for r in ok)
    tok = sum(r.get("ctok", 0) or 0 for r in ok)
    lat = [r.get("sec", 0) for r in ok]
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "ctok": tok, "sec": round(sec, 1),
        "tok_s": round(tok / sec, 1) if sec else 0,
        "lat_mean": round(sum(lat) / len(lat), 1) if lat else 0,
        "lat_p50": round(statistics.median(lat), 1) if lat else 0,
        "lat_p90": round(sorted(lat)[int(len(lat) * 0.9)], 1) if lat else 0,
    }

res = {}
e = J("eval100.jsonl")
if e:
    mm = [r for r in e if r["kind"] == "mmlu" and "error" not in r]
    dp = [r for r in e if r["kind"] == "drop" and "error" not in r]
    res["eval100"] = dict(speed(e), total=len(e),
        mmlu_n=len(mm), mmlu_acc=round(100 * sum(r["score"] for r in mm) / max(1, len(mm)), 1),
        drop_n=len(dp), drop_f1=round(100 * sum(r["score"] for r in dp) / max(1, len(dp)), 1),
        drop_em=round(100 * sum(r.get("em", 0) for r in dp) / max(1, len(dp)), 1))
r = J("ruler_qa.jsonl")
if r:
    res["ruler_qa"] = dict(speed(r), pass_n=sum(1 for x in r if x.get("score")),
                           pass_pct=round(100 * sum(x.get("score", 0) for x in r) / max(1, len(r)), 1))
p = J("polyglot100.jsonl")
if p:
    ok = [x for x in p if "error" not in x]
    by = {}
    for x in ok:
        by.setdefault(x["lang"], []).append(x["score"])
    res["polyglot100"] = dict(speed(p), pass_pct=round(100 * sum(x["score"] for x in ok) / max(1, len(ok)), 1),
        by_lang={k: [len(v), round(100 * sum(v) / len(v), 1)] for k, v in sorted(by.items())})

# thermals
tp = os.path.join(d, "temp.csv")
temps = []
if os.path.exists(tp):
    for row in csv.DictReader(open(tp)):
        try:
            temps.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()
                          if k not in ("iso",)})
        except ValueError:
            pass
if temps:
    def stat(field, f):
        vals = [t[field] for t in temps if t.get(field) is not None]
        return round(f(vals), 1) if vals else None
    res["thermal"] = {
        "rows": len(temps),
        "span_min": round((temps[-1]["epoch"] - temps[0]["epoch"]) / 60, 1),
        "edge_min": stat("edge_c", min), "edge_max": stat("edge_c", max),
        "edge_mean": stat("edge_c", lambda v: sum(v) / len(v)),
        "junction_max": stat("junction_c", max),
        "junction_mean": stat("junction_c", lambda v: sum(v) / len(v)),
        "mem_max": stat("mem_c", max),
        "power_max": stat("power_w", max),
        "power_mean_loaded": stat("power_w", lambda v: sum(x for x in v if x > 100) / max(1, sum(1 for x in v if x > 100)) or 0),
        "cpu_max": stat("cpu_tctl_c", max),
        "cpu_mean": stat("cpu_tctl_c", lambda v: sum(v) / len(v)),
        "busy_loaded_pct": round(100 * sum(1 for t in temps if (t.get("busy_pct") or 0) > 50) / len(temps), 1),
    }
    # per-phase
    phases = []
    if os.path.exists(os.path.join(d, "phases.txt")):
        for line in open(os.path.join(d, "phases.txt")):
            parts = line.split()
            if len(parts) >= 3:
                phases.append((int(parts[0]), parts[1], " ".join(parts[2:])))
    bounds = [(t, lbl) for t, _, lbl in phases if lbl.endswith("begin")]
    per = {}
    for t0, lbl in bounds:
        name = lbl.split()[0]
        t1 = min([t for t, _, l in phases if l.split()[0] == name and l.endswith("end")] + [10**12])
        seg = [t for t in temps if t0 <= t["epoch"] <= t1]
        if seg:
            jv = [t["junction_c"] for t in seg if t.get("junction_c") is not None]
            ev = [t["edge_c"] for t in seg if t.get("edge_c") is not None]
            pw = [t["power_w"] for t in seg if t.get("power_w") is not None]
            per[name] = {"min": round((t1 - t0) / 60, 1), "rows": len(seg),
                         "edge_mean": round(sum(ev) / len(ev), 1) if ev else None,
                         "junction_mean": round(sum(jv) / len(jv), 1) if jv else None,
                         "junction_max": max(jv) if jv else None,
                         "power_mean": round(sum(pw) / len(pw), 1) if pw else None}
    if per:
        res["phases"] = per

json.dump(res, open(os.path.join(d, "summary.json"), "w"), indent=1, ensure_ascii=False)

# downsampled thermal series for plotting (every 15 s, plus the exact peak rows)
if temps:
    s = [t for i, t in enumerate(temps) if i % 8 == 0]
    ts0 = temps[0]["epoch"]
    series = {"t0": ts0, "dt_s": 16,
              "edge": [[round(t["epoch"] - ts0), t.get("edge_c")] for t in s],
              "junction": [[round(t["epoch"] - ts0), t.get("junction_c")] for t in s],
              "mem": [[round(t["epoch"] - ts0), t.get("mem_c")] for t in s],
              "power": [[round(t["epoch"] - ts0), t.get("power_w")] for t in s],
              "cpu": [[round(t["epoch"] - ts0), t.get("cpu_tctl_c")] for t in s]}
    json.dump(series, open(os.path.join(d, "series.json"), "w"))
print(json.dumps(res, indent=1, ensure_ascii=False))
