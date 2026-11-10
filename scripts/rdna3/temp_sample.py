#!/usr/bin/env python3
"""temp_sample.py -- sample the RX 7900 XTX thermals during a benchmark run.

Writes one CSV row every --interval seconds:
  epoch, iso, edge_c, junction_c, mem_c, power_w, busy_pct, fan_rpm, cpu_tctl_c, cpu_util_pct, mi50_a, mi50_b

card2 is Navi 31 (0x744c), card1/card3 are MI50s (0x66a0). Reads are plain sysfs,
so this never blocks even while the GPU is saturated (rocm-smi does block -- that is
why sysfs_watch.py exists for the gfx906 side).
"""
import argparse, os, time, datetime, glob

def rd(p):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return ""

def find_hwmon(card):
    g = glob.glob(f"/sys/class/drm/{card}/device/hwmon/hwmon*")
    return g[0] if g else None

def cpu_util(prev):
    line = rd("/proc/stat").splitlines()[0].split()[1:]
    v = [int(x) for x in line]
    idle = v[3] + v[4]
    total = sum(v)
    if prev is None:
        return 0.0, (total, idle)
    dt = total - prev[0]
    di = idle - prev[1]
    return (100.0 * (dt - di) / dt if dt else 0.0), (total, idle)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--interval", type=float, default=2.0)
ap.add_argument("--tag", default="run")
a = ap.parse_args()

h7900 = find_hwmon("card2")
hmi1 = find_hwmon("card1")
hmi2 = find_hwmon("card3")
cpu = None
for h in glob.glob("/sys/class/hwmon/hwmon*"):
    if rd(os.path.join(h, "name")) in ("k10temp", "zenpower"):
        cpu = h
        break

f = open(a.out, "w", buffering=1)
f.write("epoch,iso,edge_c,junction_c,mem_c,power_w,busy_pct,fan_rpm,cpu_tctl_c,cpu_util_pct,mi50_a,mi50_b\n")
prev = None
last_print = 0.0
while True:
    now = time.time()
    util, prev = cpu_util(prev)
    def t(idx):
        v = rd(f"{h7900}/temp{idx}_input") if h7900 else ""
        return round(int(v) / 1000.0, 1) if v else ""
    p = rd(f"{h7900}/power1_average") or rd(f"{h7900}/power1_input")
    pw = round(int(p) / 1e6, 1) if p else ""
    f.write(",".join(str(x) for x in [
        int(now),
        datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        t(1), t(2), t(3),
        pw,
        rd("/sys/class/drm/card2/device/gpu_busy_percent"),
        rd(f"{h7900}/fan1_input"),
        round(int(rd(f"{cpu}/temp1_input")) / 1000.0, 1) if cpu and rd(f"{cpu}/temp1_input") else "",
        round(util, 1),
        round(int(rd(f"{hmi1}/temp1_input")) / 1000.0, 1) if hmi1 and rd(f"{hmi1}/temp1_input") else "",
        round(int(rd(f"{hmi2}/temp1_input")) / 1000.0, 1) if hmi2 and rd(f"{hmi2}/temp1_input") else "",
    ]) + "\n")
    if now - last_print >= 60:
        print(f"[{a.tag}] {datetime.datetime.now():%H:%M:%S} edge={t(1)} junction={t(2)} power={pw}W cpu={util:.0f}%", flush=True)
        last_print = now
    time.sleep(a.interval)
