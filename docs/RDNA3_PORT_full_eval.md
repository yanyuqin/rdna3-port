# Full evaluation on the production instance (2026-11-08)

One continuous 82-minute run on the RX 7900 XTX, against the **production** engine on
port 8093 — the fork build `7df6538d` with its 46 `STRATA_*` variables (`prod-rdna3.env`),
UD3.1 IQ3_XXS + MTP/dflash2 spec4, 262144 context, expert cache 7388 slots / 14.03 GiB.
Thermals were sampled every 2 s from sysfs for the whole run.

Scope requested: `eval100` full, `polyglot100` full, the `qa` slice of RULER — score,
speed, and the temperature curve.

## Result

| suite | items | score | mean latency | p50 | throughput (end-to-end) |
|---|---|---|---|---|---|
| eval100 | 100 | MMLU **90.0%** (60) · DROP **F1 91.6% / EM 82.5%** (40) | 4.3 s | 3.2 s | **65.7 tok/s** |
| RULER qa | 12 | **75.0%** — qa_1 100% (6) · qa_2 50% (6) | 48.8 s | 48.5 s | 0.1 tok/s (65 K prompt dominates) |
| polyglot100 | 100 | **84.0%** pass@1 — python 94.1% (34) · javascript 100% (33) · go 57.6% (33) | 39.1 s | 30.1 s | **88.0 tok/s** |

Zero request errors in all three suites; 344 475 completion tokens and 3 912 s of
generation in polyglot alone, 28 063 tokens in eval100.

Throughput is `sum(completion_tokens) / sum(seconds)` — end-to-end, prefill included, so
it is a floor, not the decode rate measured in `RDNA3_PORT_measurements.md`.

## Against previous full runs on the same box

`eval100` (identical data file and protocol, thinking OFF, greedy):

| run | engine | MMLU | DROP F1 / EM | tok/s | p50 |
|---|---|---|---|---|---|
| `eval100_ud31iq3_full` | upstream-class build, **no** RDNA3 flags | 86.7% | 88.5% / 82.5% | 52.4 | 3.9 s |
| `eval100_7900IQ3DF2` | 7900 + dflash2, no flags | 86.7% | 92.5% / 85.0% | 56.8 | 3.1 s |
| **this run** | fork + 46 flags (production) | **90.0%** | 91.6% / 82.5% | **65.7** | 3.2 s |

`polyglot100` (thinking ON, effort medium, temp 1.0 — stochastic between runs):

| run | engine | pass@1 | go / js / python | tok/s |
|---|---|---|---|---|
| `polyglot100_7900IQ3DF2` | 7900 + dflash2, no flags | 85.0% | 66.7 / 100 / 88.2 | 76.8 |
| `polyglot100_STRATA_Q4` | dual MI50 UD-Q4_K_XL | 88.0% | 72.7 / 100 / 91.2 | 62.2 |
| **this run** | fork + 46 flags (production) | 84.0% | 57.6 / 100 / 94.1 | **88.0** |

`RULER qa` (12 items, 65 536-token prompts):

| run | qa score | p50 | 12-item total | fails |
|---|---|---|---|---|
| `ruler100_K256_7900` | 83.3% | 91.5 s | 1098 s | qa_1_21, qa_2_59 |
| `ruler100_Q4_256k` (dual MI50) | 83.3% | 91.5 s | 1100 s | qa_2_59, qa_2_34 |
| **this run** | 75.0% | **48.5 s** | **586 s** | qa_2_93, qa_2_59, qa_2_34 |

Read this carefully, because three different things are happening:

1. **Speed is a clear win, and it matches the prefill measurement.** RULER qa went from
   91.5 s to 48.5 s per 65 K item (1.89x), which is the same direction and roughly the same
   magnitude as the 68 K prefill row in `RDNA3_PORT_measurements.md` (725 → 1584.5 tok/s,
   2.18x). eval100 end-to-end throughput rose 52.4 → 65.7 tok/s (+25%) and polyglot
   76.8 → 88.0 (+15%). Short-output suites gain less because their time is decode-bound,
   and decode is what the flags barely touch.
2. **eval100 MMLU moved +2 items** (86.7% → 90.0%, i.e. 52 → 54 of 60). Greedy decoding
   makes this deterministic per build, so it is a real difference between builds, not
   sampling — but 2 items is exactly the drift already documented for kernel-level ports in
   `RDNA3_PORT_measurements.md` (the port and upstream each fixed a *different* MMLU item).
   Two items is not evidence that the flags make the model smarter; it is evidence that
   numerics changed slightly.
3. **polyglot and RULER-qa moved *down* by 1 item each** (85.0 → 84.0; 83.3 → 75.0 on
   12 items, i.e. 10 → 9). polyglot samples at temp 1.0, so ±2–3 items between runs is
   expected variance, and the whole delta sits in `go` (22 → 19 items) — the same `go`
   exercises that fail by compile error (`other declaration of ...`, `undefined: E`), i.e.
   model behaviour, not engine corruption. RULER qa is 12 items: one flipped item is 8.3
   points. `qa_2_59` fails in **all three** runs, so it is a stable property of the data,
   not this build.

Nothing in either direction reaches significance on 60/40/33/12-item samples. The honest
summary: **no regression that survives the noise, no gain beyond the ±1–2 item band.**

## Thermals

| phase | minutes | edge mean | junction mean | junction peak | power mean |
|---|---|---|---|---|---|
| eval100 | 7.1 | 51.0 °C | 68.1 °C | 77 °C | 266.5 W |
| RULER qa | 9.8 | 56.2 °C | 76.5 °C | **80 °C** | 293.6 W |
| polyglot | 65.2 | 57.2 °C | 75.9 °C | 80 °C | 287.1 W |
| whole run | 82.1 | 56.5 °C | 75.3 °C | 80 °C | 289.7 W |

- idle before the run: edge 23 °C, junction 26 °C, 13 W (KFDQ parking holding).
- peak junction **80 °C at 06:36**, during the first 65 K RULER prompts — the only phase
  where prefill saturates the card for a full minute without gaps.
- memory junction peaked at 78 °C; package power peaked at **395 W** transient against a
  302 W steady value, i.e. Navi 31 boost spikes, not a sustained cap.
- fan peaked at 1618 rpm (~30% of range) and the loop kept edge/junction flat from 10 min
  to the end; CPU Tctl peaked at 52.2 °C (mean 41.9 °C) with the dual-MI50 engine on 8000
  still running.
- **No throttle signature.** polyglot throughput by third: 75.1 → 84.0 → **92.7 tok/s**,
  i.e. it rose as the run went on (later items simply think longer, so the token/second
  ratio improves). Items generated at junction &lt; 75 °C averaged 85.5 tok/s, at ≥ 77 °C
  82.0 tok/s — a 4% difference fully inside the per-item variance. Caveat: clocks were not
  sampled, so "no throttle" is inferred from the temperature margin (80 °C junction vs a
  ~95 °C limit) plus non-decreasing throughput, not from `pp_dpm_sclk` counters.

The curve is in `rdna3_port_thermal_full1108.svg` (edge / junction / memory / CPU on °C,
power on the right axis, phase boundaries marked).

## Method

- engine: `~/bench/run_prod.sh 8093` — fork binary, `prod-rdna3.env`, `taskset -c 8-15,24-31`,
  `--host 0.0.0.0`, TLS proxy on 9443 untouched.
- samplers: `scripts/rdna3/temp_sample.py` (2 s interval, plain sysfs — `rocm-smi` blocks
  under load) and `scripts/rdna3/run_all.sh`, which runs the three suites back to back so
  one thermal trace covers all of them.
- scoring exactly as the kit defines it: eval100 = MMLU letter match + DROP token-F1;
  RULER = official string match (qa_1 all-hits, qa_2 any-hit); polyglot = official test
  exit code with the test files supplied in the prompt.
- the 8000 instance (dual MI50) ran throughout; its CPU and RAM traffic is inside all of
  these numbers, as it was for the earlier runs.

## Reproduce

```bash
cd ~/bench
./run_all.sh 8093 full1108        # writes ~/port/logs/full1108/{*.jsonl,temp.csv,phases.txt}
python3 summarize_run.py ~/port/logs/full1108
python3 thermal_drift.py ~/port/logs/full1108
```
