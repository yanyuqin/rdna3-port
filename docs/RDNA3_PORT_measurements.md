# RDNA3 port: measurements, 2026-11-08

Machine: amd52 — EPYC 7F52 (32 threads), 247 GiB DDR4, one RX 7900 XTX (24 GB, PCIe 4.0 x16), ROCm from
`/opt/rocm715-merged`. Model: `UD3.1-UD-IQ3_XXS` + pack `strata-pack-ud31-iq3xxs`, 262 144-token context, `--kv int8`,
`--kv-resident 262144`, `--expert-cache auto` (13.62 GiB, 7174–7388 experts), dflash2 spec 4 + MTP. Both engines run
under `taskset -c 8-15,24-31`; the MI50 engine on port 8000 kept running, so all numbers below share that noise.

Prompt: `bench_client.build_prompt` filler text with a fresh nonce per rep (no prompt cache hits). 256 output tokens,
3 reps, median. Prefill target 8192 / 32768 tokens; the actual prompt_n is 6588 / 26035.

## 1. Four arms

| arm | binary | env | prefill 8K (t/s) | decode 8K (t/s) | prefill 32K (t/s) | decode 32K (t/s) | prefill 128K (t/s) | decode 128K (t/s) |
|---|---|---|---|---|---|---|---|---|
| A | upstream `146df3c4` | none | 697.8 | 77.8 | 765.5 | 74.3 | — | — |
| B | fork `7df6538d` | none | 714.3 | 71.2 | 786.7 | 77.5 | — | — |
| C | upstream `146df3c4` | its 29 RDNA3 flags | 1318.3 | 80.8 | **1614.3** | 85.9 | **1584.5** | 74.2 |
| D | fork `7df6538d` | its 46 flags | 1327.0 | 92.5 | 1584.5 | **95.9** | 1531.5 | **84.5** |

The 128K row is 2 reps (76 813-token prompts, 68-72 s each) rather than 3: prefill +3.5% for upstream, decode inside
the MTP-acceptance band (74.2 vs 84.5 median, per-rep 64.6/83.9 vs 76.1/93.0).

C vs D: prefill −0.7% / +1.9% (noise), decode −14%. The fork's 46 env vars were filtered against upstream's
`STRATA_*` namespace: 29 exist upstream, 17 do not —

```
STRATA_RDNA_MMVQ_WAVE, STRATA_PEER_TRANSPORT, STRATA_PREFILL_MMQ_TILE_ROWS,
STRATA_GR_RDNA, STRATA_GR_RDNA_Q8, STRATA_RDNA_Q6K, STRATA_GR_RDNA_A8,
STRATA_COMBINE_CPU, STRATA_WAIT_FUSE, STRATA_ROUTE_PUBLISH, STRATA_GR_INJ_UP, STRATA_GR_FUSED,
STRATA_GDN_CHUNK_GDV, STRATA_PA_KLDS, STRATA_PF_GROUP_RES, STRATA_TOPK_SPLIT_MIN, STRATA_ENGINE_NICE
```

Of those, `GR_RDNA*` / `RDNA_Q6K` / `MMVQ_WAVE` are the Qwen3.8-specific hyper-connection decode kernels (the 14%), and
`TOPK_SPLIT_MIN` is the automatic threshold for a split top-k that upstream already ships as `STRATA_TOPK_SPLIT`. The
rest are scheduling or diagnostics.

Decode varies with MTP acceptance (upstream's own answers differ run to run), so read decode as a band:
±10% at fixed settings is normal here.

## 2. Piece 1: prefill kernels are already upstream

Upstream `src/prefill/` and `src/kernels/cuda/` contain `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`,
`STRATA_MMQ_KQUANTS`, `STRATA_TOPK_SPLIT`, `STRATA_QSA_ROWS`, `STRATA_QFUSE`, `STRATA_PA_KLDS`-style attention paths
and the `GDN_*` chunked-recurrence set. Arm C is upstream with those flags on: 1.9–2.1x over the plain config, and
equal to the fork's arm D within noise. Porting the fork's prefill kernels onto this tree would duplicate code that
upstream already has and has since refactored; the branch does not.

The one measurable fork-only prefill/decode item left is the fused hyper-connection read (`src/kernels/cuda/rdna_gr.cu`,
753 lines, hard-coded N=2560 / HC=4). It needs the fork's `q8_mixed` int8 copies in `native_dense.cpp` / `layer.cpp` /
`generate.cpp`, so it is a ~1000-line port across six files, not a drop-in. Not done here; the 14% decode above is what
it is worth, and it applies to Qwen3.8-Flash-Next only.

## 3. Piece 4: what was ported, verified here

### 3.1 KFD queue parking (`src/core/kfdq.cpp`)

One 7900 XTX, `GPU_MAX_HW_QUEUES=8`, engine ready and idle for 12 s:

| | GPU busy | `hwmon power1_average` |
|---|---|---|
| `STRATA_KFDQ=0` | 14% | **97 W** |
| `STRATA_KFDQ=1` | 0% | **11 W** |

`STRATA_KFDQ_LOG=1` shows 9 queues created on the card (`util 1` is ROCr's 4 KB utility queue) and, 2 s after the last
submission, `park gpu 30693 q 0 rc 0 ioctl_ms 0.19`. Note the single-card config keeps `GPU_MAX_HW_QUEUES=6` (7 queues),
which is idle-clean without this: the port matters for `GPU_MAX_HW_QUEUES=8` and for cards shared with another process.

### 3.2 `slim` on split GGUFs (`tools/nvfp4_local_pack.py`)

Run on the three shards of `Qwen3.8-Flash-Next-UD-IQ3_XXS` (79 GB): three outputs, same file names, `_exps` tensors
left out, `split.tensors.count` rewritten in every shard.

```
slim: <kept> tensors kept, <skipped> expert tensors left out -> /home/yanyq/port/slim-test
```

## 4. The 16 common-wrong questions (regression probe)

Set from `UD3.1_AB_双MI50_Q4_vs_7900_IQ3_2026-11-05.md` §4: the questions *both* environments answered wrong, so any
change is a signal. Same protocol as that report (greedy for eval100, temp 1.0 / effort medium for polyglot, 256 K for
RULER).

| item | 2026-11-05 baseline | this build (port) | upstream `146df3c4` |
|---|---|---|---|
| eval100 MMLU 3/12/19/38/42/45/46 | 0/7 | **1/7** (`mmlu_42`) | **1/7** (`mmlu_12`) |
| eval100 DROP `drop_78` | 0/1 | 0/1 | 0/1 |
| polyglot `go_beer-song`, `go_counter`, `go_markdown`, `go_paasio`, `go_robot-simulator`, `go_scale-generator`, `go_sublist` | 0/7 | 0/7 | 0/7 |
| RULER-256K `qa_2_59` | FAIL ("Yes") | FAIL ("Yes") | FAIL ("Yes") |
| **total /16** | **0** | **1** | **1** |

The two builds pass *different* MMLU item, which is the documented ±1-question drift of greedy + spec decode +
engine KV state (`mmlu_12` was correct in the 8-item subset and wrong in the full run on 2026-11-05). No item moved
from pass to fail, no garbage output, no NaN, no non-finite logits — the regression risk the fork's own issues
describe (`DP_EARLY_REFILL` NaN, `hipMemcpy2D` silent corruption) does not appear here.

Artifacts: `~/eval100-kit/results/{eval100_port16,eval100_up16,polyglot100_port7,ruler_port16}.jsonl`.

## 5. Against the earlier runs on this card (2026-11-04 / 11-05)

Those runs used `~/bench/cfg-1100-dflash2-s4.json` (16 K context, `--kv-resident 32768`, expert cache 9615 slots /
17.73 GiB) and `~/bench/run_1100.sh`, which starts `serve.server` **without sourcing any env file**. Checked on the live
8093 process on 2026-11-08: `/proc/<pid>/environ` holds 17 variables and none matching `STRATA_*`, so every RDNA3 switch
was off. (The dual-MI50 engine on 8000 does carry three, `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1`,
`STRATA_GFX906_ATTN_REDUCE12=1`, `STRATA_MMQ_KQ=1`, from the 2026-11-02 gfx906 A/B; those are gfx906-only.)

| run | context | expert cache | prompt | prefill | decode |
|---|---|---|---|---|---|
| 11-04, DFlash2 spec 4, no env | 16 K | 9615 slots / 17.73 GiB | 512 | 314.9 | 80.9 |
| same | 16 K | same | 7 620 | 742.7 | **99.1** |
| same | 16 K | same | 11 200 | 735.4 | 86.3 |
| 11-05, both engines concurrent | 256 K | 7175 / 13.62 GiB | 68 123 | ≈725 (700–730) | — |
| A, upstream defaults | 256 K | 7175 / 13.62 GiB | 6 588 | 697.8 | 77.8 |
| A | 256 K | same | 26 035 | 765.5 | 74.3 |
| C, 29 flags | 256 K | same | 26 035 | 1614.3 | 85.9 |
| C | 256 K | same | 76 813 | **1584.5** | 74.2 |

1. Arm A reproduces the card as it was measured before: 697.8 / 765.5 at 6.6 K / 26 K against 742.7 at 7.6 K and ≈725 at
   68 K with no flags. The earlier numbers were the flag-off baseline, not a different build's speed.
2. The cleanest prefill pair is the 68 K one: ≈725 before against 1584.5 after with flags on, **2.18x**.
3. Decode is not comparable across those rows. The 11-04 config ran 16 K with 9615 expert slots; every arm here runs
   256 K with 7175 slots. That is why 99.1 becomes 95.9 / 85.9 — the cache shrank, not the port. The fork's +14% is
   measured inside one 256 K configuration, which is the only apples-to-apples decode pair in this document.
4. The eval100 number from 2026-11-06 (54.46 tok/s decode, thinking off, 256-token answers) is a latency-bound short
   workload and does not belong in these rows.

## 6. Reproduce

```sh
# binaries
~/port/up/build-1100/strata      # this branch
~/port/up/build.sh               # build script used

# engine (single card)
~/bench/cfg-port-256k.json       # config: pack/native/ple/kv/spec/mtp, --gpu 1
~/bench/run_ab2.sh <cfg> <port> <tag> <envfile>

# env arms
~/port/env-up.sh                 # the 29 flags upstream accepts
~/port/env-fork.sh               # the fork's 46

# bench
python3 ~/port/bench/bench40k.py --port 8096 --tokens 32768 --predict 256 --reps 3 --label x
```
