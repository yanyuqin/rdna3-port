# rdna3-port, rebuilt on Strata v0.2.13.1 (2026-11-10)

This branch carries **piece 4** of `nicejunjie/Strata-rdna3` on top of the current upstream
release: KFD idle queue parking (`src/core/kfdq.cpp`) and the multi-shard `slim` in
`tools/nvfp4_local_pack.py`. Piece 1 (prefill kernels) is not ported, because upstream already
has it — see "Why piece 1 stays out" below.

## Base

| | commit | date |
|---|---|---|
| upstream release | `628f7495` (tag `v0.2.13.1`) | 2026-11-09 |
| the tree the first version of this branch used | `146df3c4` (tag `v0.2.12`) | 2026-11-07 |
| commits between them | **398** (816 files, +189 955 / −6 649) | |
| `main` past the tag | 81 commits (test/docs only at the time of writing) | |
| `nicejunjie/Strata-rdna3` | `7df6538d`, unchanged since 2026-11-07 | |

## What 0.2.13 changed that this port had to follow

1. **CMakeLists.txt was split.** The single 3 000-line file became
   `cmake/backends/{cpu,cuda,hip,metal,sycl}.cmake` + `cmake/strata_gpu_engine.cmake`. The
   `strata` executable's link line now lives in `cmake/strata_gpu_engine.cmake:480`, so the kfdq
   hook moved there with it — the same nine lines, now a single
   `if(STRATA_ENABLE_HIP AND NOT WIN32)` block. This is the only conflict in the rebase.
2. **ggml is vendored.** `third_party/strata-ggml` ships in the tree; `-DSTRATA_GGML_DIR` is an
   optional override now. `scripts/rdna3/build.sh` defaults to the vendored copy.
3. **`slim` merged cleanly.** Upstream's `cmd_slim` is still single-file; the multi-shard mode
   applied without conflict and keeps its behaviour: every shard keeps its own file name,
   `split.tensors.count` is rewritten to the new total in each shard, and a data shard that has
   no `general.architecture` key writes none either.
4. **No KFD code upstream.** `git grep -n AMDKFD v0.2.13.1 -- src include` is empty, so
   `kfdq.cpp` is still fork-only and still earns its place.
5. **No defaults for this card.** `arch_defaults.cpp` grew gfx12 tables (R9700) only; gfx1100
   and gfx906 still get an empty table, so the flags are still set by hand.
6. **Switches: 850 → 848 names.** Added `STRATA_ARENA_FILE_CACHE` (community PR #5531, only
   matters when RAM, not VRAM, is short). Removed `STRATA_DF2_GATE`, `STRATA_DF2_ROW_MS`,
   `STRATA_GR_Q32_PL`. All **29** switches our env file uses still exist in 0.2.13.1.

## Measurements, 2026-11-10 — paired in one session

RX 7900 XTX, ROCm `/opt/rocm715-merged`, `UD3.1-UD-IQ3_XXS` + `strata-pack-ud31-iq3xxs`,
262 144 context, `--kv int8`, dflash2 spec 4 + MTP, `taskset -c 8-15,24-31`, fresh nonce per
rep, 256 output tokens. Both sides ran with the same 29 flags (`~/port/env-up.sh`).

Conditions that make this pair usable: the production engine on 8093 was **stopped** for the run,
so each engine sat on the whole card and filled **7174 experts / 13.62 GiB** — identical slots on
both sides, which is what the earlier four-arm A/B could not have. The dual-MI50 engine on 8000
ran throughout, as it did in every earlier number.

Prefill, 3 reps, median:

| prompt tokens | `146df3c4` + 29 flags | `v0.2.13.1` + 29 flags | delta |
|---|---|---|---|
| 6 594 | 1323.0 t/s | 1332.5 t/s | +0.7% |
| 26 038 | 1610.8 t/s | 1612.0 t/s | +0.1% |
| 103 814 | 1625.0 t/s | 1615.3 t/s | −0.6% |

Decode, 5 reps at 26 K, median (range):

| build | prefill | decode |
|---|---|---|
| `146df3c4` + 29 flags | 1602.7 t/s | 83.5 t/s (75.5–86.0) |
| `v0.2.13.1` + 29 flags | 1605.1 t/s | 81.8 t/s (68.4–86.9) |

**Verdict: 0.2.13 is speed-neutral on gfx1100.** Prefill is inside ±0.7%, decode inside the MTP
acceptance band (±10% run to run is normal here). The 2× prefill this card gained on 11-08 came
from the flags, not from any of the 398 commits — and 0.2.13 adds no switch that would move it.

KFD parking was re-checked on the new build: 12 s after `ready`, the card sits at **11 W** (the
`STRATA_KFDQ=0` figure measured on 11-08 was 97 W).

## Why piece 1 stays out

`STRATA_HIP_WMMA`, `SELECT_WMMA`, `MMQ_KQUANTS`, `TOPK_SPLIT`, `QSA_ROWS`, `QFUSE`, `GDN_*` and
the `PA_*` attention paths are all upstream, in both `146df3c4` and `v0.2.13.1`. Turning them on
is what produced 1318 → 1614 t/s prefill; porting the fork's `src/prefill/` would duplicate code
upstream has since refactored around a compiled geometry, and it would not be faster.

The one fork-only speed item that remains is the Qwen3.8 hyper-connection decode read
(`src/kernels/cuda/rdna_gr.cu`, 753 lines, N=2560 / HC=4 hard-coded): worth about **+14% decode**
in the four-arm A/B, but it needs the fork's `q8_mixed` int8 copy pipeline across
`native_dense.cpp` / `layer.cpp` / `generate.cpp` — roughly 1 000 lines over six files, and it
only applies to Qwen3.8-Flash-Next. Not part of this branch.

## Reproduce

```sh
ROCM=/opt/rocm715-merged ./scripts/rdna3/build.sh          # vendored ggml, gfx1100, ~2 min
LLAMA_CPP=~/strata/third_party/llama.cpp ./scripts/rdna3/build.sh   # override, the pre-0.2.13 way

# engine on the test port, 29 flags, whole card (stop 8093 first)
sed 's|@STRATA@|'"$PWD"'|g' ../env-up.sh > /tmp/env-up-v0213.sh
bash ~/bench/run_ab2.sh ~/bench/cfg-v0213.json 8096 v0213 /tmp/env-up-v0213.sh
python3 ~/port/bench/bench40k.py --port 8096 --tokens 32768 --predict 256 --reps 5 --label x
```

## Upgrade cost, measured

Two conflicts in total: `CMakeLists.txt` (resolved by keeping the split layout and moving the
kfdq hook into `cmake/strata_gpu_engine.cmake`) and nothing else — `kfdq.cpp` applies as a clean
addition, `slim` auto-merges, the scripts are plain additions. A future upstream bump should cost
the same unless the CMake layout moves again.
