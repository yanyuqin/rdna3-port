# rdna3-port

The RDNA3 (gfx1100) pieces of [`nicejunjie/Strata-rdna3`](https://github.com/nicejunjie/Strata-rdna3) that are
**not** in [`Niko1221/Strata`](https://github.com/Niko1221/Strata) main, ported onto upstream main and measured on a
single RX 7900 XTX.

- Base: upstream `146df3c4` (2026-11-07), branch `rdna3-port`.
- Ported from: `Strata-rdna3` `e2151f1b` (`src/core/kfdq.cpp`), `7df6538d` (`tools/nvfp4_local_pack.py slim`).
- License: MIT, same as both parents. Every ported file keeps its origin in a header comment.

## Why this branch is small

The plan for the port was "the prefill kernels (piece 1) + the small operational items (piece 4)". Measurement changed
the scope of piece 1: **upstream main already contains the prefill kernels**. `STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`,
`STRATA_MMQ_KQUANTS`, `STRATA_TOPK_SPLIT`, `STRATA_QSAROWS`, `STRATA_QFUSE` and the `GDN_*` chunked-recurrence set are
all in upstream `src/prefill/` and `src/kernels/cuda/`; what the fork still carries on top is its two-GPU tensor/DP
prefill path (not applicable here) and its Qwen3.8-specific hyper-connection decode kernels (`STRATA_GR_RDNA*`, which
need the fork's `q8_mixed` tensor plumbing in `native_dense.cpp` / `layer.cpp` / `generate.cpp`).

Measured, same model, same config, one RX 7900 XTX:

| arm | prefill 8K (t/s) | decode 8K (t/s) | prefill 32K (t/s) | decode 32K (t/s) |
|---|---|---|---|---|
| upstream, plain config | 697.8 | 77.8 | 765.5 | 74.3 |
| fork, plain config | 714.3 | 71.2 | 786.7 | 77.5 |
| upstream + its 29 RDNA3 env flags | 1318.3 | 80.8 | **1614.3** | 85.9 |
| fork + its 46 env flags | 1327.0 | 92.5 | 1584.5 | **95.9** |

Read: the 1.9-2.1x prefill comes from the env flags, and **upstream has 29 of the fork's 46**; the 17 it lacks buy
-0.7% at 8K and +1.9% at 32K (both inside run-to-run noise), plus 14% decode that belongs to the fork-only fused
hyper-connection read. So porting piece 1 wholesale would have been duplicating upstream code; the branch ports piece 4
and documents piece 1 as already upstream, with the numbers above as evidence.

## What is ported

**1. `src/core/kfdq.cpp` — KFD queue parking (idle power).**
A gfx1100 card gives a process 8 compute queue slots. `GPU_MAX_HW_QUEUES=8` opens 8 HIP queues plus ROCr's utility
queue = 9, one over the slots, so MES time-slices forever and the card sits at 100% busy / ~95 W while the engine is
idle. The executable exports `ioctl()`, records every KFD queue ROCr creates, and after `STRATA_KFDQ_IDLE_MS` (2000 ms)
without a submission deactivates the utility queue (`AMDKFD_IOC_UPDATE_QUEUE`, percentage 0) on cards holding more than
`STRATA_KFDQ_SLOTS` (8) queues; the next submission reactivates it within `STRATA_KFDQ_POLL_US` (2000 µs).

Measured here (one 7900 XTX, `GPU_MAX_HW_QUEUES=8`, engine ready and idle):

| | GPU busy | idle power |
|---|---|---|
| `STRATA_KFDQ=0` | 14% | **97 W** |
| `STRATA_KFDQ=1` (default) | 0% | **11 W** |

Cost: the first request after an idle period starts 4-10 ms later. No effect on prefill or decode. `STRATA_KFDQ_LOG=1`
logs queue creation, park and unpark. Single-card setups that keep `GPU_MAX_HW_QUEUES=6` (7 queues) are idle-clean
without it; this matters for `GPU_MAX_HW_QUEUES=8` and for cards shared with another process.

**2. `tools/nvfp4_local_pack.py slim` — every shard of a split GGUF.**
Upstream's `slim` took one shard. The fork's version takes `--gguf` once per shard (in order) and writes each shard to
`--out` under its own name, so `--native` keeps pointing at the same file names, and rewrites `split.tensors.count` to
the new total in every shard (the engine refuses a set whose shards disagree). A pack with `experts.bin` then reads no
routed expert from the GGUF: measured on the fork's side 79-104 GB -> 32 GB per model.

## Build

Same recipe as upstream's `docs/AMD_HIP.md`; only the ROCm root differs per machine.

```sh
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=/opt/rocm-merged HIP_PATH=$ROCM_PATH
cmake -G Ninja -S . -B build-1100 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_PATH \
  -DCMAKE_PREFIX_PATH=$ROCM_PATH \
  -DCMAKE_HIP_FLAGS="--rocm-path=$ROCM_PATH --rocm-device-lib-path=$ROCM_PATH/lib/llvm/amdgcn/bitcode"
ninja -C build-1100 strata
```

## Run (single card, 256K context, the numbers above)

```sh
export STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100200.txt
export STRATA_HIP_WMMA=1 STRATA_SELECT_WMMA=1 STRATA_QFUSE=1
export STRATA_MMQ_KQUANTS=1 STRATA_PREFILL_MMQ=1
export STRATA_PREFILL_RING=192 STRATA_PREFILL_AUTO_MAX=32768
export STRATA_GDN_CHUNKED=1 STRATA_PA_FAST=1 STRATA_HEAD_MIX_MULTI=1
export GPU_MAX_HW_QUEUES=6           # 7 queues: idle-clean without kfdq
build-1100/strata --serve --pack <pack> --native <shard1.gguf> --ple-gguf <last shard> --gpu 0 \
  --expert-cache auto --expert-profile data/expert-profile.bin --kv int8 \
  --kv-resident 262144 --max-context 262144 --prefill 2048 --spec 4 \
  --draft dflash2 --draft-model <dflash2.pack> --draft-cap 6 --draft-conf 0.35 --mtp <mtp/rt>
```

`--draft-fallback` exists in upstream only; the fork's configs pass it and its engine rejects it.

## Regression test: the 16 common-wrong questions

The set from `2026-11-05 UD3.1_AB_双MI50_Q4_vs_7900_IQ3`: eval100 `mmlu_3/12/19/38/42/45/46` + `drop_78`,
polyglot `go_beer-song/counter/markdown/paasio/robot-simulator/scale-generator/sublist`, RULER-256K `qa_2_59`. These
are questions *both* environments answered wrong, so the test is "does a build change any of them" — a numerical
regression would show up as garbage here before it shows up anywhere else.

Results in `bench/wrong16-port-2026-11-08.md`.

## Caveats

- The measurements are one machine (EPYC 7F52, DDR4, PCIe 4.0 x16), 3 reps, 256 output tokens, dflash2 spec 4; decode
  varies ±10% with MTP acceptance, so read decode as a band, not a point.
- `kfdq` is Linux + `linux/kfd_ioctl.h` only; the file compiles to nothing elsewhere.
- No CI here: the branch was built and run on the machine it was ported for.
