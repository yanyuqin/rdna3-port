#!/bin/bash
# build.sh -- HIP/gfx1100 build of the rdna3-port branch.
#
# Everything is overridable so this works on another gfx1100 box without editing:
#   ROCM      ROCm tree (default /opt/rocm; here it was /opt/rocm715-merged)
#   LLAMA_CPP llama.cpp source tree that supplies ggml (third_party/ggml is a
#             placeholder in upstream Strata, so point this at a real checkout)
#   SRC/BD    source and build dir (default: this repo and <repo>/build-1100)
#   JOBS      ninja parallelism
#   ARCH      HIP target (default gfx1100 -- this branch carries no gfx906 kernels)
#
# Example on the dev box:
#   ROCM=/opt/rocm715-merged LLAMA_CPP=$HOME/strata/third_party/llama.cpp ./scripts/rdna3/build.sh
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
BD=${BD:-$SRC/build-1100}
ROCM=${ROCM:-/opt/rocm}
LLAMA_CPP=${LLAMA_CPP:-$SRC/third_party/llama.cpp}
JOBS=${JOBS:-$(nproc)}
ARCH=${ARCH:-gfx1100}
LOG=${LOG:-$BD/build.log}

[ -d "$ROCM" ] || { echo "ROCM=$ROCM not found; set ROCM=" >&2; exit 1; }
[ -d "$LLAMA_CPP/ggml" ] || { echo "LLAMA_CPP=$LLAMA_CPP has no ggml/; set LLAMA_CPP=" >&2; exit 1; }

export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
export ROCM_PATH=$ROCM HIP_PATH=$ROCM
export PATH="$ROCM/bin:$PATH"

mkdir -p "$BD"
echo "=== git $(git -C "$SRC" rev-parse --short HEAD) arch=$ARCH rocm=$ROCM" | tee "$LOG"

cmake -G Ninja -S "$SRC" -B "$BD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  "-DCMAKE_PREFIX_PATH=$ROCM" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM --rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode" \
  -DSTRATA_GGML_DIR="$LLAMA_CPP" >> "$LOG" 2>&1

echo "=== build (ninja -j $JOBS)" | tee -a "$LOG"
nice -n 10 ninja -C "$BD" strata -j "$JOBS" >> "$LOG" 2>&1

echo "=== built: $(ls -la "$BD/strata")" | tee -a "$LOG"
echo DONE
