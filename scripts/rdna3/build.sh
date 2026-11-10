#!/bin/bash
# build.sh -- HIP/gfx1100 build of the rdna3-port branch (v0.2.13 layout).
#
# Everything is overridable so this works on another gfx1100 box without editing:
#   ROCM      ROCm tree (default /opt/rocm; here it was /opt/rocm715-merged)
#   LLAMA_CPP optional llama.cpp checkout that supplies ggml; empty (the default)
#             builds against the vendored third_party/strata-ggml that 0.2.13 ships
#   SRC/BD    source and build dir (default: this repo and <repo>/build-1100)
#   JOBS      ninja parallelism
#   ARCH      HIP target (default gfx1100 -- this branch carries no gfx906 kernels)
#
# Example on the dev box:
#   ROCM=/opt/rocm715-merged ./scripts/rdna3/build.sh
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:-$(cd "$HERE/../.." && pwd)}
BD=${BD:-$SRC/build-1100}
ROCM=${ROCM:-/opt/rocm}
LLAMA_CPP=${LLAMA_CPP:-}
JOBS=${JOBS:-$(nproc)}
ARCH=${ARCH:-gfx1100}
LOG=${LOG:-$BD/build.log}

[ -d "$ROCM" ] || { echo "ROCM=$ROCM not found; set ROCM=" >&2; exit 1; }
if [ -n "$LLAMA_CPP" ]; then
  [ -d "$LLAMA_CPP/ggml" ] || { echo "LLAMA_CPP=$LLAMA_CPP has no ggml/; set LLAMA_CPP=" >&2; exit 1; }
  GGML_ARG="-DSTRATA_GGML_DIR=$LLAMA_CPP"
else
  [ -f "$SRC/third_party/strata-ggml/ggml/CMakeLists.txt" ] || {
    echo "vendored ggml missing at $SRC/third_party/strata-ggml; set LLAMA_CPP=" >&2; exit 1; }
  GGML_ARG=""
fi

export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
export ROCM_PATH=$ROCM HIP_PATH=$ROCM
export PATH="$ROCM/bin:$PATH"

mkdir -p "$BD"
echo "=== git $(git -C "$SRC" rev-parse --short HEAD) arch=$ARCH rocm=$ROCM ggml=${GGML_ARG:-vendored}" | tee "$LOG"

cmake -G Ninja -S "$SRC" -B "$BD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_HIP_COMPILER="$ROCM/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  "-DCMAKE_PREFIX_PATH=$ROCM" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$ROCM --rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode" \
  ${GGML_ARG:+"$GGML_ARG"} >> "$LOG" 2>&1

echo "=== build (ninja -j $JOBS)" | tee -a "$LOG"
nice -n 10 ninja -C "$BD" strata -j "$JOBS" >> "$LOG" 2>&1

echo "=== built: $(ls -la "$BD/strata")" | tee -a "$LOG"
echo DONE
