#!/usr/bin/env bash
# Build CPU-only version on Delta with AVX-512 (Delta nodes are Xeon Platinum).
# Run on login node (fast enough) or grab a compute node via srun for heavy builds.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_setup_env.sh"

cd "$PROJECT_ROOT"
rm -rf "$BUILD_DIR_CPU"
mkdir -p "$BUILD_DIR_CPU"
cd "$BUILD_DIR_CPU"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=OFF \
    -DENABLE_AVX512=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc

cmake --build . -j $(nproc)

echo
echo "CPU build finished. Artifacts:"
ls -lh demo benchmark 2>/dev/null || true
