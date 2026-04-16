#!/usr/bin/env bash
# Build GPU version (CUDA + OpenMP + AVX-512) on Delta.
# A100 = sm_80. Build on a compute node to avoid login-node load limits; use:
#   srun -A $DELTA_ACCOUNT --partition=gpuA100x4 --gres=gpu:1 --time=0:30:00 \
#        --pty bash scripts/delta/02_build_gpu.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_setup_env.sh"

cd "$PROJECT_ROOT"
rm -rf "$BUILD_DIR_GPU"
mkdir -p "$BUILD_DIR_GPU"
cd "$BUILD_DIR_GPU"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DENABLE_AVX512=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc

cmake --build . -j $(nproc)

echo
echo "GPU build finished. Artifacts:"
ls -lh demo benchmark benchmark_gpu 2>/dev/null || true
