#!/usr/bin/env bash
# NCSA Delta environment bootstrap. Source this script on login node before
# building or submitting SLURM jobs.
#
#   source scripts/delta/00_setup_env.sh
#
# Delta uses Lmod. Modules are refreshed on login; if you hit module errors
# run `module reset` first.

set -e

# ── Modules ─────────────────────────────────────────────────────────────────
module reset >/dev/null 2>&1 || true
module load gcc/11.4.0         || module load gcc
module load cmake/3.27.9       || module load cmake
module load cuda/12.3.0        || module load cuda
module load python/3.11.6      || module load python

# ── Environment ─────────────────────────────────────────────────────────────
# Account is recorded in my Delta allocation. Override if someone else runs this.
export DELTA_ACCOUNT="${DELTA_ACCOUNT:-bfsl-delta-gpu}"

# Prefer project storage over $HOME for build + results (faster, larger quota).
export PROJECT_ROOT="${PROJECT_ROOT:-$HOME/ee451-hybrid-retrieval}"
export BUILD_DIR_CPU="${BUILD_DIR_CPU:-$PROJECT_ROOT/build_delta_cpu}"
export BUILD_DIR_GPU="${BUILD_DIR_GPU:-$PROJECT_ROOT/build_delta_gpu}"
export RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/delta}"

mkdir -p "$RESULTS_DIR"

echo "── Delta env ready ──"
echo "  Account       : $DELTA_ACCOUNT"
echo "  Project root  : $PROJECT_ROOT"
echo "  Build (CPU)   : $BUILD_DIR_CPU"
echo "  Build (GPU)   : $BUILD_DIR_GPU"
echo "  Results       : $RESULTS_DIR"
echo
echo "  gcc     : $(gcc --version | head -1)"
echo "  cmake   : $(cmake --version | head -1)"
echo "  nvcc    : $(nvcc --version | grep release || echo 'nvcc not found')"
echo "  python  : $(python --version)"
