#!/usr/bin/env bash
# Grab an interactive A100 node for development / debugging.
# Drops you on a compute node with 1 GPU and 30-minute limit.
#
#   bash scripts/delta/interactive_gpu.sh
#
# Once on the node:
#   source scripts/delta/00_setup_env.sh
#   cd $BUILD_DIR_GPU
#   ./benchmark_gpu --quick

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/00_setup_env.sh"

srun \
    --account="$DELTA_ACCOUNT" \
    --partition=gpuA100x4-interactive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gres=gpu:1 \
    --mem=64G \
    --time=00:30:00 \
    --pty bash -l
