#!/bin/bash
#SBATCH --job-name=ee451-gpu-bench
#SBATCH --account=bfsl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=results/delta/logs/gpu_bench_%j.out
#SBATCH --error=results/delta/logs/gpu_bench_%j.err

# ── Run the GPU BM25 benchmark on a single A100.
# Usage:
#   sbatch scripts/delta/benchmark_gpu.slurm                # default sweep
#   sbatch --export=CORPUS=1000000 scripts/delta/benchmark_gpu.slurm

set -euo pipefail
source scripts/delta/00_setup_env.sh

mkdir -p "$RESULTS_DIR/logs"

cd "$BUILD_DIR_GPU"

CORPUS=${CORPUS:-100000}
QUERIES=${QUERIES:-1000}
TOPK=${TOPK:-10}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "=== GPU benchmark @ $(hostname) — $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo

./benchmark_gpu \
    --corpus  "$CORPUS" \
    --queries "$QUERIES" \
    --topk    "$TOPK" \
    --csv     "$RESULTS_DIR/gpu_bench_${SLURM_JOB_ID}.csv" \
    2>&1 | tee "$RESULTS_DIR/gpu_bench_${SLURM_JOB_ID}.log"

# Also run at larger scales if GPU memory allows (A100 80GB → up to ~2M docs easily).
if [[ "$CORPUS" -ge 100000 ]]; then
    for C in 500000 1000000; do
        echo "=== scaling: corpus=$C ==="
        ./benchmark_gpu \
            --corpus  "$C" \
            --queries "$QUERIES" \
            --topk    "$TOPK" \
            --csv     "$RESULTS_DIR/gpu_scale_${C}_${SLURM_JOB_ID}.csv" \
            2>&1 | tee -a "$RESULTS_DIR/gpu_scaling_${SLURM_JOB_ID}.log"
    done
fi

echo "=== done ==="
