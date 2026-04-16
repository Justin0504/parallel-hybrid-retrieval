#!/bin/bash
#SBATCH --job-name=ee451-cpu-bench
#SBATCH --account=bfsl-delta-gpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=results/delta/logs/cpu_bench_%j.out
#SBATCH --error=results/delta/logs/cpu_bench_%j.err

# ── Run the CPU-only agent memory benchmark on a 64-core Xeon node.
# Usage: sbatch scripts/delta/benchmark_cpu.slurm

set -euo pipefail
source scripts/delta/00_setup_env.sh

mkdir -p "$RESULTS_DIR/logs"

cd "$BUILD_DIR_CPU"

# Full benchmark sweeping threads and corpus sizes.
# demo CLI: --quick | --sessions N | default
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-64}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "=== CPU benchmark @ $(hostname) — $(date) ==="
echo "  cores=$SLURM_CPUS_PER_TASK  OMP_NUM_THREADS=$OMP_NUM_THREADS"
lscpu | grep -E 'Model name|Socket|Core|Thread|MHz' | head -6
echo

# Main scan
./demo --sessions 10000 2>&1 | tee "$RESULTS_DIR/cpu_main_$SLURM_JOB_ID.log"

# Copy CSVs out of build dir
cp -v results/*.csv "$RESULTS_DIR/" 2>/dev/null || true

echo "=== done ==="
