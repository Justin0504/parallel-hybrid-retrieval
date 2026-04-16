# NCSA Delta Quickstart

This directory contains everything needed to build and benchmark the
hybrid retrieval pipeline on NCSA Delta A100 GPUs.

## Prerequisites

You already have:
- Delta SSH access (`ayuan@login.delta.ncsa.illinois.edu`)
- Allocation `bfsl-delta-gpu`
- The project cloned at `$HOME/ee451-hybrid-retrieval` (or `PROJECT_ROOT` override)

## First-time setup (on the login node)

```bash
# 1) Clone the repo (skip if you rsync'd from local)
cd $HOME
git clone <your repo> ee451-hybrid-retrieval
cd ee451-hybrid-retrieval

# 2) Make scripts executable
chmod +x scripts/delta/*.sh

# 3) Source the module environment
source scripts/delta/00_setup_env.sh
```

You should see gcc 11.4, cmake 3.27, nvcc 12.3, python 3.11.

## Build

### Option A — CPU-only (AVX-512 path, no GPU needed)

```bash
bash scripts/delta/01_build_cpu.sh
# produces: build_delta_cpu/{demo, benchmark}
```

### Option B — Full build with CUDA

Grab a GPU node first so nvcc has the right environment:

```bash
bash scripts/delta/interactive_gpu.sh
# ─ you're now on a compute node with 1 A100 ─
source scripts/delta/00_setup_env.sh
bash scripts/delta/02_build_gpu.sh
# produces: build_delta_gpu/{demo, benchmark, benchmark_gpu}
```

Login node *can* build with CUDA too but occasionally hits module issues;
compute nodes are more reliable.

## Quick smoke test (interactive A100)

```bash
# On a compute node after building:
cd $BUILD_DIR_GPU
./benchmark_gpu --quick
# should print: GPU=X.XXms (Y qps)  CPU=Z.ZZms  speedup=Wx
```

If GPU speedup ≥ 5× on the `--quick` config, the kernel is alive and correct.

## Submit full benchmarks (SLURM)

```bash
# CPU sweep on 64-core Xeon
sbatch scripts/delta/benchmark_cpu.slurm

# GPU sweep on 1× A100 (default 100K corpus, then 500K and 1M scaling)
sbatch scripts/delta/benchmark_gpu.slurm

# Override corpus size
sbatch --export=CORPUS=2000000,QUERIES=500 scripts/delta/benchmark_gpu.slurm
```

Watch jobs:

```bash
squeue -u $USER
# when finished:
ls results/delta/logs/
ls results/delta/*.csv
```

## What the outputs look like

`results/delta/gpu_bench_<jobid>.csv` columns:

```
corpus_size, batch_size, top_k,
gpu_total_ms, gpu_h2d_ms, gpu_score_ms, gpu_topk_ms, gpu_d2h_ms,
gpu_qps, gpu_latency_per_q_ms,
cpu_total_ms, cpu_qps, speedup,
top1_match_rate, kendall_tau, device
```

Target numbers for the CIKM submission:
- `speedup` ≥ 20× at batch=256, corpus=100K
- `top1_match_rate` ≥ 0.98
- `kendall_tau` ≥ 0.85
- Scaling: sub-linear `gpu_total_ms` growth from 100K → 1M corpus

## Allocation accounting

Delta charges **SUs** (service units). Costs:
- `gpuA100x4` → 1 GPU-hour = ~4 SU
- `cpu`       → 1 core-hour = 1 SU

Current balance on `bfsl-delta-gpu`: **608 GPU-hours** (as of 2026-03-31).
A single `benchmark_gpu.slurm` run at default size burns ~1.5 GPU-hour.

Monitor with:

```bash
accounts
# or
sacct --account=bfsl-delta-gpu --starttime=today
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `nvcc: command not found` | `module load cuda/12.3.0` |
| `CMake Error: CUDA_ARCHITECTURES is empty` | Upgrade cmake: `module load cmake/3.27.9` |
| `libcuda.so.1: cannot open shared object` on login node | Build on a compute node (see Option B) |
| Kernel gives wrong results on 10M docs | A100 40GB OOM — drop batch or use 80GB partition |
| `Invalid account bfsl-delta-gpu` | Check `accounts` output for the exact name |
