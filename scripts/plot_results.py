#!/usr/bin/env python3
"""
Plot benchmark results from the hybrid retrieval pipeline.
Generates presentation-quality figures for EE451 project.

Usage:
    python3 scripts/plot_results.py [results/agent_benchmark.csv]
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Style configuration for presentation.
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'sequential':     '#6c757d',
    'task_parallel':  '#0d6efd',
    'data_parallel':  '#198754',
    'full_parallel':  '#dc3545',
    'combined':       '#fd7e14',
}

MODE_LABELS = {
    'sequential':     'Sequential',
    'task_parallel':  'Task Parallel',
    'data_parallel':  'Data Parallel',
    'full_parallel':  'Full Parallel',
    'combined':       'Combined',
}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def plot_speedup(df, output_dir):
    """Speedup vs thread count — all modes on one chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = [m for m in ['task_parallel', 'data_parallel', 'full_parallel', 'combined']
             if m in df['mode'].values]

    for mode in modes:
        sub = df[df['mode'] == mode].sort_values('num_threads')
        ax.plot(sub['num_threads'], sub['speedup'],
                marker='o', linewidth=2, markersize=6,
                color=COLORS.get(mode, '#333'),
                label=MODE_LABELS.get(mode, mode))

    # Add sequential baseline point.
    seq = df[df['mode'] == 'sequential']
    if not seq.empty:
        ax.plot(seq['num_threads'].values[0], 1.0,
                marker='D', markersize=8, color=COLORS['sequential'],
                label='Sequential (baseline)', zorder=5)

    # Ideal speedup line.
    max_t = df['num_threads'].max()
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.25, label='Ideal (linear)')

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Thread Count')
    ax.legend(loc='upper left')
    ax.set_xticks(sorted(df['num_threads'].unique()))

    plt.savefig(os.path.join(output_dir, 'speedup.png'))
    plt.close()


def plot_throughput(df, output_dir):
    """Throughput (QPS) grouped bar chart — all modes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    thread_counts = sorted(df['num_threads'].unique())
    modes = df['mode'].unique()
    n_modes = len(modes)
    width = 0.8 / n_modes
    x = np.arange(len(thread_counts))

    for i, mode in enumerate(modes):
        sub = df[df['mode'] == mode]
        qps_vals = []
        for t in thread_counts:
            row = sub[sub['num_threads'] == t]
            qps_vals.append(row['throughput_qps'].values[0] if not row.empty else 0)
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar(x + offset, qps_vals, width, alpha=0.85,
               color=COLORS.get(mode, f'C{i}'),
               label=MODE_LABELS.get(mode, mode))

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Throughput (queries/sec)')
    ax.set_title('Query Throughput by Parallelism Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thread_counts])
    ax.legend(loc='upper left')

    plt.savefig(os.path.join(output_dir, 'throughput.png'))
    plt.close()


def plot_stage_breakdown(df, output_dir):
    """Stacked bar — per-stage latency breakdown."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    sparse = []
    dense = []
    fusion = []

    for _, row in df.iterrows():
        lbl = f"{row['mode'][:6]}\nT={int(row['num_threads'])}"
        labels.append(lbl)
        sparse.append(row['avg_sparse_ms'])
        dense.append(row['avg_dense_ms'])
        fusion.append(row['avg_fusion_ms'])

    x = np.arange(len(labels))
    ax.bar(x, sparse, label='Sparse (BM25)', color='#4285f4')
    ax.bar(x, dense, bottom=sparse, label='Dense (HNSW)', color='#ea4335')
    ax.bar(x, fusion,
           bottom=[s + d for s, d in zip(sparse, dense)],
           label='Fusion + Top-k', color='#34a853')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Average Per-Query Latency (ms)')
    ax.set_title('Pipeline Stage Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend()

    plt.savefig(os.path.join(output_dir, 'stage_breakdown.png'))
    plt.close()


def plot_efficiency(df, output_dir):
    """Parallel efficiency vs threads — per mode."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode in ['task_parallel', 'data_parallel', 'full_parallel', 'combined']:
        sub = df[df['mode'] == mode].sort_values('num_threads')
        if sub.empty:
            continue
        ax.plot(sub['num_threads'], sub['efficiency'],
                marker='s', linewidth=2, markersize=6,
                color=COLORS.get(mode, '#333'),
                label=MODE_LABELS.get(mode, mode))

    ax.axhline(y=100, color='k', linestyle='--', alpha=0.25, label='Ideal (100%)')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Parallel Efficiency (%)')
    ax.set_title('Parallel Efficiency')
    ax.legend()
    ax.set_ylim(0, 110)
    ax.set_xticks(sorted(df['num_threads'].unique()))

    plt.savefig(os.path.join(output_dir, 'efficiency.png'))
    plt.close()


def plot_latency(df, output_dir):
    """Per-query latency vs threads — full_parallel mode."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sequential baseline.
    seq = df[df['mode'] == 'sequential']
    if not seq.empty:
        ax.axhline(y=seq['avg_latency_ms'].values[0], color=COLORS['sequential'],
                   linestyle='--', alpha=0.5, label='Sequential baseline')

    for mode in ['full_parallel', 'combined', 'data_parallel']:
        sub = df[df['mode'] == mode].sort_values('num_threads')
        if sub.empty:
            continue
        ax.plot(sub['num_threads'], sub['avg_latency_ms'],
                marker='o', linewidth=2, markersize=6,
                color=COLORS.get(mode, '#333'),
                label=MODE_LABELS.get(mode, mode))

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Average Per-Query Latency (ms)')
    ax.set_title('Query Latency vs Thread Count')
    ax.legend()
    ax.set_xticks(sorted(df['num_threads'].unique()))

    plt.savefig(os.path.join(output_dir, 'latency.png'))
    plt.close()


def plot_amdahl(df, output_dir):
    """Amdahl's Law prediction vs measured speedup."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fp = df[df['mode'] == 'full_parallel'].sort_values('num_threads')
    if fp.empty:
        return

    threads = fp['num_threads'].values
    speedups = fp['speedup'].values

    # Estimate serial fraction from observed speedups.
    f_vals = []
    for p, S in zip(threads, speedups):
        if p > 1 and S > 0:
            f = (1.0/S - 1.0/p) / (1.0 - 1.0/p)
            f = max(0, min(1, f))
            f_vals.append(f)

    if not f_vals:
        return

    f_avg = np.mean(f_vals)

    # Predicted curve.
    t_range = np.arange(1, max(threads) * 1.5 + 1)
    S_pred = 1.0 / (f_avg + (1.0 - f_avg) / t_range)

    ax.plot(t_range, S_pred, 'b--', linewidth=2, alpha=0.6,
            label=f"Amdahl's Law (f={f_avg*100:.1f}%)")
    ax.plot(threads, speedups, 'ro-', linewidth=2, markersize=8,
            label='Measured (full_parallel)')
    ax.plot(t_range, t_range, 'k--', alpha=0.15, label='Ideal (linear)')

    # Theoretical max.
    S_max = 1.0 / f_avg
    ax.axhline(y=S_max, color='purple', linestyle=':', alpha=0.4,
               label=f'Max speedup = {S_max:.1f}x')

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title("Amdahl's Law Analysis")
    ax.legend()
    ax.set_xticks(sorted(df['num_threads'].unique()))

    plt.savefig(os.path.join(output_dir, 'amdahl.png'))
    plt.close()


def plot_memory_bound(df, output_dir):
    """Per-stage timing scaling — memory-bound vs compute-bound."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Use sequential baseline and full_parallel.
    seq = df[df['mode'] == 'sequential']
    fp = df[df['mode'] == 'full_parallel'].sort_values('num_threads')

    if seq.empty or fp.empty:
        return

    # Combine: thread=1 from sequential, rest from full_parallel.
    threads = [1] + list(fp['num_threads'].values)
    sparse_times = [seq['avg_sparse_ms'].values[0]] + list(fp['avg_sparse_ms'].values)
    dense_times = [seq['avg_dense_ms'].values[0]] + list(fp['avg_dense_ms'].values)
    fusion_times = [seq['avg_fusion_ms'].values[0]] + list(fp['avg_fusion_ms'].values)

    # Left: absolute per-query stage times.
    ax1.plot(threads, sparse_times, 'o-', linewidth=2, color='#4285f4', label='Sparse (BM25)')
    ax1.plot(threads, dense_times, 's-', linewidth=2, color='#ea4335', label='Dense (HNSW)')
    ax1.plot(threads, fusion_times, '^-', linewidth=2, color='#34a853', label='Fusion')
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('Per-Query Stage Time (ms)')
    ax1.set_title('Per-Stage Scaling')
    ax1.legend()
    ax1.set_xticks(threads)

    # Right: normalized to baseline.
    sparse_norm = [s / sparse_times[0] for s in sparse_times]
    dense_norm = [d / dense_times[0] for d in dense_times]

    ax2.plot(threads, sparse_norm, 'o-', linewidth=2, color='#4285f4',
             label='Sparse (MEMORY-BOUND)')
    ax2.plot(threads, dense_norm, 's-', linewidth=2, color='#ea4335',
             label='Dense (COMPUTE-BOUND)')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.25, label='Baseline (1x)')
    ax2.set_xlabel('Threads')
    ax2.set_ylabel('Normalized Stage Time (vs 1 thread)')
    ax2.set_title('Memory-Bound vs Compute-Bound')
    ax2.legend()
    ax2.set_xticks(threads)

    plt.savefig(os.path.join(output_dir, 'memory_bound.png'))
    plt.close()


def plot_scalability_latency(df, output_dir):
    """Per-query latency vs corpus size — all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    SCALE_COLORS = {
        'sequential':    '#6c757d',
        'full_parallel': '#dc3545',
        'combined':      '#fd7e14',
        'simd':          '#0d6efd',
        'maxscore':      '#6610f2',
        'temporal':      '#198754',
    }
    SCALE_LABELS = {
        'sequential':    'Sequential (baseline)',
        'full_parallel': 'Full Parallel (8T)',
        'combined':      'Combined (8T)',
        'simd':          'SIMD (NEON, 1T)',
        'maxscore':      'MaxScore (1T)',
        'temporal':      'Temporal Partitioned',
    }

    for strategy in ['sequential', 'combined', 'full_parallel', 'simd', 'maxscore', 'temporal']:
        sub = df[df['strategy'] == strategy].sort_values('corpus_size')
        if sub.empty:
            continue
        marker = 'o' if strategy != 'temporal' else 'D'
        lw = 2.5 if strategy in ('temporal', 'sequential') else 1.8
        ax.plot(sub['corpus_size'], sub['avg_latency_ms'] * 1000,  # convert to us
                marker=marker, linewidth=lw, markersize=6,
                color=SCALE_COLORS.get(strategy, '#333'),
                label=SCALE_LABELS.get(strategy, strategy))

    ax.set_xlabel('Corpus Size (records)')
    ax.set_ylabel('Average Per-Query Latency (us)')
    ax.set_title('Scalability: Latency vs Corpus Size')
    ax.legend(loc='upper left')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(os.path.join(output_dir, 'scalability_latency.png'))
    plt.close()


def plot_scalability_speedup(df, output_dir):
    """Speedup vs corpus size — all strategies vs sequential."""
    fig, ax = plt.subplots(figsize=(10, 6))

    SCALE_COLORS = {
        'full_parallel': '#dc3545',
        'combined':      '#fd7e14',
        'simd':          '#0d6efd',
        'maxscore':      '#6610f2',
        'temporal':      '#198754',
    }
    SCALE_LABELS = {
        'full_parallel': 'Full Parallel (8T)',
        'combined':      'Combined (8T)',
        'simd':          'SIMD (NEON, 1T)',
        'maxscore':      'MaxScore (1T)',
        'temporal':      'Temporal Partitioned',
    }

    for strategy in ['combined', 'full_parallel', 'simd', 'maxscore', 'temporal']:
        sub = df[df['strategy'] == strategy].sort_values('corpus_size')
        if sub.empty or 'speedup_vs_seq' not in sub.columns:
            continue
        marker = 'o' if strategy != 'temporal' else 'D'
        lw = 2.5 if strategy == 'temporal' else 1.8
        ax.plot(sub['corpus_size'], sub['speedup_vs_seq'],
                marker=marker, linewidth=lw, markersize=7,
                color=SCALE_COLORS.get(strategy, '#333'),
                label=SCALE_LABELS.get(strategy, strategy))

    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.2, label='Baseline (1x)')
    ax.set_xlabel('Corpus Size (records)')
    ax.set_ylabel('Speedup vs Sequential')
    ax.set_title('Scalability: Speedup vs Corpus Size')
    ax.legend(loc='upper left')
    ax.set_xscale('log')

    plt.savefig(os.path.join(output_dir, 'scalability_speedup.png'))
    plt.close()


def plot_scalability_throughput(df, output_dir):
    """Throughput vs corpus size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    SCALE_COLORS = {
        'sequential':    '#6c757d',
        'full_parallel': '#dc3545',
        'combined':      '#fd7e14',
        'temporal':      '#198754',
    }
    SCALE_LABELS = {
        'sequential':    'Sequential',
        'full_parallel': 'Full Parallel (8T)',
        'combined':      'Combined (8T)',
        'temporal':      'Temporal',
    }

    for strategy in ['sequential', 'combined', 'full_parallel', 'temporal']:
        sub = df[df['strategy'] == strategy].sort_values('corpus_size')
        if sub.empty:
            continue
        ax.plot(sub['corpus_size'], sub['throughput_qps'],
                marker='o', linewidth=2, markersize=6,
                color=SCALE_COLORS.get(strategy, '#333'),
                label=SCALE_LABELS.get(strategy, strategy))

    ax.set_xlabel('Corpus Size (records)')
    ax.set_ylabel('Throughput (queries/sec)')
    ax.set_title('Scalability: Throughput vs Corpus Size')
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.savefig(os.path.join(output_dir, 'scalability_throughput.png'))
    plt.close()


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results/agent_benchmark.csv'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(csv_path)
    n_records = df['corpus_size'].iloc[0] if 'corpus_size' in df.columns else '?'
    print(f"Loaded {len(df)} rows from {csv_path} ({n_records} records)")

    plot_speedup(df, output_dir)
    print("  -> results/speedup.png")

    plot_throughput(df, output_dir)
    print("  -> results/throughput.png")

    plot_stage_breakdown(df, output_dir)
    print("  -> results/stage_breakdown.png")

    plot_efficiency(df, output_dir)
    print("  -> results/efficiency.png")

    plot_latency(df, output_dir)
    print("  -> results/latency.png")

    plot_amdahl(df, output_dir)
    print("  -> results/amdahl.png")

    plot_memory_bound(df, output_dir)
    print("  -> results/memory_bound.png")

    # Scalability plots (from separate CSV).
    scale_csv = csv_path.replace('agent_benchmark', 'scalability')
    if os.path.exists(scale_csv):
        sdf = load_data(scale_csv)
        print(f"\nLoaded {len(sdf)} rows from {scale_csv}")

        plot_scalability_latency(sdf, output_dir)
        print("  -> results/scalability_latency.png")

        plot_scalability_speedup(sdf, output_dir)
        print("  -> results/scalability_speedup.png")

        plot_scalability_throughput(sdf, output_dir)
        print("  -> results/scalability_throughput.png")

    total = 7 + (3 if os.path.exists(scale_csv) else 0)
    print(f"\nGenerated {total} figures in {output_dir}/")


if __name__ == '__main__':
    main()
