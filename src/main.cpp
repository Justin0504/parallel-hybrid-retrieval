#include "common.h"
#include "sparse_index.h"
#include "dense_index.h"
#include "fusion.h"
#include "pipeline.h"
#include "corpus_generator.h"

#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>

using namespace hybrid;

// ============================================================================
// Configuration
// ============================================================================

struct BenchConfig {
    std::vector<int>    corpus_sizes   = {100000, 500000, 1000000};
    std::vector<int>    thread_counts  = {1, 2, 4, 8};
    int                 num_queries    = 100;
    int                 embedding_dim  = 128;
    int                 top_k          = 10;
    int                 sparse_cands   = 100;
    int                 dense_cands    = 100;
    int                 warmup_queries = 10;
    std::string         output_csv     = "results/benchmark_results.csv";
};

// ============================================================================
// Helpers
// ============================================================================

static void print_header() {
    std::cout << "\n"
              << "================================================================\n"
              << "  Hybrid Retrieval Pipeline — Parallel Systems Benchmark\n"
              << "  EE451 Project\n"
              << "================================================================\n\n";
}

static void print_system_info() {
    int max_threads = omp_get_max_threads();
    std::cout << "System Information:\n"
              << "  Max OpenMP threads: " << max_threads << "\n"
              << "  OMP_NUM_THREADS:    "
              << (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "not set")
              << "\n\n";
}

static std::string format_time(double ms) {
    std::ostringstream oss;
    if (ms < 1.0) {
        oss << std::fixed << std::setprecision(3) << (ms * 1000.0) << " us";
    } else if (ms < 1000.0) {
        oss << std::fixed << std::setprecision(2) << ms << " ms";
    } else {
        oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
    }
    return oss.str();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    print_header();
    print_system_info();

    BenchConfig config;

    // Parse optional CLI overrides.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--queries" && i + 1 < argc) {
            config.num_queries = std::stoi(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            config.embedding_dim = std::stoi(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (arg == "--corpus" && i + 1 < argc) {
            // Single corpus size
            config.corpus_sizes = {std::stoi(argv[++i])};
        } else if (arg == "--threads" && i + 1 < argc) {
            // Single thread count
            config.thread_counts = {std::stoi(argv[++i])};
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_csv = argv[++i];
        } else if (arg == "--quick") {
            // Quick mode for testing.
            config.corpus_sizes  = {10000};
            config.thread_counts = {1, 2, 4};
            config.num_queries   = 20;
        } else if (arg == "--help") {
            std::cout << "Usage: benchmark [options]\n"
                      << "  --queries N     Number of queries (default: 100)\n"
                      << "  --dim N         Embedding dimension (default: 128)\n"
                      << "  --top-k N       Top-k results (default: 10)\n"
                      << "  --corpus N      Single corpus size\n"
                      << "  --threads N     Single thread count\n"
                      << "  --output FILE   Output CSV path\n"
                      << "  --quick         Quick test mode\n";
            return 0;
        }
    }

    // Prepare CSV output.
    system("mkdir -p results");
    std::ofstream csv(config.output_csv);
    csv << "corpus_size,num_queries,num_threads,mode,"
        << "total_ms,avg_latency_ms,throughput_qps,speedup,efficiency,"
        << "avg_sparse_ms,avg_dense_ms,avg_fusion_ms\n";

    CorpusGenerator gen(config.embedding_dim);

    for (int corpus_size : config.corpus_sizes) {
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  << "Corpus size: " << corpus_size << " documents\n"
                  << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

        // Generate corpus.
        std::cout << "  Generating corpus... " << std::flush;
        Timer t;
        t.start();
        auto corpus = gen.generate_corpus(corpus_size);
        t.stop();
        std::cout << format_time(t.elapsed_ms()) << "\n";

        // Build indices.
        std::cout << "  Building sparse index... " << std::flush;
        SparseIndex sparse;
        t.start();
        sparse.build(corpus);
        t.stop();
        std::cout << format_time(t.elapsed_ms()) << "\n";

        std::cout << "  Building dense index... " << std::flush;
        DenseIndex dense(config.embedding_dim, corpus_size);
        t.start();
        dense.build(corpus);
        t.stop();
        std::cout << format_time(t.elapsed_ms()) << "\n";

        // Generate queries.
        auto queries = gen.generate_queries(config.num_queries + config.warmup_queries);

        // Warmup queries (discard results).
        std::vector<Query> warmup_qs(queries.begin(), queries.begin() + config.warmup_queries);
        std::vector<Query> bench_qs(queries.begin() + config.warmup_queries, queries.end());

        {
            Pipeline::Config pc;
            pc.top_k = config.top_k;
            pc.sparse_candidates = config.sparse_cands;
            pc.dense_candidates  = config.dense_cands;
            pc.num_threads = 1;
            Pipeline warmup_pipe(sparse, dense, pc);
            warmup_pipe.run_batch(warmup_qs, "sequential");
        }

        // Baseline: sequential with 1 thread.
        double baseline_total_ms = 0.0;

        std::cout << "\n  Running benchmarks:\n";
        std::cout << "  " << std::setw(8) << "Threads"
                  << std::setw(16) << "Mode"
                  << std::setw(14) << "Total"
                  << std::setw(14) << "Avg Latency"
                  << std::setw(14) << "QPS"
                  << std::setw(10) << "Speedup"
                  << std::setw(10) << "Eff%"
                  << "\n";
        std::cout << "  " << std::string(86, '-') << "\n";

        // For each thread count, run all three modes.
        for (int nthreads : config.thread_counts) {
            omp_set_num_threads(nthreads);

            Pipeline::Config pc;
            pc.top_k = config.top_k;
            pc.sparse_candidates = config.sparse_cands;
            pc.dense_candidates  = config.dense_cands;
            pc.num_threads = nthreads;

            Pipeline pipe(sparse, dense, pc);

            // Determine which modes to run.
            std::vector<std::string> modes;
            if (nthreads == 1) {
                modes = {"sequential"};
            } else {
                modes = {"sequential", "task_parallel", "full_parallel"};
            }

            for (const auto& mode : modes) {
                auto result = pipe.run_batch(bench_qs, mode);

                double speedup    = 0.0;
                double efficiency = 0.0;

                if (mode == "sequential" && nthreads == 1) {
                    baseline_total_ms = result.total_ms;
                    speedup    = 1.0;
                    efficiency = 100.0;
                } else if (baseline_total_ms > 0.0) {
                    speedup    = baseline_total_ms / result.total_ms;
                    efficiency = (speedup / nthreads) * 100.0;
                }

                std::cout << "  " << std::setw(8) << nthreads
                          << std::setw(16) << mode
                          << std::setw(14) << format_time(result.total_ms)
                          << std::setw(14) << format_time(result.avg_latency_ms)
                          << std::setw(14) << std::fixed << std::setprecision(1) << result.throughput_qps
                          << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
                          << std::setw(9) << std::fixed << std::setprecision(1) << efficiency << "%"
                          << "\n";

                // Write CSV row.
                csv << corpus_size << ","
                    << config.num_queries << ","
                    << nthreads << ","
                    << mode << ","
                    << std::fixed << std::setprecision(3)
                    << result.total_ms << ","
                    << result.avg_latency_ms << ","
                    << result.throughput_qps << ","
                    << speedup << ","
                    << efficiency << ","
                    << result.avg_stage.sparse_ms << ","
                    << result.avg_stage.dense_ms << ","
                    << result.avg_stage.fusion_ms << "\n";
            }
        }

        std::cout << "\n";
    }

    csv.close();
    std::cout << "Results written to: " << config.output_csv << "\n\n";

    return 0;
}
