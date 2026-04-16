// GPU vs CPU BM25 benchmark — the primary Delta / A100 driver.
//
// Measures:
//   - End-to-end latency (H2D + kernels + D2H) for a batch of queries
//   - Per-stage breakdown
//   - Correctness check against CPU baseline (top-1 match + rank correlation)
//   - Throughput scan across batch sizes and corpus sizes
//
// Outputs CSV at results/gpu_benchmark.csv for plotting.

#include "common.h"
#include "corpus_generator.h"
#include "sparse_index.h"
#include "sparse_index_gpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace hybrid;
namespace fs = std::filesystem;

// ---- CLI parsing (minimal, no external deps) -------------------------------
struct Args {
    int    corpus_size = 100000;
    int    num_queries = 500;
    int    top_k       = 10;
    int    embed_dim   = 128;
    unsigned seed      = 42;
    std::string csv    = "results/gpu_benchmark.csv";
    bool   scan_batch  = true;   // sweep batch size
    bool   scan_corpus = true;   // sweep corpus size
    bool   verify      = true;   // cross-check vs CPU
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](int& out){ if (i+1 < argc) out = std::atoi(argv[++i]); };
        auto nexts = [&](std::string& out){ if (i+1 < argc) out = argv[++i]; };
        if      (k == "--corpus")     next(a.corpus_size);
        else if (k == "--queries")    next(a.num_queries);
        else if (k == "--topk")       next(a.top_k);
        else if (k == "--csv")        nexts(a.csv);
        else if (k == "--no-scan")    { a.scan_batch = false; a.scan_corpus = false; }
        else if (k == "--no-verify")  a.verify = false;
        else if (k == "--quick")      { a.corpus_size = 20000; a.num_queries = 100; }
    }
    return a;
}

// ---- Helpers ---------------------------------------------------------------
static double kendall_tau_top_k(const std::vector<ScoredDoc>& a,
                                const std::vector<ScoredDoc>& b, int k) {
    k = std::min<int>(k, std::min(a.size(), b.size()));
    if (k < 2) return 1.0;
    int concordant = 0, discordant = 0;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            // find positions in b
            int pa_i = -1, pa_j = -1;
            for (int p = 0; p < k; ++p) {
                if (b[p].id == a[i].id) pa_i = p;
                if (b[p].id == a[j].id) pa_j = p;
            }
            if (pa_i < 0 || pa_j < 0) continue;
            if ((i < j) == (pa_i < pa_j)) concordant++; else discordant++;
        }
    }
    int total = concordant + discordant;
    return total == 0 ? 1.0 : double(concordant - discordant) / total;
}

static void append_csv(const std::string& path, const std::string& header,
                       const std::string& row, bool first_write) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream ofs(path, first_write ? std::ios::trunc : std::ios::app);
    if (first_write) ofs << header << "\n";
    ofs << row << "\n";
}

// ---- Benchmark a single config --------------------------------------------
struct Row {
    int    corpus_size;
    int    batch_size;
    int    top_k;
    double gpu_total_ms;
    double gpu_h2d_ms;
    double gpu_score_ms;
    double gpu_topk_ms;
    double gpu_d2h_ms;
    double gpu_throughput_qps;
    double gpu_latency_per_q_ms;
    double cpu_total_ms;      // sequential reference
    double cpu_throughput_qps;
    double speedup;           // cpu / gpu
    double top1_match_rate;   // [0..1]
    double avg_kendall_tau;   // [-1..1]
    std::string device_name;
};

static Row run_one(SparseIndex& cpu_idx, SparseIndexGPU& gpu_idx,
                   const std::vector<Query>& queries,
                   int corpus_size, int top_k, bool verify)
{
    // Build query text list
    std::vector<std::string> texts;
    texts.reserve(queries.size());
    for (const auto& q : queries) texts.push_back(q.text);

    Row row{};
    row.corpus_size = corpus_size;
    row.batch_size  = (int)queries.size();
    row.top_k       = top_k;

    // ---- GPU (batched) ----
    // Warm-up pass (hides driver lazy init)
    (void)gpu_idx.query_batch(std::vector<std::string>(texts.begin(),
                                                       texts.begin() + std::min<size_t>(8, texts.size())),
                              top_k);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto gpu_results = gpu_idx.query_batch(texts, top_k);
    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const auto& st = gpu_idx.last_stats();
    row.gpu_total_ms        = wall_ms;   // wall clock (includes CPU tokenize)
    row.gpu_h2d_ms          = st.h2d_ms;
    row.gpu_score_ms        = st.kernel_score_ms;
    row.gpu_topk_ms         = st.kernel_topk_ms;
    row.gpu_d2h_ms          = st.d2h_ms;
    row.gpu_throughput_qps  = 1000.0 * queries.size() / wall_ms;
    row.gpu_latency_per_q_ms= wall_ms / queries.size();
    row.device_name         = st.device_name;

    // ---- CPU baseline (sequential, top_k via best existing path) ----
    auto c0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<ScoredDoc>> cpu_results;
    cpu_results.reserve(queries.size());
    for (const auto& q : queries) {
        cpu_results.push_back(cpu_idx.query(q.text, top_k));
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    row.cpu_total_ms        = cpu_ms;
    row.cpu_throughput_qps  = 1000.0 * queries.size() / cpu_ms;
    row.speedup             = cpu_ms / wall_ms;

    // ---- Correctness ----
    if (verify) {
        int top1_match = 0;
        double tau_sum = 0.0;
        int    tau_n   = 0;
        for (size_t i = 0; i < queries.size(); ++i) {
            if (!gpu_results[i].empty() && !cpu_results[i].empty()
                && gpu_results[i][0].id == cpu_results[i][0].id) top1_match++;
            tau_sum += kendall_tau_top_k(gpu_results[i], cpu_results[i], top_k);
            tau_n++;
        }
        row.top1_match_rate = double(top1_match) / queries.size();
        row.avg_kendall_tau = tau_n ? tau_sum / tau_n : 0.0;
    } else {
        row.top1_match_rate = -1.0;
        row.avg_kendall_tau = -2.0;
    }

    return row;
}

static std::string fmt_row(const Row& r) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.4f,%.3f,%.1f,%.2f,%.4f,%.4f,%s",
        r.corpus_size, r.batch_size, r.top_k,
        r.gpu_total_ms, r.gpu_h2d_ms, r.gpu_score_ms, r.gpu_topk_ms, r.gpu_d2h_ms,
        r.gpu_throughput_qps, r.gpu_latency_per_q_ms,
        r.cpu_total_ms, r.cpu_throughput_qps, r.speedup,
        r.top1_match_rate, r.avg_kendall_tau,
        r.device_name.c_str());
    return std::string(buf);
}

// ---- Main ------------------------------------------------------------------
int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::printf("=== GPU BM25 benchmark ===\n");
    std::printf("  corpus=%d, queries=%d, topk=%d, csv=%s\n",
                a.corpus_size, a.num_queries, a.top_k, a.csv.c_str());

    // Generate corpus + queries
    CorpusGenerator cg(a.embed_dim, a.seed);
    auto corpus  = cg.generate_corpus(a.corpus_size);
    auto queries = cg.generate_queries(a.num_queries);

    // Build CPU index, then upload to GPU
    SparseIndex cpu;
    auto b0 = std::chrono::high_resolution_clock::now();
    cpu.build(corpus);
    auto b1 = std::chrono::high_resolution_clock::now();
    std::printf("  CPU index built in %.1f ms (%zu docs)\n",
                std::chrono::duration<double, std::milli>(b1 - b0).count(),
                cpu.num_docs());

    SparseIndexGPU gpu;
    gpu.upload(cpu);

    const std::string header =
        "corpus_size,batch_size,top_k,"
        "gpu_total_ms,gpu_h2d_ms,gpu_score_ms,gpu_topk_ms,gpu_d2h_ms,"
        "gpu_qps,gpu_latency_per_q_ms,"
        "cpu_total_ms,cpu_qps,speedup,top1_match_rate,kendall_tau,device";

    bool first_write = true;

    if (a.scan_batch) {
        // Sweep batch sizes
        std::vector<int> batches = {1, 4, 16, 64, 256, 1024};
        for (int B : batches) {
            if (B > (int)queries.size()) continue;
            std::vector<Query> sub(queries.begin(), queries.begin() + B);
            auto r = run_one(cpu, gpu, sub, a.corpus_size, a.top_k, a.verify);
            std::printf("  batch=%4d  GPU=%.2fms (%.0f qps)  CPU=%.2fms  speedup=%.2fx  tau=%.3f  top1=%.2f\n",
                        B, r.gpu_total_ms, r.gpu_throughput_qps, r.cpu_total_ms,
                        r.speedup, r.avg_kendall_tau, r.top1_match_rate);
            append_csv(a.csv, header, fmt_row(r), first_write);
            first_write = false;
        }
    } else {
        auto r = run_one(cpu, gpu, queries, a.corpus_size, a.top_k, a.verify);
        std::printf("  batch=%4d  GPU=%.2fms (%.0f qps)  CPU=%.2fms  speedup=%.2fx\n",
                    r.batch_size, r.gpu_total_ms, r.gpu_throughput_qps,
                    r.cpu_total_ms, r.speedup);
        append_csv(a.csv, header, fmt_row(r), first_write);
    }

    std::printf("Results written to %s\n", a.csv.c_str());
    return 0;
}
