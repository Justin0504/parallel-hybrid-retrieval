#include "pipeline.h"
#include <omp.h>
#include <numeric>
#include <iostream>

namespace hybrid {

Pipeline::Pipeline(SparseIndex& sparse, DenseIndex& dense, Config config)
    : sparse_(sparse), dense_(dense), config_(config) {}

// ============================================================================
// Sequential single query (baseline)
// ============================================================================

std::pair<std::vector<ScoredDoc>, StageTimings>
Pipeline::query_sequential(const Query& q) const {
    StageTimings timings;
    Timer t;

    t.start();
    t.stop();
    timings.preprocess_ms = t.elapsed_ms();

    // Sparse retrieval — single-threaded
    t.start();
    auto sparse_results = sparse_.query(q.text, config_.sparse_candidates);
    t.stop();
    timings.sparse_ms = t.elapsed_ms();

    // Dense retrieval — single-threaded
    t.start();
    auto dense_results = dense_.query(q.embedding, config_.dense_candidates);
    t.stop();
    timings.dense_ms = t.elapsed_ms();

    // Fusion + top-k
    t.start();
    auto fused = Fusion::rrf(sparse_results, dense_results, config_.top_k);
    t.stop();
    timings.fusion_ms = t.elapsed_ms();

    timings.total_ms = timings.preprocess_ms + timings.sparse_ms +
                       timings.dense_ms + timings.fusion_ms;

    return {fused, timings};
}

// ============================================================================
// Task-parallel single query (sparse || dense, both single-threaded internally)
// ============================================================================

std::pair<std::vector<ScoredDoc>, StageTimings>
Pipeline::query_task_parallel(const Query& q) const {
    StageTimings timings;
    Timer t_total;
    t_total.start();

    std::vector<ScoredDoc> sparse_results, dense_results;
    double sparse_time = 0.0, dense_time = 0.0;

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            Timer ts;
            ts.start();
            sparse_results = sparse_.query(q.text, config_.sparse_candidates);
            ts.stop();
            sparse_time = ts.elapsed_ms();
        }
        #pragma omp section
        {
            Timer td;
            td.start();
            dense_results = dense_.query(q.embedding, config_.dense_candidates);
            td.stop();
            dense_time = td.elapsed_ms();
        }
    }

    timings.sparse_ms = sparse_time;
    timings.dense_ms  = dense_time;

    Timer t_fusion;
    t_fusion.start();
    auto fused = Fusion::rrf(sparse_results, dense_results, config_.top_k);
    t_fusion.stop();
    timings.fusion_ms = t_fusion.elapsed_ms();

    t_total.stop();
    timings.total_ms = t_total.elapsed_ms();

    return {fused, timings};
}

// ============================================================================
// Data-parallel single query (parallel BM25 scoring within sparse retrieval)
// ============================================================================

std::pair<std::vector<ScoredDoc>, StageTimings>
Pipeline::query_data_parallel(const Query& q, int num_threads) const {
    StageTimings timings;
    Timer t;

    t.start();
    t.stop();
    timings.preprocess_ms = t.elapsed_ms();

    // Sparse retrieval — intra-query data parallelism
    t.start();
    auto sparse_results = sparse_.query_parallel(q.text, config_.sparse_candidates, num_threads);
    t.stop();
    timings.sparse_ms = t.elapsed_ms();

    // Dense retrieval — single-threaded (already fast)
    t.start();
    auto dense_results = dense_.query(q.embedding, config_.dense_candidates);
    t.stop();
    timings.dense_ms = t.elapsed_ms();

    // Fusion + top-k
    t.start();
    auto fused = Fusion::rrf(sparse_results, dense_results, config_.top_k);
    t.stop();
    timings.fusion_ms = t.elapsed_ms();

    timings.total_ms = timings.preprocess_ms + timings.sparse_ms +
                       timings.dense_ms + timings.fusion_ms;

    return {fused, timings};
}

// ============================================================================
// Batch execution
// ============================================================================

Pipeline::BatchResult Pipeline::run_batch(
    const std::vector<Query>& queries,
    const std::string& mode) const
{
    size_t n = queries.size();
    BatchResult result;
    result.results.resize(n);

    std::vector<StageTimings> all_timings(n);

    Timer batch_timer;
    batch_timer.start();

    if (mode == "sequential") {
        // Fully sequential: one query at a time, single-threaded.
        // Note: no omp_set_num_threads(1) here — sequential uses a plain loop,
        // and setting global thread count would pollute state for later calls.
        for (size_t i = 0; i < n; ++i) {
            auto [res, timing] = query_sequential(queries[i]);
            result.results[i]  = std::move(res);
            all_timings[i]     = timing;
        }
    }
    else if (mode == "task_parallel") {
        // Per-query task parallelism (sparse || dense), queries sequential.
        for (size_t i = 0; i < n; ++i) {
            auto [res, timing] = query_task_parallel(queries[i]);
            result.results[i]  = std::move(res);
            all_timings[i]     = timing;
        }
    }
    else if (mode == "data_parallel") {
        // Per-query intra-query data parallelism, queries sequential.
        // All threads used for parallel BM25 scoring within each query.
        for (size_t i = 0; i < n; ++i) {
            auto [res, timing] = query_data_parallel(queries[i], config_.num_threads);
            result.results[i]  = std::move(res);
            all_timings[i]     = timing;
        }
    }
    else if (mode == "full_parallel") {
        // Query-level parallelism: distribute queries across threads.
        // Each query runs sequentially internally (no nested parallelism).
        int total_threads = config_.num_threads;

        #pragma omp parallel for schedule(dynamic, 1) num_threads(total_threads)
        for (size_t i = 0; i < n; ++i) {
            auto [res, timing] = query_sequential(queries[i]);
            result.results[i]  = std::move(res);
            all_timings[i]     = timing;
        }
    }
    else if (mode == "combined") {
        // Query-level parallelism + intra-query data parallelism.
        // Outer: query-level. Inner: data-parallel BM25.
        int total_threads = config_.num_threads;
        omp_set_max_active_levels(2);

        // Split: half threads for query-level, half for intra-query.
        int outer_threads = std::max(1, static_cast<int>(std::sqrt(total_threads)));
        int inner_threads = std::max(1, total_threads / outer_threads);

        #pragma omp parallel for schedule(dynamic, 1) num_threads(outer_threads)
        for (size_t i = 0; i < n; ++i) {
            auto [res, timing] = query_data_parallel(queries[i], inner_threads);
            result.results[i]  = std::move(res);
            all_timings[i]     = timing;
        }
    }
    else {
        throw std::runtime_error("Unknown pipeline mode: " + mode);
    }

    batch_timer.stop();
    result.total_ms = batch_timer.elapsed_ms();

    // Compute average stage timings.
    StageTimings avg{};
    for (const auto& t : all_timings) {
        avg.preprocess_ms += t.preprocess_ms;
        avg.sparse_ms     += t.sparse_ms;
        avg.dense_ms      += t.dense_ms;
        avg.fusion_ms     += t.fusion_ms;
        avg.total_ms      += t.total_ms;
    }
    double inv = 1.0 / n;
    avg.preprocess_ms *= inv;
    avg.sparse_ms     *= inv;
    avg.dense_ms      *= inv;
    avg.fusion_ms     *= inv;
    avg.total_ms      *= inv;

    result.avg_stage      = avg;
    result.avg_latency_ms = avg.total_ms;
    result.throughput_qps = n / (result.total_ms / 1000.0);

    return result;
}

} // namespace hybrid
