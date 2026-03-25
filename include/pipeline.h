#pragma once

#include "common.h"
#include "sparse_index.h"
#include "dense_index.h"
#include "fusion.h"
#include <vector>
#include <functional>

namespace hybrid {

// End-to-end hybrid retrieval pipeline.
//
// Supports five execution modes:
//   1. sequential       — all stages single-threaded (baseline)
//   2. task_parallel    — sparse/dense branches run concurrently per query
//   3. data_parallel    — intra-query data parallelism (parallel BM25 scoring)
//   4. full_parallel    — query-level + task + data parallelism (all three)
//   5. combined         — data_parallel per query + query-level parallelism
//
// All modes produce equivalent results (modulo floating-point ordering ties).
class Pipeline {
public:
    struct Config {
        int top_k            = 10;
        int sparse_candidates = 100;  // candidates from sparse branch
        int dense_candidates  = 100;  // candidates from dense branch
        int num_threads       = 1;
    };

    Pipeline(SparseIndex& sparse, DenseIndex& dense, Config config);

    // Run a single query sequentially. Returns results + timing.
    std::pair<std::vector<ScoredDoc>, StageTimings>
    query_sequential(const Query& q) const;

    // Run a single query with task parallelism (sparse || dense).
    std::pair<std::vector<ScoredDoc>, StageTimings>
    query_task_parallel(const Query& q) const;

    // Run a single query with intra-query data parallelism.
    // Sparse BM25 scoring is partitioned across threads.
    std::pair<std::vector<ScoredDoc>, StageTimings>
    query_data_parallel(const Query& q, int num_threads) const;

    // Run a batch of queries.
    struct BatchResult {
        std::vector<std::vector<ScoredDoc>> results;
        StageTimings                        avg_stage;
        double                              total_ms;
        double                              avg_latency_ms;
        double                              throughput_qps;
    };

    BatchResult run_batch(const std::vector<Query>& queries,
                          const std::string& mode) const;

private:
    SparseIndex& sparse_;
    DenseIndex&  dense_;
    Config       config_;
};

} // namespace hybrid
