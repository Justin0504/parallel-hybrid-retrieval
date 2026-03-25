#pragma once

#include "common.h"
#include <vector>

namespace hybrid {

// Score fusion strategies for merging sparse and dense candidate sets.
class Fusion {
public:
    // Reciprocal Rank Fusion (RRF).
    // Merges two ranked lists: score = sum(1 / (k + rank)).
    // `rrf_k` is the constant (typically 60).
    static std::vector<ScoredDoc> rrf(
        const std::vector<ScoredDoc>& sparse_results,
        const std::vector<ScoredDoc>& dense_results,
        int top_k,
        int rrf_k = 60);

    // Parallel top-k selection from a large candidate set using OpenMP.
    // Partitions candidates across threads, each finds local top-k,
    // then merges partial results.
    static std::vector<ScoredDoc> parallel_top_k(
        std::vector<ScoredDoc>& candidates,
        int top_k,
        int num_threads);
};

} // namespace hybrid
