#pragma once

#include "common.h"
#include "agent_memory.h"
#include <vector>
#include <unordered_map>
#include <cmath>

namespace hybrid {

// Agent-memory-aware score fusion.
//
// Extends basic RRF with:
//   1. Temporal decay: recent memories score higher
//   2. Importance weighting: salient memories are boosted
//   3. Session/role filtering: pre-filter before fusion
//   4. Parallel top-k with metadata
class MemoryFusion {
public:
    struct Config {
        int     rrf_k           = 60;
        float   recency_weight  = 0.3f;    // blend: (1-w)*relevance + w*recency
        float   decay_halflife_ms = 86400000.0f * 30.0f; // 30 days
        bool    boost_importance = true;
    };

    // Fuse sparse + dense results with recency and importance scoring.
    // `records` is the full memory store (indexed by DocID).
    // `current_time_ms` is the query timestamp for recency computation.
    static std::vector<ScoredMemory> fuse(
        const std::vector<ScoredDoc>& sparse_results,
        const std::vector<ScoredDoc>& dense_results,
        const std::vector<MemoryRecord>& records,
        const MemoryQuery& query,
        uint64_t current_time_ms,
        int top_k,
        const Config& config);

    // Pre-filter memory records by session, agent, role, time range.
    // Returns a set of valid DocIDs.
    static std::vector<DocID> apply_filters(
        const std::vector<MemoryRecord>& records,
        const MemoryQuery& query);

private:
    // Exponential decay: score = exp(-lambda * age)
    static float compute_recency(uint64_t record_time, uint64_t current_time,
                                  float halflife_ms) {
        if (record_time >= current_time) return 1.0f;
        double age = static_cast<double>(current_time - record_time);
        double lambda = 0.693147 / halflife_ms; // ln(2) / halflife
        return static_cast<float>(std::exp(-lambda * age));
    }
};

} // namespace hybrid
