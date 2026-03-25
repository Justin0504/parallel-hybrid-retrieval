#pragma once

#include "common.h"
#include "agent_memory.h"
#include "sparse_index.h"
#include <vector>
#include <memory>

namespace hybrid {

// Time-partitioned sparse index for agent memory.
//
// Shards the inverted index into time-window partitions (e.g., per-week).
// For recency-biased queries, searches only recent partitions first and
// expands to older partitions only if top-k isn't filled with high-confidence
// results. This exploits the temporal locality of agent memory access.
//
// Key insight: 80%+ of agent memory queries target recent interactions.
// By partitioning, we search 10-20% of the index for most queries.
class TemporalIndex {
public:
    struct Config {
        uint64_t partition_width_ms = 604800000ULL; // 7 days per partition
        int      max_partitions_to_search = 4;      // expand up to N partitions
        float    early_stop_threshold = 0.8f;        // stop if top-k avg score > threshold * max_possible
    };

    TemporalIndex();
    explicit TemporalIndex(const Config& config);

    // Build from agent memory records.
    void build(const std::vector<MemoryRecord>& records);

    // Query with temporal early stopping.
    // Searches most recent partition first, expands backward if needed.
    struct TemporalResult {
        std::vector<ScoredDoc> results;
        int    partitions_searched;
        int    total_partitions;
        size_t records_searched;
        size_t total_records;
        double search_fraction; // records_searched / total_records
    };

    TemporalResult query(const std::string& query_text, int top_k) const;

    // Parallel version.
    TemporalResult query_parallel(const std::string& query_text, int top_k,
                                   int num_threads) const;

    int num_partitions() const { return static_cast<int>(partitions_.size()); }

private:
    struct Partition {
        uint64_t    time_start;
        uint64_t    time_end;
        size_t      num_records;
        SparseIndex index;
        // Map from partition-local DocID to global DocID.
        std::vector<DocID> local_to_global;
    };

    Config                      config_;
    std::vector<Partition>      partitions_; // ordered by time (oldest first)
    uint64_t                    global_time_min_ = 0;
    uint64_t                    global_time_max_ = 0;
};

} // namespace hybrid
