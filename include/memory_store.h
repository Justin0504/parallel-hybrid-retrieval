#pragma once

#include "common.h"
#include "agent_memory.h"
#include "sparse_index.h"
#include "dense_index.h"
#include "memory_fusion.h"

#include <vector>
#include <shared_mutex>
#include <atomic>

namespace hybrid {

// Thread-safe agent memory store with concurrent read/write support.
//
// Real agent pattern: the agent continuously appends new memories
// (from ongoing interactions) while simultaneously retrieving old ones.
// This creates a concurrent read/write workload that is absent from
// typical batch-retrieval benchmarks.
//
// Design:
//   - Writes append to a staging buffer, periodically flushed to indices
//   - Reads query the main indices (lock-free) + scan the staging buffer
//   - Flush is protected by a write lock; reads use a shared lock
class MemoryStore {
public:
    struct Config {
        int     embedding_dim     = 128;
        size_t  max_capacity      = 2000000;
        int     flush_threshold   = 1000;  // flush staging after N writes
        int     sparse_candidates = 100;
        int     dense_candidates  = 100;
        int     top_k             = 10;
        MemoryFusion::Config fusion_config;
    };

    explicit MemoryStore(const Config& config);
    ~MemoryStore() = default;

    // Initialize from a pre-existing corpus.
    void init(const std::vector<MemoryRecord>& initial_records);

    // Append a new memory record (thread-safe).
    void append(MemoryRecord record);

    // Retrieve top-k memories for a query (thread-safe, concurrent with writes).
    std::vector<ScoredMemory> retrieve(const MemoryQuery& query, uint64_t current_time_ms);

    // Force flush staging buffer to main indices.
    void flush();

    // Stats.
    size_t total_records() const { return record_count_.load(); }
    size_t staged_records() const;

    // Access underlying records (for display purposes).
    const MemoryRecord& get_record(DocID id) const;

private:
    Config                       config_;
    std::vector<MemoryRecord>    records_;       // main store (indexed)
    std::vector<MemoryRecord>    staging_;       // pending writes
    std::unique_ptr<SparseIndex> sparse_;
    std::unique_ptr<DenseIndex>  dense_;

    mutable std::shared_mutex    rw_mutex_;      // protects staging_ and records_
    std::atomic<size_t>          record_count_{0};
    std::atomic<bool>            indices_built_{false};

    void rebuild_indices();
};

} // namespace hybrid
