#pragma once

#include "common.h"
#include "agent_memory.h"
#include "sparse_index.h"
#include "dense_index.h"
#include "memory_fusion.h"

#include <vector>
#include <deque>
#include <shared_mutex>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace hybrid {

// ============================================================================
// Hierarchical Agent Memory — three-tier architecture
//
// Mirrors cognitive science models of human memory:
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │  WORKING MEMORY (Tier 0)                                   │
//   │  Current session buffer. Small, fast, linear scan.          │
//   │  Capacity: ~100 records. No index needed.                  │
//   │  Analogy: CPU L1 cache / human short-term memory           │
//   ├─────────────────────────────────────────────────────────────┤
//   │  SEMANTIC MEMORY (Tier 1)                                  │
//   │  Consolidated knowledge extracted from episodes.            │
//   │  Medium size, high signal-to-noise. Indexed.               │
//   │  Analogy: CPU L2 cache / human semantic memory             │
//   ├─────────────────────────────────────────────────────────────┤
//   │  EPISODIC MEMORY (Tier 2)                                  │
//   │  Full interaction history. Large, indexed.                  │
//   │  Subject to decay and forgetting.                          │
//   │  Analogy: Main memory / human episodic memory              │
//   └─────────────────────────────────────────────────────────────┘
//
// Parallel retrieval searches all three tiers concurrently (task parallelism).
// Background consolidation and decay run as concurrent writers.
// ============================================================================

enum class MemoryTier : uint8_t {
    WORKING  = 0,   // current session, linear scan
    SEMANTIC = 1,   // consolidated knowledge, indexed
    EPISODIC = 2,   // full history, indexed
};

inline const char* tier_to_string(MemoryTier t) {
    switch (t) {
        case MemoryTier::WORKING:  return "working";
        case MemoryTier::SEMANTIC: return "semantic";
        case MemoryTier::EPISODIC: return "episodic";
    }
    return "unknown";
}

// A semantic memory entry — consolidated from multiple episodic records.
struct SemanticEntry {
    DocID               id;
    std::string         content;       // consolidated text
    std::vector<float>  embedding;
    float               confidence;    // [0,1] how well-established
    int                 source_count;  // how many episodes contributed
    uint64_t            last_accessed; // for LRU-style boosting
    std::vector<DocID>  source_ids;    // episodic IDs this was derived from
};

// Result from hierarchical retrieval with tier attribution.
struct HierarchicalResult {
    DocID       id;
    MemoryTier  tier;
    float       score;
    float       tier_weight;    // how much this tier contributed
    std::string content_preview;
};

// ============================================================================
// HierarchicalMemory
// ============================================================================

class HierarchicalMemory {
public:
    struct Config {
        int     embedding_dim       = 128;
        size_t  working_capacity    = 100;    // max records in working memory
        size_t  semantic_capacity   = 10000;  // max entries in semantic memory
        size_t  episodic_capacity   = 500000; // max records in episodic memory
        float   decay_halflife_ms   = 86400000.0f * 30.0f; // 30 days
        float   forget_threshold    = 0.05f;  // importance below this -> evict
        int     consolidation_batch = 100;    // records per consolidation pass
        // Tier weights for retrieval fusion.
        float   working_weight      = 1.5f;   // boost current context
        float   semantic_weight     = 1.2f;   // consolidated knowledge is reliable
        float   episodic_weight     = 1.0f;   // raw history, baseline weight
    };

    explicit HierarchicalMemory(const Config& config);

    // ---- Write operations ----

    // Add a new interaction to working memory.
    // When working memory overflows, oldest records spill to episodic.
    void add_interaction(MemoryRecord record);

    // Start a new session: flush working memory to episodic.
    void new_session();

    // ---- Read operations ----

    // Sequential retrieval: search tiers one by one.
    std::vector<HierarchicalResult> retrieve_sequential(
        const MemoryQuery& query, uint64_t current_time, int top_k) const;

    // Parallel retrieval: search all three tiers concurrently.
    std::vector<HierarchicalResult> retrieve_parallel(
        const MemoryQuery& query, uint64_t current_time, int top_k) const;

    // ---- Background operations ----

    // Consolidation: scan episodic memory for recurring patterns,
    // merge into semantic entries. Returns number of entries created.
    int consolidate();

    // Decay: scan episodic memory, evict low-importance old records.
    // Returns number of records evicted.
    int decay(uint64_t current_time);

    // ---- Lifecycle ----

    // Initialize from a corpus of agent memory records.
    void init(const std::vector<MemoryRecord>& records);

    // Rebuild indices after bulk modifications.
    void rebuild_indices();

    // ---- Stats ----

    struct Stats {
        size_t working_size;
        size_t semantic_size;
        size_t episodic_size;
        size_t total_size;
        int    consolidation_runs;
        int    total_consolidated;
        int    total_forgotten;
    };

    Stats stats() const;

private:
    Config config_;

    // Tier 0: Working memory — simple deque, no index.
    std::deque<MemoryRecord> working_;
    mutable std::shared_mutex working_mutex_;

    // Tier 1: Semantic memory — consolidated knowledge.
    std::vector<SemanticEntry>          semantic_entries_;
    std::unique_ptr<SparseIndex>        semantic_sparse_;
    std::unique_ptr<DenseIndex>         semantic_dense_;
    mutable std::shared_mutex           semantic_mutex_;

    // Tier 2: Episodic memory — full history.
    std::vector<MemoryRecord>           episodic_records_;
    std::unique_ptr<SparseIndex>        episodic_sparse_;
    std::unique_ptr<DenseIndex>         episodic_dense_;
    mutable std::shared_mutex           episodic_mutex_;

    // Stats tracking.
    std::atomic<int> consolidation_runs_{0};
    std::atomic<int> total_consolidated_{0};
    std::atomic<int> total_forgotten_{0};

    // Internal helpers.
    std::vector<HierarchicalResult> search_working(
        const MemoryQuery& query, int top_k) const;
    std::vector<HierarchicalResult> search_semantic(
        const MemoryQuery& query, int top_k) const;
    std::vector<HierarchicalResult> search_episodic(
        const MemoryQuery& query, int top_k) const;

    std::vector<HierarchicalResult> merge_tier_results(
        std::vector<HierarchicalResult>& working_results,
        std::vector<HierarchicalResult>& semantic_results,
        std::vector<HierarchicalResult>& episodic_results,
        int top_k) const;

    void spill_working_to_episodic();
    void rebuild_episodic_index();
    void rebuild_semantic_index();
};

} // namespace hybrid
