#pragma once

#include "common.h"
#include "tokenizer.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace hybrid {

// Inverted index with BM25 scoring for sparse lexical retrieval.
//
// Optimizations over naive implementation:
//   1. SoA posting layout for SIMD-friendly memory access
//   2. SIMD-accelerated BM25 scoring (ARM NEON / SSE)
//   3. MaxScore early termination (skip low-impact posting lists)
//   4. Intra-query data parallelism (partitioned scoring)
//   5. Pre-computed IDF cache
//   6. Sorted posting lists for binary-search partitioning
class SparseIndex {
public:
    static constexpr float k1 = 1.2f;
    static constexpr float b  = 0.75f;

    SparseIndex() = default;

    void build(const std::vector<Document>& corpus);

    // Sequential baseline query.
    std::vector<ScoredDoc> query(const std::string& query_text, int top_k) const;

    // Parallel query with intra-query data parallelism.
    std::vector<ScoredDoc> query_parallel(const std::string& query_text,
                                           int top_k, int num_threads) const;

    // SIMD-accelerated query (single-threaded).
    std::vector<ScoredDoc> query_simd(const std::string& query_text, int top_k) const;

    // SIMD + parallel combined.
    std::vector<ScoredDoc> query_simd_parallel(const std::string& query_text,
                                                int top_k, int num_threads) const;

    // MaxScore early-termination query.
    std::vector<ScoredDoc> query_maxscore(const std::string& query_text, int top_k) const;

    // All optimizations combined: SIMD + MaxScore + parallel.
    std::vector<ScoredDoc> query_optimized(const std::string& query_text,
                                            int top_k, int num_threads) const;

    size_t num_docs() const { return num_docs_; }

    // Stats for benchmarking.
    struct QueryStats {
        size_t total_postings;     // total postings across all query terms
        size_t scored_postings;    // postings actually scored (after MaxScore skip)
        size_t skipped_postings;   // postings skipped by MaxScore
        double skip_rate;          // skipped / total
    };
    mutable QueryStats last_stats_ = {};

private:
    // Structure-of-Arrays posting list for SIMD-friendly access.
    struct SoAPostings {
        std::vector<DocID> doc_ids;   // aligned, contiguous
        std::vector<float> tfs;       // aligned, contiguous
        float idf;                    // pre-computed IDF for this term
        float max_score;              // pre-computed max BM25 contribution
    };

    // AoS posting for compatibility.
    struct Posting {
        DocID doc_id;
        float tf;
    };

    std::unordered_map<std::string, SoAPostings> soa_index_;
    std::unordered_map<std::string, std::vector<Posting>> index_;  // kept for baseline
    std::unordered_map<std::string, float> idf_cache_;

    std::vector<float> doc_lengths_;
    float              avg_doc_len_ = 0.0f;
    size_t             num_docs_    = 0;
    Tokenizer          tokenizer_;

    // BM25 score for a single posting.
    inline float bm25_score(float tf, float idf, float dl) const {
        float num = tf * (k1 + 1.0f);
        float den = tf + k1 * (1.0f - b + b * dl / avg_doc_len_);
        return idf * num / den;
    }
};

} // namespace hybrid
