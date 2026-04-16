#pragma once
//
// GPU-accelerated BM25 sparse retrieval for hybrid agent memory pipeline.
//
// Design:
//   - CPU builds the inverted index once (SparseIndex::build), then we
//     extract a flat, GPU-friendly CSR representation (term_offsets,
//     posting_doc_ids, posting_tfs, term_idfs).
//   - Queries are tokenized on CPU (cheap), term IDs packed into a
//     [num_queries x max_terms] matrix, then shipped to device.
//   - Kernel 1 (score_bm25): one block per (query, term) pair. Threads
//     in the block cooperatively iterate the term's posting list and
//     atomicAdd BM25 contributions into a per-query score accumulator.
//   - Kernel 2 (top_k): reduce per-query score row to top-k via
//     CUB/thrust radix sort (v1) or warp bitonic (v2 optimization).
//
// This header is safe to include from normal C++ TUs (no __device__
// qualifiers or CUDA-only types leak through). The .cu file owns all
// device memory and CUDA handles.
//

#include "sparse_index.h"
#include "common.h"

#include <memory>
#include <string>
#include <vector>

namespace hybrid {

struct GpuQueryStats {
    double h2d_ms           = 0.0;  // host-to-device transfer
    double kernel_score_ms  = 0.0;  // BM25 scoring kernel
    double kernel_topk_ms   = 0.0;  // top-k selection kernel
    double d2h_ms           = 0.0;  // device-to-host transfer
    double total_ms         = 0.0;
    size_t total_postings   = 0;    // for throughput accounting
    int    device_id        = 0;
    std::string device_name = "";
};

// -----------------------------------------------------------------------------
// GPU sparse index — holds a CSR copy of the CPU index on device memory.
// Lifetime: build once after SparseIndex is populated; free on destruction.
// -----------------------------------------------------------------------------
class SparseIndexGPU {
public:
    SparseIndexGPU();
    ~SparseIndexGPU();

    SparseIndexGPU(const SparseIndexGPU&)            = delete;
    SparseIndexGPU& operator=(const SparseIndexGPU&) = delete;

    // Upload the CPU-built index to device memory. Call once.
    // Throws std::runtime_error on CUDA failure.
    void upload(const SparseIndex& cpu_index);

    // Batched query: score `queries.size()` queries in parallel on GPU,
    // return per-query top-k results.
    //
    // `top_k` = number of results per query.
    // Returns: vector of length queries.size(), each entry has top_k results.
    std::vector<std::vector<ScoredDoc>> query_batch(
        const std::vector<std::string>& queries,
        int top_k);

    // Same as above, but the caller has already tokenized queries
    // (useful for sharing the tokenizer with the CPU pipeline).
    std::vector<std::vector<ScoredDoc>> query_batch_tokenized(
        const std::vector<std::vector<std::string>>& tokenized_queries,
        int top_k);

    // Stats from the most recent batch call.
    const GpuQueryStats& last_stats() const { return last_stats_; }

    // True iff a CUDA device is available and the index was uploaded.
    bool ready() const;

    size_t num_docs()  const;
    size_t num_terms() const;

private:
    struct Impl;                 // CUDA state lives in the .cu file
    std::unique_ptr<Impl> pimpl_;
    GpuQueryStats last_stats_{};
};

} // namespace hybrid
