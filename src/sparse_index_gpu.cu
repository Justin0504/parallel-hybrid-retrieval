// GPU-accelerated BM25 sparse retrieval kernel.
//
// Two-kernel pipeline per batch of queries:
//   (1) bm25_score_kernel — one CUDA block per (query, query-term) pair.
//       Threads cooperatively scan the term's posting list and
//       atomicAdd BM25 contributions into a per-query score row.
//   (2) top_k_kernel — one CUDA block per query. Each thread holds a
//       small register-resident top-K buffer, then a single thread
//       merges block results.
//
// This is v1 (correctness-first). Optimization notes for v2:
//   - Shared-memory posting list caching for hot terms
//   - Warp-level bitonic top-K instead of naive merge
//   - Multi-stream batching with HOST-side tokenization overlap
//   - Fused score+filter for agent-aware ranking

#include "sparse_index_gpu.h"
#include "sparse_index.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _e = (stmt);                                               \
        if (_e != cudaSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(_e));    \
        }                                                                      \
    } while (0)

namespace hybrid {

// ----------------------------------------------------------------------------
// Device kernels
// ----------------------------------------------------------------------------

// Kernel 1: BM25 scoring.
// Launch: grid=(num_queries, max_terms_per_query), block=(BLOCK_SIZE).
// Each block processes one (query, term_slot) pair.
__global__ void bm25_score_kernel(
    const int32_t* __restrict__  query_term_ids,    // [num_queries * max_terms]
    const int32_t* __restrict__  query_term_counts, // [num_queries]
    const uint32_t* __restrict__ posting_offsets,   // [num_terms + 1]
    const uint32_t* __restrict__ posting_doc_ids,   // [total_postings]
    const float*    __restrict__ posting_tfs,       // [total_postings]
    const float*    __restrict__ term_idfs,         // [num_terms]
    const float*    __restrict__ doc_lengths,       // [num_docs]
    float avg_doc_len, float k1, float b,
    int max_terms_per_query,
    int num_docs,
    float* __restrict__ scores)                     // [num_queries * num_docs]
{
    const int qid       = blockIdx.x;
    const int term_slot = blockIdx.y;

    if (term_slot >= query_term_counts[qid]) return;

    const int32_t term_id = query_term_ids[qid * max_terms_per_query + term_slot];
    if (term_id < 0) return;  // OOV term

    const uint32_t start = posting_offsets[term_id];
    const uint32_t end   = posting_offsets[term_id + 1];
    const float    idf   = term_idfs[term_id];

    float* score_row = scores + static_cast<size_t>(qid) * num_docs;

    // Threads stride across the posting list.
    for (uint32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        const uint32_t doc_id = posting_doc_ids[i];
        const float    tf     = posting_tfs[i];
        const float    dl     = doc_lengths[doc_id];
        const float    num    = tf * (k1 + 1.0f);
        const float    den    = tf + k1 * (1.0f - b + b * dl / avg_doc_len);
        const float    bm25   = idf * num / den;
        atomicAdd(&score_row[doc_id], bm25);
    }
}

// Kernel 2: per-query top-K.
// Launch: grid=(num_queries), block=(BLOCK_SIZE).
// Each thread keeps a sorted-ascending buffer of size MAX_TOPK in registers;
// then merges across threads via shared memory.
//
// Capped at 32 so static shared memory = BLOCK_SIZE*32*8 = 32KB (fits A100
// default 48KB without cudaFuncSetAttribute opt-in). For top_k > 32 use
// the CPU path or the alternative radix-sort kernel (v2).
#define GPU_MAX_TOPK 32

template <int BLOCK_SIZE>
__global__ void top_k_kernel(
    const float* __restrict__ scores,      // [num_queries * num_docs]
    int num_docs,
    int top_k,                             // runtime, <= GPU_MAX_TOPK
    uint32_t* __restrict__ out_ids,        // [num_queries * top_k]
    float*    __restrict__ out_scores)     // [num_queries * top_k]
{
    const int qid = blockIdx.x;
    const float* row = scores + static_cast<size_t>(qid) * num_docs;

    // Per-thread top-K buffer (ascending: index 0 = smallest / most replaceable).
    float local_s[GPU_MAX_TOPK];
    uint32_t local_i[GPU_MAX_TOPK];
    #pragma unroll
    for (int i = 0; i < GPU_MAX_TOPK; ++i) {
        local_s[i] = -1e30f;
        local_i[i] = 0u;
    }

    const int K = top_k;  // local name

    for (int d = threadIdx.x; d < num_docs; d += BLOCK_SIZE) {
        const float s = row[d];
        if (s > local_s[0]) {
            // Replace smallest and sift up while out of order.
            local_s[0] = s;
            local_i[0] = static_cast<uint32_t>(d);
            for (int j = 0; j + 1 < K && local_s[j] > local_s[j + 1]; ++j) {
                float  ts = local_s[j]; local_s[j] = local_s[j + 1]; local_s[j + 1] = ts;
                uint32_t ti = local_i[j]; local_i[j] = local_i[j + 1]; local_i[j + 1] = ti;
            }
        }
    }

    // Block-level merge. Dump each thread's top-K into shared memory,
    // then thread 0 does a K-way selection over BLOCK_SIZE * K candidates.
    __shared__ float    smem_s[BLOCK_SIZE * GPU_MAX_TOPK];
    __shared__ uint32_t smem_i[BLOCK_SIZE * GPU_MAX_TOPK];

    for (int i = 0; i < K; ++i) {
        smem_s[threadIdx.x * GPU_MAX_TOPK + i] = local_s[i];
        smem_i[threadIdx.x * GPU_MAX_TOPK + i] = local_i[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const int total = BLOCK_SIZE * GPU_MAX_TOPK;
        for (int k = 0; k < K; ++k) {
            int   best_idx = -1;
            float best_s   = -1e30f;
            for (int j = 0; j < total; ++j) {
                const float v = smem_s[j];
                if (v > best_s) { best_s = v; best_idx = j; }
            }
            if (best_idx >= 0) {
                out_scores[qid * K + k] = smem_s[best_idx];
                out_ids[qid * K + k]    = smem_i[best_idx];
                smem_s[best_idx] = -1e30f;  // consume
            } else {
                out_scores[qid * K + k] = 0.0f;
                out_ids[qid * K + k]    = 0u;
            }
        }
    }
}

// Simple memset kernel (cheap, avoids pulling in thrust).
__global__ void zero_floats_kernel(float* p, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

// ----------------------------------------------------------------------------
// Host-side implementation
// ----------------------------------------------------------------------------

struct SparseIndexGPU::Impl {
    // Index (uploaded once)
    uint32_t* d_posting_offsets = nullptr;
    uint32_t* d_posting_doc_ids = nullptr;   // DocID is uint32_t
    float*    d_posting_tfs     = nullptr;
    float*    d_term_idfs       = nullptr;
    float*    d_doc_lengths     = nullptr;

    std::unordered_map<std::string, uint32_t> term_to_id;

    float  k1             = 1.2f;
    float  b              = 0.75f;
    float  avg_doc_len    = 0.0f;
    size_t num_docs       = 0;
    size_t num_terms      = 0;
    size_t total_postings = 0;

    // Tokenizer reference (not owned)
    const Tokenizer* tokenizer = nullptr;

    // Per-batch workspace (grown lazily)
    float*    d_scores        = nullptr;
    uint32_t* d_out_ids       = nullptr;
    float*    d_out_scores    = nullptr;
    int32_t*  d_query_terms   = nullptr;
    int32_t*  d_query_counts  = nullptr;

    size_t batch_capacity      = 0;
    int    max_terms_capacity  = 0;
    int    topk_capacity       = 0;

    cudaStream_t stream = nullptr;
    int         device_id   = -1;
    std::string device_name;

    bool ready = false;

    ~Impl() { free_all(); }

    void free_all() {
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        cudaFree(d_posting_offsets); d_posting_offsets = nullptr;
        cudaFree(d_posting_doc_ids); d_posting_doc_ids = nullptr;
        cudaFree(d_posting_tfs);     d_posting_tfs     = nullptr;
        cudaFree(d_term_idfs);       d_term_idfs       = nullptr;
        cudaFree(d_doc_lengths);     d_doc_lengths     = nullptr;
        cudaFree(d_scores);          d_scores          = nullptr;
        cudaFree(d_out_ids);         d_out_ids         = nullptr;
        cudaFree(d_out_scores);      d_out_scores      = nullptr;
        cudaFree(d_query_terms);     d_query_terms     = nullptr;
        cudaFree(d_query_counts);    d_query_counts    = nullptr;
    }

    void ensure_workspace(size_t batch, int max_terms, int top_k) {
        if (batch > batch_capacity || num_docs == 0) {
            cudaFree(d_scores);
            CUDA_CHECK(cudaMalloc(&d_scores, batch * num_docs * sizeof(float)));
            batch_capacity = batch;
        }
        if (max_terms > max_terms_capacity) {
            cudaFree(d_query_terms);
            CUDA_CHECK(cudaMalloc(&d_query_terms, batch_capacity * max_terms * sizeof(int32_t)));
            max_terms_capacity = max_terms;
        }
        if (batch > (size_t)topk_capacity || top_k > topk_capacity) {
            cudaFree(d_out_ids);
            cudaFree(d_out_scores);
            CUDA_CHECK(cudaMalloc(&d_out_ids,    batch * top_k * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_out_scores, batch * top_k * sizeof(float)));
            topk_capacity = top_k;
        }
        if (!d_query_counts || batch > batch_capacity) {
            cudaFree(d_query_counts);
            CUDA_CHECK(cudaMalloc(&d_query_counts, batch_capacity * sizeof(int32_t)));
        }
    }
};

// ----------------------------------------------------------------------------

SparseIndexGPU::SparseIndexGPU()  : pimpl_(std::make_unique<Impl>()) {}
SparseIndexGPU::~SparseIndexGPU() = default;

bool   SparseIndexGPU::ready()     const { return pimpl_ && pimpl_->ready; }
size_t SparseIndexGPU::num_docs()  const { return pimpl_ ? pimpl_->num_docs  : 0; }
size_t SparseIndexGPU::num_terms() const { return pimpl_ ? pimpl_->num_terms : 0; }

void SparseIndexGPU::upload(const SparseIndex& cpu_index) {
    // Pick a device + query its name.
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("No CUDA device found");

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    pimpl_->device_id   = dev;
    pimpl_->device_name = prop.name;

    CUDA_CHECK(cudaStreamCreate(&pimpl_->stream));

    // Flatten.
    SparseIndexCSR csr = cpu_index.flatten();

    pimpl_->k1             = csr.k1;
    pimpl_->b              = csr.b;
    pimpl_->avg_doc_len    = csr.avg_doc_len;
    pimpl_->num_docs       = csr.num_docs;
    pimpl_->num_terms      = csr.num_terms();
    pimpl_->total_postings = csr.total_postings();
    pimpl_->term_to_id     = std::move(csr.term_to_id);
    pimpl_->tokenizer      = &cpu_index.tokenizer();

    // Device allocations + H2D copies.
    CUDA_CHECK(cudaMalloc(&pimpl_->d_posting_offsets, csr.posting_offsets.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_posting_doc_ids, csr.posting_doc_ids.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_posting_tfs,     csr.posting_tfs.size()     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_term_idfs,       csr.term_idfs.size()       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pimpl_->d_doc_lengths,     csr.doc_lengths.size()     * sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_posting_offsets, csr.posting_offsets.data(),
                               csr.posting_offsets.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_posting_doc_ids, csr.posting_doc_ids.data(),
                               csr.posting_doc_ids.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_posting_tfs,     csr.posting_tfs.data(),
                               csr.posting_tfs.size() * sizeof(float),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_term_idfs,       csr.term_idfs.data(),
                               csr.term_idfs.size() * sizeof(float),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_doc_lengths,     csr.doc_lengths.data(),
                               csr.doc_lengths.size() * sizeof(float),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream));

    pimpl_->ready = true;

    std::fprintf(stderr,
        "[GPU] Uploaded index: %zu docs, %zu terms, %zu postings to %s (device %d)\n",
        pimpl_->num_docs, pimpl_->num_terms, pimpl_->total_postings,
        pimpl_->device_name.c_str(), pimpl_->device_id);
}

// ----------------------------------------------------------------------------

std::vector<std::vector<ScoredDoc>> SparseIndexGPU::query_batch(
    const std::vector<std::string>& queries, int top_k)
{
    if (!pimpl_->tokenizer) {
        throw std::runtime_error("SparseIndexGPU::query_batch — index not uploaded");
    }
    std::vector<std::vector<std::string>> toks;
    toks.reserve(queries.size());
    for (const auto& q : queries) toks.push_back(pimpl_->tokenizer->tokenize(q));
    return query_batch_tokenized(toks, top_k);
}

std::vector<std::vector<ScoredDoc>> SparseIndexGPU::query_batch_tokenized(
    const std::vector<std::vector<std::string>>& toks, int top_k)
{
    if (!pimpl_ || !pimpl_->ready) {
        throw std::runtime_error("SparseIndexGPU not initialized (call upload first)");
    }
    if (top_k <= 0 || top_k > GPU_MAX_TOPK) {
        throw std::runtime_error("top_k out of range (1..32 for GPU path)");
    }

    const size_t B  = toks.size();
    if (B == 0) return {};

    // Determine max_terms for this batch (host-side).
    int max_terms = 0;
    for (const auto& tv : toks) max_terms = std::max<int>(max_terms, (int)tv.size());
    max_terms = std::max(1, max_terms);

    // Translate tokens -> term_ids. -1 for OOV / stopwords.
    std::vector<int32_t> h_query_terms(B * max_terms, -1);
    std::vector<int32_t> h_query_counts(B, 0);
    for (size_t q = 0; q < B; ++q) {
        int c = 0;
        for (const auto& t : toks[q]) {
            auto it = pimpl_->term_to_id.find(t);
            if (it == pimpl_->term_to_id.end()) continue;
            h_query_terms[q * max_terms + c] = (int32_t)it->second;
            ++c;
            if (c >= max_terms) break;
        }
        h_query_counts[q] = c;
    }

    pimpl_->ensure_workspace(B, max_terms, top_k);

    cudaEvent_t ev_h2d_start, ev_h2d_end, ev_score_end, ev_topk_end, ev_d2h_end;
    cudaEventCreate(&ev_h2d_start);
    cudaEventCreate(&ev_h2d_end);
    cudaEventCreate(&ev_score_end);
    cudaEventCreate(&ev_topk_end);
    cudaEventCreate(&ev_d2h_end);

    cudaEventRecord(ev_h2d_start, pimpl_->stream);

    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_query_terms, h_query_terms.data(),
                               B * max_terms * sizeof(int32_t),
                               cudaMemcpyHostToDevice, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(pimpl_->d_query_counts, h_query_counts.data(),
                               B * sizeof(int32_t),
                               cudaMemcpyHostToDevice, pimpl_->stream));

    // Zero the score accumulator for this batch.
    const size_t total_score_elts = B * pimpl_->num_docs;
    const int ZERO_BLOCK = 256;
    const size_t ZERO_GRID = (total_score_elts + ZERO_BLOCK - 1) / ZERO_BLOCK;
    zero_floats_kernel<<<(unsigned int)ZERO_GRID, ZERO_BLOCK, 0, pimpl_->stream>>>(
        pimpl_->d_scores, total_score_elts);

    cudaEventRecord(ev_h2d_end, pimpl_->stream);

    // Kernel 1: BM25 scoring.
    dim3 grid((unsigned int)B, (unsigned int)max_terms);
    const int SCORE_BLOCK = 128;
    bm25_score_kernel<<<grid, SCORE_BLOCK, 0, pimpl_->stream>>>(
        pimpl_->d_query_terms,
        pimpl_->d_query_counts,
        pimpl_->d_posting_offsets,
        pimpl_->d_posting_doc_ids,
        pimpl_->d_posting_tfs,
        pimpl_->d_term_idfs,
        pimpl_->d_doc_lengths,
        pimpl_->avg_doc_len, pimpl_->k1, pimpl_->b,
        max_terms,
        (int)pimpl_->num_docs,
        pimpl_->d_scores);

    cudaEventRecord(ev_score_end, pimpl_->stream);

    // Kernel 2: per-query top-K (one block per query).
    const int TOPK_BLOCK = 128;
    top_k_kernel<TOPK_BLOCK><<<(unsigned int)B, TOPK_BLOCK, 0, pimpl_->stream>>>(
        pimpl_->d_scores,
        (int)pimpl_->num_docs,
        top_k,
        pimpl_->d_out_ids,
        pimpl_->d_out_scores);

    cudaEventRecord(ev_topk_end, pimpl_->stream);

    // D2H.
    std::vector<uint32_t> h_ids(B * top_k);
    std::vector<float>    h_scores(B * top_k);
    CUDA_CHECK(cudaMemcpyAsync(h_ids.data(),    pimpl_->d_out_ids,
                               B * top_k * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, pimpl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(h_scores.data(), pimpl_->d_out_scores,
                               B * top_k * sizeof(float),
                               cudaMemcpyDeviceToHost, pimpl_->stream));

    cudaEventRecord(ev_d2h_end, pimpl_->stream);
    CUDA_CHECK(cudaStreamSynchronize(pimpl_->stream));

    // Collect stats.
    float ms_h2d = 0, ms_score = 0, ms_topk = 0, ms_d2h = 0;
    cudaEventElapsedTime(&ms_h2d,   ev_h2d_start, ev_h2d_end);
    cudaEventElapsedTime(&ms_score, ev_h2d_end,   ev_score_end);
    cudaEventElapsedTime(&ms_topk,  ev_score_end, ev_topk_end);
    cudaEventElapsedTime(&ms_d2h,   ev_topk_end,  ev_d2h_end);

    last_stats_.h2d_ms          = ms_h2d;
    last_stats_.kernel_score_ms = ms_score;
    last_stats_.kernel_topk_ms  = ms_topk;
    last_stats_.d2h_ms          = ms_d2h;
    last_stats_.total_ms        = ms_h2d + ms_score + ms_topk + ms_d2h;
    last_stats_.total_postings  = pimpl_->total_postings;
    last_stats_.device_id       = pimpl_->device_id;
    last_stats_.device_name     = pimpl_->device_name;

    cudaEventDestroy(ev_h2d_start);
    cudaEventDestroy(ev_h2d_end);
    cudaEventDestroy(ev_score_end);
    cudaEventDestroy(ev_topk_end);
    cudaEventDestroy(ev_d2h_end);

    // Pack results.
    std::vector<std::vector<ScoredDoc>> out(B);
    for (size_t q = 0; q < B; ++q) {
        out[q].reserve(top_k);
        for (int k = 0; k < top_k; ++k) {
            const float s = h_scores[q * top_k + k];
            if (s <= 0.0f) break;  // exhausted (empty slot)
            out[q].push_back(ScoredDoc{ (DocID)h_ids[q * top_k + k], s });
        }
    }
    return out;
}

} // namespace hybrid
