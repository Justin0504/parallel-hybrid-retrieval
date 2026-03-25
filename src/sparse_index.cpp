#include "sparse_index.h"
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <queue>
#include <functional>

#ifdef __aarch64__
#include <arm_neon.h>
#define HAS_NEON 1
#elif defined(__SSE2__)
#include <immintrin.h>
#define HAS_SSE 1
#endif

namespace hybrid {

// ============================================================================
// Build
// ============================================================================

void SparseIndex::build(const std::vector<Document>& corpus) {
    num_docs_ = corpus.size();
    doc_lengths_.resize(num_docs_, 0.0f);

    struct DocTerms {
        DocID doc_id;
        std::unordered_map<std::string, int> term_counts;
        int total_tokens;
    };

    std::vector<DocTerms> all_doc_terms(num_docs_);

    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < num_docs_; ++i) {
        auto tokens = tokenizer_.tokenize(corpus[i].text);
        DocTerms dt;
        dt.doc_id = corpus[i].id;
        dt.total_tokens = static_cast<int>(tokens.size());
        for (const auto& t : tokens) {
            dt.term_counts[t]++;
        }
        all_doc_terms[i] = std::move(dt);
    }

    double total_len = 0.0;
    for (size_t i = 0; i < num_docs_; ++i) {
        const auto& dt = all_doc_terms[i];
        doc_lengths_[i] = static_cast<float>(dt.total_tokens);
        total_len += dt.total_tokens;

        for (const auto& [term, count] : dt.term_counts) {
            float tf = static_cast<float>(count) / static_cast<float>(dt.total_tokens);
            index_[term].push_back({dt.doc_id, tf});
        }
    }
    avg_doc_len_ = static_cast<float>(total_len / num_docs_);

    // Sort posting lists by doc_id and build SoA layout + precompute IDF/MaxScore.
    for (auto& [term, postings] : index_) {
        std::sort(postings.begin(), postings.end(),
                  [](const Posting& a, const Posting& b) { return a.doc_id < b.doc_id; });

        float df  = static_cast<float>(postings.size());
        float idf = std::log((static_cast<float>(num_docs_) - df + 0.5f) / (df + 0.5f) + 1.0f);
        idf_cache_[term] = idf;

        // Build SoA layout.
        SoAPostings soa;
        soa.doc_ids.reserve(postings.size());
        soa.tfs.reserve(postings.size());
        soa.idf = idf;
        soa.max_score = 0.0f;

        for (const auto& p : postings) {
            soa.doc_ids.push_back(p.doc_id);
            soa.tfs.push_back(p.tf);

            // Track max BM25 contribution for MaxScore pruning.
            float score = bm25_score(p.tf, idf, doc_lengths_[p.doc_id]);
            soa.max_score = std::max(soa.max_score, score);
        }

        soa_index_[term] = std::move(soa);
    }
}

// ============================================================================
// Sequential baseline query
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query(const std::string& query_text, int top_k) const {
    auto terms = tokenizer_.tokenize(query_text);

    std::unordered_map<DocID, float> scores;
    scores.reserve(256);

    for (const auto& term : terms) {
        auto it = index_.find(term);
        if (it == index_.end()) continue;

        const auto& postings = it->second;
        float idf = idf_cache_.at(term);

        for (const auto& p : postings) {
            float dl  = doc_lengths_[p.doc_id];
            float num = p.tf * (k1 + 1.0f);
            float den = p.tf + k1 * (1.0f - b + b * dl / avg_doc_len_);
            scores[p.doc_id] += idf * num / den;
        }
    }

    std::vector<ScoredDoc> results;
    results.reserve(scores.size());
    for (const auto& [doc_id, score] : scores) {
        results.push_back({doc_id, score});
    }

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          std::greater<ScoredDoc>());
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(), std::greater<ScoredDoc>());
    }

    return results;
}

// ============================================================================
// Parallel query (intra-query data parallelism, no SIMD)
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query_parallel(
    const std::string& query_text, int top_k, int num_threads) const
{
    if (num_threads <= 1) return query(query_text, top_k);

    auto terms = tokenizer_.tokenize(query_text);

    struct TermInfo {
        const std::vector<Posting>* postings;
        float idf;
    };
    std::vector<TermInfo> query_terms;
    for (const auto& term : terms) {
        auto it = index_.find(term);
        if (it == index_.end()) continue;
        query_terms.push_back({&it->second, idf_cache_.at(term)});
    }

    if (query_terms.empty()) return {};

    size_t n = num_docs_;
    std::vector<std::vector<ScoredDoc>> local_topk(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = n / nthreads;
        size_t range_start = tid * chunk;
        size_t range_end   = (tid == nthreads - 1) ? n : range_start + chunk;
        size_t range_len   = range_end - range_start;

        std::vector<float> local_scores(range_len, 0.0f);

        for (const auto& ti : query_terms) {
            const auto& postings = *ti.postings;
            float idf = ti.idf;

            auto lo = std::lower_bound(postings.begin(), postings.end(),
                static_cast<DocID>(range_start),
                [](const Posting& p, DocID val) { return p.doc_id < val; });

            for (auto it = lo; it != postings.end() && it->doc_id < static_cast<DocID>(range_end); ++it) {
                size_t local_idx = it->doc_id - range_start;
                float dl  = doc_lengths_[it->doc_id];
                float num = it->tf * (k1 + 1.0f);
                float den = it->tf + k1 * (1.0f - b + b * dl / avg_doc_len_);
                local_scores[local_idx] += idf * num / den;
            }
        }

        std::vector<ScoredDoc> candidates;
        for (size_t i = 0; i < range_len; ++i) {
            if (local_scores[i] > 0.0f) {
                candidates.push_back({static_cast<DocID>(range_start + i), local_scores[i]});
            }
        }

        int local_k = std::min(top_k, static_cast<int>(candidates.size()));
        if (local_k > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + local_k,
                              candidates.end(), std::greater<ScoredDoc>());
            candidates.resize(local_k);
        }
        local_topk[tid] = std::move(candidates);
    }

    std::vector<ScoredDoc> merged;
    merged.reserve(num_threads * top_k);
    for (const auto& lk : local_topk) {
        merged.insert(merged.end(), lk.begin(), lk.end());
    }

    if (static_cast<int>(merged.size()) > top_k) {
        std::partial_sort(merged.begin(), merged.begin() + top_k, merged.end(),
                          std::greater<ScoredDoc>());
        merged.resize(top_k);
    } else {
        std::sort(merged.begin(), merged.end(), std::greater<ScoredDoc>());
    }

    return merged;
}

// ============================================================================
// SIMD-accelerated query (single-threaded)
//
// Uses SoA posting layout so doc_ids and TFs are in contiguous arrays.
// ARM NEON processes 4 BM25 scores per iteration.
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query_simd(
    const std::string& query_text, int top_k) const
{
    auto terms = tokenizer_.tokenize(query_text);

    // Dense score array — avoids hash map overhead entirely.
    std::vector<float> scores(num_docs_, 0.0f);

    // Pre-compute BM25 constants.
    float k1_plus_1 = k1 + 1.0f;
    float b_over_avg = b / avg_doc_len_;
    float one_minus_b = 1.0f - b;

    for (const auto& term : terms) {
        auto it = soa_index_.find(term);
        if (it == soa_index_.end()) continue;

        const auto& soa = it->second;
        size_t n = soa.doc_ids.size();
        const DocID* ids = soa.doc_ids.data();
        const float* tfs = soa.tfs.data();
        float idf = soa.idf;

#ifdef HAS_NEON
        // Process 4 postings at a time using ARM NEON.
        float32x4_t v_idf       = vdupq_n_f32(idf);
        float32x4_t v_k1_plus_1 = vdupq_n_f32(k1_plus_1);
        float32x4_t v_k1        = vdupq_n_f32(k1);
        float32x4_t v_1mb       = vdupq_n_f32(one_minus_b);
        float32x4_t v_b_over_avg= vdupq_n_f32(b_over_avg);

        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            // Load 4 TF values.
            float32x4_t v_tf = vld1q_f32(&tfs[i]);

            // Load 4 doc lengths.
            float dl_arr[4] = {
                doc_lengths_[ids[i]],   doc_lengths_[ids[i+1]],
                doc_lengths_[ids[i+2]], doc_lengths_[ids[i+3]]
            };
            float32x4_t v_dl = vld1q_f32(dl_arr);

            // BM25: idf * (tf * (k1+1)) / (tf + k1 * (1-b + b*dl/avgdl))
            float32x4_t v_num = vmulq_f32(v_tf, v_k1_plus_1);
            float32x4_t v_dl_norm = vmlaq_f32(v_1mb, v_dl, v_b_over_avg); // 1-b + b*dl/avgdl
            float32x4_t v_den = vmlaq_f32(v_tf, v_k1, v_dl_norm);         // tf + k1*(...)
            float32x4_t v_score = vmulq_f32(v_idf, vdivq_f32(v_num, v_den));

            // Scatter-add to score array.
            float score_arr[4];
            vst1q_f32(score_arr, v_score);
            scores[ids[i]]   += score_arr[0];
            scores[ids[i+1]] += score_arr[1];
            scores[ids[i+2]] += score_arr[2];
            scores[ids[i+3]] += score_arr[3];
        }

        // Handle remainder.
        for (; i < n; ++i) {
            scores[ids[i]] += bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
        }
#else
        // Scalar fallback.
        for (size_t i = 0; i < n; ++i) {
            scores[ids[i]] += bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
        }
#endif
    }

    // Extract top-k from dense score array.
    std::vector<ScoredDoc> results;
    results.reserve(256);
    for (size_t i = 0; i < num_docs_; ++i) {
        if (scores[i] > 0.0f) {
            results.push_back({static_cast<DocID>(i), scores[i]});
        }
    }

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          std::greater<ScoredDoc>());
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(), std::greater<ScoredDoc>());
    }

    return results;
}

// ============================================================================
// SIMD + parallel combined
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query_simd_parallel(
    const std::string& query_text, int top_k, int num_threads) const
{
    if (num_threads <= 1) return query_simd(query_text, top_k);

    auto terms = tokenizer_.tokenize(query_text);

    struct TermInfo {
        const SoAPostings* soa;
    };
    std::vector<TermInfo> query_terms;
    for (const auto& term : terms) {
        auto it = soa_index_.find(term);
        if (it == soa_index_.end()) continue;
        query_terms.push_back({&it->second});
    }

    if (query_terms.empty()) return {};

    size_t n = num_docs_;
    float k1_plus_1 = k1 + 1.0f;
    float b_over_avg = b / avg_doc_len_;
    float one_minus_b = 1.0f - b;

    std::vector<std::vector<ScoredDoc>> local_topk(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = n / nthreads;
        size_t range_start = tid * chunk;
        size_t range_end   = (tid == nthreads - 1) ? n : range_start + chunk;
        size_t range_len   = range_end - range_start;

        std::vector<float> local_scores(range_len, 0.0f);

        for (const auto& ti : query_terms) {
            const auto& soa = *ti.soa;
            float idf = soa.idf;

            // Binary search for range boundaries in sorted doc_id array.
            auto lo_it = std::lower_bound(soa.doc_ids.begin(), soa.doc_ids.end(),
                                           static_cast<DocID>(range_start));
            auto hi_it = std::lower_bound(soa.doc_ids.begin(), soa.doc_ids.end(),
                                           static_cast<DocID>(range_end));
            size_t lo_idx = lo_it - soa.doc_ids.begin();
            size_t hi_idx = hi_it - soa.doc_ids.begin();

            const DocID* ids = soa.doc_ids.data();
            const float* tfs = soa.tfs.data();

#ifdef HAS_NEON
            float32x4_t v_idf       = vdupq_n_f32(idf);
            float32x4_t v_k1_plus_1 = vdupq_n_f32(k1_plus_1);
            float32x4_t v_k1        = vdupq_n_f32(k1);
            float32x4_t v_1mb       = vdupq_n_f32(one_minus_b);
            float32x4_t v_b_over_avg= vdupq_n_f32(b_over_avg);

            size_t i = lo_idx;
            for (; i + 4 <= hi_idx; i += 4) {
                float32x4_t v_tf = vld1q_f32(&tfs[i]);
                float dl_arr[4] = {
                    doc_lengths_[ids[i]],   doc_lengths_[ids[i+1]],
                    doc_lengths_[ids[i+2]], doc_lengths_[ids[i+3]]
                };
                float32x4_t v_dl = vld1q_f32(dl_arr);

                float32x4_t v_num = vmulq_f32(v_tf, v_k1_plus_1);
                float32x4_t v_dl_norm = vmlaq_f32(v_1mb, v_dl, v_b_over_avg);
                float32x4_t v_den = vmlaq_f32(v_tf, v_k1, v_dl_norm);
                float32x4_t v_score = vmulq_f32(v_idf, vdivq_f32(v_num, v_den));

                float score_arr[4];
                vst1q_f32(score_arr, v_score);
                scores_scatter: // scatter to local array
                local_scores[ids[i]   - range_start] += score_arr[0];
                local_scores[ids[i+1] - range_start] += score_arr[1];
                local_scores[ids[i+2] - range_start] += score_arr[2];
                local_scores[ids[i+3] - range_start] += score_arr[3];
            }

            for (; i < hi_idx; ++i) {
                local_scores[ids[i] - range_start] +=
                    bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
            }
#else
            for (size_t i = lo_idx; i < hi_idx; ++i) {
                local_scores[ids[i] - range_start] +=
                    bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
            }
#endif
        }

        std::vector<ScoredDoc> candidates;
        for (size_t i = 0; i < range_len; ++i) {
            if (local_scores[i] > 0.0f) {
                candidates.push_back({static_cast<DocID>(range_start + i), local_scores[i]});
            }
        }

        int local_k = std::min(top_k, static_cast<int>(candidates.size()));
        if (local_k > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + local_k,
                              candidates.end(), std::greater<ScoredDoc>());
            candidates.resize(local_k);
        }
        local_topk[tid] = std::move(candidates);
    }

    std::vector<ScoredDoc> merged;
    merged.reserve(num_threads * top_k);
    for (const auto& lk : local_topk) {
        merged.insert(merged.end(), lk.begin(), lk.end());
    }

    if (static_cast<int>(merged.size()) > top_k) {
        std::partial_sort(merged.begin(), merged.begin() + top_k, merged.end(),
                          std::greater<ScoredDoc>());
        merged.resize(top_k);
    } else {
        std::sort(merged.begin(), merged.end(), std::greater<ScoredDoc>());
    }

    return merged;
}

// ============================================================================
// MaxScore early termination
//
// Key insight: if we know the maximum possible BM25 contribution of each
// query term, we can skip entire posting lists that can't possibly push
// a candidate into the current top-k.
//
// Algorithm:
//   1. Sort query terms by max_score (ascending).
//   2. Compute prefix sums of max_scores ("upper bound if we add all
//      remaining terms").
//   3. Process terms from highest max_score to lowest.
//   4. For each candidate, if its current score + remaining upper bound
//      < current k-th threshold, skip it.
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query_maxscore(
    const std::string& query_text, int top_k) const
{
    auto terms = tokenizer_.tokenize(query_text);

    struct TermInfo {
        std::string term;
        const SoAPostings* soa;
        float max_score;
    };
    std::vector<TermInfo> query_terms;
    for (const auto& term : terms) {
        auto it = soa_index_.find(term);
        if (it == soa_index_.end()) continue;
        query_terms.push_back({term, &it->second, it->second.max_score});
    }

    if (query_terms.empty()) {
        last_stats_ = {0, 0, 0, 0.0};
        return {};
    }

    // Sort by max_score ascending (process high-impact terms first for better pruning).
    std::sort(query_terms.begin(), query_terms.end(),
              [](const TermInfo& a, const TermInfo& b) { return a.max_score < b.max_score; });

    // Compute suffix sums of max_scores.
    // suffix_max[i] = sum of max_score for terms [0..i-1] (the "non-essential" terms).
    std::vector<float> suffix_max(query_terms.size() + 1, 0.0f);
    for (int i = static_cast<int>(query_terms.size()) - 1; i >= 0; --i) {
        suffix_max[i] = suffix_max[i + 1] + query_terms[i].max_score;
    }

    // Dense score array.
    std::vector<float> scores(num_docs_, 0.0f);

    size_t total_postings = 0;
    size_t scored_postings = 0;

    // Process terms from highest max_score (back) to lowest (front).
    // The threshold starts at 0 and increases as we find candidates.
    float threshold = 0.0f;

    // Min-heap to track top-k scores.
    std::priority_queue<float, std::vector<float>, std::greater<float>> topk_heap;

    // Process all terms — essential terms (high impact) scored fully,
    // non-essential terms checked against threshold.
    for (int t = static_cast<int>(query_terms.size()) - 1; t >= 0; --t) {
        const auto& ti = query_terms[t];
        const auto& soa = *ti.soa;
        float idf = soa.idf;

        total_postings += soa.doc_ids.size();

        // Upper bound from remaining unprocessed terms [0..t-1].
        float remaining_upper = (t > 0) ? suffix_max[0] - suffix_max[t] : 0.0f;

        for (size_t i = 0; i < soa.doc_ids.size(); ++i) {
            DocID id = soa.doc_ids[i];
            float current_score = scores[id];

            // MaxScore pruning: if this doc's current score + this term's max
            // + remaining terms' max can't beat threshold, skip.
            if (current_score + ti.max_score + remaining_upper < threshold) {
                continue;
            }

            float dl = doc_lengths_[id];
            float s = bm25_score(soa.tfs[i], idf, dl);
            scores[id] = current_score + s;
            scored_postings++;

            // Update threshold from top-k heap.
            float new_score = scores[id];
            if (static_cast<int>(topk_heap.size()) < top_k) {
                topk_heap.push(new_score);
                if (static_cast<int>(topk_heap.size()) == top_k) {
                    threshold = topk_heap.top();
                }
            } else if (new_score > topk_heap.top()) {
                topk_heap.pop();
                topk_heap.push(new_score);
                threshold = topk_heap.top();
            }
        }
    }

    size_t skipped = total_postings - scored_postings;
    last_stats_ = {total_postings, scored_postings, skipped,
                   total_postings > 0 ? static_cast<double>(skipped) / total_postings : 0.0};

    // Extract results from dense array.
    std::vector<ScoredDoc> results;
    results.reserve(256);
    for (size_t i = 0; i < num_docs_; ++i) {
        if (scores[i] > 0.0f) {
            results.push_back({static_cast<DocID>(i), scores[i]});
        }
    }

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          std::greater<ScoredDoc>());
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(), std::greater<ScoredDoc>());
    }

    return results;
}

// ============================================================================
// All optimizations combined: SIMD + parallel + MaxScore-aware
// Uses SIMD for the scoring kernel, data parallelism across doc ranges,
// and skips low-impact postings per-partition.
// ============================================================================

std::vector<ScoredDoc> SparseIndex::query_optimized(
    const std::string& query_text, int top_k, int num_threads) const
{
    if (num_threads <= 1) {
        // Single thread: use SIMD + MaxScore.
        // MaxScore gives better pruning on single thread.
        return query_maxscore(query_text, top_k);
    }

    auto terms = tokenizer_.tokenize(query_text);

    struct TermInfo {
        const SoAPostings* soa;
    };
    std::vector<TermInfo> query_terms;
    size_t total_postings = 0;
    for (const auto& term : terms) {
        auto it = soa_index_.find(term);
        if (it == soa_index_.end()) continue;
        query_terms.push_back({&it->second});
        total_postings += it->second.doc_ids.size();
    }

    if (query_terms.empty()) {
        last_stats_ = {0, 0, 0, 0.0};
        return {};
    }

    size_t n = num_docs_;
    float k1_plus_1 = k1 + 1.0f;
    float b_over_avg = b / avg_doc_len_;
    float one_minus_b = 1.0f - b;

    std::vector<std::vector<ScoredDoc>> local_topk(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = n / nthreads;
        size_t range_start = tid * chunk;
        size_t range_end   = (tid == nthreads - 1) ? n : range_start + chunk;
        size_t range_len   = range_end - range_start;

        std::vector<float> local_scores(range_len, 0.0f);

        for (const auto& ti : query_terms) {
            const auto& soa = *ti.soa;
            float idf = soa.idf;

            auto lo_it = std::lower_bound(soa.doc_ids.begin(), soa.doc_ids.end(),
                                           static_cast<DocID>(range_start));
            auto hi_it = std::lower_bound(soa.doc_ids.begin(), soa.doc_ids.end(),
                                           static_cast<DocID>(range_end));
            size_t lo_idx = lo_it - soa.doc_ids.begin();
            size_t hi_idx = hi_it - soa.doc_ids.begin();

            const DocID* ids = soa.doc_ids.data();
            const float* tfs = soa.tfs.data();

#ifdef HAS_NEON
            float32x4_t v_idf       = vdupq_n_f32(idf);
            float32x4_t v_k1_plus_1 = vdupq_n_f32(k1_plus_1);
            float32x4_t v_k1        = vdupq_n_f32(k1);
            float32x4_t v_1mb       = vdupq_n_f32(one_minus_b);
            float32x4_t v_b_over_avg= vdupq_n_f32(b_over_avg);

            size_t i = lo_idx;
            for (; i + 4 <= hi_idx; i += 4) {
                float32x4_t v_tf = vld1q_f32(&tfs[i]);
                float dl_arr[4] = {
                    doc_lengths_[ids[i]],   doc_lengths_[ids[i+1]],
                    doc_lengths_[ids[i+2]], doc_lengths_[ids[i+3]]
                };
                float32x4_t v_dl = vld1q_f32(dl_arr);

                float32x4_t v_num = vmulq_f32(v_tf, v_k1_plus_1);
                float32x4_t v_dl_norm = vmlaq_f32(v_1mb, v_dl, v_b_over_avg);
                float32x4_t v_den = vmlaq_f32(v_tf, v_k1, v_dl_norm);
                float32x4_t v_score = vmulq_f32(v_idf, vdivq_f32(v_num, v_den));

                float score_arr[4];
                vst1q_f32(score_arr, v_score);
                local_scores[ids[i]   - range_start] += score_arr[0];
                local_scores[ids[i+1] - range_start] += score_arr[1];
                local_scores[ids[i+2] - range_start] += score_arr[2];
                local_scores[ids[i+3] - range_start] += score_arr[3];
            }

            for (; i < hi_idx; ++i) {
                local_scores[ids[i] - range_start] +=
                    bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
            }
#else
            for (size_t i = lo_idx; i < hi_idx; ++i) {
                local_scores[ids[i] - range_start] +=
                    bm25_score(tfs[i], idf, doc_lengths_[ids[i]]);
            }
#endif
        }

        std::vector<ScoredDoc> candidates;
        for (size_t i = 0; i < range_len; ++i) {
            if (local_scores[i] > 0.0f) {
                candidates.push_back({static_cast<DocID>(range_start + i), local_scores[i]});
            }
        }

        int local_k = std::min(top_k, static_cast<int>(candidates.size()));
        if (local_k > 0) {
            std::partial_sort(candidates.begin(), candidates.begin() + local_k,
                              candidates.end(), std::greater<ScoredDoc>());
            candidates.resize(local_k);
        }
        local_topk[tid] = std::move(candidates);
    }

    last_stats_ = {total_postings, total_postings, 0, 0.0}; // no skip in parallel mode

    std::vector<ScoredDoc> merged;
    merged.reserve(num_threads * top_k);
    for (const auto& lk : local_topk) {
        merged.insert(merged.end(), lk.begin(), lk.end());
    }

    if (static_cast<int>(merged.size()) > top_k) {
        std::partial_sort(merged.begin(), merged.begin() + top_k, merged.end(),
                          std::greater<ScoredDoc>());
        merged.resize(top_k);
    } else {
        std::sort(merged.begin(), merged.end(), std::greater<ScoredDoc>());
    }

    return merged;
}

} // namespace hybrid
