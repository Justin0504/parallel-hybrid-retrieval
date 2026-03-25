#include "fusion.h"
#include <omp.h>
#include <algorithm>
#include <unordered_map>

namespace hybrid {

std::vector<ScoredDoc> Fusion::rrf(
    const std::vector<ScoredDoc>& sparse_results,
    const std::vector<ScoredDoc>& dense_results,
    int top_k,
    int rrf_k)
{
    std::unordered_map<DocID, float> fused_scores;
    fused_scores.reserve(sparse_results.size() + dense_results.size());

    // RRF: score(d) = sum over lists of 1/(k + rank(d))
    for (size_t rank = 0; rank < sparse_results.size(); ++rank) {
        fused_scores[sparse_results[rank].id] += 1.0f / (rrf_k + rank + 1);
    }
    for (size_t rank = 0; rank < dense_results.size(); ++rank) {
        fused_scores[dense_results[rank].id] += 1.0f / (rrf_k + rank + 1);
    }

    std::vector<ScoredDoc> results;
    results.reserve(fused_scores.size());
    for (const auto& [id, score] : fused_scores) {
        results.push_back({id, score});
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

std::vector<ScoredDoc> Fusion::parallel_top_k(
    std::vector<ScoredDoc>& candidates,
    int top_k,
    int num_threads)
{
    if (candidates.empty()) return {};

    size_t n = candidates.size();
    if (static_cast<int>(n) <= top_k) {
        std::sort(candidates.begin(), candidates.end(), std::greater<ScoredDoc>());
        return candidates;
    }

    // Each thread finds local top-k from its partition.
    std::vector<std::vector<ScoredDoc>> local_results(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk = n / nthreads;
        size_t start = tid * chunk;
        size_t end   = (tid == nthreads - 1) ? n : start + chunk;

        // Extract local partition.
        std::vector<ScoredDoc> local(candidates.begin() + start,
                                     candidates.begin() + end);

        int local_k = std::min(top_k, static_cast<int>(local.size()));
        std::partial_sort(local.begin(), local.begin() + local_k, local.end(),
                          std::greater<ScoredDoc>());
        local.resize(local_k);
        local_results[tid] = std::move(local);
    }

    // Merge all local top-k results.
    std::vector<ScoredDoc> merged;
    merged.reserve(num_threads * top_k);
    for (auto& lr : local_results) {
        merged.insert(merged.end(), lr.begin(), lr.end());
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
