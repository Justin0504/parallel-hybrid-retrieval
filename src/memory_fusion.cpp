#include "memory_fusion.h"
#include <omp.h>
#include <algorithm>
#include <unordered_set>

namespace hybrid {

std::vector<DocID> MemoryFusion::apply_filters(
    const std::vector<MemoryRecord>& records,
    const MemoryQuery& query)
{
    std::vector<DocID> valid;
    valid.reserve(records.size());

    bool has_session = !query.session_filter.empty();
    bool has_agent   = !query.agent_filter.empty();
    bool has_role    = (static_cast<uint8_t>(query.role_filter) != 255);
    bool has_time    = (query.time_start > 0 || query.time_end > 0);

    for (const auto& r : records) {
        if (has_session && r.session_id != query.session_filter) continue;
        if (has_agent && r.agent_id != query.agent_filter) continue;
        if (has_role && r.role != query.role_filter) continue;
        if (has_time) {
            if (query.time_start > 0 && r.timestamp_ms < query.time_start) continue;
            if (query.time_end > 0 && r.timestamp_ms > query.time_end) continue;
        }
        valid.push_back(r.id);
    }

    return valid;
}

std::vector<ScoredMemory> MemoryFusion::fuse(
    const std::vector<ScoredDoc>& sparse_results,
    const std::vector<ScoredDoc>& dense_results,
    const std::vector<MemoryRecord>& records,
    const MemoryQuery& query,
    uint64_t current_time_ms,
    int top_k,
    const Config& config)
{
    // Step 1: Build filter set if any filters are active.
    std::unordered_set<DocID> filter_set;
    bool use_filter = !query.session_filter.empty() ||
                      !query.agent_filter.empty() ||
                      (static_cast<uint8_t>(query.role_filter) != 255) ||
                      query.time_start > 0 || query.time_end > 0;

    if (use_filter) {
        auto valid = apply_filters(records, query);
        filter_set.insert(valid.begin(), valid.end());
    }

    // Step 2: RRF scoring with filter.
    std::unordered_map<DocID, float> rrf_scores;
    rrf_scores.reserve(sparse_results.size() + dense_results.size());

    auto add_rrf = [&](const std::vector<ScoredDoc>& results) {
        for (size_t rank = 0; rank < results.size(); ++rank) {
            DocID id = results[rank].id;
            if (use_filter && filter_set.find(id) == filter_set.end()) continue;
            rrf_scores[id] += 1.0f / (config.rrf_k + rank + 1);
        }
    };

    add_rrf(sparse_results);
    add_rrf(dense_results);

    // Step 3: Apply recency decay and importance boost.
    float rw = query.recency_weight;

    std::vector<ScoredMemory> results;
    results.reserve(rrf_scores.size());

    for (const auto& [id, rrf_score] : rrf_scores) {
        if (id >= records.size()) continue;
        const auto& rec = records[id];

        float recency = compute_recency(rec.timestamp_ms, current_time_ms,
                                         config.decay_halflife_ms);
        float relevance = rrf_score;

        // Blend relevance and recency.
        float blended = (1.0f - rw) * relevance + rw * recency * relevance;

        // Importance boost: multiply by (1 + importance * 0.5).
        if (config.boost_importance) {
            blended *= (1.0f + rec.importance * 0.5f);
        }

        ScoredMemory sm;
        sm.id = id;
        sm.relevance_score = relevance;
        sm.recency_score = recency;
        sm.final_score = blended;
        results.push_back(sm);
    }

    // Step 4: Top-k selection.
    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          [](const ScoredMemory& a, const ScoredMemory& b) {
                              return a.final_score > b.final_score;
                          });
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(),
                  [](const ScoredMemory& a, const ScoredMemory& b) {
                      return a.final_score > b.final_score;
                  });
    }

    return results;
}

} // namespace hybrid
