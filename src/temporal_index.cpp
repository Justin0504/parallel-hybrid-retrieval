#include "temporal_index.h"
#include <algorithm>
#include <unordered_set>

namespace hybrid {

TemporalIndex::TemporalIndex() : config_() {}
TemporalIndex::TemporalIndex(const Config& config) : config_(config) {}

void TemporalIndex::build(const std::vector<MemoryRecord>& records) {
    if (records.empty()) return;

    // Find time range.
    global_time_min_ = records[0].timestamp_ms;
    global_time_max_ = records[0].timestamp_ms;
    for (const auto& r : records) {
        global_time_min_ = std::min(global_time_min_, r.timestamp_ms);
        global_time_max_ = std::max(global_time_max_, r.timestamp_ms);
    }

    // Determine partition boundaries.
    uint64_t width = config_.partition_width_ms;
    uint64_t range = global_time_max_ - global_time_min_ + 1;
    int num_parts = static_cast<int>((range + width - 1) / width);
    if (num_parts < 1) num_parts = 1;

    // Assign records to partitions.
    std::vector<std::vector<size_t>> part_record_indices(num_parts);
    for (size_t i = 0; i < records.size(); ++i) {
        int p = static_cast<int>((records[i].timestamp_ms - global_time_min_) / width);
        if (p >= num_parts) p = num_parts - 1;
        part_record_indices[p].push_back(i);
    }

    // Build per-partition indices.
    partitions_.resize(num_parts);
    for (int p = 0; p < num_parts; ++p) {
        auto& part = partitions_[p];
        part.time_start = global_time_min_ + p * width;
        part.time_end   = part.time_start + width;
        part.num_records = part_record_indices[p].size();

        if (part.num_records == 0) continue;

        // Build documents for this partition with local IDs.
        std::vector<Document> docs;
        docs.reserve(part.num_records);
        part.local_to_global.reserve(part.num_records);

        for (size_t idx : part_record_indices[p]) {
            DocID local_id = static_cast<DocID>(docs.size());
            docs.push_back({local_id, records[idx].content, records[idx].embedding});
            part.local_to_global.push_back(records[idx].id);
        }

        part.index.build(docs);
    }
}

TemporalIndex::TemporalResult TemporalIndex::query(
    const std::string& query_text, int top_k) const
{
    TemporalResult result;
    result.total_partitions = static_cast<int>(partitions_.size());
    result.total_records = 0;
    for (const auto& p : partitions_) result.total_records += p.num_records;
    result.records_searched = 0;
    result.partitions_searched = 0;

    // Search from most recent partition backward.
    std::vector<ScoredDoc> all_candidates;
    all_candidates.reserve(top_k * config_.max_partitions_to_search);

    for (int p = static_cast<int>(partitions_.size()) - 1;
         p >= 0 && result.partitions_searched < config_.max_partitions_to_search;
         --p)
    {
        const auto& part = partitions_[p];
        if (part.num_records == 0) continue;

        auto local_results = part.index.query(query_text, top_k);

        // Map local DocIDs back to global.
        for (auto& sd : local_results) {
            sd.id = part.local_to_global[sd.id];
            all_candidates.push_back(sd);
        }

        result.records_searched += part.num_records;
        result.partitions_searched++;

        // Early stopping: if we have enough high-quality results, stop expanding.
        if (static_cast<int>(all_candidates.size()) >= top_k) {
            std::partial_sort(all_candidates.begin(),
                              all_candidates.begin() + top_k,
                              all_candidates.end(),
                              std::greater<ScoredDoc>());

            // Check if the k-th result is good enough to stop.
            float kth_score = all_candidates[top_k - 1].score;
            float best_score = all_candidates[0].score;
            if (best_score > 0 && kth_score / best_score > config_.early_stop_threshold) {
                break; // Good enough, no need to search older partitions.
            }
        }
    }

    // Final top-k.
    if (static_cast<int>(all_candidates.size()) > top_k) {
        std::partial_sort(all_candidates.begin(), all_candidates.begin() + top_k,
                          all_candidates.end(), std::greater<ScoredDoc>());
        all_candidates.resize(top_k);
    } else {
        std::sort(all_candidates.begin(), all_candidates.end(), std::greater<ScoredDoc>());
    }

    result.results = std::move(all_candidates);
    result.search_fraction = result.total_records > 0
        ? static_cast<double>(result.records_searched) / result.total_records : 0.0;

    return result;
}

TemporalIndex::TemporalResult TemporalIndex::query_parallel(
    const std::string& query_text, int top_k, int num_threads) const
{
    TemporalResult result;
    result.total_partitions = static_cast<int>(partitions_.size());
    result.total_records = 0;
    for (const auto& p : partitions_) result.total_records += p.num_records;
    result.records_searched = 0;
    result.partitions_searched = 0;

    // Determine which partitions to search (most recent N).
    std::vector<int> parts_to_search;
    for (int p = static_cast<int>(partitions_.size()) - 1;
         p >= 0 && static_cast<int>(parts_to_search.size()) < config_.max_partitions_to_search;
         --p)
    {
        if (partitions_[p].num_records > 0) {
            parts_to_search.push_back(p);
        }
    }

    int n_parts = static_cast<int>(parts_to_search.size());
    std::vector<std::vector<ScoredDoc>> part_results(n_parts);
    std::vector<size_t> part_records(n_parts, 0);

    // Search partitions in parallel.
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int i = 0; i < n_parts; ++i) {
        int p = parts_to_search[i];
        const auto& part = partitions_[p];

        auto local_results = part.index.query_parallel(query_text, top_k,
            std::max(1, num_threads / n_parts));

        for (auto& sd : local_results) {
            sd.id = part.local_to_global[sd.id];
        }

        part_results[i] = std::move(local_results);
        part_records[i] = part.num_records;
    }

    // Merge.
    std::vector<ScoredDoc> all_candidates;
    for (int i = 0; i < n_parts; ++i) {
        all_candidates.insert(all_candidates.end(),
                              part_results[i].begin(), part_results[i].end());
        result.records_searched += part_records[i];
    }
    result.partitions_searched = n_parts;

    if (static_cast<int>(all_candidates.size()) > top_k) {
        std::partial_sort(all_candidates.begin(), all_candidates.begin() + top_k,
                          all_candidates.end(), std::greater<ScoredDoc>());
        all_candidates.resize(top_k);
    } else {
        std::sort(all_candidates.begin(), all_candidates.end(), std::greater<ScoredDoc>());
    }

    result.results = std::move(all_candidates);
    result.search_fraction = result.total_records > 0
        ? static_cast<double>(result.records_searched) / result.total_records : 0.0;

    return result;
}

} // namespace hybrid
