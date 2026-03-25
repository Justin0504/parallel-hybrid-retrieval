#include "memory_store.h"
#include <omp.h>
#include <algorithm>

namespace hybrid {

MemoryStore::MemoryStore(const Config& config)
    : config_(config)
{
    records_.reserve(config.max_capacity);
    staging_.reserve(config.flush_threshold * 2);
}

void MemoryStore::init(const std::vector<MemoryRecord>& initial_records) {
    std::unique_lock lock(rw_mutex_);

    records_ = initial_records;
    record_count_.store(records_.size());

    rebuild_indices();
    indices_built_.store(true);
}

void MemoryStore::append(MemoryRecord record) {
    std::unique_lock lock(rw_mutex_);

    record.id = static_cast<DocID>(records_.size() + staging_.size());
    staging_.push_back(std::move(record));

    // Auto-flush when staging buffer is full.
    if (static_cast<int>(staging_.size()) >= config_.flush_threshold) {
        // Move staging into main records.
        for (auto& r : staging_) {
            records_.push_back(std::move(r));
        }
        staging_.clear();
        record_count_.store(records_.size());

        // Rebuild indices with the full record set.
        rebuild_indices();
    }
}

void MemoryStore::flush() {
    std::unique_lock lock(rw_mutex_);

    if (staging_.empty()) return;

    for (auto& r : staging_) {
        records_.push_back(std::move(r));
    }
    staging_.clear();
    record_count_.store(records_.size());

    rebuild_indices();
}

std::vector<ScoredMemory> MemoryStore::retrieve(
    const MemoryQuery& query, uint64_t current_time_ms)
{
    std::shared_lock lock(rw_mutex_);

    if (!indices_built_.load() || records_.empty()) {
        return {};
    }

    // Query main indices.
    auto sparse_results = sparse_->query(query.text, config_.sparse_candidates);
    auto dense_results  = dense_->query(query.embedding, config_.dense_candidates);

    // Fuse with recency and importance weighting.
    return MemoryFusion::fuse(
        sparse_results, dense_results,
        records_, query, current_time_ms,
        query.top_k > 0 ? query.top_k : config_.top_k,
        config_.fusion_config);
}

size_t MemoryStore::staged_records() const {
    std::shared_lock lock(rw_mutex_);
    return staging_.size();
}

const MemoryRecord& MemoryStore::get_record(DocID id) const {
    std::shared_lock lock(rw_mutex_);
    return records_.at(id);
}

void MemoryStore::rebuild_indices() {
    // Convert MemoryRecords to Documents for indexing.
    std::vector<Document> docs;
    docs.reserve(records_.size());
    for (const auto& r : records_) {
        docs.push_back(r.to_document());
    }

    sparse_ = std::make_unique<SparseIndex>();
    sparse_->build(docs);

    dense_ = std::make_unique<DenseIndex>(config_.embedding_dim, config_.max_capacity);
    dense_->build(docs);
}

} // namespace hybrid
