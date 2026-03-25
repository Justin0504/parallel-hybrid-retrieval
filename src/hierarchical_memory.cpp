#include "hierarchical_memory.h"
#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <functional>

namespace hybrid {

HierarchicalMemory::HierarchicalMemory(const Config& config) : config_(config) {}

// ============================================================================
// Initialization
// ============================================================================

void HierarchicalMemory::init(const std::vector<MemoryRecord>& records) {
    // Partition records: last N go to working, rest to episodic.
    size_t n = records.size();
    size_t working_count = std::min(n, config_.working_capacity);

    {
        std::unique_lock lock(episodic_mutex_);
        episodic_records_.assign(records.begin(), records.begin() + (n - working_count));
    }
    {
        std::unique_lock lock(working_mutex_);
        working_.clear();
        for (size_t i = n - working_count; i < n; ++i) {
            working_.push_back(records[i]);
        }
    }

    rebuild_indices();

    // Run initial consolidation to populate semantic memory.
    consolidate();
}

void HierarchicalMemory::rebuild_indices() {
    rebuild_episodic_index();
    rebuild_semantic_index();
}

void HierarchicalMemory::rebuild_episodic_index() {
    std::shared_lock lock(episodic_mutex_);

    std::vector<Document> docs;
    docs.reserve(episodic_records_.size());
    for (const auto& r : episodic_records_) {
        docs.push_back(r.to_document());
    }

    auto sp = std::make_unique<SparseIndex>();
    sp->build(docs);

    auto dn = std::make_unique<DenseIndex>(config_.embedding_dim,
        std::max(episodic_records_.size(), size_t(1)));
    if (!docs.empty()) dn->build(docs);

    lock.unlock();
    std::unique_lock wlock(episodic_mutex_);
    episodic_sparse_ = std::move(sp);
    episodic_dense_  = std::move(dn);
}

void HierarchicalMemory::rebuild_semantic_index() {
    std::shared_lock lock(semantic_mutex_);

    std::vector<Document> docs;
    docs.reserve(semantic_entries_.size());
    for (const auto& e : semantic_entries_) {
        docs.push_back({e.id, e.content, e.embedding});
    }

    if (docs.empty()) {
        lock.unlock();
        std::unique_lock wlock(semantic_mutex_);
        semantic_sparse_.reset();
        semantic_dense_.reset();
        return;
    }

    auto sp = std::make_unique<SparseIndex>();
    sp->build(docs);

    auto dn = std::make_unique<DenseIndex>(config_.embedding_dim,
        std::max(docs.size(), size_t(1)));
    dn->build(docs);

    lock.unlock();
    std::unique_lock wlock(semantic_mutex_);
    semantic_sparse_ = std::move(sp);
    semantic_dense_  = std::move(dn);
}

// ============================================================================
// Write operations
// ============================================================================

void HierarchicalMemory::add_interaction(MemoryRecord record) {
    std::unique_lock lock(working_mutex_);
    working_.push_back(std::move(record));

    if (working_.size() > config_.working_capacity) {
        spill_working_to_episodic();
    }
}

void HierarchicalMemory::new_session() {
    std::unique_lock lock(working_mutex_);

    if (working_.empty()) return;

    // Flush ALL working memory to episodic (not just half like overflow spill).
    std::vector<MemoryRecord> flushed;
    flushed.reserve(working_.size());
    while (!working_.empty()) {
        flushed.push_back(std::move(working_.front()));
        working_.pop_front();
    }

    {
        std::unique_lock elock(episodic_mutex_);
        for (auto& r : flushed) {
            r.id = static_cast<DocID>(episodic_records_.size());
            episodic_records_.push_back(std::move(r));
        }
    }

    lock.unlock();
    rebuild_episodic_index();
}

void HierarchicalMemory::spill_working_to_episodic() {
    // Move oldest half of working memory to episodic.
    size_t to_spill = working_.size() / 2;
    if (to_spill == 0) return;

    std::vector<MemoryRecord> spilled;
    spilled.reserve(to_spill);
    for (size_t i = 0; i < to_spill; ++i) {
        spilled.push_back(std::move(working_.front()));
        working_.pop_front();
    }

    {
        std::unique_lock elock(episodic_mutex_);
        for (auto& r : spilled) {
            r.id = static_cast<DocID>(episodic_records_.size());
            episodic_records_.push_back(std::move(r));
        }
    }

    // Rebuild episodic index (could be optimized with incremental updates).
    rebuild_episodic_index();
}

// ============================================================================
// Tier-specific search
// ============================================================================

std::vector<HierarchicalResult> HierarchicalMemory::search_working(
    const MemoryQuery& query, int top_k) const
{
    std::shared_lock lock(working_mutex_);

    // Working memory is small — linear scan with simple text matching.
    // Tokenize query for keyword matching.
    Tokenizer tok;
    auto query_tokens = tok.tokenize(query.text);

    std::vector<HierarchicalResult> results;
    results.reserve(working_.size());

    for (const auto& rec : working_) {
        // Simple keyword overlap score.
        auto doc_tokens = tok.tokenize(rec.content);
        std::unordered_set<std::string> doc_set(doc_tokens.begin(), doc_tokens.end());

        float match_score = 0.0f;
        for (const auto& qt : query_tokens) {
            if (doc_set.count(qt)) match_score += 1.0f;
        }
        if (match_score == 0.0f) continue;

        // Normalize by query length.
        match_score /= std::max(1.0f, static_cast<float>(query_tokens.size()));

        // Boost by recency within working memory (position-based).
        float position_boost = 1.0f; // all working memory is "recent"

        HierarchicalResult hr;
        hr.id = rec.id;
        hr.tier = MemoryTier::WORKING;
        hr.score = match_score * config_.working_weight * position_boost;
        hr.tier_weight = config_.working_weight;
        hr.content_preview = rec.content.substr(0, 80);
        results.push_back(hr);
    }

    // Top-k.
    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          [](const HierarchicalResult& a, const HierarchicalResult& b) {
                              return a.score > b.score;
                          });
        results.resize(top_k);
    } else {
        std::sort(results.begin(), results.end(),
                  [](const HierarchicalResult& a, const HierarchicalResult& b) {
                      return a.score > b.score;
                  });
    }

    return results;
}

std::vector<HierarchicalResult> HierarchicalMemory::search_semantic(
    const MemoryQuery& query, int top_k) const
{
    std::shared_lock lock(semantic_mutex_);

    if (!semantic_sparse_ || semantic_entries_.empty()) return {};

    auto sparse_results = semantic_sparse_->query(query.text, top_k * 2);
    auto dense_results  = semantic_dense_->query(query.embedding, top_k * 2);

    // RRF merge.
    std::unordered_map<DocID, float> scores;
    for (size_t r = 0; r < sparse_results.size(); ++r)
        scores[sparse_results[r].id] += 1.0f / (60 + r + 1);
    for (size_t r = 0; r < dense_results.size(); ++r)
        scores[dense_results[r].id] += 1.0f / (60 + r + 1);

    std::vector<HierarchicalResult> results;
    results.reserve(scores.size());
    for (const auto& [id, score] : scores) {
        if (id >= semantic_entries_.size()) continue;
        HierarchicalResult hr;
        hr.id = id;
        hr.tier = MemoryTier::SEMANTIC;
        // Boost by confidence and source count.
        float confidence_boost = semantic_entries_[id].confidence;
        hr.score = score * config_.semantic_weight * (1.0f + confidence_boost * 0.5f);
        hr.tier_weight = config_.semantic_weight;
        hr.content_preview = semantic_entries_[id].content.substr(0, 80);
        results.push_back(hr);
    }

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          [](const HierarchicalResult& a, const HierarchicalResult& b) {
                              return a.score > b.score;
                          });
        results.resize(top_k);
    }

    return results;
}

std::vector<HierarchicalResult> HierarchicalMemory::search_episodic(
    const MemoryQuery& query, int top_k) const
{
    std::shared_lock lock(episodic_mutex_);

    if (!episodic_sparse_ || episodic_records_.empty()) return {};

    auto sparse_results = episodic_sparse_->query(query.text, top_k * 2);
    auto dense_results  = episodic_dense_->query(query.embedding, top_k * 2);

    std::unordered_map<DocID, float> scores;
    for (size_t r = 0; r < sparse_results.size(); ++r)
        scores[sparse_results[r].id] += 1.0f / (60 + r + 1);
    for (size_t r = 0; r < dense_results.size(); ++r)
        scores[dense_results[r].id] += 1.0f / (60 + r + 1);

    std::vector<HierarchicalResult> results;
    results.reserve(scores.size());
    for (const auto& [id, score] : scores) {
        if (id >= episodic_records_.size()) continue;
        HierarchicalResult hr;
        hr.id = id;
        hr.tier = MemoryTier::EPISODIC;
        hr.score = score * config_.episodic_weight;
        hr.tier_weight = config_.episodic_weight;
        hr.content_preview = episodic_records_[id].content.substr(0, 80);
        results.push_back(hr);
    }

    if (static_cast<int>(results.size()) > top_k) {
        std::partial_sort(results.begin(), results.begin() + top_k, results.end(),
                          [](const HierarchicalResult& a, const HierarchicalResult& b) {
                              return a.score > b.score;
                          });
        results.resize(top_k);
    }

    return results;
}

// ============================================================================
// Merge tier results with priority
// ============================================================================

std::vector<HierarchicalResult> HierarchicalMemory::merge_tier_results(
    std::vector<HierarchicalResult>& working_results,
    std::vector<HierarchicalResult>& semantic_results,
    std::vector<HierarchicalResult>& episodic_results,
    int top_k) const
{
    std::vector<HierarchicalResult> merged;
    merged.reserve(working_results.size() + semantic_results.size() + episodic_results.size());

    merged.insert(merged.end(), working_results.begin(), working_results.end());
    merged.insert(merged.end(), semantic_results.begin(), semantic_results.end());
    merged.insert(merged.end(), episodic_results.begin(), episodic_results.end());

    // Sort by final score (tier weights already applied).
    std::sort(merged.begin(), merged.end(),
              [](const HierarchicalResult& a, const HierarchicalResult& b) {
                  return a.score > b.score;
              });

    if (static_cast<int>(merged.size()) > top_k) {
        merged.resize(top_k);
    }

    return merged;
}

// ============================================================================
// Sequential retrieval
// ============================================================================

std::vector<HierarchicalResult> HierarchicalMemory::retrieve_sequential(
    const MemoryQuery& query, uint64_t current_time, int top_k) const
{
    auto w_results = search_working(query, top_k);
    auto s_results = search_semantic(query, top_k);
    auto e_results = search_episodic(query, top_k);

    return merge_tier_results(w_results, s_results, e_results, top_k);
}

// ============================================================================
// Parallel retrieval — all three tiers searched concurrently
// ============================================================================

std::vector<HierarchicalResult> HierarchicalMemory::retrieve_parallel(
    const MemoryQuery& query, uint64_t current_time, int top_k) const
{
    std::vector<HierarchicalResult> w_results, s_results, e_results;

    #pragma omp parallel sections num_threads(3)
    {
        #pragma omp section
        { w_results = search_working(query, top_k); }

        #pragma omp section
        { s_results = search_semantic(query, top_k); }

        #pragma omp section
        { e_results = search_episodic(query, top_k); }
    }

    return merge_tier_results(w_results, s_results, e_results, top_k);
}

// ============================================================================
// Consolidation: episodic -> semantic
//
// Strategy: group episodic records by content similarity (simple keyword
// overlap), merge groups into single semantic entries with aggregated
// embeddings and confidence scores.
// ============================================================================

int HierarchicalMemory::consolidate() {
    std::shared_lock elock(episodic_mutex_);
    if (episodic_records_.empty()) return 0;

    Tokenizer tok;

    // Group records by their primary keywords (simple clustering).
    // Hash each record's top-2 terms as a cluster key.
    std::unordered_map<std::string, std::vector<size_t>> clusters;

    size_t start = 0;
    if (episodic_records_.size() > static_cast<size_t>(config_.consolidation_batch * 2)) {
        start = episodic_records_.size() - config_.consolidation_batch;
    }

    for (size_t i = start; i < episodic_records_.size(); ++i) {
        // Only consolidate assistant responses and tool outputs (high info density).
        if (episodic_records_[i].role != MemoryRole::ASSISTANT &&
            episodic_records_[i].role != MemoryRole::TOOL_OUTPUT) continue;

        auto tokens = tok.tokenize(episodic_records_[i].content);
        if (tokens.size() < 3) continue;

        // Use first two meaningful tokens as cluster key.
        std::string key = tokens[0] + "|" + tokens[1];
        clusters[key].push_back(i);
    }

    elock.unlock();

    // Create semantic entries from clusters with 2+ members.
    std::vector<SemanticEntry> new_entries;

    for (const auto& [key, indices] : clusters) {
        if (indices.size() < 2) continue;

        std::shared_lock lock2(episodic_mutex_);

        SemanticEntry entry;
        entry.id = static_cast<DocID>(semantic_entries_.size() + new_entries.size());
        entry.source_count = static_cast<int>(indices.size());
        entry.confidence = std::min(1.0f, static_cast<float>(indices.size()) / 5.0f);
        entry.last_accessed = 0;

        // Use the longest record's content as the consolidated text.
        size_t best_idx = indices[0];
        size_t best_len = 0;
        for (size_t idx : indices) {
            if (episodic_records_[idx].content.size() > best_len) {
                best_len = episodic_records_[idx].content.size();
                best_idx = idx;
            }
            entry.source_ids.push_back(episodic_records_[idx].id);
        }
        entry.content = episodic_records_[best_idx].content;

        // Average embedding.
        int dim = config_.embedding_dim;
        entry.embedding.resize(dim, 0.0f);
        for (size_t idx : indices) {
            const auto& emb = episodic_records_[idx].embedding;
            for (int d = 0; d < dim && d < static_cast<int>(emb.size()); ++d) {
                entry.embedding[d] += emb[d];
            }
        }
        float norm = 0.0f;
        for (float v : entry.embedding) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float& v : entry.embedding) v /= norm;
        }

        new_entries.push_back(std::move(entry));
    }

    if (new_entries.empty()) return 0;

    // Add to semantic memory.
    {
        std::unique_lock slock(semantic_mutex_);
        for (auto& e : new_entries) {
            e.id = static_cast<DocID>(semantic_entries_.size());
            semantic_entries_.push_back(std::move(e));
        }
    }

    rebuild_semantic_index();

    int count = static_cast<int>(new_entries.size());
    consolidation_runs_.fetch_add(1);
    total_consolidated_.fetch_add(count);
    return count;
}

// ============================================================================
// Decay and forgetting
// ============================================================================

int HierarchicalMemory::decay(uint64_t current_time) {
    std::unique_lock lock(episodic_mutex_);

    if (episodic_records_.empty()) return 0;

    // Mark records for eviction based on age and importance.
    float halflife = config_.decay_halflife_ms;
    float threshold = config_.forget_threshold;

    std::vector<bool> keep(episodic_records_.size(), true);
    int evicted = 0;

    for (size_t i = 0; i < episodic_records_.size(); ++i) {
        const auto& rec = episodic_records_[i];

        // Compute decay factor.
        double age = static_cast<double>(current_time - rec.timestamp_ms);
        double lambda = 0.693147 / halflife;
        float decay = static_cast<float>(std::exp(-lambda * age));

        // Effective importance = base importance * decay.
        float effective = rec.importance * decay;

        if (effective < threshold) {
            keep[i] = false;
            evicted++;
        }
    }

    if (evicted == 0) return 0;

    // Compact: keep only non-evicted records.
    std::vector<MemoryRecord> surviving;
    surviving.reserve(episodic_records_.size() - evicted);
    for (size_t i = 0; i < episodic_records_.size(); ++i) {
        if (keep[i]) {
            MemoryRecord r = std::move(episodic_records_[i]);
            r.id = static_cast<DocID>(surviving.size());
            surviving.push_back(std::move(r));
        }
    }
    episodic_records_ = std::move(surviving);

    lock.unlock();
    rebuild_episodic_index();

    total_forgotten_.fetch_add(evicted);
    return evicted;
}

// ============================================================================
// Stats
// ============================================================================

HierarchicalMemory::Stats HierarchicalMemory::stats() const {
    Stats s;
    {
        std::shared_lock lock(working_mutex_);
        s.working_size = working_.size();
    }
    {
        std::shared_lock lock(semantic_mutex_);
        s.semantic_size = semantic_entries_.size();
    }
    {
        std::shared_lock lock(episodic_mutex_);
        s.episodic_size = episodic_records_.size();
    }
    s.total_size = s.working_size + s.semantic_size + s.episodic_size;
    s.consolidation_runs = consolidation_runs_.load();
    s.total_consolidated = total_consolidated_.load();
    s.total_forgotten = total_forgotten_.load();
    return s;
}

} // namespace hybrid
