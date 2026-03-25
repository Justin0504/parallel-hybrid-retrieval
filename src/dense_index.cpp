#include "dense_index.h"
#include <omp.h>
#include <stdexcept>

namespace hybrid {

DenseIndex::DenseIndex(int dim, size_t max_elements, int M, int ef_construction)
    : dim_(dim)
{
    space_ = std::make_unique<hnswlib::L2Space>(dim);
    hnsw_  = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), max_elements, M, ef_construction);
    // Allow multi-threaded insertion
    hnsw_->setEf(200);
}

DenseIndex::~DenseIndex() = default;

void DenseIndex::build(const std::vector<Document>& corpus) {
    if (corpus.empty()) return;

    // First element must be added single-threaded (hnswlib requirement).
    hnsw_->addPoint(corpus[0].embedding.data(), corpus[0].id);

    // Remaining elements added in parallel.
    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 1; i < corpus.size(); ++i) {
        hnsw_->addPoint(corpus[i].embedding.data(), corpus[i].id);
    }
}

std::vector<ScoredDoc> DenseIndex::query(const std::vector<float>& embedding, int top_k) const {
    auto result = hnsw_->searchKnn(embedding.data(), top_k);

    std::vector<ScoredDoc> scored;
    scored.reserve(result.size());

    while (!result.empty()) {
        auto [dist, label] = result.top();
        result.pop();
        // Convert L2 distance to similarity score: 1/(1+dist)
        float score = 1.0f / (1.0f + dist);
        scored.push_back({static_cast<DocID>(label), score});
    }

    // Results come from a max-heap by distance, reverse to get best-first.
    std::reverse(scored.begin(), scored.end());
    return scored;
}

void DenseIndex::set_ef(int ef) {
    hnsw_->setEf(ef);
}

} // namespace hybrid
