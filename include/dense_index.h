#pragma once

#include "common.h"
#include <hnswlib/hnswlib.h>
#include <memory>
#include <vector>

namespace hybrid {

// Dense ANN index backed by hnswlib (HNSW graph, L2 distance).
//
// Thread safety: query() is safe for concurrent calls after build().
// hnswlib's searchKnn is internally thread-safe for reads.
class DenseIndex {
public:
    // `dim` = embedding dimension, `max_elements` = max corpus size.
    // `ef_construction` and `M` are HNSW build parameters.
    DenseIndex(int dim, size_t max_elements,
               int M = 16, int ef_construction = 200);

    ~DenseIndex();

    // Build the HNSW index from corpus embeddings. Parallelized with OpenMP.
    void build(const std::vector<Document>& corpus);

    // Query the index for `top_k` nearest neighbors.
    std::vector<ScoredDoc> query(const std::vector<float>& embedding, int top_k) const;

    // Set ef (search-time parameter). Higher ef = more accurate but slower.
    void set_ef(int ef);

    int dim() const { return dim_; }

private:
    int                                          dim_;
    std::unique_ptr<hnswlib::L2Space>            space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
};

} // namespace hybrid
