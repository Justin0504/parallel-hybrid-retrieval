#pragma once

#include "common.h"
#include <vector>
#include <cstddef>

namespace hybrid {

// Generates synthetic corpora and query batches for benchmarking.
// Documents have random text (sampled from a fixed vocabulary) and
// random dense embeddings.
class CorpusGenerator {
public:
    // `dim` = embedding dimension, `seed` = RNG seed for reproducibility.
    CorpusGenerator(int dim, unsigned seed = 42);

    // Generate `n` documents with random text and embeddings.
    std::vector<Document> generate_corpus(size_t n, int words_per_doc = 50);

    // Generate `n` queries from a subset of corpus vocabulary.
    std::vector<Query> generate_queries(size_t n, int words_per_query = 5);

private:
    int      dim_;
    unsigned seed_;
    std::vector<std::string> vocabulary_;

    void init_vocabulary();
};

} // namespace hybrid
