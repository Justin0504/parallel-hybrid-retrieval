#pragma once

#include "common.h"
#include <string>
#include <vector>

namespace hybrid {

// Simple whitespace tokenizer with lowercasing and stopword removal.
// Sufficient for BM25 on synthetic corpora; not meant to be NLP-grade.
class Tokenizer {
public:
    Tokenizer();

    // Tokenize a raw text string into terms.
    std::vector<std::string> tokenize(const std::string& text) const;

private:
    std::vector<std::string> stopwords_;
    bool is_stopword(const std::string& w) const;
};

} // namespace hybrid
