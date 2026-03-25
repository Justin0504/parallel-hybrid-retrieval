#include "tokenizer.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace hybrid {

Tokenizer::Tokenizer()
    : stopwords_{"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "shall", "can",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after",
                  "and", "but", "or", "nor", "not", "so", "yet",
                  "it", "its", "this", "that", "these", "those"} {}

bool Tokenizer::is_stopword(const std::string& w) const {
    // Linear scan is fine for ~50 stopwords; called per-token.
    for (const auto& sw : stopwords_) {
        if (w == sw) return true;
    }
    return false;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    tokens.reserve(32);

    std::string word;
    word.reserve(32);

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            word += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!word.empty()) {
            if (!is_stopword(word)) {
                tokens.push_back(std::move(word));
            }
            word.clear();
        }
    }
    if (!word.empty() && !is_stopword(word)) {
        tokens.push_back(std::move(word));
    }

    return tokens;
}

} // namespace hybrid
