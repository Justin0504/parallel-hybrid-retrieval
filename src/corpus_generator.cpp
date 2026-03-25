#include "corpus_generator.h"
#include <omp.h>
#include <random>
#include <sstream>

namespace hybrid {

CorpusGenerator::CorpusGenerator(int dim, unsigned seed)
    : dim_(dim), seed_(seed)
{
    init_vocabulary();
}

void CorpusGenerator::init_vocabulary() {
    // Fixed vocabulary of ~200 terms for synthetic document generation.
    vocabulary_ = {
        "agent", "memory", "retrieval", "query", "index", "vector", "sparse",
        "dense", "embedding", "search", "score", "rank", "document", "token",
        "pipeline", "parallel", "thread", "fusion", "candidate", "result",
        "model", "neural", "network", "attention", "transformer", "layer",
        "weight", "gradient", "loss", "training", "inference", "batch",
        "latency", "throughput", "scalability", "speedup", "efficiency",
        "cache", "memory", "bandwidth", "compute", "kernel", "operation",
        "system", "database", "storage", "disk", "buffer", "queue",
        "algorithm", "structure", "tree", "graph", "hash", "table",
        "function", "module", "interface", "library", "framework", "tool",
        "process", "schedule", "synchronize", "lock", "mutex", "atomic",
        "barrier", "reduce", "scatter", "gather", "broadcast", "communicate",
        "optimize", "profile", "benchmark", "measure", "analyze", "evaluate",
        "accuracy", "precision", "recall", "relevance", "similarity", "distance",
        "cluster", "partition", "shard", "replicate", "distribute", "balance",
        "request", "response", "service", "endpoint", "protocol", "format",
        "encode", "decode", "compress", "decompress", "serialize", "parse",
        "context", "session", "state", "history", "interaction", "feedback",
        "policy", "strategy", "heuristic", "approximate", "exact", "hybrid",
        "inverted", "posting", "frequency", "term", "vocabulary", "corpus",
        "semantic", "lexical", "syntactic", "morphological", "phonetic",
        "classification", "regression", "generation", "translation", "summary",
        "extraction", "recognition", "detection", "segmentation", "tracking",
        "planning", "reasoning", "learning", "adaptation", "exploration",
        "environment", "observation", "action", "reward", "episode", "trajectory",
        "simulation", "experiment", "configuration", "parameter", "hyperparameter",
        "convergence", "stability", "robustness", "generalization", "overfitting",
        "regularization", "normalization", "initialization", "activation",
        "convolution", "pooling", "recurrent", "sequential", "bidirectional",
        "encoder", "decoder", "discriminator", "generator", "critic",
        "objective", "constraint", "feasible", "optimal", "solution",
        "iteration", "epoch", "step", "update", "momentum", "adaptive",
        "stochastic", "deterministic", "random", "uniform", "gaussian",
        "distribution", "probability", "likelihood", "posterior", "prior",
        "marginal", "conditional", "independent", "correlated", "covariance",
        "matrix", "tensor", "scalar", "dimension", "projection", "subspace"
    };
}

std::vector<Document> CorpusGenerator::generate_corpus(size_t n, int words_per_doc) {
    std::vector<Document> corpus(n);
    size_t vocab_size = vocabulary_.size();

    #pragma omp parallel
    {
        // Per-thread RNG for reproducibility.
        unsigned thread_seed = seed_ + static_cast<unsigned>(omp_get_thread_num()) * 1000;
        std::mt19937 rng(thread_seed);
        std::uniform_int_distribution<size_t> word_dist(0, vocab_size - 1);
        std::normal_distribution<float> vec_dist(0.0f, 1.0f);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            // Re-seed per document for reproducibility across thread counts.
            std::mt19937 doc_rng(seed_ + static_cast<unsigned>(i));
            std::uniform_int_distribution<size_t> dw(0, vocab_size - 1);
            std::normal_distribution<float> dv(0.0f, 1.0f);

            Document doc;
            doc.id = static_cast<DocID>(i);

            // Generate random text.
            std::string text;
            text.reserve(words_per_doc * 10);
            for (int w = 0; w < words_per_doc; ++w) {
                if (w > 0) text += ' ';
                text += vocabulary_[dw(doc_rng)];
            }
            doc.text = std::move(text);

            // Generate random embedding.
            doc.embedding.resize(dim_);
            for (int d = 0; d < dim_; ++d) {
                doc.embedding[d] = dv(doc_rng);
            }

            // Normalize embedding to unit length.
            float norm = 0.0f;
            for (float v : doc.embedding) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (float& v : doc.embedding) v /= norm;
            }

            corpus[i] = std::move(doc);
        }
    }

    return corpus;
}

std::vector<Query> CorpusGenerator::generate_queries(size_t n, int words_per_query) {
    std::vector<Query> queries(n);
    size_t vocab_size = vocabulary_.size();

    std::mt19937 rng(seed_ + 999999);
    std::uniform_int_distribution<size_t> word_dist(0, vocab_size - 1);
    std::normal_distribution<float> vec_dist(0.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        Query q;

        // Random query text.
        std::string text;
        for (int w = 0; w < words_per_query; ++w) {
            if (w > 0) text += ' ';
            text += vocabulary_[word_dist(rng)];
        }
        q.text = std::move(text);

        // Random query embedding.
        q.embedding.resize(dim_);
        float norm = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            q.embedding[d] = vec_dist(rng);
            norm += q.embedding[d] * q.embedding[d];
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& v : q.embedding) v /= norm;
        }

        queries[i] = std::move(q);
    }

    return queries;
}

} // namespace hybrid
