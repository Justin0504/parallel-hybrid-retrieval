#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace hybrid {

// ============================================================================
// Core types
// ============================================================================

using DocID = uint32_t;

struct ScoredDoc {
    DocID   id;
    float   score;

    bool operator>(const ScoredDoc& o) const { return score > o.score; }
    bool operator<(const ScoredDoc& o) const { return score < o.score; }
};

struct Document {
    DocID                id;
    std::string          text;      // raw text for sparse indexing
    std::vector<float>   embedding; // dense vector for ANN
};

struct Query {
    std::string          text;
    std::vector<float>   embedding;
};

// ============================================================================
// Stage timing breakdown
// ============================================================================

struct StageTimings {
    double preprocess_ms  = 0.0;
    double sparse_ms      = 0.0;
    double dense_ms       = 0.0;
    double fusion_ms      = 0.0;
    double total_ms       = 0.0;
};

// ============================================================================
// High-resolution timer
// ============================================================================

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_   = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

// ============================================================================
// Benchmark result
// ============================================================================

struct BenchmarkResult {
    int    num_threads;
    int    corpus_size;
    int    num_queries;
    int    top_k;
    double total_ms;
    double avg_latency_ms;
    double throughput_qps;
    double speedup;
    double efficiency;
    StageTimings avg_stage;
};

} // namespace hybrid
