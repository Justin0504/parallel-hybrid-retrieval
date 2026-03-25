#pragma once

#include "agent_memory.h"
#include <vector>
#include <string>
#include <random>

namespace hybrid {

// Generates synthetic but realistic agent interaction histories.
//
// Produces multi-turn sessions with natural conversation flow:
//   user question → agent planning → tool call → tool output → agent response
//
// Covers diverse agent tasks: research, coding, data analysis, DevOps, etc.
class AgentCorpusGenerator {
public:
    AgentCorpusGenerator(int embedding_dim, unsigned seed = 42);

    // Generate a complete agent memory store.
    // `num_sessions`  — number of distinct conversation sessions
    // `turns_per_session` — average turns per session (varies +/- 50%)
    std::vector<MemoryRecord> generate(int num_sessions, int turns_per_session = 8);

    // Generate agent-context-aware queries.
    std::vector<MemoryQuery> generate_queries(int n, const std::vector<MemoryRecord>& corpus,
                                               float recency_weight = 0.3f, int top_k = 10);

private:
    int      dim_;
    unsigned seed_;
    std::mt19937 rng_;

    // Template pools for realistic content generation.
    struct TaskTemplate {
        std::string domain;
        std::vector<std::string> user_queries;
        std::vector<std::string> planning_thoughts;
        std::vector<std::string> tool_calls;
        std::vector<std::string> tool_outputs;
        std::vector<std::string> assistant_responses;
        std::vector<ToolType>    tools_used;
    };

    std::vector<TaskTemplate>    task_templates_;
    std::vector<std::string>     agent_ids_;
    std::vector<std::string>     system_prompts_;

    void init_templates();
    std::vector<float> make_embedding(const std::string& text);
    std::string make_session_id(int idx);
    uint64_t make_timestamp(int session_idx, int turn_idx);
};

} // namespace hybrid
