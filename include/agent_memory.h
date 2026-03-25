#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <cstdint>

namespace hybrid {

// ============================================================================
// Agent Memory Record — the fundamental unit of long-term agent memory.
//
// Each record represents one interaction in an agent's history:
// a user message, assistant response, tool invocation, tool output,
// planning note, or system observation.
// ============================================================================

enum class MemoryRole : uint8_t {
    USER        = 0,   // User query / instruction
    ASSISTANT   = 1,   // Agent response
    TOOL_CALL   = 2,   // Agent invoked a tool (function call)
    TOOL_OUTPUT = 3,   // Result returned by a tool
    SYSTEM      = 4,   // System prompt / context injection
    PLANNING    = 5,   // Agent's internal planning / reasoning
    OBSERVATION = 6,   // Environment observation / feedback
};

enum class ToolType : uint8_t {
    NONE         = 0,
    WEB_SEARCH   = 1,
    CODE_EXEC    = 2,
    FILE_READ    = 3,
    FILE_WRITE   = 4,
    API_CALL     = 5,
    DB_QUERY     = 6,
    SHELL_CMD    = 7,
    CALCULATOR   = 8,
    MEMORY_READ  = 9,   // Agent reads its own memory (meta)
};

struct MemoryRecord {
    DocID                id;
    std::string          session_id;     // groups a conversation/task
    std::string          agent_id;       // which agent produced this
    MemoryRole           role;
    ToolType             tool;
    uint64_t             timestamp_ms;   // epoch milliseconds
    std::string          content;        // the actual text
    std::vector<float>   embedding;      // dense vector representation
    float                importance;     // [0,1] salience score

    // Convert to generic Document for index compatibility.
    Document to_document() const {
        return Document{id, content, embedding};
    }
};

// ============================================================================
// Agent Memory Query — a retrieval request with agent-specific filters.
// ============================================================================

struct MemoryQuery {
    std::string          text;           // natural language query
    std::vector<float>   embedding;      // query vector
    std::string          session_filter; // "" = no filter
    std::string          agent_filter;   // "" = no filter
    MemoryRole           role_filter;    // USER/ASSISTANT/etc., or any
    uint64_t             time_start;     // 0 = no lower bound
    uint64_t             time_end;       // 0 = no upper bound
    float                recency_weight; // 0.0 = pure relevance, 1.0 = heavy recency bias
    int                  top_k;

    // Convert to generic Query for pipeline compatibility.
    Query to_query() const {
        return Query{text, embedding};
    }
};

// ============================================================================
// Scored memory result with metadata.
// ============================================================================

struct ScoredMemory {
    DocID       id;
    float       relevance_score;
    float       recency_score;
    float       final_score;
    // Pointer to original record set post-retrieval by caller.
};

// ============================================================================
// Utility: role/tool name conversions.
// ============================================================================

inline const char* role_to_string(MemoryRole r) {
    switch (r) {
        case MemoryRole::USER:        return "user";
        case MemoryRole::ASSISTANT:   return "assistant";
        case MemoryRole::TOOL_CALL:   return "tool_call";
        case MemoryRole::TOOL_OUTPUT: return "tool_output";
        case MemoryRole::SYSTEM:      return "system";
        case MemoryRole::PLANNING:    return "planning";
        case MemoryRole::OBSERVATION: return "observation";
    }
    return "unknown";
}

inline const char* tool_to_string(ToolType t) {
    switch (t) {
        case ToolType::NONE:        return "";
        case ToolType::WEB_SEARCH:  return "web_search";
        case ToolType::CODE_EXEC:   return "code_exec";
        case ToolType::FILE_READ:   return "file_read";
        case ToolType::FILE_WRITE:  return "file_write";
        case ToolType::API_CALL:    return "api_call";
        case ToolType::DB_QUERY:    return "db_query";
        case ToolType::SHELL_CMD:   return "shell_cmd";
        case ToolType::CALCULATOR:  return "calculator";
        case ToolType::MEMORY_READ: return "memory_read";
    }
    return "unknown";
}

} // namespace hybrid
