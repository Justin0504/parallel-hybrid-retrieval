#include "agent_corpus_generator.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <functional>

namespace hybrid {

AgentCorpusGenerator::AgentCorpusGenerator(int embedding_dim, unsigned seed)
    : dim_(embedding_dim), seed_(seed), rng_(seed)
{
    init_templates();

    agent_ids_ = {
        "research-agent-01", "code-agent-02", "devops-agent-03",
        "data-agent-04", "support-agent-05", "planning-agent-06"
    };

    system_prompts_ = {
        "You are a research assistant. Help the user find and synthesize information from multiple sources.",
        "You are a senior software engineer. Help the user write, debug, and review code.",
        "You are a DevOps engineer. Help the user manage infrastructure, deployments, and monitoring.",
        "You are a data analyst. Help the user explore datasets, run queries, and build visualizations.",
        "You are a customer support agent. Help resolve user issues by checking documentation and logs.",
        "You are a project manager. Help the user plan tasks, track progress, and coordinate work."
    };
}

void AgentCorpusGenerator::init_templates() {
    // ---- Research Agent Tasks ----
    task_templates_.push_back({
        "research",
        // user queries
        {
            "What are the latest developments in LLM-based autonomous agents?",
            "Compare ReAct, Reflexion, and Tree-of-Thoughts prompting strategies",
            "Find papers on long-term memory architectures for AI agents published in 2024",
            "What benchmarks exist for evaluating agent tool-use capabilities?",
            "Summarize the key findings from the GAIA agent benchmark paper",
            "How do current agents handle multi-step planning with uncertainty?",
            "What are the main failure modes of LLM agents in production?",
            "Find information about retrieval-augmented generation for agent memory",
        },
        // planning thoughts
        {
            "I need to search for recent papers and synthesize the key themes. Let me start with a web search for 2024 survey papers.",
            "This requires comparing multiple frameworks. I'll search for each one and create a comparison table.",
            "I should check arxiv and semantic scholar for the most cited papers in this area.",
            "Let me break this down: first find the benchmarks, then compare their evaluation criteria.",
        },
        // tool calls
        {
            "web_search(query='LLM autonomous agents survey 2024', num_results=10)",
            "web_search(query='ReAct prompting vs Reflexion comparison', num_results=5)",
            "api_call(endpoint='https://api.semanticscholar.org/graph/v1/paper/search', params={'query': 'agent memory architecture', 'year': '2024'})",
            "file_read(path='./references/agent-survey-notes.md')",
            "web_search(query='GAIA benchmark LLM agents results', num_results=5)",
        },
        // tool outputs
        {
            "Results: [1] 'A Survey on Large Language Model based Autonomous Agents' (citations: 487), [2] 'The Rise and Potential of LLM Based Agents' (citations: 312), [3] 'AgentBench: Evaluating LLMs as Agents' (citations: 198)...",
            "Found 8 papers comparing prompting strategies. Key finding: ReAct shows 34% improvement on HotpotQA over chain-of-thought, while Reflexion adds self-correction improving success rate by 20% on ALFWorld.",
            "API returned 23 results. Top papers: 'MemoryBank: Enhancing LLM Agents with Long-Term Memory', 'Cognitive Architectures for Language Agents', 'RecurrentGPT: Interactive Generation of Arbitrarily Long Text'.",
            "File contents: # Agent Survey Notes\n- Key challenge: catastrophic forgetting in long conversations\n- Promising approach: hierarchical memory with importance scoring\n- Open question: how to handle contradictory memories",
        },
        // assistant responses
        {
            "Based on my research, here are the key developments in LLM-based autonomous agents:\n\n1. **Memory architectures** have evolved from simple buffer storage to hierarchical systems with importance-weighted retrieval.\n2. **Tool use** has become more sophisticated with agents learning to chain multiple tools.\n3. **Self-reflection** mechanisms (Reflexion, Self-Refine) show significant improvements in task completion.\n\nThe most cited survey (487 citations) identifies three core components: memory, planning, and action.",
            "Here's a comparison of the three prompting strategies:\n\n| Strategy | Key Idea | Strength | Weakness |\n|----------|----------|----------|----------|\n| ReAct | Interleave reasoning + action | Grounded decisions | No self-correction |\n| Reflexion | Add reflection after failures | Self-improvement | Requires multiple attempts |\n| Tree-of-Thoughts | Explore multiple reasoning paths | Handles ambiguity | Higher compute cost |",
            "I found 15 relevant papers. The most impactful finding is that hybrid retrieval (combining semantic search with recency weighting) outperforms pure embedding-based retrieval by 23% on long-horizon agent tasks.",
        },
        // tools used
        {ToolType::WEB_SEARCH, ToolType::API_CALL, ToolType::FILE_READ}
    });

    // ---- Coding Agent Tasks ----
    task_templates_.push_back({
        "coding",
        {
            "Fix the memory leak in the connection pool manager",
            "Implement a thread-safe LRU cache with TTL expiration",
            "Why is this async function deadlocking? Here's the stack trace...",
            "Refactor the payment processing module to use the strategy pattern",
            "Write unit tests for the authentication middleware",
            "The CI pipeline is failing on the integration tests, can you investigate?",
            "Optimize this SQL query that's taking 12 seconds on the orders table",
            "Add rate limiting to the /api/v1/search endpoint",
        },
        {
            "The stack trace shows the deadlock occurs when two goroutines try to acquire locks in opposite order. I need to examine the lock ordering.",
            "For the LRU cache, I'll use a doubly-linked list for O(1) eviction combined with a hash map for O(1) lookup. Thread safety via a read-write lock.",
            "I should check the connection pool configuration first. The leak might be from connections not being returned after timeout.",
            "Let me trace through the payment flow to identify the right abstraction boundaries for the strategy pattern.",
        },
        {
            "code_exec(language='python', code='import tracemalloc; tracemalloc.start(); # ... profile memory usage')",
            "file_read(path='src/cache/lru_cache.go')",
            "shell_cmd(command='go test -race -v ./pkg/connpool/...')",
            "db_query(sql='EXPLAIN ANALYZE SELECT * FROM orders WHERE created_at > NOW() - INTERVAL 30 DAY ORDER BY total DESC LIMIT 100')",
            "file_read(path='src/middleware/auth.ts')",
            "shell_cmd(command='docker logs app-server --tail 200 | grep ERROR')",
        },
        {
            "Memory profile shows 3.2MB growth per minute. Top allocations:\n  1. connpool.newConn(): 45% of allocations\n  2. http.Request.Body: 23% of allocations\nThe connection finalizer is not being called when connections time out.",
            "File contents (248 lines): package cache\n\ntype LRUCache struct {\n    capacity int\n    items    map[string]*entry\n    head     *entry\n    // ...\n}\n\nfunc (c *LRUCache) Get(key string) (interface{}, bool) {\n    // BUG: no lock acquisition here\n    ...",
            "FAIL: TestConnectionPool_ConcurrentAccess (race detected)\n  goroutine 47 read at connpool.go:89\n  goroutine 52 write at connpool.go:112\n3 tests failed, 12 passed.",
            "Query plan:\n  -> Seq Scan on orders (cost=0..45230, rows=892341, actual time=0.042..11847.32)\n     Filter: created_at > '2024-01-15'\n     Rows Removed by Filter: 4521789\n  Planning Time: 0.21ms\n  Execution Time: 12034.55ms\n\nMissing index on (created_at, total).",
        },
        {
            "Found the memory leak. The connection pool's `release()` method has a bug where timed-out connections are removed from the active set but their underlying socket isn't closed.\n\n```go\n// Fix: add explicit close before removing from pool\nfunc (p *Pool) release(conn *Conn) {\n    if conn.isExpired() {\n        conn.Close()  // THIS WAS MISSING\n        p.active.Delete(conn.id)\n        return\n    }\n    p.idle.Put(conn)\n}\n```",
            "Here's the thread-safe LRU cache implementation with TTL support. Key design decisions:\n- RWMutex for concurrent reads\n- Background goroutine for TTL cleanup every 30s\n- Capacity-based eviction on Put()",
            "The deadlock is caused by inconsistent lock ordering between `ProcessPayment()` and `RefundPayment()`. Fix: always acquire locks in alphabetical order (account → transaction → wallet).",
        },
        {ToolType::CODE_EXEC, ToolType::FILE_READ, ToolType::SHELL_CMD, ToolType::DB_QUERY}
    });

    // ---- DevOps Agent Tasks ----
    task_templates_.push_back({
        "devops",
        {
            "The production API latency spiked to 5s, investigate the root cause",
            "Set up auto-scaling for the inference service based on GPU utilization",
            "Our Kubernetes pods keep getting OOMKilled, help me debug this",
            "Deploy the new model version with zero-downtime using blue-green strategy",
            "The SSL certificate for api.example.com expires in 3 days, renew it",
            "Set up monitoring alerts for error rate > 1% on the payment service",
            "Migrate the database from RDS to Aurora with minimal downtime",
        },
        {
            "Let me check the metrics dashboard first to identify when the latency spike started, then correlate with deployments and infrastructure changes.",
            "I need to check current resource utilization, then configure the HPA with appropriate thresholds. GPU metrics require the DCGM exporter.",
            "OOMKilled usually means the container memory limit is too low or there's a memory leak. Let me check the pod spec and recent memory usage.",
        },
        {
            "shell_cmd(command='kubectl top pods -n production --sort-by=memory')",
            "shell_cmd(command='kubectl describe pod api-server-7d8f9-x2k4n -n production')",
            "api_call(endpoint='https://prometheus.internal/api/v1/query', params={'query': 'rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])'})",
            "shell_cmd(command='kubectl logs deploy/api-server -n production --since=1h | grep -c ERROR')",
            "shell_cmd(command='aws rds describe-db-instances --db-instance-identifier prod-primary --query DBInstances[0].CPUUtilization')",
        },
        {
            "NAME                          CPU    MEMORY\napi-server-7d8f9-x2k4n       450m   1.8Gi\napi-server-7d8f9-m3j7p       2100m  3.2Gi  ← HIGH\nworker-5c6d8-k9p2n           180m   512Mi\ninference-gpu-8f7e6-q1w2     3800m  7.1Gi",
            "Pod events:\n  Warning  OOMKilled  4m ago   container 'api-server' exceeded memory limit (4Gi)\n  Warning  OOMKilled  18m ago  container 'api-server' exceeded memory limit (4Gi)\n  Normal   Pulled     3m ago   Successfully pulled image 'api-server:v2.3.1'",
            "Prometheus query result: {instance='api-server'} 4.82s (p99), {instance='api-server'} 0.34s (p50)\nSpike started at 14:32 UTC, correlates with deployment of api-server:v2.3.1",
        },
        {
            "Root cause identified: the v2.3.1 deployment introduced an N+1 query in the order listing endpoint. The new code fetches order items in a loop instead of using a JOIN.\n\nImmediate mitigation: rolled back to v2.3.0\nFix: PR #1247 replaces the loop with a single batch query, reducing p99 latency from 4.8s to 0.28s.",
            "Auto-scaling configured:\n- HPA target: 70% GPU utilization (DCGM exporter metrics)\n- Min replicas: 2, Max replicas: 8\n- Scale-up: 60s cooldown, Scale-down: 300s cooldown\n- Tested: load from 100 to 800 RPS scales from 2 to 6 pods in ~3 minutes.",
        },
        {ToolType::SHELL_CMD, ToolType::API_CALL, ToolType::DB_QUERY}
    });

    // ---- Data Analysis Agent Tasks ----
    task_templates_.push_back({
        "data_analysis",
        {
            "Analyze the user churn data from Q4 and identify the top predictive features",
            "Build a dashboard showing daily active users segmented by acquisition channel",
            "There's an anomaly in the revenue data for December, can you investigate?",
            "Run a cohort analysis on user retention for the last 6 months",
            "Compare the A/B test results for the new checkout flow",
            "Calculate the customer lifetime value by segment for the board presentation",
        },
        {
            "I'll start by loading the churn dataset and running feature importance analysis using gradient boosting. Then I'll visualize the top features.",
            "For the cohort analysis, I need to group users by signup month, then track their activity over subsequent months. Let me query the events table.",
            "The December anomaly could be a data pipeline issue or a real business change. Let me check both the raw data and the ETL logs.",
        },
        {
            "db_query(sql='SELECT feature, importance FROM ml.feature_importance WHERE model_id = \\'churn_v3\\' ORDER BY importance DESC LIMIT 20')",
            "code_exec(language='python', code='import pandas as pd\\ndf = pd.read_sql(\"SELECT * FROM analytics.daily_active_users WHERE date >= \\'2024-01-01\\'\", conn)\\nprint(df.describe())')",
            "db_query(sql='SELECT date, channel, COUNT(DISTINCT user_id) as dau FROM events GROUP BY date, channel ORDER BY date')",
            "code_exec(language='python', code='from scipy import stats\\ncontrol = [0.032, 0.028, 0.035]\\ntreatment = [0.041, 0.039, 0.044]\\nt_stat, p_value = stats.ttest_ind(control, treatment)\\nprint(f\"p-value: {p_value:.4f}\")')",
        },
        {
            "Top predictive features for churn:\n1. days_since_last_login (importance: 0.234)\n2. support_tickets_30d (importance: 0.189)\n3. feature_adoption_score (importance: 0.156)\n4. session_duration_trend (importance: 0.134)\n5. billing_failures_90d (importance: 0.098)",
            "Dataset: 2.3M rows, 45 columns\nDAU summary:\n  mean: 142,387\n  std: 23,451\n  min: 89,234 (Dec 25)\n  max: 198,432 (Nov 29 - Black Friday)\nMissing values: 0.02% in channel column",
            "A/B test results:\n  Control: 3.2% conversion (n=45,230)\n  Treatment: 4.1% conversion (n=44,891)\n  Uplift: +28.1%\n  p-value: 0.0003 (statistically significant)\n  95% CI for uplift: [15.2%, 41.7%]",
        },
        {
            "The churn analysis reveals that **login recency** is by far the strongest predictor (23.4% importance). Users who haven't logged in for 14+ days have a 67% probability of churning within 30 days.\n\nRecommendation: implement a re-engagement email campaign triggered at day 7 of inactivity, before the critical day-14 threshold.",
            "The December revenue anomaly is explained by two factors:\n1. A pricing migration on Dec 3 that double-counted upgrades (data issue, $234K impact)\n2. Genuine 18% increase in enterprise deals from year-end budget spending\n\nAfter correcting the double-count, December revenue was actually 12% above forecast.",
        },
        {ToolType::DB_QUERY, ToolType::CODE_EXEC, ToolType::API_CALL}
    });
}

std::vector<float> AgentCorpusGenerator::make_embedding(const std::string& text) {
    // Deterministic pseudo-embedding: hash text to seed a local RNG.
    // This gives consistent embeddings for similar-ish text and is
    // sufficient for benchmarking (we're measuring system performance, not retrieval quality).
    std::hash<std::string> hasher;
    size_t h = hasher(text);
    std::mt19937 local_rng(static_cast<unsigned>(h));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> emb(dim_);
    float norm = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        emb[i] = dist(local_rng);
        norm += emb[i] * emb[i];
    }
    norm = std::sqrt(norm);
    for (float& v : emb) v /= norm;
    return emb;
}

std::string AgentCorpusGenerator::make_session_id(int idx) {
    std::ostringstream oss;
    oss << "sess-" << std::setfill('0') << std::setw(6) << idx;
    return oss.str();
}

uint64_t AgentCorpusGenerator::make_timestamp(int session_idx, int turn_idx) {
    // Base: 2024-01-01 00:00:00 UTC = 1704067200000 ms
    // Sessions spread over ~6 months, turns within a session span minutes.
    uint64_t base = 1704067200000ULL;
    uint64_t session_offset = static_cast<uint64_t>(session_idx) * 3600000ULL; // 1 hour between sessions
    uint64_t turn_offset = static_cast<uint64_t>(turn_idx) * 15000ULL;         // 15s between turns
    return base + session_offset + turn_offset;
}

std::vector<MemoryRecord> AgentCorpusGenerator::generate(int num_sessions, int turns_per_session) {
    std::vector<MemoryRecord> records;
    records.reserve(num_sessions * turns_per_session);

    DocID next_id = 0;

    for (int s = 0; s < num_sessions; ++s) {
        std::string session_id = make_session_id(s);
        std::string agent_id = agent_ids_[s % agent_ids_.size()];
        const auto& tmpl = task_templates_[s % task_templates_.size()];

        // Vary turns per session: +/- 50%.
        std::uniform_int_distribution<int> turn_var(
            turns_per_session / 2, turns_per_session + turns_per_session / 2);
        int actual_turns = turn_var(rng_);

        int turn = 0;

        // System prompt.
        {
            MemoryRecord rec;
            rec.id = next_id++;
            rec.session_id = session_id;
            rec.agent_id = agent_id;
            rec.role = MemoryRole::SYSTEM;
            rec.tool = ToolType::NONE;
            rec.timestamp_ms = make_timestamp(s, turn);
            rec.content = system_prompts_[s % system_prompts_.size()];
            rec.embedding = make_embedding(rec.content);
            rec.importance = 0.3f;
            records.push_back(std::move(rec));
            turn++;
        }

        // Generate interaction turns.
        while (turn < actual_turns) {
            // User query.
            {
                MemoryRecord rec;
                rec.id = next_id++;
                rec.session_id = session_id;
                rec.agent_id = agent_id;
                rec.role = MemoryRole::USER;
                rec.tool = ToolType::NONE;
                rec.timestamp_ms = make_timestamp(s, turn);
                size_t qi = rng_() % tmpl.user_queries.size();
                rec.content = tmpl.user_queries[qi];
                rec.embedding = make_embedding(rec.content);
                rec.importance = 0.7f;
                records.push_back(std::move(rec));
                turn++;
                if (turn >= actual_turns) break;
            }

            // Agent planning (50% of the time).
            if (rng_() % 2 == 0 && !tmpl.planning_thoughts.empty()) {
                MemoryRecord rec;
                rec.id = next_id++;
                rec.session_id = session_id;
                rec.agent_id = agent_id;
                rec.role = MemoryRole::PLANNING;
                rec.tool = ToolType::NONE;
                rec.timestamp_ms = make_timestamp(s, turn);
                size_t pi = rng_() % tmpl.planning_thoughts.size();
                rec.content = tmpl.planning_thoughts[pi];
                rec.embedding = make_embedding(rec.content);
                rec.importance = 0.4f;
                records.push_back(std::move(rec));
                turn++;
                if (turn >= actual_turns) break;
            }

            // Tool call + tool output.
            if (!tmpl.tool_calls.empty() && !tmpl.tool_outputs.empty()) {
                size_t ti = rng_() % tmpl.tool_calls.size();
                size_t oi = rng_() % tmpl.tool_outputs.size();
                ToolType tool_type = tmpl.tools_used[rng_() % tmpl.tools_used.size()];

                // Tool call.
                {
                    MemoryRecord rec;
                    rec.id = next_id++;
                    rec.session_id = session_id;
                    rec.agent_id = agent_id;
                    rec.role = MemoryRole::TOOL_CALL;
                    rec.tool = tool_type;
                    rec.timestamp_ms = make_timestamp(s, turn);
                    rec.content = tmpl.tool_calls[ti];
                    rec.embedding = make_embedding(rec.content);
                    rec.importance = 0.5f;
                    records.push_back(std::move(rec));
                    turn++;
                    if (turn >= actual_turns) break;
                }

                // Tool output.
                {
                    MemoryRecord rec;
                    rec.id = next_id++;
                    rec.session_id = session_id;
                    rec.agent_id = agent_id;
                    rec.role = MemoryRole::TOOL_OUTPUT;
                    rec.tool = tool_type;
                    rec.timestamp_ms = make_timestamp(s, turn);
                    rec.content = tmpl.tool_outputs[oi];
                    rec.embedding = make_embedding(rec.content);
                    rec.importance = 0.6f;
                    records.push_back(std::move(rec));
                    turn++;
                    if (turn >= actual_turns) break;
                }
            }

            // Agent response.
            if (!tmpl.assistant_responses.empty()) {
                MemoryRecord rec;
                rec.id = next_id++;
                rec.session_id = session_id;
                rec.agent_id = agent_id;
                rec.role = MemoryRole::ASSISTANT;
                rec.tool = ToolType::NONE;
                rec.timestamp_ms = make_timestamp(s, turn);
                size_t ri = rng_() % tmpl.assistant_responses.size();
                rec.content = tmpl.assistant_responses[ri];
                rec.embedding = make_embedding(rec.content);
                rec.importance = 0.8f;
                records.push_back(std::move(rec));
                turn++;
            }
        }
    }

    return records;
}

std::vector<MemoryQuery> AgentCorpusGenerator::generate_queries(
    int n, const std::vector<MemoryRecord>& corpus,
    float recency_weight, int top_k)
{
    // Realistic agent memory queries: the agent is trying to recall
    // relevant past interactions to inform current decisions.
    static const std::vector<std::string> query_templates = {
        "What did I find when I searched for agent memory architectures?",
        "Show me the previous debugging session where we fixed a memory leak",
        "What was the root cause of the last production latency spike?",
        "Recall the A/B test results for the checkout flow experiment",
        "What tools did I use to investigate the OOMKilled pods?",
        "Find my previous analysis of user churn predictive features",
        "What was the fix for the connection pool issue?",
        "Show me the last time I deployed with blue-green strategy",
        "What did the feature importance analysis show?",
        "Find the SQL query optimization I did for the orders table",
        "Recall the comparison between ReAct and Reflexion strategies",
        "What were the monitoring alerts I set up for the payment service?",
        "Show me the customer lifetime value calculation",
        "What was the December revenue anomaly about?",
        "Find the thread-safe LRU cache implementation",
        "What SSL certificate renewal steps did I follow?",
        "How did I handle the database migration?",
        "What were the key findings from the agent survey papers?",
        "Show me the auto-scaling configuration for inference service",
        "What cohort retention numbers did I calculate?",
    };

    std::vector<MemoryQuery> queries;
    queries.reserve(n);

    // Collect unique sessions for session-filtered queries.
    std::vector<std::string> sessions;
    for (const auto& r : corpus) {
        if (sessions.empty() || sessions.back() != r.session_id) {
            sessions.push_back(r.session_id);
        }
    }

    uint64_t max_time = 0;
    for (const auto& r : corpus) {
        max_time = std::max(max_time, r.timestamp_ms);
    }

    for (int i = 0; i < n; ++i) {
        MemoryQuery mq;
        mq.text = query_templates[i % query_templates.size()];
        mq.embedding = make_embedding(mq.text);

        // 30% of queries have session filter.
        if (rng_() % 10 < 3 && !sessions.empty()) {
            mq.session_filter = sessions[rng_() % sessions.size()];
        }

        // 20% of queries have time range filter.
        if (rng_() % 10 < 2) {
            mq.time_end = max_time;
            mq.time_start = max_time - 86400000ULL * 7; // last 7 days
        } else {
            mq.time_start = 0;
            mq.time_end = 0;
        }

        mq.agent_filter = "";
        mq.role_filter = static_cast<MemoryRole>(255); // no filter sentinel
        mq.recency_weight = recency_weight;
        mq.top_k = top_k;

        queries.push_back(std::move(mq));
    }

    return queries;
}

} // namespace hybrid
