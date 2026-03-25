# Hybrid Retrieval Pipeline for Long-Term Agent Memory

**EE451 Project**: Parallelizing a Hybrid Retrieval Pipeline for Long-Term Agent Memory

## Overview

Modern AI agents rely on long-term memory to retrieve relevant past interactions, tool outputs, and planning notes. This project implements and parallelizes a hybrid retrieval pipeline that combines **sparse lexical search** (BM25) with **dense ANN search** (HNSW), augmented with **recency decay** and **importance weighting** for agent-specific workloads.

The main contribution is a **parallel systems study** of the end-to-end pipeline, analyzing speedup, efficiency, and bottlenecks across three levels of parallelism.

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │         Agent Memory Store              │
                          │  (concurrent read/write, thread-safe)   │
                          └────────────────┬────────────────────────┘
                                           │
  MemoryQuery ──► [Preprocessing] ──► ┬────┴────────────────────────┐
                                      │                             │
                              [Sparse Branch]              [Dense Branch]
                              BM25 Inverted Index          HNSW Graph (L2)
                              Lexical term matching        Semantic similarity
                                      │                             │
                                      └──────────┬─────────────────┘
                                                  │
                                    [Agent-Aware Score Fusion]
                                    RRF + Recency Decay + Importance Boost
                                    + Session/Role/Time Filtering
                                                  │
                                          [Parallel Top-k]
                                          Partitioned partial sort
                                                  │
                                        Ranked Memory Results
```

### Three Levels of Parallelism

| Level | Strategy | OpenMP Construct |
|-------|----------|-----------------|
| **Query-level** | Multiple queries processed concurrently | `#pragma omp parallel for` |
| **Task-level** | Sparse + dense branches run in parallel per query | `#pragma omp sections` |
| **Data-level** | Top-k selection partitioned across threads | Thread-local partial sort + merge |

### Agent Memory Model

Each memory record represents one interaction in an agent's history:

| Field | Description |
|-------|-------------|
| `role` | USER, ASSISTANT, TOOL_CALL, TOOL_OUTPUT, SYSTEM, PLANNING, OBSERVATION |
| `tool` | web_search, code_exec, file_read, api_call, db_query, shell_cmd, ... |
| `session_id` | Groups a conversation/task |
| `agent_id` | Which agent produced this record |
| `timestamp_ms` | For recency-weighted retrieval |
| `importance` | Salience score for importance boosting |
| `content` | Text for sparse indexing |
| `embedding` | Dense vector for ANN search |

### Agent-Aware Fusion

Beyond standard RRF, the fusion layer adds:
- **Temporal decay**: `score *= exp(-lambda * age)` with configurable half-life
- **Importance boost**: `score *= (1 + importance * 0.5)`
- **Session filtering**: restrict retrieval to a specific conversation
- **Role/time filtering**: query only tool outputs, or memories from last 7 days

### Concurrent Read/Write

Real agents continuously write new memories while retrieving old ones. The `MemoryStore` supports this pattern with:
- Staging buffer for writes, periodically flushed to main indices
- `shared_mutex` for concurrent read access during writes
- Benchmarked under mixed R/W workload

## Build

```bash
# Prerequisites: C++17 compiler, CMake >= 3.16, OpenMP
# macOS: brew install cmake libomp

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Run

```bash
# Agent memory demo (recommended)
./demo --quick               # quick demo (~6K records)
./demo                       # full demo (~40K records, 5000 sessions)
./demo --sessions 10000      # larger scale

# Generic retrieval benchmark
./benchmark --quick           # 10K docs
./benchmark                   # 100K/500K/1M docs

# All options
./demo --help
./benchmark --help
```

## Output

Results are written to `results/`. Generate plots:

```bash
pip install pandas matplotlib
python3 scripts/plot_results.py results/agent_benchmark.csv
```

## Project Structure

```
ee451-hybrid-retrieval/
├── CMakeLists.txt
├── include/
│   ├── common.h                # Core types, Timer, BenchmarkResult
│   ├── tokenizer.h             # Query preprocessing
│   ├── sparse_index.h          # BM25 inverted index
│   ├── dense_index.h           # hnswlib HNSW wrapper
│   ├── fusion.h                # RRF + parallel top-k
│   ├── pipeline.h              # Generic pipeline orchestrator
│   ├── corpus_generator.h      # Synthetic data (generic)
│   ├── agent_memory.h          # MemoryRecord, MemoryQuery, roles, tools
│   ├── agent_corpus_generator.h # Realistic agent interaction generator
│   ├── memory_fusion.h         # Agent-aware fusion (recency + importance)
│   └── memory_store.h          # Thread-safe memory store (concurrent R/W)
├── src/
│   ├── tokenizer.cpp
│   ├── sparse_index.cpp
│   ├── dense_index.cpp
│   ├── fusion.cpp
│   ├── pipeline.cpp
│   ├── corpus_generator.cpp
│   ├── agent_corpus_generator.cpp
│   ├── memory_fusion.cpp
│   ├── memory_store.cpp
│   ├── main.cpp                # Generic benchmark driver
│   └── demo.cpp                # Agent memory demo (primary entry point)
├── scripts/
│   └── plot_results.py         # Result visualization
└── third_party/
    └── hnswlib/                # Header-only ANN library
```
