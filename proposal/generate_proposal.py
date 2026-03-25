#!/usr/bin/env python3
"""Generate EE451 Final Project Proposal PDF using fpdf2."""

from fpdf import FPDF

class ProposalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128)
            self.cell(0, 8, "EE451 Project Proposal: Parallelizing a Hybrid Retrieval Pipeline for Long-Term Agent Memory", align="C")
            self.ln(4)
            self.set_draw_color(200)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"{self.page_no()}", align="C")

    def section_title(self, num, title):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0)
        self.cell(0, 8, f"{num}   {title}")
        self.ln(6)
        self.set_draw_color(0)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40)
        self.cell(0, 7, title)
        self.ln(6)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30)
        x = self.get_x()
        self.cell(8, 5.2, "-")
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.cell(self.get_string_width(bold_prefix) + 1, 5.2, bold_prefix)
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 5.2, text)
        else:
            self.multi_cell(0, 5.2, text)
        self.ln(0.5)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_text_color(50)
        self.set_fill_color(245, 245, 245)
        for line in text.strip().split("\n"):
            self.cell(0, 4.8, "  " + line, fill=True)
            self.ln(4.8)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None, highlight_col=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(50, 50, 50)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6.5, h, border=1, align="C", fill=True)
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(248, 248, 248)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                if highlight_col is not None and i == highlight_col:
                    self.set_font("Helvetica", "B", 9)
                    self.set_text_color(0, 120, 60)
                self.cell(col_widths[i], 6, str(val), border=1, align="C", fill=True)
                if highlight_col is not None and i == highlight_col:
                    self.set_font("Helvetica", "", 9)
                    self.set_text_color(30)
            self.ln()
        self.ln(2)


def main():
    pdf = ProposalPDF()
    pdf.set_left_margin(22)
    pdf.set_right_margin(22)

    # ======== Title Page ========
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0)
    pdf.cell(0, 12, "EE451 Project Proposal", align="C")
    pdf.ln(14)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, "Parallelizing a Hybrid Retrieval Pipeline", align="C")
    pdf.ln(10)
    pdf.cell(0, 10, "for Long-Term Agent Memory", align="C")
    pdf.ln(18)

    pdf.set_draw_color(0)
    cx = pdf.w / 2
    pdf.line(cx - 40, pdf.get_y(), cx + 40, pdf.get_y())
    pdf.ln(12)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60)
    pdf.cell(0, 7, "Team Members", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0)
    pdf.cell(0, 7, "Ziyao Wang,  Aojie Yuan,  Haiyue Zhang,  Weixiao Wang,  Joyce Meng", align="C")
    pdf.ln(14)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60)
    pdf.cell(0, 7, "Category: Implement and Evaluate", align="C")
    pdf.ln(8)
    pdf.cell(0, 7, "EE451 - Parallel and Distributed Computing", align="C")
    pdf.ln(8)
    pdf.cell(0, 7, "Spring 2026", align="C")
    pdf.ln(25)

    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(80)
    pdf.multi_cell(0, 5.5,
        "Abstract: This project conducts a systematic parallel systems study of hybrid retrieval "
        "for AI agent long-term memory. We implement a complete C++ pipeline combining BM25 sparse "
        "search with HNSW dense vector search, study five parallelization strategies across up to "
        "12 threads, and complement multi-threaded parallelism with single-thread algorithmic "
        "optimizations (SIMD, MaxScore pruning, temporal partitioning). We further extend the system "
        "with a three-tier hierarchical memory architecture and a multi-corpus-size scalability study. "
        "Our best parallel strategy achieves 8.3x throughput speedup at 12 threads, while temporal "
        "partitioning achieves 47x latency reduction that scales sub-linearly with corpus size.",
        align="J")

    # ======== Section 1: Introduction ========
    pdf.add_page()
    pdf.section_title("1", "Introduction and Motivation")

    pdf.body_text(
        "Modern AI agents rely on long-term memory to retrieve relevant past interactions, tool outputs, "
        "and accumulated knowledge. In production systems such as ChatGPT, Claude, and autonomous coding "
        "agents, this memory can span hundreds of thousands of interaction records across months of "
        "operation. Retrieval over this memory is typically implemented as a hybrid pipeline that combines "
        "sparse lexical search (BM25 over inverted indices) with dense vector search (approximate nearest "
        "neighbors over embeddings), then fuses the candidate lists into a final ranking.")

    pdf.body_text(
        "While effective for relevance, this pipeline presents significant parallelization challenges:")

    pdf.bullet("BM25 scoring traverses posting lists of varying length, causing unpredictable cache behavior.",
               "Irregular memory access: ")
    pdf.bullet("Sparse retrieval is compute-bound while dense ANN search is memory-bound, creating load "
               "imbalance when run concurrently as parallel tasks.",
               "Stage heterogeneity: ")
    pdf.bullet("Candidate lists from independent branches must be merged via Reciprocal Rank Fusion, "
               "requiring barrier synchronization.",
               "Synchronization at fusion: ")
    pdf.bullet("As the memory store grows from 10^3 to 10^5 records, the dominant bottleneck shifts, "
               "requiring different optimization strategies at different scales.",
               "Scalability limits: ")

    pdf.ln(1)
    pdf.body_text(
        "This project conducts a systematic parallel systems study of the end-to-end hybrid retrieval "
        "pipeline. We implement a complete C++ prototype with multiple parallelization strategies, "
        "measure performance across thread counts and corpus sizes, and analyze where speedup is gained "
        "and where scaling saturates. Beyond the basic pipeline, we extend the system with single-thread "
        "algorithmic optimizations (SIMD vectorization, MaxScore pruning), a temporal partitioned index "
        "for exploiting recency bias, a three-tier hierarchical memory architecture with concurrent "
        "cross-tier retrieval, and a comprehensive multi-corpus-size scalability study.")

    # ======== Section 2: Related Work ========
    pdf.section_title("2", "Related Work and Context")

    pdf.body_text(
        "Hybrid retrieval is the dominant paradigm in modern information retrieval. Sparse methods like "
        "BM25 [Robertson & Zaragoza, 2009] excel at exact term matching, while dense methods using learned "
        "embeddings capture semantic similarity. Reciprocal Rank Fusion (RRF) [Cormack et al., 2009] is "
        "a standard technique for merging ranked lists from heterogeneous retrieval systems.")

    pdf.body_text(
        "Existing libraries provide efficient primitives for individual components: HNSWlib [Malkov & "
        "Yashunin, 2018] for approximate nearest neighbor search via hierarchical navigable small world "
        "graphs, and libraries like Lucene and Tantivy for inverted index search. However, these libraries "
        "do not address the end-to-end parallelization question: how to schedule sparse and dense branches, "
        "how to overlap computation with fusion, and how query-level versus intra-query parallelism trade "
        "off at different scales.")

    pdf.body_text(
        "For agent memory specifically, recent work on MemGPT [Packer et al., 2023] and LangChain memory "
        "modules highlights the need for multi-tier memory architectures that mirror cognitive science "
        "models of human memory (working, semantic, episodic). Our project bridges these domains by "
        "studying the parallel systems implications of hierarchical agent memory retrieval.")

    # ======== Section 3: System Architecture ========
    pdf.section_title("3", "System Architecture")

    pdf.body_text(
        "Our system implements a four-stage retrieval pipeline augmented with domain-specific optimizations "
        "for agent memory. The pipeline flow is:")

    pdf.code_block(
        "Query --> [Preprocessing] --> [Sparse BM25]  --\\\n"
        "                                                --> [RRF Fusion] --> Top-k\n"
        "      --> [Preprocessing] --> [Dense HNSW]   --/")

    pdf.body_text(
        "The full codebase comprises approximately 5,700 lines of C++17 across 13 source files and 13 "
        "header files, built with CMake and linked against OpenMP and HNSWlib. The architecture is "
        "organized into two static libraries:")

    pdf.bullet("hybrid_retrieval: Core pipeline components including tokenizer, sparse index (inverted "
               "index with BM25), dense index (HNSW wrapper), RRF fusion, and the multi-strategy pipeline.",
               "libhybrid_retrieval: ")
    pdf.bullet("agent_memory: Agent-specific components including memory store with importance scoring "
               "and recency decay, temporal partitioned index, three-tier hierarchical memory with "
               "consolidation and decay, and the corpus generator.",
               "libagent_memory: ")

    pdf.ln(1)
    pdf.body_text("Key source modules:")
    pdf.bullet("Inverted index with BM25 scoring, SIMD-accelerated scoring (ARM NEON), MaxScore pruning, "
               "and data-parallel scoring.",
               "sparse_index: ")
    pdf.bullet("HNSW-based approximate nearest neighbor search via HNSWlib.",
               "dense_index: ")
    pdf.bullet("Reciprocal Rank Fusion for merging sparse and dense candidate lists.",
               "fusion: ")
    pdf.bullet("Five parallelization strategies (sequential, task-parallel, data-parallel, full-parallel, combined).",
               "pipeline: ")
    pdf.bullet("Time-partitioned inverted index with recency-biased partition selection.",
               "temporal_index: ")
    pdf.bullet("Three-tier memory (working/semantic/episodic) with consolidation and decay.",
               "hierarchical_memory: ")

    # ======== Section 4: Parallelization Strategies ========
    pdf.section_title("4", "Parallelization Strategies")

    pdf.body_text(
        "We study five distinct parallelization strategies, each targeting a different level of the pipeline:")

    pdf.subsection_title("4.1  Query-Level Parallelism (Full Parallel)")
    pdf.body_text(
        "Distributes independent queries across threads using #pragma omp parallel for with dynamic "
        "scheduling. Each thread processes a complete query (sparse + dense + fusion) independently. "
        "Because queries share no mutable state, this strategy achieves near-linear speedup and is "
        "the best strategy for throughput optimization.")

    pdf.subsection_title("4.2  Task Parallelism")
    pdf.body_text(
        "For each query, runs the sparse and dense retrieval branches concurrently using #pragma omp "
        "parallel sections. The two branches have fundamentally different computational profiles: sparse "
        "BM25 is compute-bound (arithmetic over posting lists), while dense HNSW is memory-bound (random "
        "graph traversal with cache misses). Speedup is bounded by the slower branch, limiting this "
        "strategy to approximately 1.3x regardless of thread count.")

    pdf.subsection_title("4.3  Data Parallelism")
    pdf.body_text(
        "Partitions the BM25 scoring loop across threads using #pragma omp parallel for reduction. Each "
        "thread scores a subset of posting list entries and maintains a local top-k heap. Partial results "
        "are merged after a barrier. This reduces per-query latency but incurs synchronization overhead "
        "at the merge step, peaking at 4 threads before degrading.")

    pdf.subsection_title("4.4  Combined (Nested) Parallelism")
    pdf.body_text(
        "Combines query-level and intra-query data parallelism using OpenMP nested parallelism "
        "(omp_set_max_active_levels(2)). The outer loop distributes queries across sqrt(T) threads; "
        "each query internally uses T/sqrt(T) threads for data-parallel BM25 scoring. This strategy "
        "balances throughput and latency but suffers from nested parallelism overhead.")

    pdf.subsection_title("4.5  Cross-Tier Parallel Retrieval")
    pdf.body_text(
        "In the hierarchical memory architecture, all three memory tiers (working, semantic, episodic) "
        "are searched concurrently using #pragma omp parallel sections num_threads(3). Each tier has its "
        "own shared_mutex for reader-writer concurrency. Results are merged with tier-specific weighting "
        "(working: 1.5x, semantic: 1.2x, episodic: 1.0x) to prioritize recent and consolidated knowledge.")

    # ======== Section 5: Single-Thread Optimizations ========
    pdf.section_title("5", "Single-Thread Optimizations")

    pdf.body_text(
        "Beyond multi-threaded parallelism, we implement three single-thread algorithmic optimizations "
        "that exploit instruction-level parallelism and algorithmic pruning. These complement the "
        "multi-threaded strategies and can be composed with them.")

    pdf.subsection_title("5.1  SIMD-Accelerated BM25 (ARM NEON)")
    pdf.body_text(
        "Vectorizes BM25 term-frequency accumulation using 128-bit ARM NEON intrinsics (vld1q_f32, "
        "vfmaq_f32), processing 4 float scores per cycle. The inner scoring loop over posting list "
        "entries is restructured to operate on aligned float vectors. This achieves 6-18x per-query "
        "latency reduction depending on corpus size, with higher speedup at smaller corpora where "
        "the posting lists fit in L1/L2 cache.")

    pdf.subsection_title("5.2  MaxScore Pruning")
    pdf.body_text(
        "Implements the MaxScore algorithm [Turtle & Flood, 1995] for safe early termination during "
        "top-k retrieval. Posting lists are pre-sorted by their maximum possible BM25 contribution. "
        "During scoring, once the k-th best accumulated score exceeds the sum of remaining maximum "
        "contributions from unprocessed terms, scoring terminates safely without missing any true "
        "top-k results. Achieves 8-22x latency reduction.")

    pdf.subsection_title("5.3  Temporal Partitioned Index")
    pdf.body_text(
        "Exploits the strong recency bias in agent memory access patterns by partitioning the inverted "
        "index into time-window buckets (e.g., weekly). At query time, partitions are searched from "
        "newest to oldest, stopping when sufficient high-quality results are found. At 64K records, "
        "this searches only 2.6% of the corpus while maintaining result quality, achieving 47x latency "
        "reduction. Crucially, this speedup increases with corpus size, providing sub-linear scaling.")

    # ======== Section 6: Hierarchical Memory ========
    pdf.section_title("6", "Hierarchical Memory Architecture")

    pdf.body_text(
        "To model realistic agent memory, we implement a three-tier hierarchical architecture inspired "
        "by cognitive science models of human memory:")

    cw = [30, 35, 55, 46]
    pdf.add_table(
        ["Tier", "Name", "Cognitive Analogy", "Capacity"],
        [
            ["0", "Working", "Short-term / L1 cache", "~100 records"],
            ["1", "Semantic", "Semantic memory / L2 cache", "~10K entries"],
            ["2", "Episodic", "Episodic memory / main RAM", "~500K records"],
        ],
        col_widths=cw
    )

    pdf.body_text("Key operations in the hierarchical memory system:")

    pdf.bullet("Scans episodic records for recurring keyword patterns, clusters them, and "
               "merges clusters into compact semantic entries with confidence scores. Each consolidation "
               "pass processes a configurable batch of records.",
               "Consolidation: ")
    pdf.bullet("Applies exponential forgetting: effective_importance = base_importance * exp(-lambda * age), "
               "where lambda = ln(2) / halflife. Records whose effective importance falls below a threshold "
               "(default 0.01) are evicted. With a 30-day halflife, 97.5% of records are forgotten after 180 days.",
               "Decay: ")
    pdf.bullet("When working memory exceeds capacity (default 100), the oldest half spills to episodic. "
               "Starting a new session flushes all working memory to episodic with index rebuild.",
               "Overflow and spill: ")
    pdf.bullet("Each tier has its own std::shared_mutex, enabling concurrent reads across all three "
               "tiers and exclusive writes during consolidation or decay operations.",
               "Concurrent access: ")

    # ======== Section 7: Experimental Setup ========
    pdf.section_title("7", "Experimental Setup")

    pdf.subsection_title("7.1  Platform")
    pdf.bullet("Apple M-series ARM64, 4 performance + 4 efficiency cores", "CPU: ")
    pdf.bullet("Apple Clang with -O3 -march=native", "Compiler: ")
    pdf.bullet("Homebrew libomp (OpenMP 5.0)", "OpenMP: ")
    pdf.bullet("C++17, CMake 3.16+", "Standard: ")
    pdf.bullet("HNSWlib (header-only, vendored in third_party/)", "Dependencies: ")

    pdf.subsection_title("7.2  Workload")
    pdf.body_text(
        "Corpus: Synthetically generated agent interaction records with realistic session/turn structure. "
        "Each session contains 6-8 turns with multiple agent roles (user, assistant, tool_call, tool_output, "
        "system, planning, observation) and embedded topic keywords from domains including code review, "
        "research, debugging, deployment, and data analysis.")

    pdf.bullet("12K records (2,000 sessions x 6 turns), 100-200 queries", "Main benchmark: ")
    pdf.bullet("5 corpus sizes: 3.9K, 12K, 24K, 40K, 64K records (500-8,000 sessions)", "Scalability study: ")
    pdf.bullet("1, 2, 4, 8, 12 threads", "Thread sweep: ")
    pdf.bullet("128-dimensional float vectors", "Embedding dimension: ")

    pdf.subsection_title("7.3  Evaluation Metrics")
    pdf.bullet("End-to-end latency (per-query, microseconds) and throughput (queries/sec)")
    pdf.bullet("Speedup and parallel efficiency relative to single-thread sequential baseline")
    pdf.bullet("Stage-level timing breakdown: sparse retrieval, dense retrieval, fusion")
    pdf.bullet("Scalability with respect to corpus size (3.9K to 64K records)")
    pdf.bullet("Amdahl's Law analysis: serial fraction estimation from speedup curve")
    pdf.bullet("Compute-bound vs. memory-bound characterization per pipeline stage")
    pdf.bullet("Correctness validation: parallel results match sequential within epsilon")

    # ======== Section 8: Results ========
    pdf.section_title("8", "Results")

    pdf.subsection_title("8.1  Thread Scaling (12K records, 100 queries)")

    cw2 = [32, 25, 25, 25, 25, 27]
    pdf.add_table(
        ["Strategy", "2T", "4T", "8T", "12T", "Peak Eff."],
        [
            ["Task Parallel",  "1.27x", "1.28x", "1.30x", "1.30x", "64%"],
            ["Data Parallel",  "1.80x", "1.95x", "1.56x", "1.51x", "90%"],
            ["Full Parallel",  "1.97x", "3.89x", "6.72x", "8.28x", "98%"],
            ["Combined",       "1.87x", "3.90x", "3.50x", "5.15x", "97%"],
        ],
        col_widths=cw2
    )

    pdf.body_text("Key observations from the thread scaling experiment:")

    pdf.bullet("Full Parallel achieves 8.28x at 12 threads with 69% efficiency, the best overall "
               "throughput strategy. Near-linear scaling up to the physical core count (4 performance "
               "cores achieve 3.89x).",
               "Best throughput: ")
    pdf.bullet("Task Parallel saturates at ~1.3x regardless of thread count because it is bounded "
               "by the slower branch (sparse BM25, which takes 2.7x longer than dense HNSW).",
               "Task Parallel ceiling: ")
    pdf.bullet("Data Parallel peaks at 1.95x with 4 threads, then degrades at 8+ threads due to "
               "synchronization overhead in the partial-result merge step.",
               "Data Parallel peak: ")
    pdf.bullet("Stage breakdown reveals sparse BM25 accounts for 68% of per-query time (0.111ms), "
               "dense HNSW for 26% (0.042ms), and fusion for 6% (0.011ms).",
               "Stage breakdown: ")

    pdf.subsection_title("8.2  Scalability Study (3.9K - 64K records)")

    cw3 = [22, 24, 27, 24, 27, 27]
    pdf.add_table(
        ["Corpus", "FullPar", "Combined", "SIMD", "MaxScore", "Temporal"],
        [
            ["3,933",  "4.1x",  "2.5x",  "18.0x", "21.9x", "8.0x"],
            ["12,135", "4.0x",  "4.0x",  "6.8x",  "11.8x", "12.4x"],
            ["24,390", "4.1x",  "7.1x",  "5.7x",  "9.9x",  "19.1x"],
            ["39,984", "4.0x",  "8.7x",  "6.3x",  "8.4x",  "29.6x"],
            ["63,621", "3.4x", "10.7x",  "6.0x",  "9.2x",  "46.7x"],
        ],
        col_widths=cw3,
        highlight_col=5,
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 100, 50)
    pdf.cell(0, 6, "Key insight: Sub-linear scaling of temporal partitioning")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30)
    pdf.body_text(
        "When corpus size grows 16x (3,933 to 63,621 records), sequential BM25 latency grows 6.1x "
        "(117us to 717us). However, temporal partitioned latency grows only 1.1x (14.6us to 15.3us), "
        "achieving near-constant-time retrieval. This is because the temporal index searches only the "
        "most recent partitions: at 64K records, only 2.6% of the corpus is searched. This property "
        "is critical for long-running AI agents whose memory grows unboundedly over time.")

    pdf.body_text(
        "The Combined strategy shows the opposite scaling behavior: it improves from 2.5x at 3.9K to "
        "10.7x at 64K because intra-query data parallelism becomes more effective as the per-query "
        "workload grows with corpus size, amortizing the nested parallelism overhead.")

    pdf.subsection_title("8.3  Amdahl's Law Analysis")
    pdf.body_text(
        "From the Full Parallel speedup curve (1.97x at 2T, 3.89x at 4T, 8.28x at 12T), we estimate "
        "the serial fraction using Amdahl's Law: S(N) = 1 / (f + (1-f)/N). Fitting the data yields "
        "f = 7-10%, giving a theoretical maximum speedup of 10-14x. The primary serial bottleneck is "
        "the RRF fusion stage, which requires a global merge of candidate lists and cannot be parallelized "
        "across queries.")

    pdf.subsection_title("8.4  Compute-Bound vs. Memory-Bound Analysis")
    pdf.body_text(
        "Stage-level timing with data parallelism reveals the fundamental difference between pipeline "
        "stages. Under data-parallel BM25, sparse latency drops from 0.111ms (1T) to 0.032ms (4T), "
        "a 3.5x reduction, confirming BM25 is compute-bound. Dense HNSW latency remains at 0.042-"
        "0.045ms regardless of thread count, confirming it is memory-bound (limited by cache misses "
        "during random graph traversal). This explains why task parallelism (sparse || dense) is "
        "ineffective: the memory-bound dense stage cannot be accelerated by additional compute.")

    # ======== Section 9: Correctness ========
    pdf.section_title("9", "Correctness Validation")

    pdf.body_text(
        "All parallel strategies are validated against the sequential baseline on every run. For each "
        "of 200 queries, we verify two properties:")

    pdf.bullet("The top-1 document matches by ID, or if tie-breaking in partial_sort produces a "
               "different ordering, the top-1 scores match within epsilon = 10^-3 for pipeline "
               "strategies and 10^-4 for sparse-only strategies.",
               "Top-1 match: ")
    pdf.bullet("Kendall-tau rank correlation of the full top-10 result list exceeds 0.8, ensuring "
               "the overall ranking is preserved even when individual positions differ due to "
               "floating-point non-associativity in parallel reductions.",
               "Rank correlation: ")

    pdf.ln(1)
    pdf.body_text(
        "All strategies (task_parallel, data_parallel, full_parallel, combined at 4T and 8T; "
        "SIMD, MaxScore, data_parallel_sparse, SIMD+parallel) pass 200/200 queries on both metrics.")

    # ======== Section 10: Deliverables ========
    pdf.section_title("10", "Deliverables")

    pdf.bullet("Complete C++17 codebase (~5,700 LOC) with CMake build system, "
               "organized as two static libraries (hybrid_retrieval, agent_memory) plus demo executable.",
               "Source code: ")
    pdf.bullet("Automated demo that generates two CSV outputs (agent_benchmark.csv, scalability.csv) "
               "and runs all benchmarks including correctness validation, hierarchical memory, and "
               "scalability study.",
               "Benchmark suite: ")
    pdf.bullet("Python plotting script producing 10 publication-quality figures: speedup, throughput, "
               "efficiency, stage breakdown, latency distribution, Amdahl's Law fit, memory-bound "
               "analysis, scalability latency (log-log), scalability speedup, scalability throughput.",
               "Visualization: ")
    pdf.bullet("Performance analysis with bottleneck identification, scaling discussion, and "
               "Amdahl's Law estimation.",
               "Final report: ")

    pdf.ln(2)
    pdf.body_text("Generated plots (saved to build/results/):")
    pdf.code_block(
        "speedup.png              - Speedup vs thread count\n"
        "throughput.png           - Throughput vs thread count\n"
        "stage_breakdown.png      - Per-stage timing breakdown\n"
        "efficiency.png           - Parallel efficiency curves\n"
        "latency.png              - Per-query latency distribution\n"
        "amdahl.png               - Amdahl's Law fit\n"
        "memory_bound.png         - Compute vs memory bound analysis\n"
        "scalability_latency.png  - Latency vs corpus size (log-log)\n"
        "scalability_speedup.png  - Speedup vs corpus size\n"
        "scalability_throughput.png - Throughput vs corpus size")

    # ======== Section 11: Conclusion ========
    pdf.section_title("11", "Conclusion")

    pdf.body_text(
        "This project demonstrates that hybrid retrieval for agent memory is a rich parallelization "
        "problem with multiple interacting bottlenecks. Our key findings are:")

    pdf.bullet("Query-level parallelism (Full Parallel) provides the most consistent throughput scaling, "
               "achieving 8.3x at 12 threads with 69% efficiency. It scales near-linearly up to the "
               "physical core count.",
               "1. Best parallel strategy: ")
    pdf.bullet("Task parallelism is fundamentally limited (~1.3x) by the asymmetry between compute-"
               "bound sparse search and memory-bound dense search. Data parallelism peaks at 4 threads "
               "before synchronization overhead dominates.",
               "2. Strategy-specific limits: ")
    pdf.bullet("SIMD vectorization (6-18x) and MaxScore pruning (8-22x) provide substantial per-query "
               "latency reductions that are orthogonal to multi-threaded parallelism and can be "
               "composed with it.",
               "3. Algorithmic optimizations: ")
    pdf.bullet("The temporal partitioned index achieves sub-linear scaling with corpus size: as the "
               "corpus grows 16x, temporal latency grows only 1.1x (47x speedup at 64K records). "
               "This is the single most impactful optimization for long-running agents.",
               "4. Temporal partitioning: ")
    pdf.bullet("The three-tier hierarchical memory adds cognitive-inspired organization with concurrent "
               "cross-tier retrieval, consolidation, and decay, demonstrating task-parallel design "
               "patterns beyond simple loop parallelism.",
               "5. Hierarchical memory: ")

    pdf.ln(2)
    pdf.body_text(
        "Future extensions could include NUMA-aware thread placement for server-class hardware, "
        "distributed sharding with MPI for corpora exceeding single-machine memory, GPU-accelerated "
        "dense retrieval, and integration with production agent frameworks.")

    # ======== References ========
    pdf.section_title("", "References")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30)
    refs = [
        "[1] Robertson, S. and Zaragoza, H. \"The Probabilistic Relevance Framework: BM25 and Beyond.\" "
        "Foundations and Trends in Information Retrieval, 3(4), 2009.",
        "[2] Cormack, G., Clarke, C., and Buettcher, S. \"Reciprocal Rank Fusion outperforms Condorcet "
        "and individual Rank Learning Methods.\" SIGIR 2009.",
        "[3] Malkov, Y. and Yashunin, D. \"Efficient and Robust Approximate Nearest Neighbor Search "
        "Using Hierarchical Navigable Small World Graphs.\" IEEE TPAMI, 2018.",
        "[4] Turtle, H. and Flood, J. \"Query Evaluation: Strategies and Optimizations.\" "
        "Information Processing & Management, 31(6), 1995.",
        "[5] Packer, C. et al. \"MemGPT: Towards LLMs as Operating Systems.\" arXiv:2310.08560, 2023.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 4.5, ref)
        pdf.ln(1.5)

    # Save
    out_path = "/Users/justin/ee451-hybrid-retrieval/proposal/EE451_final_proposal.pdf"
    pdf.output(out_path)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
