# Engram Improvement Roadmap

Based on analysis of 15 recent papers (2025-2026) on AI agent memory systems.

## Tier 1 — High Impact, Low Effort (do first)

### 1. Enriched Embeddings (A-Mem)
**Paper:** A-Mem (2502.12110)
**What:** At write time, generate keywords + categorical tags + contextual summary via LLM, then embed `concat(content, keywords, tags, summary)` instead of just content. The richer vector captures semantic dimensions the raw text misses.
**Impact:** Multi-hop F1 nearly doubled in A-Mem benchmarks.
**Files:** `mcp_server.py` (_remember), `embeddings.py`
**Effort:** Small — one extra LLM call at write time, concat before embed.

### 2. Memory Evolution on Write (A-Mem)
**Paper:** A-Mem (2502.12110)
**What:** When surprise gate detects near-neighbors at write time, feed new memory + neighbors to LLM and ask: "should any existing memories be updated?" If yes, rewrite their contextual descriptions and re-embed. Memories get smarter over time instead of going stale.
**Impact:** Removing evolution dropped multi-hop F1 from 45.85% to 31.24% in ablation.
**Files:** `mcp_server.py` (_remember), new `evolution.py`
**Effort:** Medium — extends the existing surprise gate flow.

### 3. Temporal Edges on Entity Graph (Zep/Graphiti)
**Paper:** Zep (2501.13956)
**What:** Add temporal dimension to relationships. Every entity edge gets `valid_from` and `valid_to` timestamps. Relationships can expire. "Ari uses Flask" was true in March, not anymore. Query-time filtering: "what did X use in March?" traverses only temporally-valid edges.
**Impact:** Temporal reasoning accuracy jumped significantly in Zep benchmarks.
**Files:** `store.py` (relationships table), `retrieval.py` (temporal filter)
**Effort:** Medium — schema migration + retrieval filter.

### 4. Quality Metrics from Access Patterns (AgeMem)
**Paper:** AgeMem (2601.01885)
**What:** Track three ongoing health metrics:
- Storage quality: fraction of stored memories that get retrieved later
- Curation ratio: sessions with updates/invalidations vs just writes
- Retrieval relevance: how well retrieved memories match actual queries
**Impact:** Gives a real signal on whether the memory system is *useful*, not just *big*.
**Files:** `lifecycle.py`, `mcp_server.py` (health tool), `web/routes.py`
**Effort:** Small — mostly instrumentation on existing access_log.

## Tier 2 — High Impact, Medium Effort

### 5. Intent-Aware Retrieval Router (MAGMA)
**Paper:** MAGMA (2601.03236)
**What:** Classify query intent (Why/When/Entity/What) and dynamically weight retrieval signals. "Why" queries boost causal/procedural memories. "When" queries boost temporal matching. "What" queries boost semantic similarity. Currently engram treats all queries the same.
**Formula:** `S(n|q) = exp(λ₁·φ(edge_type, intent) + λ₂·sim(n, q))`
**Impact:** Removing adaptive policy dropped MAGMA's score from 0.700 to 0.637.
**Files:** `retrieval.py` (add intent classification before search)
**Effort:** Medium — lightweight classifier + weight adjustment.

### 6. Causal Edge Inference (MAGMA)
**Paper:** MAGMA (2601.03236)
**What:** During consolidation, analyze memory neighborhoods and infer causal edges: "X happened because of Y." Store as directed CAUSED_BY relationships. Enables "why did this happen?" queries via causal traversal.
**Files:** `consolidator.py` (new step), `store.py` (edge type)
**Effort:** Medium — LLM inference during dream cycle.

### 7. Memory-to-Memory Links (A-Mem + Zep)
**Paper:** A-Mem (2502.12110), Zep (2501.13956)
**What:** Beyond entity-mediated relationships, create direct memory-to-memory links when LLM confirms meaningful connection. Different from entity graph — captures conceptual links between memories that don't share entities.
**Files:** `store.py` (new table), `mcp_server.py`
**Effort:** Medium — new relation type + LLM gating.

### 8. Proactive Working Memory Management (AgeMem)
**Paper:** AgeMem (2601.01885)
**What:** During sessions, periodically check working memories against current focus. Auto-demote anything below cosine similarity θ=0.6. When working memory accumulates N items from same session, auto-summarize into one episodic memory. Don't wait for dream cycle.
**Files:** `mcp_server.py` (_sweep_working), `lifecycle.py`
**Effort:** Medium — extends existing sweep.

### 9. ACT-R Base-Level Activation (Human-Like Memory)
**Paper:** ACT-R Inspired Memory (HAI 2025)
**What:** Replace or supplement importance scoring with ACT-R's base-level activation:
`B_i = ln(Σ t_j^(-d))` where t_j is time since jth access and d=0.5 is decay.
Also: spreading activation — when entity X is activated, boost connected entities by `S_ji * W_j`.
**Impact:** Produces more human-like recall patterns than flat importance scores.
**Files:** `lifecycle.py`, `retrieval.py`
**Effort:** Medium — new scoring function.

## Tier 3 — Transformative, High Effort

### 10. Multi-Graph Architecture (MAGMA)
**Paper:** MAGMA (2601.03236)
**What:** Four orthogonal graph views per memory: semantic (cosine similarity edges), temporal (ordered sequence), causal (directed cause-effect), entity (shared entities). Each graph enables different traversal patterns. Currently engram has one merged entity graph.
**Data structure:** Each node `n_i = ⟨content, timestamp, embedding, metadata⟩`. Four edge sets maintained independently.
**Impact:** MAGMA scored 0.700 vs next best 0.590 on LoCoMo.
**Files:** Major refactor of `store.py`, `retrieval.py`, `consolidator.py`
**Effort:** High — fundamental architecture change.

### 11. Dual-Stream Write (MAGMA + Zep)
**Paper:** MAGMA (2601.03236), Zep (2501.13956)
**What:** Fast path: immediate ingestion (embed, temporal link, vector index). Slow path: async consolidation (LLM infers causal/semantic edges, densifies graph). Decouples responsiveness from reasoning depth.
**Files:** `mcp_server.py`, new `async_consolidator.py`
**Effort:** High — async processing pipeline.

### 12. Community Detection (Zep/Graphiti)
**Paper:** Zep (2501.13956)
**What:** Three-tier graph: episodes → semantic entities → communities. Communities are clusters of related entities discovered via graph algorithms (Leiden/Louvain). Enables "what's the big picture around X?" queries.
**Files:** `entities.py`, `consolidator.py`, new community detection
**Effort:** High — graph clustering algorithm.

### 13. Learnable Memory Operations (AgeMem)
**Paper:** AgeMem (2601.01885)
**What:** Instead of hardcoded thresholds for forgetting/promotion, train a policy (via RL) to decide when to store/retrieve/update/summarize/delete. Three-stage training with terminal reward. Our deep reranker already learns retrieval preferences — extend to lifecycle.
**Files:** Major new training pipeline
**Effort:** Very high — RL training infrastructure.

### 14. MemoryBench Evaluation (MemoryBench)
**Paper:** MemoryBench (2510.17281)
**What:** Standardized benchmark for memory systems. Adversarial distractors, multi-hop reasoning, temporal queries, knowledge updates. Run this against engram to get real numbers.
**Files:** New eval/ directory
**Effort:** Medium — setup benchmark, run eval.

## Tier 4 — Research Ideas Worth Tracking

### 15. Cognitive Quantization (SuperLocalMemory V3.3)
**Paper:** SuperLocalMemory (2604.04514)
**What:** Multi-channel retrieval with biologically-inspired forgetting. Zero-LLM approach. Interesting for reducing LLM dependency in the retrieval pipeline.

### 16. MemOS Abstraction (MemOS)
**Paper:** MemOS (2505.22101)
**What:** Memory as an OS — unified API over parametric, contextual, and external memory. Clean abstraction but mostly relevant for multi-model systems.

### 17. Procedural Memory Templates (Mem^p)
**Paper:** Mem^p (2508.06433)
**What:** Deep dive on how agents should learn and store procedures. Our pattern extraction is basic compared to what's possible — templated procedures with variable slots, preconditions, postconditions.

## Implementation Order

**Week 1:** Items 1 (enriched embeddings) + 4 (quality metrics) — quick wins
**Week 2:** Items 2 (memory evolution) + 3 (temporal edges) — core improvements  
**Week 3:** Items 5 (intent router) + 8 (proactive working memory) — retrieval upgrade
**Week 4:** Items 6 (causal edges) + 7 (memory-to-memory links) — graph enrichment
**Month 2:** Items 9 (ACT-R) + 14 (MemoryBench) — scoring + evaluation
**Month 3+:** Items 10-13 — architectural transformation

## Papers Referenced

| # | Paper | Year | Key Contribution |
|---|-------|------|-----------------|
| 1 | [A-Mem](https://arxiv.org/abs/2502.12110) | 2025 | Zettelkasten memory with evolution |
| 2 | [AgeMem](https://arxiv.org/abs/2601.01885) | 2026 | RL-trained memory operations |
| 3 | [Zep/Graphiti](https://arxiv.org/abs/2501.13956) | 2025 | Temporal knowledge graph |
| 4 | [Graph Memory Taxonomy](https://arxiv.org/abs/2602.05665) | 2026 | Graph memory classification |
| 5 | [MAGMA](https://arxiv.org/abs/2601.03236) | 2026 | Multi-graph architecture |
| 6 | [ACT-R Memory](https://dl.acm.org/doi/10.1145/3765766.3765803) | 2025 | Cognitive activation model |
| 7 | [SuperLocalMemory](https://arxiv.org/abs/2604.04514) | 2026 | Bio-inspired forgetting |
| 8 | [Mem^p](https://arxiv.org/abs/2508.06433) | 2025 | Procedural memory |
| 9 | [Mem0](https://arxiv.org/abs/2504.19413) | 2025 | Production memory at scale |
| 10 | [Memoria](https://arxiv.org/abs/2512.12686) | 2025 | Session summarization + KG |
| 11 | [MemOS](https://arxiv.org/abs/2505.22101) | 2025 | Memory as OS abstraction |
| 12 | [Autonomous LLM Memory](https://arxiv.org/abs/2603.07670) | 2026 | Latest comprehensive survey |
| 13 | [Rethinking Memory](https://arxiv.org/abs/2602.06052) | 2026 | Foundation model for memory |
| 14 | [MemoryBench](https://arxiv.org/abs/2510.17281) | 2025 | Evaluation benchmark |
| 15 | [Anatomy of Agentic Memory](https://arxiv.org/abs/2602.19320) | 2026 | Evaluation limitations |
| + | [Titans](https://arxiv.org/abs/2501.00663) | 2025 | Surprise-based memorization (already in engram) |
| + | [Miras](https://arxiv.org/abs/2504.13173) | 2025 | Retention regularization (already in engram) |
| + | [SkillsBench](https://arxiv.org/abs/2602.12670) | 2026 | Skill injection (already in engram) |
| + | [Your Brain on ChatGPT](https://arxiv.org/abs/2506.08872) | 2025 | Cognitive scaffolding (already in engram) |
