# Retrieval Pipeline

engram runs an 8-stage hybrid retrieval pipeline that fuses four parallel search channels.

## the pipeline

```
query
  │
  ├── intent classification (why/when/who/how/what)
  │         → dynamic signal weights per intent type
  │
  ├── dense HNSW search (bge-small, 384-dim, hnswlib)       → top 3k candidates
  ├── BM25 via sqlite FTS5 (content + hypothetical queries)  → top 3k candidates
  ├── entity graph BFS (1-hop traversal, strength-weighted)  → top k candidates
  └── Hopfield associative (pattern completion, β=8.0)       → top k candidates
           │
           ▼
     intent-weighted reciprocal rank fusion (k=60)
           │
           ▼
     temporal + importance boosting
           │
           ▼
     cross-encoder reranking (ms-marco-MiniLM or Voyage rerank-2.5)
           │
           ▼
     deep MLP reranker (optional, trained on access patterns)
           │
           ▼
     gaussian noise (ACT-R, σ=0.02) + threshold gate
           │
           ▼
     final top-k results
```

## stage 0: intent classification

queries are classified into 5 intent types using regex patterns:

| intent | triggers | effect |
|--------|----------|--------|
| `why` | "why", "because", "reason" | boost graph (causal reasoning) |
| `when` | "when", "date", "timeline" | boost BM25 (date matching) |
| `who` | "who", "person", "built" | boost graph (entity lookup) |
| `how` | "how to", "steps", "fix" | boost dense (procedural) |
| `what` | default | balanced weights |

## stage 1: parallel candidate generation

four channels run independently:

### dense search (HNSW)

embeds the query with bge-small-en-v1.5, searches the HNSW index for approximate nearest neighbors. O(log n) at any scale.

### BM25 (FTS5)

full-text search via SQLite's FTS5 extension. matches on content text and hypothetical queries (generated at ingestion time via docTTTTTquery).

### entity graph BFS

extracts entity names from the query, finds matching entities, retrieves their memories (hop 0, score 1.0), then traverses 1-hop related entities (score 0.5 * relationship strength).

### Hopfield associative

pattern completion via modern Hopfield network: `ξ_new = X^T · softmax(β · X · ξ)`. retrieves memories by associative recall, not just similarity.

## stage 2: RRF fusion

reciprocal rank fusion combines all four channels:

```
score(doc) = Σ weight_intent · 1/(60 + rank) across channels
```

the k=60 constant comes from [Cormack et al. 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf). intent-specific weights adjust channel importance per query type.

## stage 3: temporal + importance boosting

- temporal boost for date-matching memories
- importance scaling: `score *= (0.8 + 0.4 * importance)`
- recency decay (Ebbinghaus): `exp(-0.693 * age_days / half_life)` for episodic memories
- access frequency boost: `1.0 + 0.1 * log(1 + access_count)`

## stage 4: cross-encoder reranking

top-20 candidates are jointly scored by a cross-encoder (query + document → relevance score). more accurate than bi-encoder similarity but O(n*q) so only applied to the shortlist.

supports local (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and cloud (`rerank-2.5` via Voyage API).

## stage 5: deep MLP reranker

optional learned reranker trained on access patterns. 2-layer MLP taking 10 features (cosine similarity, importance, access count, age, layer, retention score) → relevance prediction. <1ms per query.

train with `train_reranker` after accumulating usage data.

## stage 6: noise + threshold

small gaussian noise (σ=0.02, [ACT-R](https://dl.acm.org/doi/10.1145/3765766.3765803) inspired) for beneficial retrieval variation. minimum score threshold gates out low-quality results.

## tuning

all parameters are in `config.yaml`:

```yaml
retrieval:
  top_k: 10              # final results
  rrf_k: 60              # RRF constant
  min_confidence: 0.60   # threshold gate
  rerank_candidates: 20  # cross-encoder shortlist
  dense_multiplier: 3    # dense candidates = top_k * 3
  bm25_multiplier: 3     # BM25 candidates = top_k * 3
```

## debugging

use `--debug` on CLI or `debug=true` on the API to see per-stage breakdown:

```bash
engram search "deployment strategy" --debug
```

returns dense candidates, BM25 candidates, graph candidates, RRF scores, boosted scores, reranked scores, and total latency.
