# Retrieval Internals

deep dive into the 8-stage hybrid retrieval pipeline. see [Retrieval Pipeline](../guides/retrieval-pipeline.md) for the user-facing guide.

## intent classification

regex-based, zero cost:

```python
INTENT_PATTERNS = {
    "why": r'\b(why|because|reason|cause|led to|resulted in)\b',
    "when": r'\b(when|date|time|before|after|during|timeline|history)\b',
    "who": r'\b(who|person|people|team|built|created|wrote)\b',
    "how": r'\b(how to|steps|procedure|process|workflow|debug|fix)\b',
}
# default: "what" (balanced weights)
```

each intent adjusts channel weights:

| intent | dense | BM25 | graph |
|--------|-------|------|-------|
| why | 1.0 | 0.8 | **1.5** |
| when | 0.8 | **1.2** | 0.8 |
| who | 0.8 | 0.8 | **1.8** |
| how | **1.2** | 1.0 | 0.8 |
| what | 1.0 | 1.0 | 1.0 |

## RRF fusion

reciprocal rank fusion from [Cormack et al. 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf):

```
score(doc) = Σ weight / (k + rank + 1)
```

k=60 is the standard constant. each channel contributes independently — a document ranked #1 in dense and #5 in BM25 gets a higher fused score than a document ranked #2 in both.

## temporal boost

date detection via regex, then boost based on proximity to the query's temporal signal:

- matched date in memory: 2x boost
- episodic memories get recency decay: `exp(-0.693 * age_days / half_life)`, floored at 50%
- access frequency: `1.0 + 0.1 * log(1 + access_count)`

## cross-encoder

top-20 candidates from RRF get jointly scored by a cross-encoder. the cross-encoder sees (query, document) together, not just embeddings — much more accurate for nuanced relevance.

local: `ms-marco-MiniLM-L-6-v2` (~300ms for 20 docs)
cloud: `rerank-2.5` via Voyage API (better quality, 32k context)

## deep MLP reranker

optional 7th stage. 2-layer MLP trained on access patterns:

- input: 10 features (cosine sim, importance, access count, age, layer one-hot, retention)
- output: relevance prediction
- persisted to `~/.local/share/engram/reranker.npz`
- trains on which memories actually get accessed after being returned in search

## ACT-R noise + threshold

gaussian noise (σ=0.02) for beneficial retrieval variation — prevents the same top results every time. minimum score threshold gates out garbage (only applies when cross-encoder is active, since RRF scores are on a different scale).

## key files

- `engram/retrieval.py` — the full pipeline
- `engram/embeddings.py` — dense search + cross-encoder
- `engram/ann_index.py` — HNSW wrapper
- `engram/hopfield.py` — associative channel
- `engram/deep_retrieval.py` — learned reranker
