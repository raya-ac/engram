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
     optional cross-encoder reranking (BGE by default; MiniLM or Voyage optional)
           │
           ▼
     confidence gate + prior coverage when cross-encoder reranking is on
           │
           ▼
     deep MLP reranker (optional, trained on access patterns)
           │
           ▼
     noise only when cross-encoder reranking is off
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

production search and the benchmark share date parsing and relative windows in
`engram/temporal.py`, including slash-separated dates, hyphenated dates and ISO
timestamps.

## stage 4: cross-encoder reranking

the configured shortlist, 20 candidates by default, is scored by a cross-encoder
using each query/document pair. the default local model is
`BAAI/bge-reranker-base`; MiniLM and supported Voyage models are optional.

with `rerank_passage_fallback: true` (the default), a local reranker can retry
one excerpt per long document when every full-document sigmoid score is below
`rerank_passage_floor` (default 0.001). each excerpt contains a matching sentence and nearby context,
is copied directly from the source, and is capped at 160 words. shorter
documents and documents without matching query terms keep their original
scores. the same semantic query scores both versions; the larger raw score
survives. matching supports conservative regular English singular/plural forms
and counts each original query term once per sentence. hosted rerankers skip
the retry.

this adds inference work and may omit useful context. set
`rerank_passage_fallback: false` to disable it. [retrieval internals](../architecture/retrieval.md#focused-excerpt-retry)
describe date handling and the raw-score/source-offset traces. the final
`min_confidence` gate independently controls returned results, with a default
of 0.6, even when the excerpt improves a result's rank. the activation floor was
chosen during development on LongMemEval; evaluation on that dataset is not
held-out accuracy, and neither threshold is a calibrated probability.

local logits pass through one sigmoid. hosted relevance scores keep their
normalized scale. temporal evidence applies in logit space before an optional
blend with reciprocal pre-rerank position. set `rerank_fusion_alpha` between 0
and 1 to enable that blend; 0 is the default. final scores stay between 0 and 1,
without separate lexical bonuses. these scores are not measured probabilities
that a memory is correct.

## stage 5: confidence gate and prior coverage

cross-encoder results receive no random ranking noise. `min_confidence` filters
their final scores before coverage. with `preserve_prior_candidate: true`, a
request for at least two results retains the best eligible hybrid candidate.
if missing, that candidate moves to the last requested position while the
rerank winner stays first. coverage cannot admit a candidate rejected by the
confidence gate.

scores stay unchanged, so preserve the returned order instead of sorting again
by score. fusion weight 0 disables score blending; candidate coverage can still
adjust order. set `preserve_prior_candidate: false` to disable that step.

## stage 6: deep MLP reranker

an optional learned reranker uses access patterns after candidate selection.
its two-layer MLP combines features including cosine similarity, importance,
access count, age, layer and retention score. train with `train_reranker` after
accumulating usage data.

## stage 7: noise and cache

searches with reranking off retain the gaussian noise term (σ=0.02). the rerank
cache includes fusion, confidence, coverage, passage fallback and its separate
activation floor, so policy changes take effect on the next search.

## retrieval profiles

the `recall` tool accepts a `mode` parameter that filters candidates by memory type *before* cross-encoder reranking:

| mode | types included | use case |
|------|---------------|----------|
| `facts_only` | fact | statuses, states, structured answers |
| `facts_plus_rules` | fact + procedure | methodology, how-to queries |
| `full_context` | fact + procedure + narrative | exhaustive recall (default) |

filtering happens after RRF fusion (stage 2) but before the cross-encoder (stage 4). non-active memories (status != active) are also excluded at this stage.

this means `facts_only` mode still benefits from the full 4-channel candidate generation — it just removes narrative-typed candidates before the expensive reranking step.

## tuning

all parameters are in `config.yaml`:

```yaml
retrieval:
  top_k: 10              # final results
  rrf_k: 60              # RRF constant
  min_confidence: 0.60   # threshold gate
  rerank_candidates: 20  # cross-encoder shortlist
  rerank_fusion_alpha: 0.0 # optional prior rank blend, from 0 to 1
  preserve_prior_candidate: true # keep the best eligible hybrid candidate
  rerank_passage_fallback: true # retry one focused excerpt when all local scores are low
  rerank_passage_floor: 0.001 # activation floor, independent of min_confidence
  dense_multiplier: 3    # dense candidates = top_k * 3
  bm25_multiplier: 3     # BM25 candidates = top_k * 3
```

## explain a result or a miss

```bash
engram config show
engram search "deployment strategy" --rerank --explain --json
```

`--explain` keeps the CLI's normal reranking choice: include `--rerank` to inspect
the cross-encoder and confidence gate. `--debug` remains an alias. the JSON object
contains `results` and `explanation`, even if every candidate is rejected.

each observed candidate has an outcome and reason: returned, below the confidence
cutoff, outside the result budget, or filtered before scoring. eligible candidates
include their observed signals, raw local logits or hosted normalized scores,
temporal adjustments and any excerpt retry. forgotten, inactive and
profile-filtered candidates never include their content.

diagnostics bypass the result cache and leave access history and dormant
evaluations unchanged. MCP `recall_explain`, native `search_explain`, and
`/api/search/explain` use the same report; the web search's **Explain retrieval**
option displays it. MCP explanation also leaves session handoffs unchanged.

an empty result can mean a relevant memory failed the confidence cutoff. inspect
that reason before changing a threshold. a score is not a calibrated probability,
and an explanation does not establish that the memory's claim is true.
