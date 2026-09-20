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

`engram/temporal.py` provides the date parser and relative windows used by both
production retrieval and the benchmark. it accepts slash-separated dates,
hyphenated dates and ISO timestamps. candidates are boosted based on proximity
to the query's temporal signal:

- matched date in memory: 2x boost
- episodic memories get recency decay: `exp(-0.693 * age_days / half_life)`, floored at 50%
- access frequency: `1.0 + 0.1 * log(1 + access_count)`

## cross-encoder

up to `retrieval.rerank_candidates` candidates are scored as query/document
pairs; the default shortlist is 20. the default model is
`BAAI/bge-reranker-base`. `cross-encoder/ms-marco-MiniLM-L-6-v2` remains available
locally, and Voyage rerankers provide a hosted option.

### focused excerpt retry

`retrieval.rerank_passage_fallback` defaults to `true`. after scoring the full
documents, a local reranker retries excerpts only when **every** base logit,
converted through sigmoid, is below `rerank_passage_floor` (default 0.001). this check
precedes temporal adjustments and prior blending. one score at or above that
floor prevents the retry. setting the floor to 0 also disables it; hosted
rerankers do not use this fallback. `min_confidence` independently controls
returned results and defaults to 0.6.

each document longer than 160 words can contribute at most one excerpt. the
selector finds a sentence with matching query terms, favoring terms that occur
in fewer sentences, and includes its immediate neighbors. matching includes
conservative regular English singular/plural forms. each distinct original
query term contributes once per sentence, including when it is repeated or
matches through multiple surface forms. the excerpt remains
a contiguous slice of the source, capped at 160 whitespace-separated words.
documents of 160 words or fewer and documents without a lexical match retain
their full-document scores.

both model calls receive the same semantic query. when an explicit reference
date resolves a relative-time phrase, that phrase is excluded only from the
lexical excerpt-selection query. it stays in semantic inference and temporal
scoring. the retained raw score is `max(full_document, excerpt)`.

score traces include `base_raw_score` and, for a retried document,
`excerpt_raw_score`, `source_start`, `source_end` and `passage_words`. offsets
are character positions in the document supplied to the reranker, with an
exclusive end. the returned memory remains the full memory.

this adds up to one extra model pair per eligible document. an excerpt can
omit qualifications or other useful context; the word cap is not a tokenizer
token limit. set `rerank_passage_fallback: false` to skip this work. the normal
confidence gate still runs after score adjustments, so improving a document's
rank does not ensure production search returns it.

the 0.001 activation floor was selected during development on LongMemEval.
evaluation on that same dataset measures development performance, not held-out
accuracy. neither this floor nor the final confidence gate is a calibrated
probability of correctness.

### score conversion and selection

local encoders return raw logits. ordinary retrieval applies one sigmoid to
map them into the 0–1 range. hosted relevance scores are already normalized and
do not pass through another sigmoid. a matching resolved temporal window adds
5 in logit space; hosted scores are converted to log-odds for this adjustment.

`retrieval.rerank_fusion_alpha` optionally blends this score with the reciprocal
pre-rerank position. with zero-based rank `r` and fusion weight `a`:

```text
final_score = (1 - a) * model_score + a / (r + 1)
```

the default weight is 0; valid weights run from 0 to 1. the resulting scores are
bounded, without separate lexical or top-three bonuses. these scores express
ranking relevance, not measured probabilities of correctness.

after the confidence gate, `preserve_prior_candidate: true` keeps the best
eligible hybrid candidate in requests for at least two results. when that ID is
absent from the requested prefix, it moves to the last requested position.
the model winner remains first and every other candidate keeps its relative
order. requests for one result keep the model winner.

this coverage step does not change score values or make rejected candidates
eligible. result order can therefore differ from numerical score order, even
with `rerank_fusion_alpha: 0.0`. consumers should preserve the returned order.
set `preserve_prior_candidate: false` to disable coverage.

## deep MLP reranker

optional 7th stage. 2-layer MLP trained on access patterns:

- input: 10 features (cosine sim, importance, access count, age, layer one-hot, retention)
- output: relevance prediction
- persisted to `~/.local/share/engram/reranker.npz`
- trains on which memories actually get accessed after being returned in search

## noise, threshold and cache

cross-encoder results receive no random ranking noise. `min_confidence` gates
their final scores before the coverage step chooses from eligible candidates.
searches with reranking off retain the gaussian noise term
(σ=0.02); their RRF scores use a different scale and do not use this gate.

the rerank cache includes the fusion weight, minimum confidence, coverage flag,
passage fallback flag and its independent activation floor, so changes cannot
return results calculated under the earlier policy.
given the same candidates, model scores and settings, the rerank scoring step is
deterministic.

## key files

- `engram/retrieval.py` — the full pipeline
- `engram/embeddings.py` — dense search + cross-encoder
- `engram/rerank_scoring.py` — shared score conversion and optional prior blend
- `engram/rerank_passages.py` — bounded local excerpt selection and retry traces
- `engram/rerank_selection.py` — eligible hybrid leader coverage without score changes
- `engram/temporal.py` — shared date parsing and relative windows
- `engram/ann_index.py` — HNSW wrapper
- `engram/hopfield.py` — associative channel
- `engram/deep_retrieval.py` — learned reranker
