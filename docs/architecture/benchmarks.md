# benchmarks and their limits

## what the retrieval benchmark measures

`benchmarks/longmemeval/run_engram.py` evaluates session retrieval. its
`recall_any@5` metric succeeds when at least one gold session appears among the
first five retrieved sessions. this is not generated-answer accuracy, evidence
verification, or the usefulness of a dormant suggestion.

the standard dataset contributes 470 answerable retrieval questions; the 30
abstention questions ending in `_abs` are excluded. the runner also skips
entries without `answer_session_ids` and builds a separate corpus for each
question.

the runner ranks candidates without the production `min_confidence` gate.
its result is session ranking recall, not production search recall or answer
accuracy. production search applies its configured confidence gate before
prior coverage and may return fewer results or none. this benchmark does not
measure that abstention behavior, the complete memory lifecycle or the current
production database.

## latest fresh validation

the September 20, 2026 development candidate completed a fresh full run with
**470/470 session recall-any@5 (100.0%)** and **0 top-five misses**.
all embeddings and cross-encoder scores were recomputed in a new process;
the run did not use saved inference scores or resume earlier result rows.
independent recomputation checked every expected question ID, all recorded
recall and NDCG metrics, dataset integrity, and unchanged source hashes.
those hashes identify the measured retrieval candidate published in commit
`c3b8cea`. 0.7.0 subsequently adds configuration validation, retrieval
explanations and the website redesign. the artifact is preserved unchanged;
it is not a claim that the full release package was benchmarked again.

| metric | fresh result |
| --- | ---: |
| recall-any@1 | 439/470 (93.40%) |
| recall-any@3 | 462/470 (98.30%) |
| recall-any@5 | 470/470 (100.0%) |
| recall-any@10 | 470/470 (100.0%) |
| recall-all@5 | 428/470 (91.06%) |
| NDCG@5 | 94.74% |

| question type | evaluated | hits in top 5 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: |
| knowledge-update | 72 | 72 | 100.0% | 100.0% |
| multi-session | 121 | 121 | 100.0% | 100.0% |
| single-session-assistant | 56 | 56 | 100.0% | 100.0% |
| single-session-preference | 30 | 30 | 100.0% | 100.0% |
| single-session-user | 64 | 64 | 100.0% | 100.0% |
| temporal-reasoning | 127 | 127 | 100.0% | 100.0% |

the run used BGE reranker base, MLX embeddings, zero prior score blending,
hybrid candidate coverage and a passage activation floor of 0.001. the
dataset was used during development and threshold selection; this is a
development measurement rather than held-out accuracy. production confidence
filtering remains disabled in this ranking benchmark.

the run fingerprint is
`a93ec267e563044b6e9ae9b714bac672a6da083298173ce565f731017a892ef9`.

[download the verified result and provenance](../assets/benchmarks/2026-09-20-fresh.json)
and [question-level session rankings](../assets/benchmarks/2026-09-20-session-rankings.json).
the result includes the dataset hash, relevant source hashes, model revisions,
package versions, exact settings, counts and the raw local result checksum.
the public rankings contain question/session IDs without conversation text.

the earlier development validation also reached 470/470, reusing 16,904 exact
query/document/model raw-score pairs and performing 85 fresh pair inferences.
the fresh confirmation above is the published headline measurement.

the isolated native JSONL API returned two short facts and abstained on an
unrelated query. an additional long-memory fact was still rejected with the
production cutoff of 0.6. this remains a known production false negative.

## other systems and comparison scope

external results below are vendor-reported figures checked against their
linked sources on September 20, 2026. they were not rerun in Engram's harness.
different metrics and evaluation populations prevent a head-to-head ranking.

| system | reported result | metric and evaluation scope |
| --- | ---: | --- |
| **Engram — September 20 retrieval candidate** | **100.0% (470/470)** | **session recall-any@5; fresh local run, 470 answerable questions** |
| Engram 0.6.2 — historical | 99.4% (467/470) | previously published session recall-any@5; not rerun here |
| [MemPalace — raw](https://github.com/MemPalace/mempalace/blob/develop/benchmarks/BENCHMARKS.md) | 96.6% | reported retrieval R@5; 500 evaluated questions in its report |
| [MemPalace — hybrid v4 + Haiku](https://github.com/MemPalace/mempalace/blob/develop/benchmarks/BENCHMARKS.md) | 100% | reported retrieval R@5 on 500 questions; explicitly tuned using failure cases |
| [EmergenceMem Internal](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag) | 86.0% | reported answer accuracy on 500 LongMemEval-S questions |
| [Mem0 — new algorithm](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) | 94.4% | reported LongMemEval answer accuracy; evaluation count not stated in that report |
| dense RAG — OpenAI embeddings | — | no verified source for the earlier 68.4% figure; not measured in this run |
| BM25 baseline | — | no verified source for the earlier 58.2% figure; not measured in this run |

the old comparison incorrectly labelled Emergence's answer accuracy as R@5,
used an unsupported overall Mem0 value, and converted unrelated percentages
into misses out of 470. these claims have been removed. Mem0's linked report
places 79.5% in the old algorithm's knowledge-update category. the earlier
dense-RAG and BM25 percentages have no verified primary source.

[LongMemEval's retrieval instructions](https://github.com/xiaowu0162/LongMemEval#memory-retrieval)
skip 30 abstention cases because they lack a ground-truth answer location.
MemPalace reports 500 evaluated questions, while Engram evaluates the 470
answerable questions. even the retrieval rows therefore have different
denominators. comparing generated answers would require running each system
under a shared QA protocol.

## earlier published benchmark results

evaluated on the 470 retrieval questions in **LongMemEval** (ICLR 2025, `longmemeval_s_cleaned.json`): **99.4% session recall-any@5** (467 / 470).

| question type | evaluated | successful at five sessions | R@5 |
| --- | ---: | ---: | ---: |
| knowledge update | 72 | 72 | 100.0% |
| multi-session | 121 | 121 | 100.0% |
| single-session assistant | 56 | 56 | 100.0% |
| single-session user | 64 | 64 | 100.0% |
| temporal reasoning | 127 | 126 | 99.2% |
| single-session preference | 30 | 28 | 93.3% |

## run a new evaluation

obtain the appropriate LongMemEval dataset separately, then run the checked-in
runner from the repository root:

```sh
python benchmarks/longmemeval/run_engram.py data/longmemeval_s_cleaned.json \
  --rerank --reranker-model BAAI/bge-reranker-base \
  --embedding-backend sentence_transformers --output results.jsonl
```

the runner starts from package defaults unless `--config path.yaml` is supplied;
it does not automatically load the working directory's config. an explicit
`--reranker-model` overrides that model choice. `--embedding-backend` defaults
to `sentence_transformers`; use `mlx` when evaluating that local backend.
`--fusion-alpha` and `--passage-floor` accept values from 0 to 1. when omitted,
they preserve the selected configuration's values; package defaults are 0 for
fusion and 0.001 for passage activation. an explicit CLI value, including 0,
overrides the corresponding config value.

local rerankers also use the shared focused-excerpt retry when every base
sigmoid score is below `rerank_passage_floor`, independently of the production
`min_confidence` gate. use
`--no-passage-fallback` to disable it for a comparison. at most one source
excerpt of up to 160 words is scored for each eligible long document. the
semantic query is unchanged, and the larger full/excerpt raw score survives.
this adds inference work and can omit context; it is not a separate confidence
guarantee. the benchmark still ranks without a final confidence filter, while
production can reject a recovered result after score adjustments. excerpt
selection includes conservative regular English singular/plural matches and
counts each original query term once per sentence.

the 0.001 activation floor was selected during development using this
benchmark. results on the same dataset are development measurements, not
held-out accuracy. the activation floor and the production confidence gate
are relevance thresholds, not calibrated probabilities of correctness.

`--result-k` defaults to 5 and sets the requested window for hybrid candidate
coverage. coverage keeps the rerank winner and, when the window is at least
two, includes the best scored hybrid candidate by moving it to the window's
last position if needed. scores stay unchanged. use `--no-preserve-prior` for
a comparison without that selection step. the runner records both choices in
its fingerprint and still reports metrics over the retrieved ranking.

the benchmark shares date helpers and rerank score conversion with production.
its candidate generation is still specific to this evaluation: it combines
user-session dense retrieval with user and assistant BM25, then reranks up to
35 sessions. a temporal multiplier is applied once per session, even when that
session contributes both user and assistant text. NDCG discounts position two
below position one and counts all gold sessions in its ideal ranking, including
answers missing from the retrieved list.

queries retain the spelling supplied by the dataset. the runner does not apply
a special-case spelling substitution before retrieval or reranking.

each run writes `results.jsonl.metadata.json` beside the raw result file. the
sidecar records the dataset SHA256, relevant source hashes, model names,
backend, selection settings, `confidence_filter: false` and a run fingerprint.
it includes the `engram/rerank_passages.py` source hash,
`rerank_passage_fallback` flag and `passage_confidence_floor`, whose value comes
from the separate `rerank_passage_floor` configuration field.
per-session results include the final score and, for reranked candidates,
the retained raw model score, prior rank and temporal adjustment. passage
traces retain `base_raw_score` plus `excerpt_raw_score`, `source_start`,
`source_end` and `passage_words` when an excerpt was retried. offsets refer to
the concatenated user/assistant document supplied to the reranker, rather than
the result row's displayed text.

fresh runs refuse an existing result file or metadata sidecar. the runner
claims the result filename with exclusive creation before writing its sidecar,
so a new fingerprint cannot relabel an older result file. choose a new output
filename for a new experiment.

to continue an interrupted run, repeat its command with `--resume`. the runner
checks provenance before loading models and rejects missing sidecars,
different sources/models/settings/datasets, malformed JSON and duplicate
question IDs. use `--question-id` or `--limit` for diagnostics; their selections
are recorded in the fingerprint.

save model revisions, dependency versions, hardware and the dataset revision
alongside the generated files when reporting a score. model names alone do not
pin downloaded weights or the execution environment. model downloads and a
full run are separate from the unit and integration test suite.

## dormant recall evidence

the dormant experiment has its own targeted evaluation. a rarely accessed
historical release note was absent from the persisted ANN index and recovered
from current database embeddings. the final cosine and query/content gates
produced three evaluator-assessed useful additions and five abstentions across
eight selected queries. no usefulness feedback was submitted on behalf of the
user.

this small, partly retrospective sample guided a repair. it does not estimate
real-world precision, prove improved task outcomes, or imply that an old note is
still true. see [dormant recall](../dormant-recall.md) for the method and limits.
