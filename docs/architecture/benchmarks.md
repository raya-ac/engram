# benchmarks and their limits

## what the retrieval benchmark measures

`benchmarks/longmemeval/run_engram.py` evaluates session retrieval. its
`recall_any@5` metric succeeds when at least one gold session appears among the
first five retrieved sessions. this is not generated-answer accuracy, evidence
verification, or the usefulness of a dormant suggestion.

the runner excludes question IDs ending in `_abs` and skips entries without
`answer_session_ids`. its retrieval implementation builds a corpus separately
for each question; it is not a measurement of the complete deployed memory
lifecycle or the current production database.

## historical local result

a saved local result file, `engram_v2_rerank_results.jsonl`, contains 470 evaluated
questions. recomputing its stored session metrics on 14 September 2026 gives
461 successes at five sessions: **98.1% session recall-any@5**. the benchmark was
not rerun for this release. the result file is not published with the repository,
so this number is historical local evidence, not an independently reproducible
release gate or a claim of superiority over another system.

| question type | evaluated | successful at five sessions |
| --- | ---: | ---: |
| knowledge update | 72 | 72 |
| multi-session | 121 | 120 |
| single-session assistant | 56 | 54 |
| single-session preference | 30 | 28 |
| single-session user | 64 | 64 |
| temporal reasoning | 127 | 123 |

local artifact SHA-256:
`a39105bd467b8b6aa1071cbf57ea87179d9c7b1dd94aefd24ec469d7bdec3291`.

competitor rankings and historical latency tables have been removed because
the available artifacts do not establish a controlled, current comparison.

## run a new evaluation

obtain the appropriate LongMemEval dataset separately, then run the checked-in
runner from the repository root:

```sh
python benchmarks/longmemeval/run_engram.py data/longmemeval_s_cleaned.json --rerank --output results.jsonl
```

record the source revision, dataset revision and hash, model versions, hardware,
configuration, sample exclusions and raw results alongside any reported score.
model downloads and a full run take resources; they are separate from the unit
and integration test suite.

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
