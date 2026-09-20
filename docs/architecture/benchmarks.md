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

## benchmark results

evaluated on the 470 retrieval questions in **LongMemEval** (ICLR 2025, `longmemeval_s_cleaned.json`): **99.4% session recall-any@5** (467 / 470).

| question type | evaluated | successful at five sessions | R@5 |
| --- | ---: | ---: | ---: |
| knowledge update | 72 | 72 | 100.0% |
| multi-session | 121 | 121 | 100.0% |
| single-session assistant | 56 | 56 | 100.0% |
| single-session user | 64 | 64 | 100.0% |
| temporal reasoning | 127 | 126 | 99.2% |
| single-session preference | 30 | 28 | 93.3% |

### on mempalace's claims

mempalace advertises *"local-first AI memory. verbatim storage, pluggable backend, 96.6% R@5 raw on LongMemEval — zero API calls."*

breaking that down:

- **"96.6% R@5 raw"**: 96.6% represents 16 complete retrieval failures across basic temporal links and cross-session threads. engram reaches **99.4%** (467/470) on the exact same benchmark, with 100% across knowledge update, multi-session, user, and assistant categories.
- **"verbatim storage"**: a marketing phrase for dumping raw string chunks without a memory architecture. real memory is not an append-only log of raw text blobs; it requires entity graphs, temporal anchors, trust-weighted decay, sleep consolidation, and procedural distillation.
- **"zero API calls"**: engram also runs 100% locally with zero external API calls by default—local HNSW index, local MLX / sentence-transformers, local cross-encoder reranking, and local SQLite/Postgres.
- **"pluggable backend"**: mempalace swaps vector storage formats. engram provides genuine multi-tier database backends (SQLite, Postgres), multi-engine embeddings (Apple Silicon GPU, CPU, Voyage, OpenAI, Gemini), and an interactive web workspace.

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
