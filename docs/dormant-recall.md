# dormant recall (shadow experiment)

Dormant recall looks for a relevant connection that ordinary retrieval left out.
It evaluates at most one candidate per hybrid search, or none. It is off by
default. Shadow mode writes a separate bounded review log; it does not insert a
suggestion into recall, hints, context, debug results or normal answers. There is
no automatic visible suggestion mode in this version.

## enable, disable, review

Add this to the config used by the process (quote `off` in YAML):

```yaml
dormant_recall:
  mode: shadow                 # "off" disables evaluation and logging
  candidate_limit: 50          # independent of ordinary top_k; maximum 200
  dormancy_days: 30
  min_relevance: 0.75          # raw embedding cosine, not truth/confidence
  max_bonus: 0.05              # maximum allowed setting: 0.1
  rerank_candidates: 12        # separate query/content relevance checks
  min_rerank_score: 0.6        # model score, not truth confidence
  cooldown_days: 7
  feedback_cooldown_days: 30
  log_max_events: 1000         # maximum allowed setting: 10000
  log_retention_days: 30
```

`ENGRAM_DORMANT_RECALL_MODE=off` overrides the file on process start. Existing
processes keep their loaded configuration and Python code. Editing the file
does not hot-reload or restart an MCP service. A fresh CLI process uses the new
code/config immediately. Use the same explicit config for search and review:

```sh
engram --config /path/to/config.yaml search "current task query" -k 5 --json
engram --config /path/to/config.yaml dormant review --limit 20
engram --config /path/to/config.yaml dormant inspect EVENT_ID
engram --config /path/to/config.yaml dormant feedback EVENT_ID useful
```

Feedback choices are `useful`, `irrelevant`, and `dismissed`. Use `useful` only
when the inspected connection was actually used. A silent user, a log row, and
an inspection do not establish usefulness. An event accepts one feedback
category; repeating the same category is idempotent. Different feedback for the
same event is rejected. The equivalent MCP tools are `dormant_review`,
`dormant_inspect`, and `dormant_feedback`. They require an explicit review request;
they are not instructions to insert candidates into ordinary answers.

Review returns metadata only, including event ID/time, candidate ID (or null),
candidate count, raw relevance, bounded bonus, dormancy, overlap counts, exposure
time, and explicit feedback. Inspect fetches the current eligible memory content
and marks a separate exposure timestamp. It describes the retrieval evidence,
including whether there was any literal overlap, without claiming that an old
blocker is now solved. Original queries and matching words are not saved, so
review a result alongside the original task. This privacy choice limits later
blind relevance audits; do not infer usefulness from the metadata alone.

## selection and reinforcement boundaries

The separate dense search uses the original query and streams current eligible
database embeddings into a bounded heap. Active status, memory type and the
dormancy threshold are checked before the candidate limit. Vectors are normalized
for exact cosine comparison. It does not reuse ordinary ANN contents, embedding
caches, the final list, RRF cutoff, query expansion, recency/frequency boosts or
the deep reranker. This matters when a persisted ANN index misses newer records:
an old index must not silently make those memories unreachable to this experiment.

Up to 12 cosine-qualified memories receive a separate query/content check using
the configured cross-encoder. The default minimum score is 0.6, on that model's
own scale. A broad topic match that cannot address the query is rejected. Scores
are never described as probabilities or evidence that a historical claim remains
true. The memory content is checked again before recording, so an edit during
reranking cannot inherit a stale score. If the relevance check fails, the shadow
evaluation fails open without changing ordinary recall.

Exact scanning costs O(N × embedding dimension) over eligible records, with a
bounded candidate heap rather than a full matrix. This is suitable for the
current small store; it is not a million-record performance claim. It adds an
embedding call and a bounded rerank batch; configured API backends have their
usual cost. No index is rebuilt, saved or repaired by dormant evaluation. Ordinary
retrieval can still have a stale index and needs separate maintenance.

Candidates must pass the raw relevance floor before any dormancy bonus. They
must be non-forgotten with active status (legacy NULL matches ordinary recall)
and fit the same fact/procedure/narrative retrieval profile. Missing, forgotten,
archived, superseded, merged, challenged and otherwise inactive memories are
excluded. Version one does not distinguish lifecycle archival from explicit
forgetting: both use the existing forgotten flag and neither is revived.
Existing source/trust metadata and importance are read without modification;
no trust or confidence is manufactured from age.

Among eligible dormant candidates, ranking is:

```text
relevance + max_bonus * clamp(importance, 0, 1)
                      * min(1, dormant_days / (4 * dormancy_days))
```

Raw cosine and the separate relevance-check score are stored separately and never
increased. With the defaults,
age/importance can change selection only within a 0.05 cosine gap among already
relevant candidates. Low similarity can never be rescued by age. Exact ties
use raw relevance then memory ID for deterministic selection. Literal overlap
is supporting evidence only, not a requirement: differently worded semantic
connections are allowed. Neither cosine nor the ranking score is a probability
that the memory is true or the connection useful. Threshold calibration varies
by embedding model and is not established by the contract tests.

Dormancy starts at the latest of creation, ordinary `last_accessed`, and explicit
`useful` feedback. Historical ordinary access timestamps are a conservative proxy
for meaningful use: this store does not distinguish historical retrieval from
actual use. The experiment preserves that ordinary behavior. A shadow candidate
updates only its separate `retrieved_at` and cooldown; explicit inspection updates
only `shown_at` and cooldown. Useful feedback records only `used_at` and cooldown.
None changes `access_count`, `last_accessed`, importance, ordinary access logs,
search caches, or the deep reranker's training history.

All feedback pauses the candidate for at least 30 days by default. Useful also
restarts its dormancy threshold. Log rotation does not erase use or cooldown
state. After cooldown expiry, an unhelpful candidate can be considered again;
negative feedback is not global suppression or a new forgetting operation.

## persistence and failure behavior

Two additive tables, `dormant_recall_events` and `dormant_recall_state`, are created
lazily on first evaluation/review. This does not run legacy memory backfills or
rewrite the memories table. The second version adds nullable `rerank_score` and an
`algorithm` label without rewriting old events, which retain `ann-cosine-v1`.
The SQL works on SQLite and PostgreSQL. A separate
connection/transaction prevents a failed shadow write from poisoning ordinary
recall. Concurrent evaluators serialize the selection/cooldown decision. Short
database lock waits fail open; failures report only an exception type, not query,
memory text or connection secrets. Normal recall still returns its usual results.

History is capped by event count and retention age, pruned on evaluation/review.
There is no timer while the feature is off: retained rows remain until the next
review/evaluation. State has at most one row per extant eligible memory and
survives event rotation. Deleted/forgotten/inactive references are pruned on the
next operation, and inspect always rechecks current eligibility. No full memory
text, query, matching terms, query hash, embedding, or LLM explanation is copied
into these tables. Existing ordinary recall logging is unchanged.

SQLite-to-PostgreSQL legacy export/migration commands do not currently transfer
experiment telemetry. Back up these tables with native database tooling if
continuity of cooldown/feedback is needed across a storage migration. Disabling
the feature needs no schema rollback; older code can ignore the additive tables.

## verification and limits

`tests/test_dormant.py` uses temporary SQLite files and optionally a disposable
PostgreSQL schema. It covers eligibility, independent candidates, semantic-only
connections, bounded bonus effects, relevance gating, one-or-none selection,
cooldown, concurrency, feedback, no reinforcement, privacy/retention, failure
isolation, persistence, unchanged ordinary output, and MCP review handlers.
Set `ENGRAM_TEST_POSTGRES_DSN` only to a disposable cluster to run both backends.
The fixture creates and drops its own randomly named schema.

The synthetic contract tests verify mechanics. They do not establish whether
real suggestions help with actual work, whether a stored claim remains current, or whether
the default cosine threshold is appropriate for every embedding model. A small
shadow pilot needs explicit usefulness feedback before any visible rollout.

For an actual CLI/stdio MCP check using cached local models and synthetic stores:

```sh
python tests/dormant_smoke.py --work-dir /path/to/new/disposable/directory
```

This starts fresh processes, compares ordinary result IDs with shadow off/on,
exercises review/inspect/feedback, and checks persisted access fields and cooldown.
Its prototype-note fixture scores about 0.736 with BGE small: the 0.75 default
abstains; an explicit 0.70 threshold only in the isolated pilot admits it after
the separate relevance check. The report records this limitation instead of
treating an abstention as useful recall. The earlier sparse semantic-only fixture
is rejected by the added relevance check; pure cosine proximity was insufficient.

The real-memory target-first check used a Junkstep installer/backend-release note
with one prior access, about 37 days of dormancy, active status and `forgotten=0`.
The old live ANN path omitted it; exact current-store cosine ranked it first at
0.817. The repaired algorithm selected it in an isolated snapshot at the unchanged
0.75 floor, its relevance-check score was 3.84, and its access/importance fields
stayed unchanged. It added installer packaging and release-verification details
missing from ordinary results. This is an evaluator judgment about historical
technical context, not user-submitted `useful` feedback or a live release audit.
