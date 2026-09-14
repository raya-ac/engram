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

The separate dense search uses the original query and its own candidate limit.
It does not reuse the ordinary final list, RRF cutoff, query expansion, recency
boost, frequency boost, deep reranker or result cache. It uses the configured
embedding backend and existing ANN index (or brute-force fallback). It can
therefore find candidates missing from ordinary recall even on a cache hit.
This adds a dense search/embedding call and latency; an API embedding backend
has its usual cost. The embedding backend is not changed by this feature.

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

Raw relevance is stored separately and is never increased. With the defaults,
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
rewrite the memories table. The SQL works on SQLite and PostgreSQL. A separate
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
Its semantic-only fixture scores about 0.734 with BGE small: the 0.75 default
abstains; an explicit 0.70 threshold only in the isolated pilot admits it. The
report records this limitation instead of treating an abstention as useful recall.
