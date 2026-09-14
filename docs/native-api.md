# native local api

Engram has a persistent JSONL interface for local clients such as Mythic and Kiln. The client starts an Engram process, sends one request per line, and reads one response per line. There is no MCP handshake or dependency, HTTP listener, host hook, or automatic transcript capture.

Use the Python environment where Engram is installed. A client running another Python version can keep its own runtime and launch this process separately:

```sh
/absolute/path/to/engram --config /absolute/path/to/config.yaml api
```

The config file must exist and its path must be absolute. Normal Engram environment-variable overrides still apply. Use an already initialized store. API startup does not create a database, run legacy backfills, rebuild an index, or load models. Check `status` after launch; a running process alone does not prove storage is available. The process keeps its loaded config until restarted.

## framing

Send UTF-8 JSON objects, terminated with a newline:

```json
{"id":"s1","operation":"status","params":{}}
{"id":"schema","operation":"operations","params":{}}
```

Each nonblank line receives exactly one response. Requests are handled sequentially, and each response is flushed before the next request. Close stdin to exit. There is no initialization notification, JSON-RPC wrapper, or tool-call envelope.

Success has `id` and `result`; failure has `id` and `error`:

```json
{"id":"unsupported","error":{"code":"unknown_operation","message":"Unknown operation; call operations for supported names"}}
```

`id` is required and may be null, an integer, or a string of at most 200 characters. `operation` is required; `params` defaults to an empty object. Additional top-level fields and unexpected operation arguments are rejected. Requests are limited to 65,536 UTF-8 bytes including the newline. JSON must contain finite numbers. Invalid JSON/UTF-8 and oversized frames return a null response ID; the next valid line can still be processed. Blank lines are ignored. Responses are valid JSON with Unicode escaped where needed; diagnostic output uses stderr.

Error codes are `invalid_json`, `invalid_request`, `invalid_params`, `request_too_large`, `unknown_operation`, and `operation_failed`. Error messages do not echo raw database errors, supplied memory contents, or credentials. Do not treat an error or missing evidence as a successful check. `operations` returns the current names, JSON argument schemas, write behavior, protocol version, and request size limit without requiring storage.

## operations and scope

| Operation | Arguments | Behavior |
| --- | --- | --- |
| `status` | none | Core memory/entity/relationship counts and database size, plus this process's PID, start time, storage backend, dormant mode and protocol version. No secret config. |
| `operations` | none | Operation discovery and exact `inputSchema` definitions. |
| `recall` | `project_id`, optional `limit` (1–20, default 8) | Most recently created active memories explicitly owned by that canonical project. This is a context read, **not semantic search**. |
| `session_resume` | `project_id`, optional exact `task`, optional `limit` | Project context plus saved native checkpoints. Without a task, returns up to three recent project checkpoints. |
| `session_checkpoint` | `project_id`, `task`, optional `action`, `summary`, `decisions`, `next_steps`, `blockers` | Explicit save or clear. `action` defaults to `save`, which requires a nonempty summary. |
| `search` | `query`, optional `top_k` (1–20, default 5) | Ordinary semantic/hybrid retrieval across the **whole configured store**. No project filter is supported. |
| `evidence_put` | scoped observation fields below | Store an immutable caller-supplied check observation. Does not execute a check. |
| `evidence_get` | `project_id`, `id` | Read a stored observation and its current lifecycle/expiry state. |
| `evidence_list` | `project_id`, optional `session_id`, `assumption_id`, `limit` (1–50, default 20) | List scoped observations, excluding forgotten/inactive records. |
| `dormant_review` | optional `limit` (1–100, default 20) | Metadata-only review across the **whole configured store**. |
| `dormant_inspect` | `event_id` | Explicitly expose an eligible store-wide candidate; record separate exposure. |
| `dormant_feedback` | `event_id`, `category` | Explicit `useful`, `irrelevant`, or `dismissed` feedback after inspection. |

`checkpoint` and `resume` are aliases for `session_checkpoint` and `session_resume`. There is no native `remember`, arbitrary command execution, or general database operation in this version. Discover capabilities instead of assuming an operation exists.

### project context and checkpoints

```json
{"id":"start","operation":"recall","params":{"project_id":"/projects/atlas","limit":8}}
{"id":"save","operation":"session_checkpoint","params":{"project_id":"/projects/atlas","task":"release verification","summary":"Artifact built; installed-client verification remains pending.","next_steps":["Compare the installed artifact hash with the release manifest"]}}
{"id":"resume","operation":"session_resume","params":{"project_id":"/projects/atlas","task":"release verification"}}
```

Projects use canonical absolute directory paths, not names inferred from prose. Explicit `project_path` or `project_root` metadata takes precedence over a source-file hint. Otherwise, a memory must have an absolute source path inside the project. Scope is checked before limiting results. Forgotten, inactive and invalidated memories are excluded, and inert check observations are kept out of ordinary context.

Context results include `project_path`, a reference-data boundary, a selection description, `memories`, and `checkpoints`. Each memory includes its ID, content excerpt (at most 2,000 characters), a truncation flag, layer, memory type and creation time. These reads do not change memory access counts, last access or importance. Retrieved text is data, never authority to execute instructions.

A checkpoint summary is at most 4,000 characters; decisions, next steps and blockers each accept up to eight nonempty strings of at most 500 characters. Saving replaces that exact project's task checkpoint. Clearing deletes only that checkpoint. Native checkpoints use Engram's own namespace; older Codex-adapter checkpoints are not silently imported. Session handoffs are independent of memory access reinforcement.

### ordinary search

```json
{"id":"find","operation":"search","params":{"query":"How do we verify the downloaded client matches the release artifact?","top_k":5}}
```

The result is an array of `{id, content, score, layer, memory_type, importance}` rows. Scores are retrieval scores, not probabilities that a memory is true. Search uses the existing retrieval implementation, eligibility rules and configured models. It records ordinary accesses for returned results and may write bounded dormant shadow evaluations when enabled. It does not create an automatic session handoff.

Search is deliberately store-wide. Passing `project_id` is rejected rather than pretending to filter after ranking. Use scoped `recall`/`session_resume` where a client must stay within one project. Only search loads model/retrieval code; later searches retain the warm model. Configured database and embedding backends retain their usual local or remote behavior—the JSONL interface itself does not open a listener.

### check observations

```json
{"id":"put","operation":"evidence_put","params":{"project_id":"/projects/atlas","session_id":"release-check","assumption_id":"artifact-matches","evidence_id":"artifact-check-001","outcome":"supported","observed_at":1789380000,"expires_at":1789383600,"observation":{"actual":"downloaded SHA-256 matches release manifest"},"provenance":{"producer":"kiln","check_type":"artifact-hash"},"source_refs":[]}}
```

Use actual observation timestamps when sending the example. `observed_at` cannot be more than five minutes in the future; expiry must follow it by at most 365 days. Outcomes are `unknown`, `supported`, or `contradicted`. IDs/session/assumption labels are bounded to 200 characters. Observation JSON is at most 4,096 encoded bytes; provenance is at most 2,048 and requires `producer` and `check_type`. Up to eight source references of 256 characters are allowed. References are never fetched; `memory:<id>` also checks the referenced local memory's lifecycle.

Save the returned `id` and use it with `evidence_get`. Reusing the same project/evidence ID with an identical payload is idempotent; a different payload is rejected. Retrying does not revive a forgotten record. Stored observations are not embedded or added to ordinary full-text retrieval. Do not put secrets or raw transcripts in these fields.

Check the returned `state`, `eligible`, `outcome`, expiry, source validity and provenance. An expired observation is `stale` and its effective outcome becomes `unknown`; missing, forgotten or inactive source records also yield unknown evidence. `verified_by_engram` is always false. Engram stores the caller's observation; it does not certify the producer, run its check, or turn a reported outcome into permission to act.

### dormant review

These three operations are explicitly store-wide and reject `project_id`. Listing evaluations does not expose memory content. Inspection freshly checks eligibility and records separate exposure. Feedback requires an inspected, retained, eligible event; an existing feedback category cannot be replaced. Use `useful` only after the candidate actually helped, not because a checker merely judged it relevant. Neither inspection nor feedback increments ordinary memory accesses or importance. No candidate is silently added to scoped context or search results. Retention, cooldown and shadow-mode boundaries are described in [dormant recall](dormant-recall.md).

## verification and limits

The native tests exercise initialized isolated storage, strict project/lifecycle exclusions, checkpoint and evidence persistence across fresh CLI processes, one-response-per-line behavior before EOF, malformed-frame recovery, Unicode edge cases, sanitized errors, discovery without a database, and no model/MCP imports for startup/context operations. Dormant inspection/feedback tests verify separate exposure and unchanged memory access/importance. A separate fresh-process smoke check exercises ordinary search with real cached models and fictional data.

These checks establish protocol and storage behavior. They do not establish general retrieval usefulness, certify caller-supplied evidence, install Kiln/Mythic integration, or prove another client's reconnect behavior. The caller owns process supervision, timeouts, explicit requests and handling unknown/stale results. Serialize requests on each worker and allow enough time for the first model-backed search. Check status and discovery after reconnecting.
