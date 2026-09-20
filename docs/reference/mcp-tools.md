# MCP tools

all tools available via the engram MCP server (`engram serve --mcp`).

## recall & search

| tool | params | description |
|------|--------|-------------|
| `recall` | `query` (required), `top_k` (default: `retrieval.top_k`), `mode` (default: "full_context") | relevance-ranked hybrid search with configured retrieval channels and reranking. mode filters by memory type: `facts_only`, `facts_plus_rules`, `full_context` |
| `recall_by_type` | `memory_type` (required: fact/procedure/narrative), `limit` (default: 20) | get memories filtered by semantic type |
| `recall_entity` | `name` (required) | everything about a person/project/tool — memories, relationships, timeline |
| `recall_timeline` | `start` (required, YYYY-MM-DD or YYYY-MM), `end` | memories by recorded `fact_date`, rather than creation time; without `end`, matches the `start` prefix |
| `recall_related` | `name` (required), `max_hops` (default: 2) | multi-hop graph traversal from an entity |
| `recall_recent` | `limit` (default: 20) | newest non-forgotten memories by creation time, across the store; no semantic query |
| `recall_layer` | `layer` (required: working/episodic/semantic/procedural), `limit` (default: 20) | list non-forgotten memories in a layer, ordered by importance |
| `recall_context` | `query` (required), `max_tokens` (default: 2000) | retrieve a formatted block for prompt context within a token budget |
| `recall_code` | `query` (required), `project`, `top_k` (default: 10) | search the codebase layer for functions, classes, files |
| `recall_hints` | `query` (required), `top_k` (default: 10), `hint_length` (default: 60 characters) | relevance-ranked retrieval with short snippets, memory IDs and entity names; can record accesses |
| `recall_explain` | `query` (required), `top_k` (default: `retrieval.top_k`), `mode` (default: "full_context"), optional `reference_date` | returned and rejected candidates, scores, confidence decisions, passage retries and effective settings; no access reinforcement |
| `find_similar` | `memory_id` (required), `top_k` (default: 5) | find memories most similar by embedding distance |
| `find_duplicates` | `threshold` (default: 0.92), `limit` (default: 20) | preview near-duplicate pairs without merging |
| `search_entities` | `query` (required), `limit` (default: 20) | fuzzy search for entities by partial name |
| `compress` | `query` (required), `max_tokens` (default: 2000) | compressed version of retrieved memories |
| `get_skills` | `query` (required), `max_skills` (default: 3), `format` (default: true) | select relevant procedural guides up to the requested maximum |

### retrieval explanations

`recall_explain` returns `results` and an `explanation`, alongside the existing
query and debug summaries. it uses the configured reranker and confidence gate;
an optional `reference_date`, such as `2026-09-20`, anchors relative-time phrases.
omitting `top_k` uses the loaded `retrieval.top_k`, whose package default is 10.

explanation calls bypass result-cache reads and writes, and leave access history,
importance, dormant evaluations and session handoffs unchanged. they can load
models. rejected candidates carry an outcome and reason; forgotten, inactive or
profile-filtered candidates have no content exposed. the report covers only the
bounded candidates retrieved for this query. its `final_ids` gives the returned
order, which can differ from sorting scores. see
[the explanation fields](../architecture/retrieval.md#explanations).

## store & organize

| tool | params | description |
|------|--------|-------------|
| `remember` | `content` (required), `source_type` (default: "remember:human"), `layer` (default: "episodic"), `memory_type` (default: "narrative", enum: fact/procedure/narrative), `importance` (default: 0.7) | store a memory with surprise scoring |
| `remember_interaction` | `question` (required), `answer` (required), `importance` (default: 0.5) | store a Q+A pair → episodic |
| `remember_decision` | `decision` (required), `rationale`, `importance` (default: 0.8) | decision + rationale → procedural |
| `remember_error` | `error` (required), `prevention`, `importance` (default: 0.7) | error pattern + prevention → procedural |
| `remember_project` | `name` (required), `status`, `location`, `notes` | structured project info → semantic |
| `remember_negative` | `content` (required), `context`, `scope`, `importance` (default: 0.75) | record explicit exclusions, absent features and assumptions to avoid |
| `forget` | `memory_id` (required) | soft-delete a memory |
| `bulk_forget` | `confirm` (required: true), `source_file`, `layer`, `older_than` (YYYY-MM-DD) | mass cleanup by criteria |
| `tag` | `memory_id` (required), `add` (array), `remove` (array) | add or remove tags |
| `batch_tag` | `query` (required), `tags` (required, array), `top_k` (default: 10) | add tags to all memories matching a search |
| `link_memories` | `memory_id_1` (required), `memory_id_2` (required), `relation` (default: "RELATED_TO") | manually relate two memories |
| `backlinks` | `memory_id` (required) | find all memories linked via shared entities |
| `annotate` | `memory_id` (required), `note` (required) | add a note without changing content |

## lifecycle

| tool | params | description |
|------|--------|-------------|
| `invalidate` | `memory_id` (required), `reason` | record invalidation metadata; use `update_status` to change lifecycle status |
| `update_status` | `memory_id` (required), `new_status` (required: active/challenged/invalidated/merged/superseded), `reason` | transition lifecycle status with audit trail |
| `status_history` | `memory_id` (required) | full status transition history — what changed, when, why |
| `promote` | `memory_id` (required), `target_layer` (required) | move to a higher layer |
| `demote` | `memory_id` (required), `target_layer` (required) | move to a lower layer |
| `edit_memory` | `memory_id` (required), `new_content` (required) | edit content, auto re-embeds |
| `pin` | `memory_id` (required) | immune to dream cycle forgetting |
| `unpin` | `memory_id` (required) | remove pin |

## entities & graph

| tool | params | description |
|------|--------|-------------|
| `entity_graph` | `name` (required) | relationship subgraph as JSON |
| `entity_timeline` | `name` (required) | entity's memories chronologically |
| `update_entity` | `name` (required), `alias`, `metadata` | add an alias; the current handler does not apply `metadata` or change entity type |
| `merge_entities` | `source_name` (required), `target_name` (required) | combine duplicates, moves all links |

## codebase

| tool | params | description |
|------|--------|-------------|
| `scan_codebase` | `path` (required), `project_name` | extract compressed code knowledge from a project |
| `recall_code` | `query` (required), `project`, `top_k` (default: 10) | search codebase layer |
| `list_projects` | — | show all scanned projects with counts |

## drift & patterns

| tool | params | description |
|------|--------|-------------|
| `drift_check` | `search_roots` (array), `project_root`, `layers` (array), `check_functions` (default: true) | verify memories against filesystem, drift score 0-100 |
| `drift_fix` | `search_roots`, `project_root`, `dry_run` (default: true) | auto-fix drift issues |
| `extract_patterns` | `hours` (default: 4.0), `novelty_threshold` (default: 0.25), `dry_run` (default: false) | distill procedural patterns from session activity |

## dedup & maintenance

| tool | params | description |
|------|--------|-------------|
| `dedup` | `threshold` (default: 0.92), `max_merges` (default: 50) | find and merge near-duplicates |
| `recompute_importance` | — | recalculate all importance scores (9-factor) |
| `train_reranker` | `epochs` (default: 50), `learning_rate` (default: 0.01) | train deep MLP reranker on access patterns |
| `reranker_status` | — | check if reranker is trained |
| `compress_embeddings` | `dry_run` (default: true) | lifecycle-aware quantization (32/8/4/2-bit) |
| `detect_communities` | `min_size` (default: 3), `generate_summaries` (default: false) | label propagation over entity graph |
| `quality_metrics` | — | storage quality, curation ratio, enrichment coverage |
| `explain_importance` | `memory_id` (required) | break down importance into 9 factors |

## ingest & sessions

| tool | params | description |
|------|--------|-------------|
| `ingest` | `path` (required) | ingest a file or directory |
| `ingest_sessions` | `limit` (default: 20) | ingest recent Claude Code sessions |
| `session_summary` | — | build and save the current handoff, returning its summary and selected sections |
| `session_handoff` | `session_id` (optional), `save` (default: true), `limit` (default: 8 items per section) | build a packet for the current or specified MCP session; save it unless `save` is false |
| `session_checkpoint` | `note` (optional), `limit` (default: 8 items per section) | append a checkpoint note when supplied, then build and save the current MCP session's packet |
| `resume_context` | `session_id` (optional), `limit` (default: 3 handoffs) | read an exact saved handoff, or list handoffs by last update; return a generated, unsaved fallback if none exists |
| `focus_brief` | `query` (required), `top_k` (default: 8) | build a compact briefing with dominant entities, layer mix, key memories, and suggested follow-up pulls |
| `compare_queries` | `query_a` (required), `query_b` (required), `top_k` (default: 8) | compare overlap, divergence, and entity differences across two retrieval paths |
| `hotspots` | `hours` (default: 72.0), `limit` (default: 8) | surface the hottest entities, layers, sources, and memories in a recent window |

## system & context

| tool | params | description |
|------|--------|-------------|
| `status` | — | memory counts, entities, DB size |
| `config_show` | — | loaded effective configuration, source of each setting, warnings and version; credentials redacted |
| `health` | — | cache, FTS index, orphaned entities, ANN status, embedding backend |
| `layers` | `query`, `max_tokens` (default: 4000) | L0-L3 graduated prompt context |
| `access_patterns` | `limit` (default: 20) | most-recalled memories, hit rates |
| `memory_map` | — | high-level map of entire system |
| `count_by` | `group_by` (required: layer/source_type/entity/month) | group counts |
| `consolidate` | — | run full dream cycle |
| `export` | `format` (default: "markdown"), `layer`, `limit` (default: 100) | export as markdown or JSON |

`config_show` returns `config_file`, nested `values`, dotted-path `sources`,
`warnings`, and `version`. it does not reload or change settings. use
`engram config check`, `show`, or `schema` to inspect configuration before server
startup; see [configuration](config.md).

## diary

| tool | params | description |
|------|--------|-------------|
| `diary_write` | `entry` (required) | append a persistent note tagged with the current MCP session ID |
| `diary_read` | — | read up to 50 recent diary entries across the store, newest first; use the in-process diary if storage has none |

## continuity pattern

clients must invoke this workflow through their instructions or explicit user
requests. connecting a server does not install startup or shutdown hooks.

1. call `recall_recent` with `limit: 5` for chronology, then `recall_hints` with a concrete project-and-task query.
2. when resuming work, read `resume_context` before ordinary `recall` refreshes the current packet. use `recall` for specific semantic questions, not to find the latest session.
3. save useful outcomes, decisions and progress during work. use `recall_explain` for non-reinforcing retrieval diagnosis.
4. before stopping, explicitly save an authored summary with `remember` (`memory_type: "narrative"`, `importance: 0.8` for substantial work), then `session_checkpoint` or `session_handoff` to save the final packet.

automatic handoff refresh runs after ordinary `recall`, successful stored/updated
`remember` calls (including their specialized wrappers), `diary_write`, and
`edit_memory`. skipped writes and other mutations do not trigger it.
`recall_recent`, `recall_hints`, `recall_explain` and `resume_context` do not
refresh handoffs; hints still use ordinary retrieval and can record accesses.

the core MCP session ID belongs to the server process. handoffs combine its
diary with bounded recent store-wide memories and events, so they are not
project-filtered conversation transcripts. saved `resume_context` results have
`latest` and `handoffs`, with each packet under the saved row's `metadata`;
generated fallbacks contain the packet directly and are not saved by that read.
`session_handoff` rebuilds rather than loading a saved snapshot.

see [session continuity](../guides/session-continuity.md) for examples and scope.
the [native local API](../native-api.md) has a separate, project-and-task
checkpoint contract; its arguments are not interchangeable with this MCP
`session_checkpoint(note, limit)`.

## dormant review

The core server also exposes `dormant_review`, `dormant_inspect` and
`dormant_feedback`. Review lists bounded metadata, inspect explicitly exposes a
still-eligible candidate, and feedback records Useful/Irrelevant/Dismissed without
ordinary memory reinforcement. None is automatically inserted into normal recall.
See [dormant recall](../dormant-recall.md) for schemas, defaults and limitations.

## separate Codex adapter

`codex_context`, `codex_checkpoint` and `codex_diagnostics` belong to the separate
project-bound adapter server, not the core tool list. See the
[adapter guide](../codex-adapter.md) for supported setup and task workflows.

## structured check evidence

`evidence_put`, `evidence_get` and `evidence_list` expose the same harness-neutral
contract as the [native local API](../native-api.md). records are caller-supplied
observations, not independently verified claims. reads preserve access history;
expiry and referenced memory lifecycle affect whether evidence remains eligible.
Engram does not execute check instructions embedded in these records.

the server's `tools/list` is the authoritative schema for the running revision.
