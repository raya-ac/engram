# MCP Tools (68)

all tools available via the engram MCP server (`engram serve --mcp`).

## recall & search

| tool | params | description |
|------|--------|-------------|
| `recall` | `query` (required), `top_k` (default: 10), `mode` (default: "full_context") | hybrid search — HNSW + BM25 + graph + Hopfield + RRF + cross-encoder. mode filters by memory type: `facts_only`, `facts_plus_rules`, `full_context` |
| `recall_by_type` | `memory_type` (required: fact/procedure/narrative), `limit` (default: 20) | get memories filtered by semantic type |
| `recall_entity` | `name` (required) | everything about a person/project/tool — memories, relationships, timeline |
| `recall_timeline` | `start` (required, YYYY-MM-DD or YYYY-MM), `end` | memories in a date range |
| `recall_related` | `name` (required), `max_hops` (default: 2) | multi-hop graph traversal from an entity |
| `recall_recent` | `limit` (default: 20) | last N memories by creation time |
| `recall_layer` | `layer` (required: working/episodic/semantic/procedural), `limit` (default: 20) | search within a specific layer |
| `recall_context` | `query` (required), `max_tokens` (default: 2000) | formatted context block ready for prompt injection |
| `recall_code` | `query` (required), `project`, `top_k` (default: 10) | search the codebase layer for functions, classes, files |
| `recall_hints` | `query` (required), `top_k` (default: 10), `hint_length` (default: 60) | truncated snippets + entity names for recognition without replacing cognition |
| `recall_explain` | `query` (required), `top_k` (default: 10), `mode` (default: "full_context") | hybrid search with retrieval intent, expansions, cache status, candidate counts, and score breakdowns |
| `find_similar` | `memory_id` (required), `top_k` (default: 5) | find memories most similar by embedding distance |
| `find_duplicates` | `threshold` (default: 0.92), `limit` (default: 20) | preview near-duplicate pairs without merging |
| `search_entities` | `query` (required), `limit` (default: 20) | fuzzy search for entities by partial name |
| `compress` | `query` (required), `max_tokens` (default: 2000) | compressed version of retrieved memories |
| `get_skills` | `query` (required), `max_skills` (default: 3), `format` (default: true) | task-aware skill selection — 2-3 focused procedural guides |

## store & organize

| tool | params | description |
|------|--------|-------------|
| `remember` | `content` (required), `source_type` (default: "remember:human"), `layer` (default: "episodic"), `memory_type` (default: "narrative", enum: fact/procedure/narrative), `importance` (default: 0.7) | store a memory with surprise scoring |
| `remember_interaction` | `question` (required), `answer` (required), `importance` (default: 0.5) | store a Q+A pair → episodic |
| `remember_decision` | `decision` (required), `rationale`, `importance` (default: 0.8) | decision + rationale → procedural |
| `remember_error` | `error` (required), `prevention`, `importance` (default: 0.7) | error pattern + prevention → procedural |
| `remember_project` | `name` (required), `status`, `location`, `notes` | structured project info → semantic |
| `remember_negative` | `content` (required), `context`, `scope`, `importance` (default: 0.75) | what does NOT exist — prevents hallucinated recommendations |
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
| `invalidate` | `memory_id` (required), `reason` | mark a fact as no longer true |
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
| `update_entity` | `name` (required), `alias`, `metadata` | add aliases, change type |
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
| `session_summary` | — | generate summary from diary + recent events |
| `session_handoff` | `session_id`, `save` (default: true), `limit` (default: 8) | build a structured handoff packet for the current or specified session and optionally persist it |
| `session_checkpoint` | `note`, `limit` (default: 8) | append an optional checkpoint note and persist a richer handoff packet for the current session |
| `resume_context` | `session_id`, `limit` (default: 3) | load the latest saved handoff packet so a new agent session can resume quickly |

## system & context

| tool | params | description |
|------|--------|-------------|
| `status` | — | memory counts, entities, DB size |
| `health` | — | cache, FTS index, orphaned entities, ANN status, embedding backend |
| `layers` | `query`, `max_tokens` (default: 4000) | L0-L3 graduated context for prompt injection |
| `access_patterns` | `limit` (default: 20) | most-recalled memories, hit rates |
| `memory_map` | — | high-level map of entire system |
| `count_by` | `group_by` (required: layer/source_type/entity/month) | group counts |
| `consolidate` | — | run full dream cycle |
| `export` | `format` (default: "markdown"), `layer`, `limit` (default: 100) | export as markdown or JSON |

## diary

| tool | params | description |
|------|--------|-------------|
| `diary_write` | `entry` (required) | append to session diary |
| `diary_read` | — | read current session diary |

## continuity pattern

for resumable agent work, the default flow is:

1. `resume_context` at session startup
2. normal `remember`, `remember_decision`, `remember_negative`, and `diary_write` calls during work
3. `recall_explain` when retrieval quality needs debugging or tuning
4. `session_checkpoint` or `session_handoff` near a stop point if you want to explicitly persist the current packet

the active MCP session also refreshes its handoff automatically after recalls, memory writes, diary writes, and memory edits.
