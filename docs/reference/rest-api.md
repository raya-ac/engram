# REST API (57 endpoints)

the web dashboard (`engram serve --web`) exposes a full JSON API at `http://127.0.0.1:8420`.

if `web.auth_token` is set in config, include `Authorization: Bearer <token>` header or `?token=<token>` query param.

## memories

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/memories` | paginated list, optional `?layer=` filter |
| GET | `/api/memories/:id` | full memory with hypothetical queries, entities, access history |
| GET | `/api/memories/:id/similar` | find similar by embedding distance |
| GET | `/api/memories/:id/importance` | 9-factor importance breakdown |
| GET | `/api/memories/:id/history` | importance score over time |
| POST | `/api/memories/:id/edit` | edit content (re-embeds) |
| POST | `/api/memories/:id/annotate` | add timestamped note |
| POST | `/api/memories/:id/promote` | change layer upward |
| POST | `/api/memories/:id/demote` | change layer downward |
| POST | `/api/memories/:id/forget` | soft-delete |
| POST | `/api/memories/:id/invalidate` | mark as no longer true |
| POST | `/api/memories/:id/pin` | pin (prevent forgetting) |
| POST | `/api/memories/:id/unpin` | unpin |
| POST | `/api/memories/bulk` | bulk promote/forget/tag/demote |

## search

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/search?q=...` | hybrid search, optional `&debug=true` |
| GET | `/api/search/filtered?q=...` | search with layer, importance, date, source filters |
| GET | `/api/search/hints?q=...` | truncated hints for cognitive scaffolding |
| POST | `/api/remember` | store a memory with surprise scoring |

## entities

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/entities` | all entities with memory counts |
| GET | `/api/entities/:id/graph` | relationship subgraph |
| GET | `/api/entities/:id/timeline` | memories ordered chronologically |
| POST | `/api/entities/:id/alias` | add entity alias |
| POST | `/api/entities/:id/type` | change entity type |

## visualization

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/neural` | full entity graph for neural map |
| GET | `/api/neural/fires?since=...` | recent access events for animations |
| GET | `/api/timeline?start=...&end=...` | temporal query |
| GET | `/api/events?limit=...` | recent events from all processes |
| GET | `/api/pulse` | hourly activity counters + sparkline |
| GET | `/api/heatmap?days=30` | GitHub-style activity grid |
| GET | `/api/retention/curves` | L2/Huber/elastic curve data |
| GET | `/api/retention/scatter` | real memory age vs retention |

## analytics & system

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/stats` | memory counts, entities, DB size |
| GET | `/api/health` | cache, FTS, orphans, ANN index, embedding backend |
| GET | `/api/memory-map` | per-layer top entities, date range |
| GET | `/api/analytics` | layer distribution, top accessed, top entities |
| GET | `/api/context?query=...` | L0-L3 graduated context with token counts |

## advanced features

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/drift` | drift check report |
| POST | `/api/drift/fix` | auto-fix drift issues (supports `?dry_run=true`) |
| GET | `/api/patterns` | extract patterns from session |
| POST | `/api/patterns/extract` | extract and store patterns |
| GET | `/api/bridges` | cross-domain bridge memories |
| GET | `/api/skills?query=...` | task-aware skill selection |
| GET | `/api/duplicates?threshold=0.92` | preview duplicate pairs |
| POST | `/api/dedup` | auto-merge duplicates |
| GET | `/api/surprise/:id` | surprise score for a memory |
| POST | `/api/surprise/preview` | compute surprise for text |
| GET | `/api/reranker/status` | deep reranker training state |
| POST | `/api/reranker/train` | trigger reranker training |
| GET | `/api/export?format=json` | export memories |

## ingest & diary

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/ingest/log` | recent ingestion log |
| POST | `/api/ingest/path` | ingest a file/directory |
| POST | `/api/ingest/sessions` | ingest Claude Code sessions |
| GET | `/api/diary` | session diary entries |
| POST | `/api/diary` | append diary entry |

## SSE

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/api/stream` | server-sent events for real-time updates |

## root

| method | endpoint | description |
|--------|----------|-------------|
| GET | `/` | web dashboard HTML |
