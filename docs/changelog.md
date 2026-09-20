# Changelog

## unreleased

- independently confirm 470/470 session recall-any@5 in a fresh full LongMemEval development run without saved-score reuse; publish text-free question rankings and provenance, with abstention, confidence and native API limitations
- correct the comparison table: label external answer-accuracy and retrieval metrics separately, link primary reports and remove unsupported misses-out-of-470 claims
- use `BAAI/bge-reranker-base` as the default local reranker; keep MiniLM available through config
- return raw local reranker logits and apply a sigmoid once; preserve hosted normalized scores
- retry at most one focused source excerpt per eligible long document when every full-document local sigmoid score is below `rerank_passage_floor` (default 0.001); keep the 0.6 production confidence gate independent and retain max full/excerpt aggregation
- match conservative regular English singular/plural forms during excerpt selection, counting each original query term once per sentence
- retain full/excerpt raw scores and source offsets; keep the semantic query unchanged and exclude resolved relative-time text only from lexical selection
- remove extra lexical/prior bonuses and random noise from cross-encoder results; honor the optional bounded `rerank_fusion_alpha` blend
- keep the best confidence-eligible hybrid candidate in requests for at least two results, without changing scores or replacing the rerank winner; expose `preserve_prior_candidate` to disable coverage
- separate cached rerank results when the fusion weight, confidence gate, coverage setting, passage fallback flag or activation floor changes
- preserve query spelling instead of applying a single-word substitution
- share date parsing and relative windows with the benchmark, and apply its temporal boost once per session
- correct NDCG rank discounts and include unretrieved answers in the ideal ranking
- record benchmark dataset/source hashes, model names, result window, passage fallback flag/floor and settings; support `--no-passage-fallback` and `--passage-floor`, and reject incompatible or duplicate resume records
- preserve configured fusion and passage activation values unless the corresponding CLI option overrides them; document the activation floor as a benchmark development choice, not a calibrated probability or held-out accuracy claim
- refuse existing output paths for fresh benchmark runs and claim the result file before writing new provenance

## 0.6.2 (September 19, 2026)

- add `hf_token` config and env fallback so model downloads authenticate without manual huggingface-cli logins
- resolve relative dates ("last Saturday", "N days ago") and keep the temporal boost after cross-encoder reranking
- switch license to Engram Public Use License 1.0

## 0.6.1 (September 18, 2026)

- widen cross-encoder candidate pool from 20 to 35 so deep semantic hits don't get buried by lexical BM25
- use regex tokenization in BM25 so trailing punctuation doesn't kill keyword matches

## 0.6.0 (September 18, 2026)

### new features
- **dormant recall in shadow mode** — tracks queries with low confidence or empty retrieval results in an associative shadow buffer; when new memories are ingested or consolidated, the system automatically checks for newly satisfied associations and reactivates dormant queries (`engram dormant` CLI, config controls, and background evaluation)
- **native memory service layer** — introduced `EngramService` (`engram/service.py`), providing a high-level programmatic interface that unifies memory CRUD, evidence tracking, project context assembly, and hybrid retrieval
- **evidence graph & provenance tracking** — added first-class evidence extraction and verification (`engram/evidence.py`) to trace ground-truth source references, causal antecedents, and factual reliability
- **project context provider** — added session-aware context distillation (`engram/project_context.py`) for assembling working memory state, open decisions, and active task parameters for coding agents
- **codex adapter** — added native adapter (`engram/adapters/codex.py`) bridging Codex tool calls, memory handoffs, and skill execution directly into Engram
- **neural graph & connections overhaul** — complete overhaul of the web dashboard's Connections workspace into an interactive neural exploration canvas:
  - 2D camera with cursor-anchored zoom (0.25x–4.0x), smooth click-drag panning, and reset controls
  - dual layout modes: concentric layer rings (Working → Semantic → Episodic → Procedural) with glowing orbits and force-directed spring cluster physics
  - constellation mode: click-to-spotlight 1-hop and 2-hop subgraphs with camera targeting and unrelated node dimming
  - in-graph inspector drawer showing direct entity connections with jump links, linked memory excerpts, and inline type/alias management
  - live entity search with auto-suggest dropdown and keyboard navigation (`/`, `Esc`, `Enter`)
  - real-time synaptic pulse cascades with shockwave bursts and edge particle flows (`⚡ pulse` button, `F` hotkey, and live access log polling)
  - dual sizing metric combining memory mention volume and relationship degree
- **zigcho-infra design language** — full UI reskin of the dashboard to the refined slate-black palette (`#09090a` base, `#111113` surface, `#292427` borders, `#ed83b6` signature pink, `#8fd4b1` mint, `#7bbcff` blue)
- **unbounded exports** — `/api/export` now supports exporting all memories without forced pagination caps

### fixes
- **cross-database SQL compatibility** — replaced postgres-specific cast syntax with standard `CAST(e.aliases AS TEXT)` in entity search so MCP entity discovery runs safely across SQLite and Postgres
- **defensive consolidation** — access log and event pruning now defensively handles differing driver rowcount behaviors across database backends
- **web script parsing** — eliminated duplicate state declaration in the web dashboard preventing browser reference errors
- **standard package metadata** — pinned build-system hatchling to ensure strict compliance with packaging metadata standards across build environments

## 0.5.2 (April 24, 2026)

### new features
- **memory intelligence workbench** — added query briefing, query comparison, and recent hotspot surfacing to the web dashboard so Engram can summarize what matters instead of only exposing raw tables and graph views
- **3 new MCP tools** — `focus_brief`, `compare_queries`, and `hotspots` expose the same higher-level intelligence surface to agent clients over MCP

### fixes
- **postgres-safe drift dashboard** — web drift check and drift fix now avoid SQLite-only `PRAGMA busy_timeout` and `BEGIN IMMEDIATE` calls when Engram is running on Postgres
- **drift metadata compatibility** — invalidated-memory drift checks no longer rely on SQLite `json_extract(...)`, so the drift panel works correctly against Postgres `jsonb`
- **better drift API errors** — drift routes now return clean JSON errors for unexpected backend failures instead of surfacing broken frontend syntax errors
- **postgres compatibility sweep** — neural map aggregation, bridges, community summaries, quality metrics, and benchmark helpers now avoid SQLite-only query functions on Postgres-backed installs

## 0.5.1 (April 24, 2026)

### fixes
- **postgres migration hardening** — `engram migrate-postgres` now tolerates stale SQLite references during real-world migrations instead of failing on orphaned `entity_mentions`, `importance_history`, or `status_history` rows
- **migration verification** — post-copy verification now compares against valid migratable rows, so installations with historical SQLite inconsistencies can still complete a clean verified move to postgres

## 0.5.0 (April 24, 2026)

### new features
- **postgres backend** — engram can now use postgres as its primary storage backend for concurrent web + MCP deployments, while keeping sqlite as the default local-first option
- **sqlite to postgres migration command** — new `engram migrate-postgres` command copies an existing sqlite store into postgres, rebuilds full-text search rows, verifies key row counts, and can switch `config.yaml` automatically
- **config-level backend selection** — `storage_backend` and `postgres_dsn` are now first-class config fields and can also be driven by `ENGRAM_STORAGE_BACKEND` / `ENGRAM_POSTGRES_DSN`

### docs
- added postgres migration guide
- updated install, config, CLI, and README docs for mixed sqlite/postgres deployments

## 0.4.1 (April 23, 2026)

### fixes
- **web dashboard stability** — high-traffic dashboard routes now use short-lived store connections instead of the shared app store connection, which reduces SQLite contention under concurrent polling
- **drift check + fix responsiveness** — web drift operations now run off the async request thread, return fast `503` errors on lock contention, and avoid pinning the whole UI while a check or fix runs
- **lighter dashboard boot** — the web UI defers more work until it is needed and reduces background polling pressure during startup

### docs
- release metadata updated for the `0.4.1` patch release

## 0.4.0 (April 23, 2026)

### new features
- **retrieval explainability** — search now exposes query intent, expansion terms, exact-match boosting, cache-hit status, and candidate counts through `recall_explain` and `/api/search/explain`
- **query expansion + exact match boosts** — retrieval can expand common operator queries (`auth`, `deploy`, `code`, `graph`, `memory`) and reward exact phrase/token matches without overwhelming the dense pipeline
- **search result caching** — repeated hybrid searches are cached in-memory with automatic invalidation on writes, edits, forgets, layer changes, and bulk updates
- **richer continuity tooling** — `session_checkpoint` adds explicit stop-point checkpoints on top of automatic handoff refreshes, and `resume_context` now falls back to generating a handoff when no saved packet exists yet
- **continuity APIs** — the web server now exposes `/api/session-handoffs` and `/api/session-handoffs/:session_id` for dashboard and agent clients that want resumable session state without MCP

### docs
- MCP tools reference now covers `recall_explain` and `session_checkpoint`
- REST API reference now documents the search explain endpoint and session handoff endpoints
- release metadata updated for the new feature release

## 0.3.2 (April 23, 2026)

### new features
- **structured session handoffs** — active MCP sessions now maintain a resumable handoff packet with recent work, decisions, open loops, touched entities, and recall history
- **2 new MCP tools** — `session_handoff` builds and optionally persists a structured handoff packet, `resume_context` loads the latest saved handoff for fast startup
- **session continuity skill** — added `examples/skills/session-continuity/SKILL.md` so agents have a concrete default pattern for startup, in-session writes, and stop-point handoffs

### docs
- docs site now covers the session continuity flow on the home page, MCP tools reference, and Claude Code setup guide
- README and examples docs now point to the new continuity skill and handoff workflow

## 0.3.1 (April 14, 2026)

### new features
- **LLM API backends** — `anthropic` and `openai` backends for fact extraction, memory enrichment, consolidation. configure with `llm.backend` + `llm.api_key` in config.yaml or env vars. `pip install 'engram-memory-system[anthropic]'` or `[openai]`
- **memory types** — `fact`, `procedure`, `narrative` column on every memory. indexed, filterable. auto-backfilled from existing metadata on migration
- **retrieval profiles** — `recall` accepts `mode`: `facts_only` (structured knowledge only), `facts_plus_rules` (+ procedures), `full_context` (everything). filters before cross-encoder reranking
- **status tracking** — lifecycle states: `active`, `challenged`, `invalidated`, `merged`, `superseded`. `status_history` audit table with timestamps and reasons. non-active memories excluded from retrieval
- **3 new MCP tools**: `update_status`, `recall_by_type`, `status_history` (66 total)
- `remember` tool accepts `memory_type` param. `remember_decision`, `remember_error`, `remember_project`, `remember_negative` auto-set the correct type

### fixes
- hnswlib `knn_query` crash when MCP passes float `k` (JSON numbers are floats)
- schema init on existing DBs — tolerates missing columns, runs migration before index creation
- config.yaml gitignored (contains API keys), config.example.yaml tracked

## 0.2.0 (April 11, 2026)

### new features
- **HNSW ANN index** — approximate nearest neighbor via hnswlib, 100% recall@10, 0.09ms search, scales to 1M vectors
- historically reported **98.1% session recall-any@5** on a 470-question LongMemEval subset; see [measurement scope and limitations](architecture/benchmarks.md)
- **multi-backend embeddings** — Voyage AI, OpenAI, Google Gemini alongside local MLX/sentence-transformers
- **Voyage cloud reranker** — rerank-2.5/2.5-lite alongside local cross-encoder
- **SSE MCP transport** — `engram serve --mcp-sse` for HTTP clients
- **CLI: reembed** — re-embed all memories after switching embedding model
- **CLI: watch** — poll a directory for new files and auto-ingest
- **CLI: export/import** — portable JSON backup with optional embeddings
- **web auth** — bearer token auth via `web.auth_token` config
- **Docker** — Dockerfile + docker-compose.yml
- **72 pytest tests** across 6 modules
- **13 examples** — setup guides + Python scripts for every major feature
- **GitHub Actions CI** — tests on push/PR, auto-publish to PyPI on release
- **PyPI** — `pip install engram-memory-system`
- **docs site** — MkDocs Material at engram-memory.dev

### fixes
- ANN count tracks active ids (hnswlib `mark_delete` doesn't decrement)
- debug mode with `rerank=False` no longer crashes (unbound `reranked` variable)
- 9-factor importance (was incorrectly documented as 7-factor)
- 63 MCP tools (was incorrectly documented as 52)

## 0.1.0 (April 9, 2026)

initial release.

- 5-channel hybrid retrieval (dense + BM25 + graph BFS + Hopfield + RRF)
- memory layers (working, episodic, semantic, procedural, codebase)
- entity graph with co-occurrence relationships
- surprise-based importance scoring at write time
- retention regularization (L2/Huber/elastic)
- deep MLP reranker trained on access patterns
- dream cycle consolidation
- drift detection and auto-fix
- pattern extraction from sessions
- negative knowledge
- enriched embeddings (A-Mem)
- memory evolution
- intent-aware retrieval (MAGMA)
- trust-weighted decay
- 63 MCP tools
- web dashboard with neural map
- MLX GPU embedding backend
