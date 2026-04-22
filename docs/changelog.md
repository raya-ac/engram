# Changelog

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
- **98.1% R@5 on LongMemEval** — new SOTA, beating MemPalace (96.6%) and all others
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
