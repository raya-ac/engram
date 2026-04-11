# Changelog

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
