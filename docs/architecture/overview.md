# Architecture Overview

local installs can live in one sqlite file. service-style deployments can switch to postgres when they need concurrent clients.

## module map

```
engram/
├── store.py          # sqlite schema, CRUD, FTS5, entity graph, ANN lifecycle
├── ann_index.py      # HNSW approximate nearest neighbor (hnswlib)
├── embeddings.py     # multi-backend: mlx, sentence-transformers, voyage, openai, gemini
├── retrieval.py      # 8-stage hybrid pipeline
├── extractor.py      # LLM fact extraction + hypothetical query generation
├── entities.py       # regex entity extraction, relationship graph
├── surprise.py       # k-NN novelty scoring at write time
├── deep_retrieval.py # learned MLP reranker
├── skill_select.py   # task-aware skill selection gate
├── lifecycle.py      # retention regularization, 9-factor importance, promotion
├── consolidator.py   # dream cycle (7 steps)
├── codebase.py       # project scanner → codebase layer
├── conversations.py  # conversation ingest + classification
├── dedup.py          # semantic deduplication
├── layers.py         # L0-L3 graduated context
├── compress.py       # token-budget compression
├── formats.py        # parsers: markdown, JSON, PDF, slack
├── llm.py            # claude CLI + mlx backend
├── evolution.py      # memory enrichment, evolution, CRUD, trust, canonicalization
├── drift.py          # memory drift detection + auto-fix
├── patterns.py       # procedural pattern extraction
├── quantize.py       # lifecycle embedding compression (FRQAD)
├── communities.py    # label propagation community detection
├── hopfield.py       # Hopfield associative retrieval
├── mcp_server.py     # 63-tool MCP server (stdio + SSE)
├── cli.py            # 15 CLI commands
├── config.py         # yaml config with env overrides, auto-dim
└── web/
    ├── app.py        # FastAPI with auth, model warmup
    ├── routes.py     # 57 REST endpoints
    └── templates/
        └── index.html  # single-page dashboard
```

## data flow

### write path

```
content → canonicalize → enrich (keywords+tags+summary) → embed
    → surprise gate (k-NN novelty check)
    → CRUD classification (ADD/UPDATE/NOOP)
    → memory evolution (update neighbors if context changed)
    → save to SQLite + update FTS5 + add to ANN index
    → extract entities + build relationships
    → compute importance (9-factor)
```

### read path

```
query → intent classification → 4 parallel channels
    → RRF fusion → temporal boost → cross-encoder rerank
    → deep MLP rerank → noise + threshold gate
    → record access → return results
```

### lifecycle

```
dream cycle (consolidate):
    forgetting curve → cluster + merge → peer cards
    → cross-domain bridges → belief probing
    → drift detection → archive old + prune logs
```

## storage

one SQLite database with WAL mode:

- `memories` — content, embedding (BLOB), importance, layer, timestamps
- `memories_fts` — FTS5 virtual table for BM25
- `entities` — canonical name, aliases, type, metadata
- `entity_mentions` — memory ↔ entity links
- `relationships` — entity ↔ entity with type, strength, temporal validity
- `access_log` — every recall recorded for reranker training
- `events` — all reads/writes for the web dashboard
- `diary_entries` — session notes
- `importance_history` — importance score over time
- `hypothetical_queries` — generated questions per memory (docTTTTTquery)
- `ingest_log` — file hash tracking for dedup

HNSW index persists separately at `~/.local/share/engram/hnsw.index`.

## design decisions

- **SQLite over Postgres** — single file, no ops, WAL handles concurrent readers. good enough for 1M+ memories
- **hnswlib over FAISS** — lighter, pip-installable, cosine space native, simpler API
- **hybrid retrieval over single-channel** — each channel catches what others miss. RRF fusion is provably better than any single signal
- **local-first** — everything runs on your machine by default. API backends are optional
- **surprise at write time** — prevents garbage in, not just garbage out
