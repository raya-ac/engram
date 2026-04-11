# Configuration

lives at `config.yaml` (project root), `~/.config/engram/config.yaml` (user), or any path passed with `--config`. env vars override everything with `ENGRAM_` prefix.

## full reference

```yaml
db_path: ~/.local/share/engram/memory.db

# embedding model — auto-detects backend from model name
# local:  BAAI/bge-small-en-v1.5 (384d), BAAI/bge-base-en-v1.5 (768d)
# voyage: voyage-3.5 (1024d), voyage-3.5-lite (1024d), voyage-code-3 (1024d)
# openai: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
# gemini: gemini-embedding-001 (768d)
embedding_model: BAAI/bge-small-en-v1.5

# reranker — local or API
# local:  cross-encoder/ms-marco-MiniLM-L-6-v2
# voyage: rerank-2.5, rerank-2.5-lite
cross_encoder_model: cross-encoder/ms-marco-MiniLM-L-6-v2

# auto | mlx | sentence_transformers | voyage | openai | gemini
embedding_backend: auto

# auto-detected from model name if known
embedding_dim: 384

retrieval:
  top_k: 10                # final results returned
  rrf_k: 60                # RRF fusion constant
  min_confidence: 0.60     # threshold gate (cross-encoder scores only)
  rerank_candidates: 20    # candidates sent to cross-encoder
  dense_multiplier: 3      # dense candidates = top_k * multiplier
  bm25_multiplier: 3       # BM25 candidates = top_k * multiplier

lifecycle:
  forgetting_half_life_days: 30
  archive_after_days: 90
  archive_min_importance: 0.3
  archive_min_accesses: 3
  promote_importance: 0.7
  promote_accesses: 5
  cluster_threshold: 0.8
  cluster_min_size: 5
  retention_mode: huber     # l2 | huber | elastic
  huber_delta: 0.5
  elastic_l1_ratio: 0.3

llm:
  backend: claude_cli       # claude_cli | mlx | llamacpp
  model: claude-sonnet-4-20250514
  mlx_model: mlx-community/Qwen2.5-3B-Instruct-4bit

web:
  host: 127.0.0.1
  port: 8420
  auth_token: ""            # set to enable bearer token auth

ann:
  enabled: true
  m: 32                     # HNSW graph connectivity
  ef_construction: 200      # build-time search depth
  ef_search: 100            # query-time search depth
  max_elements: 500000      # pre-allocated capacity
  index_path: ~/.local/share/engram/hnsw.index
```

## environment variables

any config field can be overridden with `ENGRAM_` prefix:

```bash
export ENGRAM_DB_PATH=/custom/path/memory.db
export ENGRAM_EMBEDDING_MODEL=voyage-3.5
export ENGRAM_EMBEDDING_DIM=1024
export ENGRAM_EMBEDDING_BACKEND=voyage
```

API keys (not in config, env-only):

```bash
export VOYAGE_API_KEY=your-key
export OPENAI_API_KEY=your-key
export GEMINI_API_KEY=your-key
```

## load priority

1. environment variables (highest)
2. config file (first found from: `--config` path, `./config.yaml`, project root, `~/.config/engram/config.yaml`)
3. defaults (lowest)
