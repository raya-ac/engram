# Configuration

lives at `config.yaml` (project root), `~/.config/engram/config.yaml` (user), or any path passed with `--config`. env vars override everything with `ENGRAM_` prefix.

## full reference

```yaml
storage_backend: sqlite
db_path: ~/.local/share/engram/memory.db
postgres_dsn: ""

# embedding model — auto-detects backend from model name
# local:  BAAI/bge-small-en-v1.5 (384d), BAAI/bge-base-en-v1.5 (768d)
# voyage: voyage-3.5 (1024d), voyage-3.5-lite (1024d), voyage-code-3 (1024d)
# openai: text-embedding-3-small (1536d), text-embedding-3-large (3072d)
# gemini: gemini-embedding-001 (768d)
embedding_model: BAAI/bge-small-en-v1.5

# reranker — local or API
# local:  BAAI/bge-reranker-base (default), cross-encoder/ms-marco-MiniLM-L-6-v2
# voyage: rerank-2.5, rerank-2.5-lite
cross_encoder_model: BAAI/bge-reranker-base

# optional Hugging Face token for downloads / rate limits
hf_token: ""

# auto | mlx | sentence_transformers | voyage | openai | gemini
embedding_backend: auto

# auto-detected from model name if known
embedding_dim: 384

retrieval:
  top_k: 10                # final results returned
  rrf_k: 60                # RRF fusion constant
  min_confidence: 0.60     # threshold gate (cross-encoder scores only)
  rerank_candidates: 20    # candidates sent to cross-encoder
  rerank_fusion_alpha: 0.0 # optional prior rank blend, from 0 to 1
  preserve_prior_candidate: true # keep one eligible hybrid leader in the requested results
  rerank_passage_fallback: true # bounded local excerpt retry when all base scores are low
  rerank_passage_floor: 0.001 # activation floor, independent of the final result gate
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
  backend: anthropic         # claude_cli | anthropic | openai | mlx
  model: claude-haiku-4-5-20251001
  api_key: ""                # or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var
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

## rerank scores

`BAAI/bge-reranker-base` is the default local reranker. an existing config file
can continue selecting `cross-encoder/ms-marco-MiniLM-L-6-v2` or a supported
hosted model.

`rerank_passage_fallback: true` enables one local excerpt retry when all
full-document sigmoid scores are below `rerank_passage_floor`, before temporal
or prior adjustments. the default activation floor is 0.001; `min_confidence`
independently controls returned results and defaults to 0.6. each eligible document longer
than 160 words contributes at most one source-contiguous excerpt, capped at
160 words, selected from a matching sentence and its neighbors. short documents
and documents without a lexical match keep their full-document scores. hosted
rerankers are unaffected. lexical selection recognizes conservative regular
English singular/plural forms and counts each original query term once per
sentence, across repetitions and surface variants.

both model calls keep the same semantic query, and the larger full/excerpt raw
score survives. with an explicit reference date, a resolved relative-time span
is removed only from lexical excerpt selection. the later confidence filter
can still reject the result. retries add model work and can lose context; set
`rerank_passage_fallback: false` or `rerank_passage_floor: 0` to disable retries.
changing `min_confidence` does not change the activation floor.

the activation floor was chosen during development on LongMemEval. evaluation
on that same dataset is a development result, not held-out accuracy. these
thresholds are not calibrated probabilities.

local cross-encoders return logits, which ordinary retrieval maps into the 0–1
range with one sigmoid. hosted rerankers already return normalized relevance
scores. when temporal evidence applies, it shifts log-odds before the final
score. these are relevance scores, not measured probabilities of correctness.

`rerank_fusion_alpha: 0.0` leaves the model score unchanged apart from temporal
evidence. a value from 0 to 1 blends it with `1 / (prior_rank + 1)`, where
`prior_rank` starts at zero. `min_confidence` filters the resulting score.

`preserve_prior_candidate: true` then keeps the best confidence-eligible hybrid
candidate within a requested result count of at least two. if needed, it moves
that candidate into the last requested position and retains the rerank winner.
a request for one result keeps the winner. rejected candidates remain excluded;
coverage never bypasses `min_confidence`.

coverage changes order without changing scores, even with fusion weight 0.
keep the returned order when displaying or consuming results. set
`preserve_prior_candidate: false` for model ordering without this coverage step.
changing fusion, confidence, coverage, passage fallback or its activation floor
prevents reuse of results cached under the old policy. random ranking noise applies only
to searches with cross-encoder reranking off.

## storage

`storage_backend` controls which database Engram uses:

- `sqlite` — local file-backed default
- `postgres` — concurrent service backend

when `storage_backend: postgres`, `postgres_dsn` is required and `db_path` is ignored for live reads/writes.

example:

```yaml
storage_backend: postgres
postgres_dsn: postgresql://user:pass@localhost:5432/engram
```

## environment variables

any config field can be overridden with `ENGRAM_` prefix:

```bash
export ENGRAM_DB_PATH=/custom/path/memory.db
export ENGRAM_STORAGE_BACKEND=postgres
export ENGRAM_POSTGRES_DSN=postgresql://user:pass@localhost:5432/engram
export ENGRAM_EMBEDDING_MODEL=voyage-3.5
export ENGRAM_EMBEDDING_DIM=1024
export ENGRAM_EMBEDDING_BACKEND=voyage
```

API keys (env vars or `llm.api_key` in config):

```bash
export ANTHROPIC_API_KEY=your-key   # for llm.backend: anthropic
export OPENAI_API_KEY=your-key      # for llm.backend: openai (or embedding)
export VOYAGE_API_KEY=your-key      # for embedding backend
export GEMINI_API_KEY=your-key      # for embedding backend
export HF_TOKEN=your-token          # or ENGRAM_HF_TOKEN for Hugging Face downloads
```

### LLM backends

| backend | auth | notes |
|---------|------|-------|
| `claude_cli` | Claude Code login | uses `claude` CLI subprocess |
| `anthropic` | `ANTHROPIC_API_KEY` or `llm.api_key` | direct API, any Claude model |
| `openai` | `OPENAI_API_KEY` or `llm.api_key` | any OpenAI/compatible model |
| `mlx` | local | runs Qwen/Llama/etc on Apple Silicon GPU |

install the backend you need:

```bash
pip install 'engram-memory-system[anthropic]'   # anthropic SDK
pip install 'engram-memory-system[openai]'      # openai SDK
pip install 'engram-memory-system[api]'         # all backends
```

## load priority

1. environment variables (highest)
2. config file (first found from: `--config` path, `./config.yaml`, project root, `~/.config/engram/config.yaml`)
3. defaults (lowest)

## dormant recall

```yaml
dormant_recall:
  mode: "off"                 # opt into shadow; no automatic visible mode
  candidate_limit: 50
  dormancy_days: 30
  min_relevance: 0.75
  max_bonus: 0.05
  rerank_candidates: 12
  min_rerank_score: 0.6
  cooldown_days: 7
  feedback_cooldown_days: 30
  log_max_events: 1000
  log_retention_days: 30
```

`ENGRAM_DORMANT_RECALL_MODE` overrides mode at process start. Current eligible
vectors are searched independently of ordinary ANN/cache state, and the bounded
query/content relevance check precedes selection. Configuration and code do not
hot-reload into an existing Python process. Full behavior and bounds are in
[the dormant guide](../dormant-recall.md). The separate Codex adapter binds its
project through `codex serve --project`; it uses the same explicit Engram config
without changing host configuration automatically.
