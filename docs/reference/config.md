# Configuration

## create a new configuration

```sh
engram init
engram --config /absolute/new/config.yaml init --yes --preset local \
  --db-path /absolute/new/memory.db
```

`init` guides storage/model selection and initializes a new store. its global
`--config` argument names a new output file; existing config, database and index
files are refused. without it, setup uses `~/.config/engram/config.yaml`.
`local`, `portable` and `light` are local model presets. setup does not download
models or edit agent settings; it prints connection settings and a
`doctor --full` command. see [setup and doctor](cli.md#setup-and-checks).

environment variables keep their usual precedence during setup. their names are
reported so you can preserve them in the agent environment, and inherited
credentials are not copied into the file or printed snippet. Postgres setup
requires `ENGRAM_POSTGRES_DSN`, an existing database and an empty current schema.
the saved config continues to depend on that environment variable.

## inspect settings

inspect the settings Engram will use before starting a server or search:

```sh
engram --config /absolute/config.yaml config check
engram --config /absolute/config.yaml config show --json
engram config show --defaults
engram config schema
```

`check` validates settings. `show` returns effective values and where each came
from. both accept `--json` and `--defaults`; defaults bypass files and environment
variables and cannot be combined with `--config`. `schema` always returns JSON
with every field's default, type, bounds, environment name and help. it does not
read or validate a config file. these commands do not open storage, load models,
or rewrite configuration.

`show --json` contains `config_file`, nested `values`, dotted-path `sources`, and
`warnings`. sources distinguish `default`, `file`, `env:VARIABLE`, and
`derived:reason`, including model-derived dimensions and runtime overrides.
nonempty `hf_token`, `postgres_dsn`, `llm.api_key` and `web.auth_token` values are
replaced with `<redacted>`; empty credentials remain empty. reports from a running
server describe its loaded configuration, which stays in effect until restart.

unknown fields, invalid sections or scalar types, unsupported backends,
nonfinite numbers and out-of-range values fail validation. numeric fields need
numbers, boolean fields need booleans, and postgres requires a nonempty DSN.
errors identify the field or section and rule without quoting supplied values or
YAML snippets. invalid file values are rejected even when an environment override
exists. `check --json` returns `{"valid": false, "error": "..."}` and exits with
status 2 on failure.

## full reference

these values match the package defaults. a supplied `embedding_dim` is explicit;
omit it to use the dimension registry when changing to a known model.

```yaml
storage_backend: sqlite
db_path: ~/.local/share/engram/memory.db
postgres_dsn: ""

# known hosted models select their provider from the model name
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

# omit this field to derive it from a known model
embedding_dim: 384

retrieval:
  top_k: 10                # final results returned
  rrf_k: 60                # RRF fusion constant
  min_confidence: 0.60     # threshold gate (cross-encoder scores only)
  rerank_candidates: 20    # candidates sent to cross-encoder
  rerank_fusion_alpha: 0.0 # optional prior rank blend, from 0 to 1
  preserve_prior_candidate: true # keep one eligible hybrid leader in the requested results
  rerank_passage_fallback: true # one excerpt for weak scores or rejected long memories
  rerank_passage_floor: 0.001 # early activation floor; zero disables excerpt retries
  dense_multiplier: 3      # dense candidates = top_k * multiplier
  bm25_multiplier: 3       # BM25 candidates = top_k * multiplier
  enable_query_expansion: true
  exact_match_boost: 1.22
  search_cache_size: 128

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
  backend: claude_cli        # claude_cli | anthropic | openai | mlx
  model: claude-sonnet-4-20250514
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
or prior adjustments. production search also retries eligible long memories
that fall below the final `min_confidence` gate after those adjustments, even
when another candidate passes. already scored excerpts are not repeated.
the default early activation floor is 0.001; `min_confidence` controls returned
results and defaults to 0.6. each eligible document longer
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
changing `min_confidence` changes which candidates qualify for the production
retry, without changing the early activation floor. a zero confidence gate
disables the production retry; the early retry still follows its own floor.

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

every field has an environment override: uppercase its dotted path, replace dots
with underscores, and add `ENGRAM_`. for example, `retrieval.min_confidence`
becomes `ENGRAM_RETRIEVAL_MIN_CONFIDENCE`:

```bash
export ENGRAM_DB_PATH=/custom/path/memory.db
export ENGRAM_STORAGE_BACKEND=postgres
export ENGRAM_POSTGRES_DSN=postgresql://user:pass@localhost:5432/engram
export ENGRAM_EMBEDDING_MODEL=voyage-3.5
export ENGRAM_EMBEDDING_DIM=1024
export ENGRAM_EMBEDDING_BACKEND=voyage
export ENGRAM_RETRIEVAL_MIN_CONFIDENCE=0.7
export ENGRAM_RETRIEVAL_RERANK_PASSAGE_FALLBACK=false
export ENGRAM_ANN_EF_SEARCH=200
export ENGRAM_WEB_PORT=9000
export ENGRAM_DORMANT_RECALL_MODE=off
```

environment booleans accept `true`, `false`, `1` or `0` (case-insensitive).
other strings such as `yes` are rejected. numeric overrides must parse as the
required type and satisfy the same bounds as file values. an explicitly empty
string overrides a file string; an empty numeric or boolean value is invalid.

credential aliases and provider keys:

```bash
export ANTHROPIC_API_KEY=your-key   # for llm.backend: anthropic
export OPENAI_API_KEY=your-key      # for llm.backend: openai (or embedding)
export VOYAGE_API_KEY=your-key      # for embedding backend
export GEMINI_API_KEY=your-key      # for embedding backend
export HF_TOKEN=your-token         # or ENGRAM_HF_TOKEN / HUGGING_FACE_HUB_TOKEN
```

`ENGRAM_HF_TOKEN` takes precedence over the nonempty `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` aliases, in that order. for LLMs,
`ENGRAM_LLM_API_KEY` overrides the file; when the effective `llm.api_key` is
empty, the selected provider's `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is used.
embedding providers read their own API-key variables; `llm.api_key` does not
configure embeddings. Gemini also accepts `GOOGLE_API_KEY` as a fallback.

### LLM backends

| backend | auth | notes |
|---------|------|-------|
| `claude_cli` | Claude Code login | uses `claude` CLI subprocess |
| `anthropic` | `ANTHROPIC_API_KEY` or `llm.api_key` | direct API, any Claude model |
| `openai` | `OPENAI_API_KEY` or `llm.api_key` | a model supported by the OpenAI backend |
| `mlx` | local | runs Qwen/Llama/etc on Apple Silicon GPU |

install the backend you need:

```bash
pip install 'engram-memory-system[anthropic]'   # anthropic SDK
pip install 'engram-memory-system[openai]'      # openai SDK
pip install 'engram-memory-system[api]'         # all backends
```

## load priority

1. environment variables (highest)
2. one config file
3. defaults (lowest)

an explicit `--config` path must be readable; a missing or unreadable file fails
instead of falling back. without it, Engram uses the first existing file from
`./config.yaml`, the package's project root, then
`~/.config/engram/config.yaml`. files are not merged. if none exists, defaults and
environment overrides apply. a known model supplies `embedding_dim` only when
that field was omitted from both file and environment. unknown models keep the
default dimension and produce a warning to set it explicitly.

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
