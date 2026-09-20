# Installation

## from PyPI

```bash
pip install --upgrade engram-memory-system
engram --version
```

## from source

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

requires python 3.11+. local models download on first use and reuse cached
weights on later runs. the defaults are:

- `BAAI/bge-small-en-v1.5` — embeddings
- `BAAI/bge-reranker-base` — reranking

download size and inference time depend on the selected model, backend and
hardware. `cross-encoder/ms-marco-MiniLM-L-6-v2` remains available as an optional
local reranker through `cross_encoder_model`.

## set up a new store

engram 0.8.0 includes `init` for guided setup and `doctor` for checking the result.

```sh
engram init
```

the guided setup chooses storage, a local model preset and new config/database
paths. it creates a private config and initializes storage, then prints an MCP
command and connection snippet for your agent. it does not edit client settings,
install dependencies or download models.

for the default choices without prompts:

```sh
engram init --yes
```

the default config is `~/.config/engram/config.yaml`; SQLite data lives at
`~/.local/share/engram/memory.db`. to choose new paths:

```sh
engram --config /absolute/new/config.yaml init --yes \
  --db-path /absolute/new/memory.db --preset portable
```

`--config` comes before `init` and names the file to create. existing config,
database and index files are refused. use your existing config with `doctor`
instead of running setup over it. inherited `ENGRAM_*` overrides retain their
usual priority and are listed without exposing credentials.

| preset | local models and runtime |
| --- | --- |
| `local` | BGE embeddings and reranker; automatic local embedding runtime |
| `portable` | same models with `sentence_transformers` |
| `light` | `sentence_transformers` with the smaller MiniLM reranker |

the runtime selects its device; `portable` does not force CPU execution.

## storage backends

engram supports:

- `sqlite` for local-first installs
- `postgres` for concurrent web + MCP deployments

sqlite is still the default and requires no extra setup.

for a new Postgres store, set `ENGRAM_POSTGRES_DSN` in the environment first:

```sh
engram --config /absolute/new/postgres.yaml init --storage postgres --yes
```

the selected database must already exist and its current schema must be empty.
setup creates the Engram tables; it never clears or migrates existing tables.
the DSN stays in the environment and must also be available to the agent process.

if you're already using sqlite and want to move later, use the migration guide:

- [Postgres Migration](../guides/postgres-migration.md)

## optional: API embedding backends

use cloud embedding APIs for higher quality:

```bash
pip install engram-memory-system[voyage]   # voyage-3.5, voyage-3.5-lite
pip install engram-memory-system[openai]   # text-embedding-3-small/large
pip install engram-memory-system[gemini]   # gemini-embedding-001
pip install engram-memory-system[api]      # all three
```

set API keys:

```bash
export VOYAGE_API_KEY="your-key"    # https://dash.voyageai.com/
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

see [Embedding Backends](../guides/embedding-backends.md) for model comparison and switching.

## docker

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
docker compose up -d
# → http://localhost:8420
```

see [Docker Guide](../guides/docker.md) for configuration.

## build the ANN index

after installing, build the HNSW index for fast dense search:

```bash
engram index rebuild
```

this auto-updates on write/forget. only needed once on first install or after bulk operations.

## verify

```bash
engram --config /absolute/path/to/config.yaml doctor
engram --config /absolute/path/to/config.yaml doctor --full
```

use the path printed by `init`. the basic check reads configuration, existing
SQLite storage, package metadata and model-cache files. `incomplete` means
optional checks were skipped or readiness remains unverified.

`--full` runs the configured models, checks an isolated local MCP process, and
saves/retrieves fictional data in a temporary database without an LLM. it can
download model weights or call the configured embedding/reranker provider with
synthetic text. Postgres inspection makes a read-only connection. stored memories
and client settings stay unchanged; SQLite reads may use normal WAL/SHM
bookkeeping. the local MCP check establishes endpoint behavior, not attachment
by an external agent. see [doctor flags and exit codes](../reference/cli.md#doctor).
