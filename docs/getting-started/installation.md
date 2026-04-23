# Installation

## from PyPI

```bash
pip install engram-memory-system
```

## from source

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

requires python 3.11+. first run downloads two small models (~100MB total):

- `BAAI/bge-small-en-v1.5` (33MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB) — reranking

## storage backends

engram supports:

- `sqlite` for local-first installs
- `postgres` for concurrent web + MCP deployments

sqlite is still the default and requires no extra setup.

if you want postgres from day one, point config at it:

```yaml
storage_backend: postgres
postgres_dsn: postgresql://user:pass@localhost:5432/engram
```

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
engram status
```

should show your database path, memory counts, and ANN index status.
