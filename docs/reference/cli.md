# CLI commands

all commands available via `engram <command>`.

## setup and checks

`init` and `doctor` are available from 0.8.0.

### init

```sh
engram init
engram --config /absolute/new/config.yaml init --yes --preset portable \
  --db-path /absolute/new/memory.db
engram --config /absolute/new/config.yaml init --yes --json
```

guided setup selects storage, model preset and new paths. `--yes` accepts the
chosen options without prompts and is required without a terminal; `--json` also
requires `--yes`. the global `--config` argument names a **new output file** for
this command. otherwise it uses `~/.config/engram/config.yaml`, with SQLite at
`~/.local/share/engram/memory.db`.

`--preset local` uses BGE embeddings/reranking with automatic local embedding
runtime selection. `portable` selects `sentence_transformers`; `light` also
switches the reranker to MiniLM. these are local presets, with device selection
left to the runtime. `--embedding-backend auto|mlx|sentence_transformers`,
`--embedding-model` and `--cross-encoder-model` can override the preset. init
accepts known local embedding models and local rerankers.

setup validates before writing, creates private config/database files, and
refuses existing files or SQLite sidecars. it never edits agent/global client
settings or installs models. the output includes a portable MCP launch command,
connection JSON and a `doctor --full` command. inherited environment overrides
remain active; their names are reported without copying secrets into the output.
make required variables available to the agent process too.

for Postgres, set `ENGRAM_POSTGRES_DSN` in the environment and use
`--storage postgres`. the database must exist with an empty current schema;
setup does not clear or migrate existing tables. the environment-sourced DSN is
not copied into the new config. use the migration command for an existing store.

### doctor

```sh
engram --config /absolute/config.yaml doctor [--json]
engram --config /absolute/config.yaml doctor --full [--json]
```

the basic check validates effective configuration, reads existing SQLite schema
and counts, and inspects package metadata, credential presence and local model
files. it does not initialize storage, load models or contact providers.
Postgres reachability is skipped until requested.

| flag | additional checks |
| --- | --- |
| `--check-models` | real configured embedding/reranker inference on synthetic text; may download weights or contact a provider |
| `--check-connection` | isolated local stdio MCP handshake, tool discovery and redacted config read; also permits read-only inspection of configured Postgres storage |
| `--smoke` | model checks plus a real synthetic save/retrieve cycle in temporary SQLite, using the configured retrieval gates; no LLM |
| `--full` | all three checks above |

doctor does not alter stored memories or client settings. temporary stores use
isolated paths; normal SQLite read bookkeeping can still touch WAL/SHM sidecars.
the MCP check verifies a local Engram endpoint, not whether another agent has
attached. the synthetic SQLite check does not certify production Postgres writes.

the report lists each check and an overall `pass`, `incomplete` or `fail`.
`incomplete` means an optional check was skipped or a readiness warning remains;
it is expected from a basic run. passing and incomplete reports exit 0, failed
checks exit 1, and invalid configuration exits 2. `--json` returns the structured
report for scripts.

## configuration

```sh
engram --config /absolute/config.yaml config check [--json]
engram --config /absolute/config.yaml config show [--json]
engram config show --defaults
engram config check --defaults --json
engram config schema
```

`check` validates the selected file, environment overrides and defaults. invalid
settings exit with status 2; JSON failures contain `valid: false` and a safe
`error`. `show` includes effective values and their sources, with credentials
redacted. `--defaults` bypasses files and environment and cannot accompany
`--config`. `schema` always emits JSON describing supported fields; it does not
load configuration. none of these commands connects storage, loads models, or
rewrites a file. see [configuration](config.md) for precedence and all overrides.

## memory operations

### ingest
```bash
engram ingest <paths...> [-j JOBS] [--no-queries]
```
ingest files or directories. supports markdown, plaintext, JSON (Claude Code, ChatGPT, Slack), PDF. `-j` for parallel extraction. `--no-queries` skips hypothetical query generation.

### search
```bash
engram search <query> [-k TOP_K] [--explain] [--rerank] [--json]
```
hybrid search across all layers. omitting `-k` uses `retrieval.top_k` (package
default 10). `--rerank` enables the cross-encoder; CLI search leaves it off unless
requested. `--json` returns an array of results.

```sh
engram search "why did we change the storage backend?" --rerank --explain --json
```

`--explain` and its existing alias `--debug` explain the actual run, including
returned and rejected candidates, observed scores, confidence decisions,
passage retries and effective retrieval settings. they do not enable reranking
themselves. JSON output becomes `{results, explanation}`. explanations require an
initialized store and can load the configured models.

this diagnostic search bypasses the result cache and leaves access history,
importance and dormant evaluations unchanged. it explains the bounded candidates
produced by the search, not every memory in storage. lifecycle/profile-filtered
candidates have no content in the report. keep the returned order; coverage can
change rank without changing scores. see [retrieval internals](../architecture/retrieval.md#explanations).

### remember
```bash
engram remember <content> [--source SOURCE] [--layer LAYER] [--importance IMPORTANCE]
```
store a memory directly. default layer is episodic, default importance 0.7.

### entity
```bash
engram entity <name> [--graph]
```
look up an entity. `--graph` shows the relationship graph and 2-hop traversal.

## maintenance

### consolidate
```bash
engram consolidate
```
run the dream cycle — cluster, summarize, peer cards, cross-domain bridges, belief probing, drift detection, archival.

### drift
```bash
engram drift [--search-roots DIRS] [--project-root DIR] [--fix] [--dry-run] [--json] [--no-functions]
```
check memory drift against filesystem. `--fix` auto-invalidates dead refs. `--dry-run` previews fixes.

### patterns
```bash
engram patterns [--hours N] [--threshold N] [--dry-run]
```
extract reusable procedural patterns from recent session activity.

## index management

### index
```bash
engram index rebuild    # full HNSW index rebuild
engram index status     # show index size, vector count, last built
```

### reembed
```bash
engram reembed [--batch-size N] [--dry-run]
```
re-embed all memories with the current model. use after switching embedding models.

## data management

### export
```bash
engram export <output> [--layer LAYER] [--include-embeddings]
```
export to JSON or JSONL. `--include-embeddings` adds base64 vectors for portable backup.

### import
```bash
engram import <input> [--skip-duplicates]
```
restore from exported file. `--skip-duplicates` skips memories with matching content hash.

### migrate-postgres
```bash
engram migrate-postgres --dsn postgresql://user:pass@localhost:5432/engram [--from-sqlite PATH] [--switch-config] [--force-reset]
engram migrate-postgres --verify-only --dsn postgresql://user:pass@localhost:5432/engram
```
copy an existing sqlite Engram store into postgres, verify the migrated counts, and optionally rewrite `config.yaml` to switch the default backend. `--verify-only` just checks connectivity and shows source/target counts. `--force-reset` truncates the target tables before copying.

### watch
```bash
engram watch <path> [--interval SECONDS]
```
poll a directory for new/changed files and auto-ingest. default 30s interval.

## server

### serve
```bash
engram serve --web [--port PORT]      # web dashboard (default 8420)
engram serve --mcp                     # MCP server (stdio, for Claude Code)
engram serve --mcp-sse [--port PORT]  # MCP server (HTTP/SSE, default 8421)
```

## info

### status
```bash
engram status
```
memory counts by layer, entities, relationships, DB size, ANN index status.

### demo
```bash
engram demo [--keep] [--web] [--port PORT]
```
interactive walkthrough with sample data. `--keep` preserves the demo database. `--web` starts the dashboard.

## dormant review and Codex adapter

```sh
engram --config /absolute/config.yaml dormant review --limit 20
engram --config /absolute/config.yaml dormant inspect EVENT_ID
engram --config /absolute/config.yaml dormant feedback EVENT_ID dismissed
engram --config /absolute/config.yaml codex setup --project /absolute/project
engram --config /absolute/config.yaml codex serve --project /absolute/project
engram --config /absolute/config.yaml codex context --project /absolute/project
engram --config /absolute/config.yaml codex diagnostics --project /absolute/project
```

Dormant commands are an explicit review flow; the adapter is a separate stdio
server for a bound project. `codex setup` only prints a registration command.
See [dormant recall](../dormant-recall.md) and the [Codex adapter](../codex-adapter.md).

## native local API

```sh
engram --config /absolute/path/to/config.yaml api
```

serves sequential JSONL requests until EOF, with no network listener or MCP
framing. the config must be an existing absolute path. use the `operations`
request for the running [native API schemas](../native-api.md). storage
operations require an initialized database; startup does not run legacy
backfills or load models. semantic search loads its models on demand.
