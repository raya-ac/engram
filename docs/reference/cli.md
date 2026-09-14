# CLI commands

all commands available via `engram <command>`.

## memory operations

### ingest
```bash
engram ingest <paths...> [-j JOBS] [--no-queries]
```
ingest files or directories. supports markdown, plaintext, JSON (Claude Code, ChatGPT, Slack), PDF. `-j` for parallel extraction. `--no-queries` skips hypothetical query generation.

### search
```bash
engram search <query> [-k TOP_K] [--debug] [--rerank] [--json]
```
hybrid search across all layers. `--debug` shows per-stage breakdown. `--rerank` enables cross-encoder. `--json` for machine-readable output.

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
