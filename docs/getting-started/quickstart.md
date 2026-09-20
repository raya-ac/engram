# Quick Start

## initialize and check

after [installing Engram 0.8.0 or newer](installation.md), start with:

```sh
engram --version
engram init
```

follow the prompts for a new store, or use `engram init --yes` for the local
defaults. `--preset portable` chooses `sentence_transformers`; `--preset light`
also selects the smaller MiniLM reranker. setup prints the config path and MCP
connection settings without editing your agent's configuration. it refuses
existing config/database files and does not load models.

run the command printed at the end of setup:

```sh
engram --config /absolute/path/to/config.yaml doctor --full
```

this checks real model inference, an isolated local MCP endpoint and a synthetic
save/retrieve cycle without an LLM. first use may download local weights; a
configured hosted model can contact its provider with synthetic text. diagnostic
writes stay in temporary storage, and your memory records remain unchanged.
the MCP self-test checks the local endpoint; use the printed snippet to connect
your agent separately.

plain `doctor` performs the lighter configuration/storage/package checks. its
`incomplete` status means optional checks were skipped or remain unverified;
failed checks exit with status 1, invalid configuration with status 2.

with a custom setup path, keep passing `--config /absolute/path/to/config.yaml`
before the commands below. environment overrides still apply to both setup and
the agent process. [the CLI reference](../reference/cli.md#init) covers unattended
setup, all presets and Postgres.

## ingest some files

file extraction uses your configured LLM backend; the default is the Claude CLI.
configure that backend before ingestion. the setup and doctor checks above do
not require an LLM.

```bash
engram ingest ~/notes/
engram ingest ~/projects/docs/ ~/journal/
```

supports markdown, plaintext, JSON (Claude Code JSONL, Claude.ai JSON, ChatGPT JSON, Slack), PDF.

## search

```bash
engram search "what happened on march 28"
engram search "melee garden architecture" --debug
engram search "apple sandbox bypass" --rerank
```

`--debug` shows the retrieval stage breakdown (dense, BM25, graph, RRF scores).
`--rerank` enables the configured cross-encoder, `BAAI/bge-reranker-base` by
default. its additional inference time depends on the model and hardware.

## remember something

```bash
engram remember "deploy command: npm run build && rsync" --layer procedural
engram remember "Ari prefers casual tone" --importance 0.9
```

## check status

```bash
engram status
```

shows memory counts by layer, entity count, relationships, DB size, and ANN index status.

## entity lookup

```bash
engram entity Ari --graph
```

## start the web dashboard

```bash
engram serve --web
# → http://127.0.0.1:8420
```

17 panels: neural map, search, memories, entities, timeline, remember, analytics, heatmap, context, ingest, health, dedup, cognition, bridges, drift, patterns, plus an inspector panel.

## start the MCP server

```bash
engram serve --mcp       # stdio (for Claude Code)
engram serve --mcp-sse   # HTTP/SSE (for remote clients)
```

## watch a directory

```bash
engram watch ~/notes/ --interval 30
```

polls for new/changed files and auto-ingests.

## export and import

```bash
engram export backup.json --include-embeddings
engram import backup.json --skip-duplicates
```

## run tests

```bash
pytest tests/ -v    # 72 tests, ~3s
```
