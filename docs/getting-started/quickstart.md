# Quick Start

## ingest some files

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

`--debug` shows the retrieval stage breakdown (dense, BM25, graph, RRF scores). `--rerank` enables the cross-encoder for better precision (~300ms slower).

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
