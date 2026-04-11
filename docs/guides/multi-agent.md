# Multi-Agent

multiple agents can share the same engram database. they automatically share memories — what one agent stores, all agents can recall.

## setup

point all agents at the same database:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram", "serve", "--mcp"],
      "env": {
        "ENGRAM_DB_PATH": "/shared/path/memory.db"
      }
    }
  }
}
```

## how it works

- SQLite WAL mode handles concurrent readers with a single writer
- the access_log tracks which memories get used by which process
- the deep reranker learns from all agents' access patterns
- entity graphs naturally bridge across agents' domains

## cross-domain recall

when agent A stores knowledge about deployment and agent B searches for "infrastructure", they find each other's memories through:

- dense similarity (semantic overlap)
- BM25 (keyword matching)
- entity graph (shared entity references)

the dream cycle's cross-domain synthesis step explicitly looks for entity pairs that appear in different contexts and creates bridge memories.

## web dashboard

start the dashboard to watch all agents in real time:

```bash
ENGRAM_DB_PATH=/shared/path/memory.db engram serve --web
```

the neural map shows entity activations as any agent reads or writes.

## example

see [`examples/multi-agent.py`](https://github.com/raya-ac/engram/blob/main/examples/multi-agent.py) for a runnable experiment with 3 specialized agents (CodeBot, ResearchBot, OpsBot) sharing a database.

## isolation

if you want agents to have separate memories, use different DB paths:

```bash
ENGRAM_DB_PATH=~/.local/share/engram/agent-a.db engram serve --mcp
ENGRAM_DB_PATH=~/.local/share/engram/agent-b.db engram serve --mcp
```
