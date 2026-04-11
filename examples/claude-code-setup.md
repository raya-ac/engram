# claude code setup

wire engram into claude code as an MCP server so your agent has persistent memory across sessions.

## 1. install

```bash
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

first run downloads two small models (~100MB total):
- `BAAI/bge-small-en-v1.5` (33MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (22MB) — reranking

optional: install cloud embedding backends for higher quality embeddings:

```bash
pip install -e ".[voyage]"   # voyage-3.5 ($0.18/1M tokens, Anthropic recommended)
pip install -e ".[openai]"   # text-embedding-3-small ($0.02/1M tokens)
pip install -e ".[api]"      # all three (voyage + openai + gemini)
```

## 2. build the ANN index

```bash
engram index rebuild
```

this creates the HNSW approximate nearest neighbor index for fast dense search. takes a few seconds. the index auto-updates when you store new memories.

## 3. add to claude code settings

edit `~/.claude/settings.json` (or create it):

```json
{
  "mcpServers": {
    "engram": {
      "command": "/absolute/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram", "serve", "--mcp"]
    }
  }
}
```

use the absolute path to the venv python. find it with:

```bash
echo "$(cd /path/to/engram && pwd)/.venv/bin/python"
```

restart claude code. you should see `engram` in your MCP servers with 63 tools.

## 4. add memory instructions to CLAUDE.md

put this in the `CLAUDE.md` at the root of your project (or `~/.claude/CLAUDE.md` for global):

```markdown
## Memory

You have a persistent memory system via the `engram` MCP server.

**When to recall:** At the start of complex tasks, when you need context
about past work, or when the user references something from a previous session.
Use `recall_hints` first for lightweight recognition, then `recall` for full context.

**When to remember:** After learning something worth keeping — user preferences,
project decisions, error patterns, architecture choices. Use:
- `remember` for general facts
- `remember_decision` for decisions with rationale
- `remember_error` for error patterns with prevention steps
- `remember_negative` for things that do NOT exist or should NOT be done

The system scores novelty automatically — redundant memories get flagged.

**Key tools:**
- `recall` — hybrid search across all memory layers (HNSW + BM25 + entity graph + Hopfield)
- `recall_hints` — lightweight hints to check if you know something before full recall
- `recall_entity` — everything about a person/project/tool
- `recall_timeline` — what happened in a date range
- `remember` — store with automatic surprise scoring
- `remember_decision` — decisions with rationale (procedural layer)
- `remember_error` — error patterns with prevention (procedural layer)
- `remember_negative` — what does NOT exist (prevents hallucinated recommendations)
- `get_skills` — get focused procedural guidance for a task
- `layers` — graduated L0-L3 context for system prompt injection
- `status` — check memory stats
```

## 5. seed some memories

ingest existing notes, docs, or conversations:

```bash
# ingest markdown files
engram ingest ~/notes/

# ingest claude code session logs
engram ingest ~/.claude/projects/*/sessions/*.jsonl

# or use the MCP tool from within claude code
# (just ask: "ingest my session logs")
```

## 6. verify it works

in claude code, try:

```
> what do you remember about this project?
```

the agent should use `recall` or `recall_hints` to search its memory.

## 7. start the web dashboard (optional)

```bash
engram serve --web
# → http://127.0.0.1:8420
```

watch the neural map light up as the MCP server reads and writes memories. the dashboard polls the same database so everything shows up in real time.

to lock down the dashboard on a network:

```yaml
# config.yaml
web:
  auth_token: "your-secret-token"
```

## 8. watch a directory for auto-ingest (optional)

```bash
engram watch ~/notes/ --interval 30
```

polls the directory every 30s for new or changed files and auto-ingests them.

## tips

- **train the reranker** after a few days of use. it learns which memories are actually useful from your access patterns: `train_reranker`
- **run the dream cycle** periodically to consolidate, deduplicate, and bridge cross-domain connections: `consolidate`
- **use `recall_hints` before `recall`** when you just want to check if memory exists. returns truncated snippets instead of full content, keeps context windows lean.
- **watch surprise scores** on `remember` responses. if surprise is low (< 0.3), you're storing redundant info.
- **use `drift_check`** periodically to find memories that reference dead file paths or missing functions.
- **export your memories** before major changes: `engram export backup.json --include-embeddings`
- **switch to API embeddings** for better quality: set `embedding_model: voyage-3.5` in config.yaml, then `engram reembed`
