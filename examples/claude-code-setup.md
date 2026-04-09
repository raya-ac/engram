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

## 2. add to claude code settings

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

restart claude code. you should see `engram` in your MCP servers.

## 3. add memory instructions to CLAUDE.md

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

The system scores novelty automatically — redundant memories get flagged.

**Key tools:**
- `recall` — search across all memory layers
- `recall_hints` — lightweight hints to check if you know something
- `recall_entity` — everything about a person/project/tool
- `remember` — store with automatic surprise scoring
- `remember_decision` — decisions with rationale (procedural layer)
- `remember_error` — error patterns (procedural layer)
- `train_reranker` — periodically retrain the retrieval model on usage patterns
- `status` — check memory stats
```

## 4. seed some memories

ingest existing notes, docs, or conversations:

```bash
# ingest markdown files
engram ingest ~/notes/

# ingest claude code session logs
engram ingest ~/.claude/projects/*/sessions/*.jsonl

# or use the MCP tool from within claude code
# (just ask: "ingest my session logs")
```

## 5. verify it works

in claude code, try:

```
> what do you remember about this project?
```

the agent should use `recall` or `recall_hints` to search its memory.

## tips

- **train the reranker** after a few days of use. it learns which memories are actually useful from your access patterns: `train_reranker`
- **run the dream cycle** periodically to consolidate, deduplicate, and bridge cross-domain connections: `consolidate`
- **use `recall_hints` before `recall`** when you just want to check if memory exists. it returns truncated snippets instead of full content, which keeps context windows lean.
- **watch surprise scores** on `remember` responses. if surprise is low (< 0.3), you're storing redundant info.
