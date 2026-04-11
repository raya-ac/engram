# Claude Code Setup

wire engram into Claude Code as an MCP server for persistent memory across sessions.

## 1. add to settings

edit `~/.claude/settings.json`:

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

find the absolute path:

```bash
echo "$(cd /path/to/engram && pwd)/.venv/bin/python"
```

restart Claude Code. you should see `engram` with 63 tools.

## 2. add memory instructions to CLAUDE.md

put this in your project's `CLAUDE.md` or `~/.claude/CLAUDE.md` for global:

```markdown
## Memory

You have a persistent memory system via the `engram` MCP server.

**When to recall:** At the start of complex tasks, or when the user
references something from a previous session.

**When to remember:** After learning something worth keeping — user
preferences, project decisions, error patterns, architecture choices.

**Key tools:**
- `recall` — hybrid search across all memory layers
- `recall_hints` — lightweight check before full recall
- `recall_entity` — everything about a person/project/tool
- `remember` — store with automatic surprise scoring
- `remember_decision` — decisions with rationale
- `remember_error` — error patterns with prevention
- `remember_negative` — what does NOT exist
- `get_skills` — focused procedural guidance for a task
```

## 3. seed memories

```bash
engram ingest ~/notes/
engram ingest ~/.claude/projects/*/sessions/*.jsonl
```

## 4. auto-extract with hooks

add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Stop",
        "hooks": [
          {
            "type": "command",
            "command": "ENGRAM_VENV=/path/to/engram/.venv /path/to/engram/hooks/save_hook.sh"
          }
        ]
      }
    ]
  }
}
```

this auto-extracts memories from every conversation — decisions, corrections, facts, Q+A pairs.

## 5. verify

in Claude Code:

```
> what do you remember about this project?
```

the agent should use `recall` or `recall_hints` to search.

## tips

- **`recall_hints` before `recall`** — check if memory exists before pulling full content
- **train the reranker** after a few days: `train_reranker`
- **run the dream cycle** periodically: `consolidate`
- **watch surprise scores** — low surprise (< 0.3) means redundant storage
- **use `drift_check`** to find stale memories referencing dead paths
