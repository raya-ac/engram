# hooks: auto-extract memories from conversations

engram can automatically extract memories from your claude code sessions after each conversation. you don't have to manually `remember` things — the system picks up decisions, corrections, and facts on its own.

## claude code hooks

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

this fires the save hook whenever a conversation ends (the Stop tool is called). the hook:

1. finds recent claude code JSONL session files
2. skips files it's already ingested (tracked by content hash)
3. parses exchanges into Q+A pairs
4. classifies them (decisions, corrections, errors, task completions)
5. stores them in the right layer with appropriate importance
6. runs surprise scoring to avoid duplicates

## alternative: directory watcher

instead of hooks, you can use the `watch` command to continuously monitor for new session files:

```bash
engram watch ~/.claude/projects/ --interval 60
```

this polls every 60s and auto-ingests any new or changed files. simpler setup but slightly delayed.

## manual ingestion

you can also ingest sessions manually:

```bash
# from the CLI
engram ingest ~/.claude/projects/*/sessions/*.jsonl

# from the MCP server (in claude code)
# just say: "ingest my recent sessions"
# it will call: ingest_sessions(limit=20)
```

## what gets extracted

the conversation parser classifies exchanges into categories:

- **decisions**: "we decided to...", "let's go with...", "the approach is..."
  → stored in procedural layer with high importance

- **corrections**: "no, that's wrong", "actually...", "don't do that"
  → stored as error patterns in procedural layer

- **facts about the user**: preferences, roles, responsibilities
  → stored in semantic layer

- **project state**: what was built, deployed, changed
  → stored in episodic layer with date tags

- **Q+A pairs**: general exchanges
  → stored in episodic layer with lower importance

every stored memory goes through surprise scoring — if it's too similar to something already stored, importance gets reduced and it's flagged as a potential duplicate. no manual dedup needed.

## supported conversation formats

engram auto-detects the format from file structure:

| format | extension | how to ingest |
|--------|-----------|--------------|
| claude code | `.jsonl` | `engram ingest session.jsonl` |
| claude.ai | `.json` | `engram ingest conversation.json` |
| chatgpt | `.json` | `engram ingest chatgpt_export.json` |
| slack | `.json` | `engram ingest slack_channel.json` |
| markdown | `.md` | `engram ingest notes.md` |
| plaintext | `.txt` | `engram ingest log.txt` |
| PDF | `.pdf` | `engram ingest document.pdf` |

## tips

- **run `engram status`** after the first hook fires to verify memories were created
- **check surprise scores** in the web dashboard (Remember tab) to see if auto-ingestion is storing redundant content
- **set up drift detection** to catch when auto-ingested memories reference things that later change: `engram drift --fix --dry-run`
- **export before bulk ingestion** in case you want to roll back: `engram export pre-ingest-backup.json`
