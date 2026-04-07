#!/bin/bash
# Engram Save Hook for Claude Code
# Runs every N messages to extract and store memories from the current session.
# Add to .claude/settings.json hooks.postToolUse or run manually.
#
# Usage: ENGRAM_VENV=~/Ash/engram/.venv ./save_hook.sh <session_jsonl>

VENV="${ENGRAM_VENV:-$HOME/Ash/engram/.venv}"
PYTHON="$VENV/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "engram: python not found at $PYTHON" >&2
    exit 0  # don't block Claude Code
fi

# extract from the most recent Claude Code session
"$PYTHON" -c "
from engram.config import Config
from engram.store import Store
from engram.conversations import ingest_all_sessions

config = Config.load()
store = Store(config)
store.init_db()
stats = ingest_all_sessions(store, limit=5)
if stats['memories_created'] > 0:
    import sys
    print(f'engram: extracted {stats[\"memories_created\"]} memories from {stats[\"sessions\"]} sessions', file=sys.stderr)
store.close()
" 2>&1 | head -5
