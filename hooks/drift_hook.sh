#!/bin/bash
# Post-commit hook for engram drift checking.
# Runs drift_check after every commit. Silent on perfect score.
#
# Install:
#   cp hooks/drift_hook.sh .git/hooks/post-commit
#   chmod +x .git/hooks/post-commit
#
# Or use engram's hook installer:
#   engram watch --install
#
# Requires ENGRAM_VENV to point to the engram virtualenv:
#   export ENGRAM_VENV=~/Ash/engram/.venv

ENGRAM_VENV="${ENGRAM_VENV:-$HOME/Ash/engram/.venv}"
PYTHON="$ENGRAM_VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
    exit 0  # engram not installed, skip silently
fi

# Run drift check in background so it doesn't block the commit
(
    RESULT=$("$PYTHON" -c "
from engram.config import Config
from engram.store import Store
from engram.drift import run_drift_check
import json

config = Config.load()
store = Store(config)
store.init_db()
report = run_drift_check(store, check_functions=False)
errors = sum(1 for i in report.issues if i.severity == 'error')
warnings = sum(1 for i in report.issues if i.severity == 'warning')
print(json.dumps({'score': report.score, 'errors': errors, 'warnings': warnings}))
store.close()
" 2>/dev/null)

    if [ -z "$RESULT" ]; then
        exit 0
    fi

    SCORE=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['score'])" 2>/dev/null)
    ERRORS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['errors'])" 2>/dev/null)
    WARNINGS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['warnings'])" 2>/dev/null)

    # Silent on perfect score
    if [ "$SCORE" = "100" ]; then
        exit 0
    fi

    # Show drift score in terminal
    if [ "$SCORE" -lt 50 ]; then
        COLOR="\033[31m"  # red
    elif [ "$SCORE" -lt 80 ]; then
        COLOR="\033[33m"  # yellow
    else
        COLOR="\033[32m"  # green
    fi
    RESET="\033[0m"

    echo ""
    echo -e "${COLOR}engram: drift score ${SCORE}/100${RESET} (${ERRORS} errors, ${WARNINGS} warnings)"
    if [ "$ERRORS" -gt 0 ]; then
        echo "  run 'engram drift' for details, 'engram drift --fix' to auto-repair"
    fi
) &
