# Drift Detection

memories reference file paths, function names, commands, and dependencies. those references go stale when the codebase changes. drift detection finds and fixes them.

## how it works

`drift_check` extracts verifiable claims from memory content and validates them against the actual filesystem:

- **file paths** — does `/path/to/file.py` exist?
- **function names** — does `def my_function` appear in the codebase?
- **npm scripts** — does `package.json` have a `build` script?
- **commands** — are referenced CLI commands valid?

zero AI cost — pure filesystem checks.

## running a check

```bash
engram drift                                    # basic check
engram drift --search-roots ~/project/src       # also grep for function names
engram drift --project-root ~/project           # check package.json scripts
engram drift --json                             # output as JSON
```

returns a drift score (0-100) with per-issue breakdown:

```
Drift score: 85/100
  Memories checked: 42
  Claims extracted: 128
  Claims verified: 96 (82 valid)
  Stale memories: 3

  Issues (3):
    [ERROR] dead_path: /src/old_module.py no longer exists
    [WARNING] missing_function: calculate_total not found in codebase
    [INFO] stale_command: npm run legacy-build not in package.json
```

## auto-fix

```bash
engram drift --fix --dry-run    # preview changes
engram drift --fix              # apply fixes
```

auto-fix actions:

- **invalidate** memories with dead file paths
- **flag stale** memories with missing functions/commands
- **forget** memories that are completely invalidated

## MCP tools

- `drift_check(search_roots, project_root, check_functions)` — returns drift report
- `drift_fix(dry_run)` — auto-fix drift issues

## in the dream cycle

drift detection runs automatically as step 6 of the dream cycle (`consolidate`). stale memories get invalidated without manual intervention.

## tips

- run `drift_check` after major refactors
- set `--search-roots` to your source directories for function verification
- use `--dry-run` before `--fix` to preview what will change
- export a backup before bulk fixes: `engram export pre-drift-backup.json`
