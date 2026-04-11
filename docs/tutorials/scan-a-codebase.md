# Scan a Codebase

index a project directory into engram's codebase layer. extracts file trees, function/class signatures, import graphs, and config files — ~10x fewer tokens than raw code.

## scan

```bash
engram ingest ~/projects/myapp
```

or via MCP:

```
scan_codebase(path="~/projects/myapp", project_name="myapp")
```

## what gets extracted

- **file tree** — directory structure with file sizes
- **function signatures** — `def function_name(params) -> return_type`
- **class definitions** — class names, methods, inheritance
- **import graphs** — what imports what
- **config files** — package.json, pyproject.toml, Makefile targets
- **dependency lists** — requirements.txt, package.json deps

stored in the `codebase` layer with compressed content (~10x fewer tokens than raw source).

## search

```bash
engram search "authentication middleware" --debug
```

or via MCP:

```
recall_code(query="auth middleware", project="myapp")
```

the codebase layer participates in the normal hybrid search pipeline — it shows up alongside episodic and semantic memories when relevant.

## drift detection

memories about code go stale when you refactor. drift detection catches this:

```bash
engram drift --search-roots ~/projects/myapp/src --project-root ~/projects/myapp
```

this checks:
- do referenced file paths still exist?
- do mentioned function names appear in the codebase?
- are package.json scripts still valid?

auto-fix stale references:

```bash
engram drift --fix --dry-run    # preview
engram drift --fix              # apply
```

## list scanned projects

```bash
# via MCP
list_projects()
```

shows all scanned projects with file counts and memory counts.

## tips

- re-scan after major refactors to update the codebase layer
- use `drift_check()` regularly to catch stale code references
- combine with `remember_decision()` to capture *why* the code is structured that way
- the codebase layer is exempt from forgetting — it persists permanently
