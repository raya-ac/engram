# Scan a Codebase

scan a project directory into Engram's codebase layer using its MCP tool.
the scanner stores structural summaries and selected configuration text; output
size depends on the project and is not a fixed compression ratio.

## scan

connect an agent using the [client setup guide](../guides/client-configs.md),
then ask it to call the MCP tool with the project directory:

```text
scan_codebase(path="/absolute/path/to/myapp", project_name="myapp")
```

`engram ingest` is a separate text/document ingestion path; it does not invoke
this structural code scanner.

## what gets extracted

- **file tree** — directory structure with file sizes
- **function signatures** — `def function_name(params) -> return_type`
- **class definitions** — class names, methods, inheritance
- **dependency summary** — counts of extracted external import names
- **config files** — package.json, pyproject.toml, Makefile targets
- **dependency lists** — requirements.txt, package.json deps

stored in the `codebase` layer with project metadata. extracted signatures are
pattern-based summaries, not a complete language-aware representation of the
source. selected config-file content is included, so choose the scan directory
with the intended memory store and model provider in mind.

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

shows project names, memory counts and a breakdown of stored record types.

## tips

- re-scan after major refactors, then inspect stale notes: a scan adds records
  and does not replace or remove every previous project snapshot
- use `drift_check()` regularly to catch stale code references
- combine with `remember_decision()` to capture *why* the code is structured that way
- the normal forgetting cycle processes episodic/working layers, not codebase
  records; explicit edits, invalidation or forgetting can still change them
