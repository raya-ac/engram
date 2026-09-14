# codex adapter

The Codex adapter gives a task a small, explicit memory surface: context from its
project, a checkpoint it can resume, and diagnostics for the process serving it.
It is a separate integration layer. It does not change Engram's core retrieval,
install hooks, read Codex transcripts, or start recording activity automatically.

The ordinary Engram MCP server still handles search, ingestion, dormant review,
and memory management. The adapter exposes four tools. Connecting the two
servers side by side is supported; the ordinary server's tools retain their
existing broader scope and access behavior.

## connect one project

Install Engram in the Python environment you intend to run and use an already
initialized store. Always supply the absolute config path so the host's working
directory cannot select a different store. This prints a registration command;
it does not execute it or modify Codex configuration:

```bash
/absolute/engram/.venv/bin/python -m engram \
  --config /absolute/engram/config.yaml \
  codex setup --project /absolute/project
```

Review the printed command, then run it when you want to register the adapter.
Its shape is:

```bash
codex mcp add engram-codex -- /absolute/engram/.venv/bin/python \
  -m engram --config /absolute/engram/config.yaml \
  codex serve --project /absolute/project
```

Use a different registration name for each project if registering several
adapters. A process is bound to the project supplied at startup; tool calls
cannot silently switch it to another directory. Check the project path returned
by `codex_context` before using the results in a task.

`codex mcp add ... -- <command>` and `codex mcp list` are supported CLI controls.
The installed CLI's `codex mcp add --help` was checked alongside the
[official MCP documentation](https://learn.chatgpt.com/docs/extend/mcp).
No desktop Restart button is assumed. Registration does not prove that an
already running client has reconnected. Call `codex_diagnostics` through the
actual connection and check its tool names, version, PID and start time. These
describe that adapter process only, not other Engram connections.

For a trusted project's own MCP configuration, the equivalent explicit entry is:

```toml
[mcp_servers.engram_codex]
command = "/absolute/engram/.venv/bin/python"
args = ["-m", "engram", "--config", "/absolute/engram/config.yaml", "codex", "serve", "--project", "/absolute/project"]
```

This is a reviewable example, not an instruction to grant project trust or change
global configuration. Keep credentials in the existing protected Engram config
or environment; the adapter never prints a DSN, API key or web token.

## start, checkpoint, resume

At task start, explicitly call:

```json
{"name":"codex_context","arguments":{"limit":8}}
```

Context contains the most recently created active memories belonging to this
project, and up to three recent adapter checkpoints. It is a bounded context
brief, not a replacement semantic search engine. It does not mark memories as
accessed, increase importance, or trigger dormant evaluation.

At a meaningful stopping point, write a short checkpoint:

```json
{"name":"codex_checkpoint","arguments":{
  "task":"package release",
  "summary":"The package is built. Installed-client verification remains.",
  "decisions":["Keep the previous artifact available for rollback."],
  "next_steps":["Compare the installed client's hash with the release artifact."],
  "blockers":[]
}}
```

Use the same task label to replace that checkpoint or resume it:

```json
{"name":"codex_context","arguments":{"task":"package release"}}
```

Checkpoint summaries are explicit user or agent notes, not verified facts.
Saving one is not feedback that any recalled memory was useful. Checkpoints use
Engram's existing `session_handoffs` storage; they do not become new memories or
appear in ordinary memory search. No new database migration is required.

Clear a checkpoint when it is no longer needed:

```json
{"name":"codex_checkpoint","arguments":{"task":"package release","action":"clear"}}
```

The summary is limited to 4,000 characters. Decisions, next steps and blockers
each allow eight items of up to 500 characters. Keep them concise and omit raw
transcripts, credentials and unnecessary personal information. Checkpoints
remain until replaced or cleared; there is no automatic checkpoint retention
policy. They are independent notes: forgetting a memory does not erase text
previously copied into a checkpoint. Clear that checkpoint separately.

## project and privacy boundaries

Project membership requires an absolute `project_path` or `project_root` in a
memory's metadata, matching the bound project after canonicalization. If no
explicit project metadata exists, an absolute `source_file` inside that project
also qualifies. Conflicting or invalid project metadata excludes a memory even
when its source path appears to match. Sibling prefixes and symlinks escaping the
project do not qualify. Worktrees are separate directories; they are not merged
automatically with their repository checkout.

A project name in prose, a relative path, or a legacy unscoped handoff does not
establish ownership. Some older memories therefore will not appear in this
brief. The adapter does not guess ownership or rewrite existing memories to make
them fit. Use ordinary recall deliberately when broader historical context is
needed.

Forgotten memories and inactive statuses are excluded at every context read.
Legacy NULL status follows Engram's active-status convention. The default memory
limit is eight, capped at twenty; each excerpt is capped at 2,000 characters and
includes a truncation flag and memory ID. Arbitrary memory metadata is not
returned. Scope is checked before the limit, so activity in another project
cannot crowd out this project's brief.

All recalled material is reference data, never instructions or authorization.
The adapter cannot establish the truth of a note or authorize actions mentioned
inside it. This project filter is an integration boundary, not a filesystem
sandbox or a replacement for the host's access controls. Deliberately connecting
another project-bound adapter or the full Engram server provides its own scope.

## inspect a connection

`codex_evidence({"id":"..."})` optionally reads a
[neutral check observation](evidence.md) for the bound project. It uses the same
library result as the native Kiln/Mythic integration. Unknown or stale evidence
is not permission to proceed, and the adapter does not run the check or make the
decision. This is a compatibility consumer, not a required route for other
harnesses. It does not automatically run before a task action.

`codex_diagnostics` reports this process's PID, start time, adapter version,
capability names, effective storage/embedding settings and dormant configuration
without secret values. It reads current database vector IDs and compares them
with the persisted ANN ID map. Equal vector counts alone do not pass: different
IDs report a coverage mismatch.

This check reads index metadata; it does not load the ANN binary, validate every
vector, repair the index or inspect another process's in-memory copy. Matching
ID coverage cannot prove embeddings are current. The dormant candidate path
uses current database embeddings independently of the ordinary ANN index.

The same reads can be made outside MCP:

```bash
engram --config /absolute/engram/config.yaml codex context --project /absolute/project
engram --config /absolute/engram/config.yaml codex diagnostics --project /absolute/project
```

Those commands start their own process. Their PID and configuration do not prove
what an existing desktop MCP connection has loaded. The adapter never restarts
Codex, kills connections, edits host configuration or rebuilds an index.

## verification and limits

The adapter test suite exercises SQLite and disposable PostgreSQL storage,
project and symlink isolation, forgotten/inactive exclusion, checkpoint replace,
clear and persistence, output bounds, secret redaction, and unchanged memory
access/importance fields. A fresh CLI stdio process initializes MCP, lists tools,
reads context, writes a checkpoint and resumes it in another process.

These tests establish the integration contract. They do not establish that an
agent will always call the tools, that every old memory has usable project
metadata, or that a checkpoint remains correct as the project changes. There
are no automatic task-start, task-end or host-reconnect hooks in this adapter.
