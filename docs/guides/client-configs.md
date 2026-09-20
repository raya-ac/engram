# Connect your agent

Engram runs beside your agent as a local MCP server. the client starts it, calls
its memory tools and receives results. choose your client below; their config
formats differ even when they launch the same Engram command.

## choose your client

| Client | Guide | Connection |
|---|---|---|
| Codex CLI, IDE and desktop | [Codex](../getting-started/codex.md) | stdio; TOML or CLI registration |
| Claude Code | [Claude Code](../getting-started/claude-code.md) | stdio; CLI with user/project/local scope |
| Claude Desktop | [Claude Desktop](../getting-started/claude-desktop.md) | stdio; desktop JSON settings |
| Cursor | [Cursor](../getting-started/cursor.md) | stdio; user or project MCP config |
| VS Code / GitHub Copilot | [VS Code](../getting-started/vscode.md) | stdio; `servers` config |
| Windsurf / legacy Cascade | [Windsurf](../getting-started/windsurf.md) | stdio; Cascade MCP config |
| Cline | [Cline](../getting-started/cline.md) | stdio; MCP settings |
| OpenCode | [OpenCode](../getting-started/opencode.md) | local MCP; version-specific schema |
| Gemini CLI | [Gemini CLI](../getting-started/gemini-cli.md) | stdio; CLI or settings JSON |
| Other clients and custom agents | [Other clients](../getting-started/other-clients.md) | stdio MCP, native API or Python |

start with [installation](../getting-started/installation.md) and the
[quick start](../getting-started/quickstart.md) if you have not created a store.
these recipes describe configuration, not a certification that every client
version has been tested end to end. each guide links its client's official docs.

## the shared command

```sh
/absolute/path/to/engram-venv/bin/python -m engram --config /absolute/path/to/engram.yaml serve --mcp
```

use the Python environment where Engram is installed. `--config` comes before
`serve`. use absolute paths so a different working directory does not select a
different config. the client manages this process; you do not also need to
leave a second server running in a terminal.

the server communicates MCP over stdin/stdout. a quiet process waiting for
input is normal. the web workspace and native JSONL API are separate interfaces.

## one store across clients

point clients at the same explicit Engram config to share memories. check
`config_show` in each client: inherited `ENGRAM_*` variables can override that
file, including the database or backend. GUI apps, WSL and remote IDE hosts
may have different environments and filesystems.

client scope decides where the client loads a server. it does not restrict
Engram's full memory tools to that project. if you need explicit project-bound
context and checkpoints, see the [Codex adapter](../codex-adapter.md) or
[native API](../native-api.md). do not treat a project label as an access boundary.

## verify the connection

1. Run `config check` and `doctor --full` with the chosen interpreter and config.
2. Add the client configuration, then restart or reconnect its MCP server.
3. Ask the agent to call `config_show`, then `recall_recent` with `limit: 5`.
4. Confirm the config path and actual tool result. an empty new store is valid.
5. Deliberately save a small test fact with `remember`, then recall it from
   another session using the same store.

doctor checks Engram's endpoint with an isolated client. it cannot confirm that
your editor attached successfully; step 3 checks that connection.

## give the agent a memory routine

connection and memory use are separate steps. put a short workflow in your
client's supported instruction file: recent context first, a specific task
query next, useful decisions saved during work, then a handoff before stopping.
the [session continuity guide](session-continuity.md) explains the tools and
gives a reusable routine.

`recall_recent` returns chronological context. `recall` searches by relevance.
`recall_hints` provides short cues before requesting full memories. for a
read-only explanation of search ranking and rejected candidates, use
`recall_explain`. the installed server's `tools/list` is the current tool catalog.

manual save/search works without a working LLM, but save paths may attempt
optional enrichment through the configured backend. importing documents and
conversation extraction also use LLM processing. read the
[configuration reference](../reference/config.md) before ingesting private text.

if the agent sees no tools, starts the wrong store or times out, continue with
[troubleshooting](../getting-started/troubleshooting.md).
