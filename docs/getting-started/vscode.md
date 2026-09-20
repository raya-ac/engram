# visual studio code

this guide connects Engram to VS Code's built-in chat MCP integration. Cline and
other agent extensions have their own connection settings; see the
[Cline guide](cline.md) for that extension.

start with [installation](installation.md) and the [quick start](quickstart.md).
use the absolute Python and Engram config paths printed by `engram init`.
Engram runs as a local stdio process, independently of your chat model provider.

## choose the configuration scope

for your personal setup, run **MCP: Open User Configuration** from the Command
Palette. VS Code opens the MCP file for your current user profile.
for one workspace, create `.vscode/mcp.json` in that workspace.
[VS Code documents both scopes](https://code.visualstudio.com/docs/agent-customization/mcp-servers#configure-the-mcpjson-file).

project-scoped tool registration does not partition the Engram database. use
different Engram config/store paths when projects need separate memories.

## add the stdio entry

VS Code uses a top-level `servers` object. merge this entry into that object:

```json
{
  "servers": {
    "engram": {
      "type": "stdio",
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "-m", "engram",
        "--config", "/absolute/path/to/config.yaml",
        "serve", "--mcp"
      ]
    }
  }
}
```

replace both paths. `command` must identify the Python environment containing
Engram; each argument stays separate. Windows paths need escaped backslashes
in JSON, for example `C:\\Users\\you\\engram-env\\Scripts\\python.exe`.
the field definitions are in the
[MCP configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration#standard-io-stdio-servers).

## local, remote and environment settings

user-profile servers run locally. for a server on an SSH host or inside a
remote workspace, use the workspace or **MCP: Open Remote User Configuration**
location and paths that exist there. the Python installation, config, database
and model cache must be accessible in that execution environment.

the absolute executable avoids relying on a terminal's activated environment.
if Engram needs provider credentials or a Postgres DSN, use the server's `env`
or `envFile` support. keep credential files outside source control. VS Code's
reference also covers [secret inputs and remote configuration](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).

VS Code's newer Agent Host sessions receive supported MCP configurations from
VS Code; interactive-input configurations have additional restrictions. consult
the official guide when moving this setup between execution environments.

## start it and check a tool

run **MCP: List Servers**, select `engram`, then start it. review any server trust
prompt against the executable and config you chose. in Chat, use **Configure
Tools** to make Engram's tools available, then ask:

> call Engram's `status` and `config_show` tools and show the connected store.

check the tool-call output, including the config path and storage settings.
the [server management guide](https://code.visualstudio.com/docs/agent-customization/mcp-servers#manage-mcp-servers)
covers starting, stopping and inspecting servers.

## troubleshoot or restart

use **MCP: List Servers → engram → Show Output** to inspect startup errors.
restart the server from the same menu after changing paths or updating Engram.
if an upgrade changes the tool list, **MCP: Reset Cached Tools** refreshes discovery.

check Engram independently with the same executable:

```sh
/absolute/path/to/venv/bin/python -m engram --config /absolute/path/to/config.yaml doctor --full
```

see [doctor's behavior](../reference/cli.md#doctor) before the full check; it may
download configured models. a successful doctor result verifies the local
endpoint, while the tool call above verifies VS Code's connection. use
`recall_explain` when connection succeeds but a search returns nothing.

this setup uses stdio. Engram's optional HTTP transport is legacy SSE; do not
replace this entry with an assumed Streamable HTTP `/mcp` endpoint.

sources checked on 20 september 2026: [server setup](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
and [configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).
