# opencode

connect an existing Engram store over local MCP stdio after completing
[installation](installation.md) and the [quick start](quickstart.md).
use the absolute Python and config paths printed by `engram init`.

## choose the matching schema

check `opencode --version`. the official site documents different MCP layouts
for v1 and v2; choose the matching example below. both use a command array
containing the executable followed by its arguments.

put personal settings in `~/.config/opencode/opencode.json`, or use
`opencode.json` / `opencode.jsonc` in a project. project configuration can
override the global entry. see the official
[v1 config locations](https://opencode.ai/docs/config/#locations) and
[v2 config locations](https://opencode.ai/v2/docs/config#locations).
in v2, an override replaces the whole named server, so repeat its required fields.

## OpenCode v2

v2 places server names under `mcp.servers`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "engram": {
        "type": "local",
        "command": [
          "/absolute/path/to/venv/bin/python",
          "-m", "engram",
          "--config", "/absolute/path/to/config.yaml",
          "serve", "--mcp"
        ]
      }
    }
  }
}
```

v2 connects configured servers unless `disabled` is true. keep its default
protocol setting for Engram's standard MCP handshake. these fields are covered
by the [v2 MCP reference](https://opencode.ai/v2/docs/mcp-servers).

## OpenCode v1

v1 places the entry directly under `mcp` and uses `enabled`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "engram": {
      "type": "local",
      "command": [
        "/absolute/path/to/venv/bin/python",
        "-m", "engram",
        "--config", "/absolute/path/to/config.yaml",
        "serve", "--mcp"
      ],
      "enabled": true
    }
  }
}
```

this matches the [v1 local-server reference](https://opencode.ai/docs/mcp-servers/#local).
merge the appropriate entry into your existing file; preserve unrelated settings.

## environment and paths

replace the executable and config paths. no shell activation is needed when
the command points directly to the right Python environment. on Windows, use
the environment's `Scripts\\python.exe` and escape backslashes in JSON.

OpenCode calls its per-server environment object `environment`, not `env`.
when needed, an entry such as `"ENGRAM_POSTGRES_DSN": "{env:ENGRAM_POSTGRES_DSN}"`
belongs inside that object. ensure the variable exists in OpenCode's environment
and keep credential values out of source control.

the agent's chosen model and Engram's models are configured separately. a
project-local MCP entry still shares memory with other connections using the
same Engram config.

## verify and troubleshoot

restart OpenCode after changing the entry, then inspect connection status:

```sh
opencode mcp list
```

v2 also provides `/mcps` to manage connections. ask the agent to call Engram's
`status` and `config_show` tools; inspect the returned store settings. tool
presentation can differ between OpenCode versions, so identify the server as
`engram` in the request.

if connection fails, check the schema version and absolute paths first. run
the same executable with `-m engram --config /absolute/path/to/config.yaml
doctor --full`; the [doctor reference](../reference/cli.md#doctor) explains the
local checks. restart the client after correcting errors. use `recall_explain`
for a connected server whose searches return no results.

Engram's HTTP option is legacy SSE. use the local command above instead of
assuming a Streamable HTTP endpoint for OpenCode's remote-server entry.

sources checked on 20 september 2026: [v1 MCP](https://opencode.ai/docs/mcp-servers/),
[v1 CLI](https://opencode.ai/docs/cli/#mcp), and
[v2 MCP](https://opencode.ai/v2/docs/mcp-servers).
