# cursor

connect Cursor to your existing Engram store over local MCP stdio. Cursor starts
the Python process and uses its tools from chat; no separate HTTP service is
needed. your Cursor model choice is independent of Engram's model settings.

complete [installation](installation.md) and the [quick start](quickstart.md)
first. keep the absolute Python executable and config path printed by `engram init`.

## choose where to register it

use `~/.cursor/mcp.json` for your personal configuration across projects, or
`.cursor/mcp.json` inside one project for project-specific availability.
these locations and the stdio fields are documented in
[Cursor's MCP reference](https://cursor.com/docs/mcp#configuration-locations).

a project entry controls where the tools appear. it does not isolate memories:
clients using the same Engram config connect to the same store. keep personal
paths and credentials out of a shared project configuration.

## add the server

merge this entry into the existing `mcpServers` object, replacing both paths:

```json
{
  "mcpServers": {
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

`command` is the executable, not a whole shell command. keep arguments as
separate array elements, including paths containing spaces. on Windows, use
your environment's `Scripts\\python.exe` and escape backslashes in JSON.

## environment and paths

an editor launched from the desktop may not inherit your terminal's `PATH` or
activated virtual environment. the absolute executable avoids that ambiguity.
use the same explicit `--config` path for Cursor and terminal checks.

if your Engram setup needs environment credentials, add an `env` object inside
the `engram` entry. Cursor supports environment interpolation, for example:

```json
{
  "env": {
    "ENGRAM_POSTGRES_DSN": "${env:ENGRAM_POSTGRES_DSN}"
  }
}
```

this is an optional fragment, not another server. the variable must exist in
Cursor's environment; omit it for a local SQLite setup. see
[config interpolation](https://cursor.com/docs/mcp#config-interpolation).

## check the connection

open **Customize** in Cursor, find `engram`, and enable it. its tools should
appear under **Available Tools**. ask in chat:

> use Engram's `status` tool, then `config_show`, and show which store is connected.

inspect the actual tool result. `config_show` reports effective settings with
secrets redacted; a chat response without a tool call is not a connection check.
these tool-discovery controls are described in
[Cursor's chat integration](https://cursor.com/docs/mcp#using-mcp-in-chat).

## if it does not connect

- restart Cursor after changing the command or updating Engram.
- open **Output → MCP Logs** for process and connection errors.
- check that the selected Python can import Engram and read the config file.
- for missing tools, check that the server and required tools are enabled.

run the local check with the exact executable from the entry:

```sh
/absolute/path/to/venv/bin/python -m engram --config /absolute/path/to/config.yaml doctor --full
```

the [doctor reference](../reference/cli.md#doctor) explains model downloads and
isolated checks. doctor tests the local endpoint; repeat the Cursor tool call
to verify the client connection. if the server works but recall is empty, use
`recall_explain` to inspect retrieval gates.

source checked: [official Cursor MCP documentation](https://cursor.com/docs/mcp),
20 september 2026. configuration reviewed against the docs; no Cursor session
is claimed as tested by this guide.
