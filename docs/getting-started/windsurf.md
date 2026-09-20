# windsurf / cascade

connect Engram to **Cascade** using its local MCP configuration. begin with
[installation](installation.md) and the [quick start](quickstart.md), then keep
the absolute Python and config paths printed by `engram init`.

the official Windsurf documentation now redirects to Devin Desktop. it
explicitly identifies this MCP configuration as **Cascade-only**; the newer
Devin Local agent uses separate settings. check which agent your tab runs before
editing the file below. see the
[official Cascade scope note](https://docs.devin.ai/desktop/cascade/mcp).

## open Cascade's MCP configuration

open the **MCPs** panel in Cascade, or the editor's **Settings → Cascade → MCP
Servers** section. edit the raw MCP configuration at:

```text
~/.codeium/windsurf/mcp_config.json
```

this is the user's Cascade server configuration. using the same Engram config
across projects shares the same memory store; opening another workspace does
not create a separate Engram database.

## add Engram

merge this entry into `mcpServers`, replacing the executable and config paths:

```json
{
  "mcpServers": {
    "engram": {
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

the editor starts this process over stdio. it does not need an HTTP URL or a
separately launched server. the
[official configuration example](https://docs.devin.ai/desktop/cascade/mcp)
uses the same `command`, `args` and optional `env` structure.

## environment and paths

use the actual Python executable from your Engram environment. a desktop
application may have a different `PATH` from your terminal. on Windows, use
the environment's `Scripts\\python.exe` and escape JSON backslashes.

if your setup needs credentials from the editor's environment, add this optional
fragment inside the `engram` server entry:

```json
{
  "env": {
    "ENGRAM_POSTGRES_DSN": "${env:ENGRAM_POSTGRES_DSN}"
  }
}
```

omit it for SQLite. the variable must exist in the editor process's environment;
do not store its value in a shared project file. Cascade's documented
[interpolation syntax](https://docs.devin.ai/desktop/cascade/mcp#config-interpolation)
uses `${env:NAME}`.

## verify tools and connection

save the file, restart the editor, and open the `engram` entry in the MCPs panel.
check which tools are enabled. ask Cascade:

> use Engram's `status` and `config_show` tools and show the connected memory store.

inspect a real tool result before assuming the connection works. Engram's
memory tools do not require Claude Code; file ingestion uses Engram's separately
configured [LLM backend](../reference/config.md).

## if something is missing

- no server entry: check the JSON file and that you are using Cascade.
- server starts but tools are missing: inspect the entry's enabled tools.
- command failure: check the executable and both absolute paths.
- managed team setup: check whether MCP access or the server is restricted by policy.

run the exact executable with `-m engram --config /absolute/path/to/config.yaml
doctor --full` in a terminal. see [doctor](../reference/cli.md#doctor) for its
model and local-endpoint checks. after correcting an error, restart the editor
and repeat the tool call. `recall_explain` distinguishes retrieval rejection
from a connection failure.

source checked on 20 september 2026: [official Cascade MCP guide](https://docs.devin.ai/desktop/cascade/mcp),
reached through [Windsurf's documentation URL](https://docs.windsurf.com/windsurf/cascade/mcp).
