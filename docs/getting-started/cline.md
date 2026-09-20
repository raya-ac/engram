# cline

Engram connects to Cline through local MCP stdio. complete
[installation](installation.md) and the [quick start](quickstart.md), then use
the absolute Python executable and config path printed by `engram init`.
the connection does not depend on choosing Claude as Cline's model.

## open the right settings

in the Cline editor panel, open **MCP Servers → Configure → Configure MCP
Servers**. this opens the extension's own MCP settings JSON. use that file,
rather than VS Code's separate `.vscode/mcp.json`; the editor profile and
extension installation determine its actual storage location.

for **Cline CLI**, the documented file is `~/.cline/mcp.json`. the `cline mcp`
wizard can add or edit entries there. extension and CLI setup are described in
[Cline's official MCP guide](https://docs.cline.bot/mcp/mcp-overview#manual-config).

## add the local server

merge this into the file's existing `mcpServers` object:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/absolute/path/to/venv/bin/python",
      "args": [
        "-m", "engram",
        "--config", "/absolute/path/to/config.yaml",
        "serve", "--mcp"
      ],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

replace both paths. `command` names the executable; the array contains its
arguments. the empty `autoApprove` list leaves individual tool approvals in
place. this entry follows Cline's
[local-server schema](https://docs.cline.bot/mcp/mcp-overview#local-server-stdio).

Cline starts the server itself. do not start another copy in a terminal just
to make the connection work, and do not enter the web dashboard's URL as an
MCP endpoint.

## environment and shared stores

the Python path must belong to the environment where Engram is installed.
an editor process may not inherit your shell's virtual-environment activation.
Windows JSON paths need escaped backslashes, such as
`C:\\Users\\you\\engram-env\\Scripts\\python.exe`.

if you use hosted embeddings or Postgres, make the required environment
variables available to the Cline process or configure the server's `env`
mapping. keep secrets out of checked-in settings. use the same explicit Engram
config when testing from a terminal.

all clients pointed at that config use its configured store. if two projects
need separate memories, give their connections separate Engram configs and
databases; changing the chat model does not switch the store.

## check it from Cline

save the configuration, then enable or restart `engram` in MCP settings.
confirm its tools appear and ask:

> call Engram's `status`, then `config_show`, and show the connected store.

look for the actual tool response. Cline CLI can also show configured servers:

```sh
cline config mcp --json
```

this lists configuration; follow it with a tool call to verify the running
connection. the official guide describes
[server management and CLI inspection](https://docs.cline.bot/mcp/mcp-overview#managing-servers).

## troubleshoot

for connection errors, check Cline's MCP server error details and the absolute
executable/config paths. for timeouts, run the local diagnostic first:

```sh
/absolute/path/to/venv/bin/python -m engram --config /absolute/path/to/config.yaml doctor --full
```

see [doctor's checks and model behavior](../reference/cli.md#doctor). restart
the Cline server after fixing an error or upgrading Engram. when tools connect
but recall is empty, `recall_explain` shows candidate scores and rejection reasons.

this guide uses stdio. Engram's optional remote transport is legacy SSE,
not a generic Streamable HTTP endpoint.

source checked: [official Cline MCP documentation](https://docs.cline.bot/mcp/mcp-overview),
20 september 2026. no editor-global storage path or client session is assumed.
