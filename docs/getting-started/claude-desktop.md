# Claude Desktop

connect the desktop chat application to a local Engram stdio process. this is
the manual MCP configuration route. Claude's remote custom connectors are a
different mechanism; a local Desktop entry is not a connector for claude.ai or
Cowork. see [Claude's connector documentation](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp).

## 1. prepare your store

follow [installation](installation.md) and [quick start](quickstart.md). use
`engram init` for a new store; keep the existing config when connecting a store
you already use. init prints an MCP snippet but does not change Desktop settings.

from the Python environment where Engram is installed, find its interpreter:

```sh
python -c "import sys; print(sys.executable)"
python -m engram --config /absolute/path/to/engram.yaml config check
python -m engram --config /absolute/path/to/engram.yaml doctor --full
```

doctor checks models, an isolated MCP endpoint and synthetic save/retrieve.
it can download local weights or call a configured model provider. it does not
register Engram with Desktop or verify Desktop attachment.

## 2. add the local server

open **Settings → Developer → Edit Config**, or open the configuration file
directly:

| system | Desktop configuration |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

these locations and the manual configuration flow are documented in the
[official MCP local-server guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

merge an `engram` entry into `mcpServers`, preserving other entries. on macOS:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/absolute/path/to/engram-venv/bin/python",
      "args": [
        "-m", "engram",
        "--config", "/absolute/path/to/engram.yaml",
        "serve", "--mcp"
      ]
    }
  }
}
```

on Windows, use the installed environment's `python.exe` and escape backslashes
in JSON:

```json
{
  "mcpServers": {
    "engram": {
      "command": "C:\\Users\\you\\engram-venv\\Scripts\\python.exe",
      "args": [
        "-m", "engram",
        "--config", "C:\\Users\\you\\engram\\config.yaml",
        "serve", "--mcp"
      ]
    }
  }
}
```

replace every placeholder with a real absolute path. Engram's `--config` must
precede `serve`. Desktop starts the server itself; you do not need to leave a
separate terminal server running.

## 3. restart and verify a tool call

fully quit Desktop, then reopen it. open the chat's **+ → Connectors** menu to
find Engram and inspect its tools. Developer settings also show connection
status and logs. see [Claude's local MCP guide](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop).

ask in a new chat:

```text
Use Engram's config_show and report the version and config path, with credentials
redacted. Then call recall_recent with limit 5 and summarize what it returned.
```

check for actual tool results and the intended config path. a new store can
return no recent memories. the discovered tools reflect your installed Engram
version; there is no fixed count to expect.

use `recall_recent` for chronology, `recall_hints` for a specific topic, and
`recall` for a semantic question. `resume_context` and `session_handoff` support
explicit handoffs. memory writes can invoke the configured LLM for enrichment;
review those settings before storing private material. the
[Claude Code guide's workflow](claude-code.md#4-give-claude-a-memory-workflow) can
be reused as a chat instruction. keep using the Desktop registration above.

## environment and troubleshooting

Desktop's launch environment can differ from your shell. absolute interpreter
and config paths avoid depending on shell activation or `PATH`. if the config
requires `ENGRAM_POSTGRES_DSN` or provider keys, make them available through the
server entry's `env` object or an appropriate private Engram config field. do not
commit credentials. see the [official host setup notes](https://py.sdk.modelcontextprotocol.io/get-started/real-host/)
and [Engram configuration](../reference/config.md).

- **Server missing or disconnected:** check JSON syntax, both absolute paths,
  and that Desktop was fully restarted.
- **Wrong effective settings:** ask for `config_show`; its sources identify
  environment overrides. compare them with the terminal's `config show`.
- **Models unavailable:** run doctor from the exact interpreter in `command`.
  Desktop may need time for the first local model download or warmup.
- **Need connection details:** inspect `mcp.log` and `mcp-server-engram.log` in
  `~/Library/Logs/Claude` on macOS or `%APPDATA%\Claude\logs` on Windows.
  those log locations are listed in the
  [MCP troubleshooting guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers#troubleshooting).

organization policies may restrict local servers; the
[Claude local MCP guide](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)
describes those controls. this setup does not install session-capture hooks or
import conversations automatically.
