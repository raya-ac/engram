# gemini cli

Gemini CLI can launch Engram as a local MCP stdio server. complete
[installation](installation.md) and the [quick start](quickstart.md), and keep
the Python executable and config path printed by `engram init`.
Engram's connection and memory tools do not require Claude Code.

## choose the scope

use `~/.gemini/settings.json` for user-wide availability or
`.gemini/settings.json` for the current project. project settings override user
settings; managed system settings can override both. see the official
[configuration reference](https://geminicli.com/docs/reference/configuration/#settings-files).

tool availability is separate from storage: connections using the same Engram
config share its store, even from different projects.

## add the server

merge this entry into the settings file's existing `mcpServers` object:

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

replace both paths. JSON arguments are separate values, so a path containing
spaces remains one string. Windows paths need escaped backslashes.
the [official Python stdio example](https://geminicli.com/docs/tools/mcp-server/#python-mcp-server-stdio)
uses this `command` and `args` structure.

alternatively, register it with the CLI from your chosen project:

```sh
gemini mcp add --scope user --transport stdio engram \
  "/absolute/path/to/venv/bin/python" -- \
  -m engram --config "/absolute/path/to/config.yaml" serve --mcp
```

use `--scope project` for project settings. the `--` separates the Python
arguments from Gemini CLI's options. review the generated entry before starting
the server. the [MCP command reference](https://geminicli.com/docs/tools/mcp-server/#adding-a-server-gemini-mcp-add)
documents these flags.

## environment and startup

the absolute executable selects the environment containing Engram, regardless
of the shell's `PATH`. if a Postgres or hosted-model setup needs credentials,
pass the required variables explicitly through the server's `env` object.
for example, this optional fragment belongs inside the `engram` entry:

```json
{
  "env": {
    "ENGRAM_POSTGRES_DSN": "${ENGRAM_POSTGRES_DSN}"
  }
}
```

Gemini CLI supports environment substitution and sanitizes inherited variables;
explicit mappings make the intended credential available. omit this fragment
for SQLite. keep actual secret values out of project settings.

## verify the connection

start a new Gemini CLI session after editing settings. inspect the server list:

```sh
gemini mcp list
```

inside a session, `/mcp` shows connection state and discovered tools. ask Gemini
to call Engram's `status` and `config_show` tools and inspect the actual result.
these checks are described in the
[MCP status documentation](https://geminicli.com/docs/tools/mcp-server/#using-the-mcp-command).

## troubleshoot

- disconnected: verify executable/config paths and any required environment variables.
- no tools: inspect `/mcp` diagnostics and confirm `engram` is enabled.
- untrusted folder: Gemini may leave stdio servers disconnected; review the
  workspace trust decision instead of bypassing tool confirmations.
- startup or model error: run the same executable with `-m engram --config
  /absolute/path/to/config.yaml doctor --full`, then restart Gemini CLI.

see [doctor](../reference/cli.md#doctor) for model downloads and local endpoint
checks. doctor alone does not prove Gemini has attached. for empty recall after
a successful connection, use `recall_explain` to inspect the confidence gate.

this guide uses stdio. Engram's separate HTTP option is legacy SSE; it is not
the Streamable HTTP transport selected by `gemini mcp add --transport http`.

sources checked on 20 september 2026: [Gemini CLI MCP](https://geminicli.com/docs/tools/mcp-server/)
and [settings](https://geminicli.com/docs/reference/configuration/).
