# Other clients and custom agents

any client that can launch a local stdio MCP server can connect to Engram.
the required process is the same; use your client's own configuration schema.

## local stdio MCP

first complete [installation](installation.md) and create or choose an Engram
config. enter these fields in the client's MCP server setup:

| Field | Value |
|---|---|
| Name | `engram` |
| Transport | stdio / local process |
| Executable | the absolute Python path in your Engram environment |
| Arguments | `-m`, `engram`, `--config`, your absolute config path, `serve`, `--mcp` |

arguments are separate array items, not one long quoted shell command. if the
client asks for a full command instead, quote paths containing spaces according
to its shell. on Windows choose `Scripts/python.exe`; in a container or remote
host use paths that exist inside that environment.

```sh
/absolute/path/to/engram-venv/bin/python -m engram \
  --config /absolute/path/to/engram.yaml serve --mcp
```

do not paste a `mcpServers` JSON object into a client that expects another
schema. [the client hub](../guides/client-configs.md) links individual recipes
and official configuration references.

the client starts and stops its subprocess. stdout is reserved for MCP traffic;
startup diagnostics belong on stderr. avoid shell wrappers that print banners
or environment-loader messages to stdout before starting the server.

## verify discovery and calls

after reconnecting, inspect the client's MCP status and discovered tools. ask
it to call `config_show` and verify the effective config path, then call
`recall_recent` with `limit: 5`. a successful process launch alone does not show
that the client completed the MCP handshake or can call tools.

for your own MCP host, implement the standard initialization and tool discovery
flow, then call the names and argument schemas returned by `tools/list`. keep
the process alive between calls so models can remain loaded. see the
[MCP tool reference](../reference/mcp-tools.md) for Engram's operations.

## HTTP and remote clients

Engram also exposes a **legacy SSE** MCP server:

```sh
engram --config /absolute/path/to/engram.yaml serve --mcp-sse --port 8421
```

its event endpoint is `http://127.0.0.1:8421/sse`. this requires a client that
explicitly supports the older SSE transport. it is not a Streamable HTTP `/mcp`
endpoint; changing the URL does not convert the protocol.

the built-in SSE server binds to loopback and does not provide a hosted OAuth
connector. a cloud-only client cannot launch your local Python process or reach
your machine's localhost. remote hosting needs a separately designed transport
and authentication boundary; the stdio instructions do not create one.

## custom integrations without MCP

| Interface | Use it for | Contract |
|---|---|---|
| Native JSONL service | long-lived subprocess with explicit project context and checkpoints | [Native API](../native-api.md) |
| Python | direct embedding/storage/retrieval integration | [Build an agent](../tutorials/build-an-agent.md) |
| CLI | shell scripts and deliberate one-shot operations | [CLI reference](../reference/cli.md) |
| Web REST API | authenticated workspace operations over HTTP | [REST reference](../reference/rest-api.md) |

the native JSONL service is not MCP. use its discovery operation and documented
operation names; it does not accept arbitrary MCP tool names or offer a native
`remember` operation. choose the interface for the scope you need.

## credentials and continuity

inherited `ENGRAM_*` overrides take precedence over the config file. provide
required provider or Postgres credentials through your client's supported
environment/secret mechanism. keep private config and secrets out of shared
project settings.

give the agent a deliberate [memory workflow](../guides/session-continuity.md).
retrieved text supplies context, not permission to run commands. registration
does not enable transcript capture or guarantee that the agent uses memory on
every turn.

for PATH errors, model downloads, mismatched stores and connection timeouts,
see [troubleshooting](troubleshooting.md).
