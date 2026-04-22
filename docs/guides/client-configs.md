# Client Configs

Copy-pasteable setup snippets for common Engram clients.

## Claude Code

```json
{
  "mcpServers": {
    "engram": {
      "command": "/absolute/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram", "serve", "--mcp"]
    }
  }
}
```

Good first tools to teach the agent:

- `resume_context`
- `recall`
- `remember`
- `remember_decision`
- `remember_negative`
- `session_handoff`

## Codex

Add Engram as an MCP server in your Codex config:

```toml
[mcp_servers.engram]
command = "/absolute/path/to/engram/.venv/bin/python"
args = ["-m", "engram", "serve", "--mcp"]
```

For continuity-focused work, pair it with:

- [`examples/skills/session-continuity/SKILL.md`](https://github.com/raya-ac/engram/blob/main/examples/skills/session-continuity/SKILL.md)

## Generic MCP client

If your client supports stdio MCP servers:

```bash
/absolute/path/to/engram/.venv/bin/python -m engram serve --mcp
```

If your client supports HTTP/SSE instead:

```bash
engram serve --mcp-sse --port 8421
```

## Web dashboard

Run the dashboard:

```bash
engram serve --web
```

Default address:

- `http://127.0.0.1:8420`

## Continuity-first tool set

If you want the smallest high-value default surface for a coding agent, start with:

- `resume_context`
- `recall`
- `recall_hints`
- `remember`
- `remember_decision`
- `remember_negative`
- `diary_write`
- `session_handoff`
