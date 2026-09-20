# Build an agent with persistent context

give your agent a small memory boundary: load the context it needs, keep that
context separate from instructions, and save an explicit handoff when the user
chooses to stop. the same local interface works with a desktop app, command-line
assistant or background task runner, regardless of its model provider.

this tutorial uses Engram's native JSONL API and a stdlib Python client. it needs
no agent framework, provider key or LLM to run the memory workflow.

## choose the interface

| need | interface |
| --- | --- |
| a client that already supports MCP and wants semantic remember/recall tools | [full MCP server](../reference/mcp-tools.md) |
| project context and explicit task checkpoints in your own application | [native JSONL API](../native-api.md), used below |
| store-wide semantic retrieval or a retrieval explanation | native `search` / `search_explain`, with configured models |
| an HTTP application | [REST API](../reference/rest-api.md), with its own authentication and endpoint contract |

these are different contracts. native `recall` reads project-owned recent
context; MCP `recall` performs semantic retrieval. the native API has no
`remember` operation.

## prepare Engram and the client

follow [installation](../getting-started/installation.md) and [quickstart](../getting-started/quickstart.md).
keep the absolute config path and the Python interpreter where Engram is
installed. for an existing store, reuse its config rather than running init over
it.

copy [native_client.py](https://github.com/raya-ac/engram/blob/main/examples/integrations/native_client.py)
beside your application script. it starts:

```sh
/absolute/venv/bin/python -m engram --config /absolute/engram.yaml api
```

the client does not discover or create a store for you. it preserves inherited
environment overrides, so inspect the effective config if your application
launch environment differs from your terminal.

## load context for one task

save this as `agent_context.py` beside `native_client.py`, replacing the three
paths:

```python
import json
from pathlib import Path

from native_client import NativeClient

project = str(Path("/absolute/projects/lantern").resolve())
task = "issue:APP-42"

with NativeClient(
    config="/absolute/engram.yaml",
    python="/absolute/venv/bin/python",
    timeout=30,
) as memory:
    capabilities = memory.call("operations")
    names = {item["name"] for item in capabilities["operations"]}
    if "session_resume" not in names:
        raise RuntimeError("This Engram version lacks native task context")
    memory.call("status")
    context = memory.call("session_resume", project_id=project, task=task, limit=5)
    print(json.dumps(context, indent=2))
```

run it with `python agent_context.py`. the response includes recent eligible
project memories, the exact task's checkpoint if one exists, and a `boundary`
describing the returned text as reference data. an empty result is valid for a
new project.

your app can present this alongside the user's current request or pass it as
clearly labeled reference context to its chosen model. preserve the boundary:
a memory can describe a command or a claimed result without authorizing the
command or proving the result. this example only reads and prints context.

## save the handoff deliberately

after reviewing a summary with the user, save it through the same client:

```python
saved = memory.call(
    "session_checkpoint",
    project_id=project,
    task=task,
    summary="Retry limit updated; the network-failure check is still pending.",
    decisions=["Stop after three attempts"],
    next_steps=["Run the real network-failure check"],
    blockers=[],
)
if saved.get("status") != "saved":
    raise RuntimeError("Checkpoint was not confirmed")
```

this snippet belongs inside the `with NativeClient(...)` block. saving replaces
the checkpoint for that project and task. a new process can retrieve it with
`session_resume`. checkpoints do not become semantic memories and reads do not
reinforce memory importance.

for a complete runnable command with reviewed note files and explicit write
confirmation, use the [task assistant tutorial](task-assistant.md). to store
general facts, decisions or narrative memories, use the full MCP server's
`remember` tools separately. do not invent a native `remember` call or silently
save every conversation turn.

## choose filters before retrieving

| operation | supported selection |
| --- | --- |
| native `recall` | canonical `project_id`, recent eligible memories, `limit` 1–20 |
| native `session_resume` | the same project context plus an exact `task` checkpoint; omit `task` for up to three recent project checkpoints |
| native `search` / `search_explain` | `query` and `top_k` only, across the configured store |
| MCP `recall` | semantic `query`, `top_k`, and `mode`: `facts_only`, `facts_plus_rules` or `full_context` |
| MCP `recall_recent` | newest memories across the store by creation time, using `limit` |

native project context requires explicit ownership metadata or an absolute
source path inside the project. mentioning a project name in prose does not
establish ownership. the task key selects a checkpoint; it does not further
filter project memories.

when store-wide search is appropriate, make that choice explicit:

```python
explanation = memory.call(
    "search_explain",
    query="How did we decide to verify release artifacts?",
    top_k=5,
)
```

`search_explain` leaves access history and result caches unchanged, but loads
the configured embedding/reranking models. first use can download weights or
contact a configured provider; allow a longer client timeout. ordinary
`search` records accesses. neither supports a project filter, and filtering
its results afterward is not a project isolation boundary. see
[retrieval explanations](../architecture/retrieval.md#explanations).

## process and error handling

keep one client alive while working and serialize its calls. the shared client
matches response IDs, rejects invalid or non-finite JSON, limits requests to
65,536 bytes including the newline, and caps responses at 2 MiB by default.
`max_response_bytes` can set an explicit alternative response bound.

`NativeClientError` exposes an error `code`. API errors are distinct from
empty results. timeouts and invalid responses close the worker; create a fresh
client and check its status before continuing. a timed-out write may have
completed, so inspect the task before deciding to retry. the client never
replays a write automatically.

the context manager closes stdin and waits briefly for exit, then terminates
or kills a worker that does not stop. it drains and discards stderr separately
from the protocol. if startup fails, run `engram --config /absolute/engram.yaml
config check` or [doctor](../reference/cli.md#doctor) directly for diagnostics.

extend this boundary with [application integration patterns](integration-patterns.md)
or the [Minecraft server example](minecraft-server.md). those examples make
specific memory calls; they do not require a new agent framework.
