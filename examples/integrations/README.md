# app integrations

small adapters that put Engram behind a deliberate app action: recall context,
review it, then save when the user or application explicitly asks. start with the
[integration guide](../../docs/tutorials/integration-patterns.md) for interface
selection and scope.

| example | tutorial | use |
| --- | --- | --- |
| [native JSONL client](native_client.py) | [native API](../../docs/native-api.md) | supervised local subprocess requests using the installed Engram interpreter |
| [task assistant](task_assistant.py) | [task assistant](../../docs/tutorials/task-assistant.md) | project context, exact-task resume and reviewed checkpoint saves |
| [Open WebUI filter source](../../integrations/open_webui/engram_filter.py) | [chat filter setup](../../docs/tutorials/open-webui-filter.md) | add reference context through the app's filter interface |
| [Paper plugin source](../../integrations/minecraft/paper/) and [local bridge](../../integrations/game_server_bridge.py) | [Minecraft setup](../../docs/tutorials/minecraft-server.md) | explicit in-game commands connected to Engram |

the tutorials list exact files, dependencies, build commands and checks. these
are source integrations to run or adapt; installing `engram-memory-system` does
not install plugins into other applications. follow [installation](../../docs/getting-started/installation.md)
to prepare Engram, then use a dedicated initialized config/store for the app.

## choose the boundary first

native `recall`/`session_resume` use a canonical absolute project path and return
recent scoped context. native `search` is semantic retrieval across the whole
configured store and has no project filter. MCP and native operations with the
same name do not necessarily have the same meaning. native JSONL has no general
`remember` operation; checkpoints are separate from searchable memories.

the app must authorize a user's scope. neither a project name in a query nor a
memory layer isolates customers, players or channels. use separate stores for
separate trust boundaries, and keep credentials and arbitrary filesystem paths
out of client-controlled requests. model-backed operations retain the selected
provider's normal local or remote behavior.

keep retrieved text as reference data, use bounded requests with timeouts, and
test save/restart/recall through the real app. empty recall, a connection failure
and a confirmed save are three different outcomes. ordinary semantic search can
record access activity; native context reads and search explanations do not
reinforce memories.

for another game platform, use the
[server plugin pattern](../../docs/tutorials/game-server-plugins.md). the Paper
example does not imply that Rust, Valheim or other platform plugins are already
included.
