# build memory into an app

start with one useful moment: reopen a task, look up a server note, or recover a
decision while answering a question. decide who may see that context and what
event is allowed to save it. then choose the Engram interface that fits your app.

## choose a working example

| build | tutorial | what you implement |
| --- | --- | --- |
| a chat filter | [Open WebUI filter](open-webui-filter.md) | add recalled reference context to a chat request with deliberate save behavior |
| a Discord bot | [Discord channel notes](discord-bot.md) | save and recall exact notes through slash commands, with channel and moderator-role checks |
| a Minecraft server plugin | [Minecraft server](minecraft-server.md) | build the Paper plugin source and connect it to the local Engram bridge |
| NPC memory | [NPC reference snapshots](npc-memory.md) | keep public persona/lore separate from an exact player's latest interaction state |
| another game server plugin | [game server integrations](game-server-plugins.md) | adapt server commands and trusted identities to the bridge contract |
| a local task assistant | [task assistant](task-assistant.md) | resume a project's task and explicitly save its next checkpoint |
| your own Python agent | [build an agent](build-an-agent.md) | connect a long-lived native client to your app's memory workflow |

the [integration source](https://github.com/raya-ac/engram/tree/main/integrations)
contains the Open WebUI filter, Discord bot, game bridge, NPC adapter and Paper plugin. the
[local client examples](https://github.com/raya-ac/engram/tree/main/examples/integrations)
contain the task assistant and reusable JSONL client. each tutorial names its
dependencies, setup and verification steps; connecting Engram alone does not
install a chat filter or game plugin.

## choose an interface

| your app already has… | use | important boundary |
| --- | --- | --- |
| an MCP client | [stdio MCP](../guides/client-configs.md) | discover the actual tools; general `recall` searches the configured store |
| a server plugin or chat bot with named notes | [checkpoint bridge](game-server-plugins.md) | exact world/kind/key reads and replacements; no semantic search or arbitrary operations |
| control of local subprocesses | [native JSONL](../native-api.md) | scoped context/checkpoints and store-wide semantic search are different operations |
| a server-side HTTP integration | [web REST API](../reference/rest-api.md) | authenticated workspace access, not per-user or per-project authorization |
| a small scheduled or manual script | [CLI](../reference/cli.md) | explicit one-shot commands with a selected config |
| a Python process that needs storage internals | [Python examples](https://github.com/raya-ac/engram/tree/main/examples) | your code owns model setup, store lifetime and filtering |

native JSONL is a local request/response protocol, not MCP or an HTTP endpoint.
it has no general `remember` operation. use `session_checkpoint` for a reviewed
task summary; use a documented MCP, REST, CLI or Python write path when you need
an ordinary searchable memory. checkpoints are separate from the semantic memory
pool and are retrieved through native context/resume operations.

the interface also determines what “remember” means. the Open WebUI filter saves
an ordinary memory that can later match a semantic question. the Discord bot and
game bridge replace an exact named checkpoint. the NPC adapter stores bounded
structured snapshots inside those checkpoints. choose the behavior before
copying a save call; a checkpoint key will not appear automatically in semantic
search.

## extend the source you already have

| feature you want | start here | add in your application |
| --- | --- | --- |
| a reviewed release note in chat | `discord_bot.py`: `CommandPolicy` and `NoteCommands` | a trusted channel mapping and moderator role; `/engram-save release` replaces the current note |
| a Save to memory button in a notes app | `open_webui/engram_filter.py`: explicit save HTTP path | selected text, source-note reference and an authorized click handler |
| an issue-tracker handoff | `examples/integrations/task_assistant.py` | map a trusted project directory and stable issue ID to checkpoint/resume |
| an NPC remembers one player | `npc_memory.py`: `NPCMemory.save_event` and `dialogue_context` | an authenticated player ID, NPC ID and game-confirmed event |
| a Paper quest/build plugin uses memory | `EngramMemoryService` and `recipes/NpcMemoryHooks` | permission-checked event handlers and a main-thread continuation for game changes |
| another Python server or worker | `checkpoint_client.py`: sync and async clients | trusted ID mapping, explicit save events and a UI that distinguishes missing notes from errors |

the [source directory](https://github.com/raya-ac/engram/tree/main/integrations)
contains these adapters; the task assistant is under
[examples/integrations](https://github.com/raya-ac/engram/tree/main/examples/integrations).
the notes-app and issue-tracker rows describe places to attach the existing code,
not prebuilt plugins for a particular service.

for example, a helpdesk integration can save an approved resolution under a
stable issue key when a responder clicks “save handoff.” reopening that issue
recalls the exact note. if the goal is finding similar fixes across tickets,
use a separate authorized semantic-retrieval path instead. keep account routing
in application code in either case.

## design the event before writing the adapter

use this sequence for each feature:

1. **event:** a user opens a task, asks a question or presses “save note.” avoid
   starting with “record every message.”
2. **scope:** select the permitted store and, when applicable, the canonical
   project directory and task. resolve these from trusted app state.
3. **Engram request:** choose one documented operation with bounded inputs and a
   timeout. discover the native `operations` or MCP `tools/list` contract first.
4. **reference context:** preserve result IDs, source information when present,
   truncation flags and returned order. present remembered text as reference data.
5. **verified output:** check the current app state before applying a remembered
   instruction or claiming a task succeeded. show empty results and connection
   failures honestly; save only after the selected write event is confirmed.

for example, opening a release task can send this native request:

```json
{"id":"resume-1","operation":"session_resume","params":{"project_id":"/srv/projects/lantern","task":"release checklist","limit":5}}
```

after a person reviews the summary, the save action can send:

```json
{"id":"save-1","operation":"session_checkpoint","params":{"project_id":"/srv/projects/lantern","task":"release checklist","summary":"Package built. Installed-client verification is still pending.","next_steps":["Install the candidate and verify its reported version"]}}
```

the same project/task key updates that checkpoint; it is not an append-only event
log. the saved statement records what your app supplied, not a check Engram ran.
see the [task assistant](task-assistant.md) for a runnable version with an explicit
save action and a fresh-process resume.

## choose what an app remembers

use these record choices when adapting the source above:

| app | recall when | save when | useful record | keep outside memory |
| --- | --- | --- | --- | --- |
| notes app | reopening a project or drafting a related note | the author selects “remember this decision” | a short decision, rationale and source-note ID | private notebooks the current reader cannot access |
| helpdesk | drafting a reply for an authorized queue | a responder approves a reusable resolution | symptom, verified fix, product version and ticket reference | credentials, full customer transcripts and unrelated accounts |
| game NPC | starting a conversation or quest transition | the game confirms a durable event | a witnessed encounter or completed objective | authoritative inventory, permissions and hidden state the NPC should not know |
| chat bot | an explicit recall command or an opted-in conversation | an authorized user issues a save command | agreed rules or a reviewed answer with its source | unrelated channels, private messages and bot-generated guesses |
| task board | opening a task | the owner saves a handoff | current state, decisions, next steps and blockers | live assignee/status fields that should come from the task system |

keep the application's database authoritative for access rights, inventory,
ticket state and task completion. Engram supplies context about those things.
when a recalled fact conflicts with current state, show or correct that conflict
through a deliberate workflow rather than silently trusting the older note.

## scope is part of the interface

**store-wide search:** native `search`/`search_explain`, general MCP `recall`,
CLI search and normal web search can retrieve across the configured store.
adding a username or project name to the query does not enforce isolation.
native search rejects a `project_id` argument; it is not a scoped search API.

**project context:** native `recall` and `session_resume` select recent active
records with explicit project ownership, plus native checkpoints. their
`project_id` is a canonical absolute directory path. this is chronological
context, not semantic search. merely writing a project name into a memory's
prose does not give it that ownership.

**authorization:** project selection is still not an account permission system.
the app must map an authenticated user or server identity to allowed projects.
do not let a caller choose an arbitrary filesystem path. use separate stores,
configs and processes for separate trust boundaries; give each SQLite store its
own ANN index path as well. different configs pointing at the same database do
not isolate data.

for a multi-user chat service, either keep the entire dedicated store suitable
for everyone who can query it, or build authenticated routing to separate stores.
filtering displayed results afterward does not make a store-wide query private.
the [game server pattern](game-server-plugins.md) shows a bounded adapter rather
than exposing every Engram operation to players.

## keep context separate from commands

memory text can contain mistakes, quoted instructions or old plans. label it as
untrusted reference data in the prompt or UI, and keep app instructions and tool
permissions outside that text. a delimiter helps presentation; it does not
replace authorization in code. a remembered command is not permission to execute
it, and a recalled “done” is not proof of current success.

model-backed retrieval uses the configured embedding/reranking providers.
generation in your chat app is a separate provider choice: selected memory text
sent with a prompt goes there too. review both paths before using private data.
project context and checkpoint operations do not require model inference.

## verify the complete feature

use a new disposable store with fictional data first. validate its config and
run [doctor](../getting-started/troubleshooting.md#understand-doctor-results),
then test through the actual app:

- save one approved item, restart the client, and recover it through the intended
  context or search operation;
- try a second project/user and confirm the app rejects an unauthorized scope;
- disconnect Engram and check that the app reports unavailable context instead
  of inventing a saved result;
- submit memory text that looks like an instruction and verify it cannot trigger
  writes or privileged actions;
- inspect the store after an ordinary read: context/resume are non-reinforcing,
  while ordinary semantic search can record accesses and affect importance.

for diagnostic search without those retrieval side effects, use
`search_explain` or the corresponding explanation interface. a successful
`doctor --full` checks Engram's own paths; it does not prove your app's permission
checks, event hooks or reconnect behavior work.

the source contracts are
[`service.py`](https://github.com/raya-ac/engram/blob/main/engram/service.py),
[`project_context.py`](https://github.com/raya-ac/engram/blob/main/engram/project_context.py),
[`mcp_server.py`](https://github.com/raya-ac/engram/blob/main/engram/mcp_server.py)
and the [REST reference](../reference/rest-api.md). use discovery and these
contracts when adapting an example to a new app.
