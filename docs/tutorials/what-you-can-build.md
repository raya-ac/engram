# what you can build with Engram

imagine returning to a Minecraft town after a week away. the builder remembers
where you stopped, the harbour keeper remembers the delivery you made, and the
staff channel still has the decision about moving spawn. those are three
different uses of memory. each needs a different moment to save and a different
way to retrieve the right context.

Engram supplies that persistence. your game, chat app or bot decides what a
record means, who can read it and what to do with it. a language model is
optional: an NPC can choose a scripted line from saved context, or your dialogue
system can use that context to generate a response.

the repository has [usable source integrations](https://github.com/raya-ac/engram/tree/main/integrations).
the recipes below show how to turn them into features. the examples describe
things you can build; they are not claims that complete quest systems, NPC
engines or support platforms are included.

## 1. a Minecraft building companion

**what a player sees:** they return to a half-finished lighthouse, ask for the
build note and get its location, the latest progress and the next job.

start with the [Paper plugin](minecraft-server.md). while standing at the build:

```text
/engram save build east-lighthouse Walls complete. The north staircase needs one landing; bring copper for the roof.
/engram recall build east-lighthouse
```

the plugin adds coordinates from the server. the note survives the player
leaving and the bridge restarting. save the same key again after the next work
session to replace the progress report. use another key for another building.

to make this a feature in your own building plugin, call the registered
`EngramMemoryService` when a player selects **Save project note**. retrieve the
same key when they reopen that project. show the saved note beside the current
build status; your plugin still supplies live claims, ownership and block state.

you could add a project menu, a list of active builds in your own database, a
reviewed materials plan or a builder-to-builder handoff. Engram stores the
context; the source bridge currently retrieves exact keys, so your app must
keep its own project catalogue rather than assuming a list-all endpoint exists.

**first useful check:** save a note, restart the bridge, recover it, then update
the same key and confirm only the new checkpoint returns.

## 2. an NPC who recognises a returning player

**what a player sees:** the harbour keeper greets them with “you brought the
timber yesterday; the west pier is repaired now.” a different player gets their
own greeting, not someone else's history.

use the [NPC memory adapter](npc-memory.md) for a game-independent example, or
the compiled `NpcMemoryHooks` source with the Paper service. keep a stable NPC
identifier and use the authenticated player's persistent ID. a display name is
presentation, not the memory key.

the useful records are different:

| record | example | who supplies it |
| --- | --- | --- |
| shared persona/lore | the harbour keeper manages ferries and knows the public pier closure | a reviewed authoring tool or confirmed world event |
| player observation | this player delivered the requested timber | the server after validating the delivery |
| player claim | the player says they know a hidden route | dialogue input, explicitly labelled as a claim |
| handoff | next conversation should mention the repaired pier | a reviewed continuation note |

on interaction, retrieve the shared NPC record and this player's record. pass
those references to a scripted dialogue tree or your chosen text generator.
ask the generator to stay within the NPC's knowledge; the player's current
message is separate from remembered facts. an NPC should not learn every event
on the server simply because it shares a store.

the included Python adapter keeps a current bounded snapshot, not an unlimited
diary. refresh that snapshot after meaningful events instead of storing every
line of chat. it does not spawn an NPC, animate one, supply a model or decide
whether a quest reward is earned. the [NPC walkthrough](npc-memory.md) shows
the save/read flow and two-player isolation in detail.

**first useful check:** teach the NPC different facts about two test players,
restart the memory client and verify each conversation receives only its
intended context. repeat with another NPC using the same two players.

## 3. quests with a remembered story

**what a player sees:** their journal explains why they are helping the town,
what they learned and what they planned to do next, even after changing sessions.

your quest system owns the quest ID, stage, inventory checks and reward ledger.
after it confirms a stage transition, save a short narrative checkpoint such as:

```text
Quest harbor-repair: timber delivery was accepted by the quest system.
The player chose to help the west pier first. Next conversation should explain
the lantern repair route. Reward status must be checked in the quest ledger.
```

use a stable player + quest identity in your adapter's key mapping. read that
checkpoint when opening the journal or starting related dialogue. supply the
current quest stage separately so an old summary cannot make the game pay the
same reward twice.

this gives you narrative continuity across dialogue, journals and staff-assisted
recovery. it is also useful for tabletop campaigns: a session recap and a
character's knowledge can persist while the campaign system owns authoritative
rules and character stats.

**first useful check:** replay the same quest-completed event. the narrative
checkpoint may be replaced, but the authoritative reward ledger must reject a
second payout independently of Engram.

## 4. server lore and faction knowledge

**what players see:** different characters know different parts of the world.
a public guide knows a town's history; a faction contact knows its approved
briefing; an individual NPC remembers only relevant encounters.

model those audiences before writing prompts. public lore can use an approved
`rule` or `note` checkpoint. faction and player records need your plugin's own
membership checks and stable ID mapping. the bridge token grants access to
every allowlisted world on that bridge; a hashed key alone is not authorization.

build a small lore editor that lets staff review a paragraph and explicitly
publish its replacement checkpoint. when an event changes the world, update
the lore record deliberately. give the dialogue system the current game state
alongside those references so it can distinguish “the bridge used to be closed”
from “the bridge is closed now.”

the [game-server guide](game-server-plugins.md) explains the shared HTTP contract.
the Python client works outside Minecraft; adapt the game-facing event and
permission code to your engine. separate stores/processes are appropriate when
knowledge must remain private between independently trusted applications.

**first useful check:** a player outside a faction cannot request its briefing,
even if they know its record key. check this in the game adapter, before HTTP.

## 5. a Discord handoff bot for a gaming community

**what a group sees:** a moderator saves the next raid plan or server decision,
then another member retrieves it later without scrolling through a long channel.

the [Discord source](discord-bot.md) provides explicit slash commands:

```text
/engram-save key:weekend-build summary:Finish the north staircase, then fit copper roof panels. Meet at the harbour.
/engram-recall key:weekend-build
```

each configured channel maps to its own checkpoint namespace. a configured
role can save; recalls stay in that channel's scope. the bot returns ephemeral
results and does not listen to ordinary messages. use stable keys for current
plans, onboarding notes, known server issues or an agreed rules summary.

to share selected public notes with a Minecraft server, build a deliberate
publisher that copies an approved note into a public game key. keep Discord
channel notes and private NPC records in separate namespaces by default. do not
point a general chat command at private player memory and rely on obscure keys.

you could extend the bot with a review button, task-specific commands or a
scheduled reminder that reads an approved key. those event handlers are additions
to the source; the current bot only saves and recalls when someone invokes it.

**first useful check:** save from the configured role, recall in that channel,
then try another channel and an account without the save role.

## 6. a chat assistant that remembers selected decisions

**what a user sees:** they tell the assistant to remember a project choice,
then ask about it in a later chat using different words.

the [Open WebUI filter](open-webui-filter.md) already supports that path:

```text
/remember My Lantern project uses SQLite for local development.
```

a later “what database did I choose?” runs semantic retrieval and adds matching
notes as reference context before the model answers. the selected notes go to
the chat model you configured; retrieval and reply generation are separate steps.

adapt this into a Save decision button in another chat app. send the reviewed
sentence to the documented memory write endpoint and recall it before related
questions. retain the app's existing system prompt and keep recalled text in a
reference block. do not silently treat a generated reply as a fact worth saving.

this differs from the Discord/NPC checkpoint examples: semantic memory helps
with related wording; an exact checkpoint helps when you already know the task,
NPC or note key. general semantic search covers the configured store. a multi-user
app needs trusted per-user store routing before it can safely reuse this pattern.

**first useful check:** recover the selected decision in a new conversation,
then verify an unrelated account cannot reach that store.

## 7. support notes and moderation assistance

**what staff see:** an assistant can recover an approved troubleshooting answer
or a current community guideline while they draft a response.

start from the Discord command source for exact approved answers, or the chat
filter for a dedicated searchable knowledge store. save the symptom, the tested
fix, the version it applies to and a reference to the original report. save a
guideline's rationale as well as its wording so a future answer can explain it.

the app should retrieve only knowledge the current staff member is allowed to
see. review generated answers before sending them. reports about a player remain
reports; a remembered accusation is not evidence for an automatic ban. your
moderation system owns actions, permissions and audit history.

**first useful check:** change a known fix or guideline, then confirm the current
record appears and the assistant identifies any older conflicting context.

## 8. continuity across tools

**what a user sees:** a task begun in one app can be resumed elsewhere without
retyping its decisions and next steps.

use the [task assistant](task-assistant.md) and reusable native client when your
app can launch a local process. use the shared HTTP checkpoint client when your
server needs a narrow connection. store a reviewed checkpoint with a stable
project/task identity, then have each app request that exact context on resume.

you can combine these with code editors, notes apps or game-admin tools. choose
which records should be shared explicitly; sharing a database does not mean
every tool should receive every record. a saved checkpoint also does not update
the original issue tracker or game state—the integration must do its own work
and save the verified result afterward.

## choose the smallest useful starting point

| if the feature needs… | start here |
| --- | --- |
| exact named notes from game or chat code | [shared checkpoint client](checkpoint-client.md) |
| Paper commands or calls from another Paper plugin | [Minecraft source and service API](minecraft-server.md) |
| shared NPC context plus one player's latest interaction | [NPC memory adapter](npc-memory.md) |
| channel-scoped commands without listening to chat | [Discord bot](discord-bot.md) |
| related memories added to a model's next answer | [Open WebUI filter](open-webui-filter.md) |
| a local subprocess and project/task handoffs | [task assistant](task-assistant.md) |

build one save → restart → recall loop first. once that works, connect the real
app event and UI. add generation, richer menus or automation only when the app
can distinguish an empty record, a failed request and a confirmed save.
