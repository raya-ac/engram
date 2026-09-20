# minecraft server integration

this tutorial connects real Paper commands to Engram. staff can save a server
rule, record a build at their current coordinates, or leave a handoff for the
next group. recall reads the named note for the current world. other plugins can
use the same connection through the included Java service, including the
[NPC memory hooks](#npc-memory-from-an-existing-plugin).

the repository includes the [Java plugin source](https://github.com/raya-ac/engram/tree/main/integrations/minecraft/paper)
and [Python bridge source](https://github.com/raya-ac/engram/blob/main/integrations/game_server_bridge.py).
you can build and adapt them for your server; there is no separate plugin
download or automatic installation into Minecraft.

## what runs where

```text
staff /engram command
  → Paper plugin worker → authenticated loopback HTTP bridge
  → Engram native checkpoint operation → configured store
```

the plugin targets **Paper 1.21.11 and Java 21**. it uses ordinary `plugin.yml`
registration and Paper's scheduler; compatibility with newer 26.x releases,
Folia, Fabric or Bedrock servers is not established. Paper's
[Java requirements](https://docs.papermc.io/paper/getting-started/#requirements)
and [plugin project setup](https://docs.papermc.io/paper/dev/project-setup/)
explain the platform versions.

Engram and the bridge run on the same host as Paper. the plugin accepts a numeric
loopback HTTP address, refuses redirects, and bypasses system HTTP proxies.
each bridge has one fixed server project and an explicit world allowlist. run
separate bridge processes and ports for independent server projects.

## 1. prepare Engram and the source

complete [installation](../getting-started/installation.md) and
[initial setup](../getting-started/quickstart.md) with a dedicated Minecraft
config/store. keep its absolute config path and Python executable. the bridge
needs an initialized store; it does not initialize or migrate one for you.

use a checkout of this repository for the source commands below. the required
files are `integrations/game_server_bridge.py` and
`integrations/minecraft/paper/`. the bridge uses FastAPI/Uvicorn already included
with Engram. checkpoint operations do not call an LLM or embedding model.

choose the actual, existing server directory as the project, for example
`/srv/minecraft/survival`. that path is a scope identifier; the bridge does not
read or modify world files. moving the directory changes its canonical project
identity, so plan the scope before saving notes.

## 2. start the bridge

generate one token for this bridge:

```sh
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

set `GAME_MEMORY_TOKEN` to that value in the bridge process environment. use the
same value for the Minecraft process or its private plugin config. keep it out
of source control and player-visible commands.

from the repository root, with Engram's Python environment:

```sh
/absolute/path/to/venv/bin/python integrations/game_server_bridge.py \
  --config /srv/engram/minecraft.yaml \
  --project /srv/minecraft/survival \
  --world world --world world_nether --world world_the_end \
  --host 127.0.0.1 --port 8422
```

use the actual Bukkit world names from your server. this example runs in the
foreground; your process supervisor can run the same command persistently.
normal Engram environment overrides still apply to the selected config.

check initialized storage through the authenticated endpoint:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $GAME_MEMORY_TOKEN" \
  http://127.0.0.1:8422/health
```

expect `status: "ok"` and storage metadata. authentication failure, an unavailable
store and an empty note are separate outcomes. all bridge routes require the
token; this service is separate from Engram's web dashboard and MCP transports.

## 3. build and configure the plugin source

with Java 21 and Maven available, build from the repository root:

```sh
mvn -B -ntp -f integrations/minecraft/paper/pom.xml verify
```

this compiles against Paper's 1.21.11 API and produces
`integrations/minecraft/paper/target/engram-paper-0.1.0.jar`. use that shaded JAR,
which includes its relocated JSON dependency. the Paper API is supplied by the
server. its upstream `SNAPSHOT` dependency can change; record the resolved
dependency versions when reproducing a build.

place your built JAR in the server's `plugins` directory and restart Paper, as
described in [Paper's plugin guide](https://docs.papermc.io/paper/adding-plugins/).
the plugin creates `plugins/EngramMemory/config.yml`. if no token is configured
it disables itself; configure the file and restart the server.

```yaml
bridge-url: 'http://127.0.0.1:8422'
token-env: 'GAME_MEMORY_TOKEN'
token: ''
request-timeout-seconds: 5
allowed-worlds:
  - world
  - world_nether
  - world_the_end
console-world: world
```

`token-env` reads the Minecraft server process's environment. when the host
cannot supply that variable, put the generated token in `token` instead and
restrict access to that file. a nonempty environment value takes precedence.
the plugin and bridge must allow the same world names. console commands use
`console-world`; player commands use the player's current server world.

## 4. use the in-game commands

the permission nodes `engram.status`, `engram.save` and `engram.recall` default to
operators. grant only the needed nodes through your server's permission system.
`engram.*` grants all three. command blocks are not supported.

```text
/engram status
/engram save rule harbor-builds Keep the marked public dock route open.
/engram save build east-lighthouse Shell complete; north staircase and lantern glass remain.
/engram recall build east-lighthouse
/engram save handoff weekend-build Walls finished. Next group should bring twelve copper blocks.
/engram recall handoff weekend-build
```

save a `build` note while standing at the build: the plugin captures the world
identifier and block coordinates from the server, before handing work to its
HTTP worker. console users can save `rule`, `handoff` and `note` entries; a
`build` save requires an in-game player location.

worlds and keys use case-sensitive ASCII labels of at most 64 characters:
letters, digits, `.`, `_`, `-`, starting with a letter or digit. note text is
limited to 3,000 characters. `/engram save note <key> <text...>` covers other
short references.

saving the same **project + world + kind + key** replaces its previous checkpoint.
different worlds and kinds keep separate notes. this is a current checkpoint,
not an append-only event history or semantic search over chat. the plugin adds
no chat listener and performs no moderation or world writes. server or admin
plugins may still log commands under their own configuration.

## 5. verify it on your server

1. run `/engram status` and confirm initialized storage is available.
2. save a build, then recall it and compare the reported coordinates with your location.
3. restart the bridge and recall again to check persistence.
4. request the same key from another allowed world; it should be missing there.
5. verify that a player without `engram.save` cannot save a note.

the plugin displays results only to the requesting player or console. retrieved
text is plain reference text: slash commands and formatting tags inside notes
are never executed. output is capped at ten short messages and marks truncation.

## call Engram from another Paper plugin

the plugin registers `dev.engram.paper.api.EngramMemoryService` with Bukkit's
[ServicesManager](https://jd.papermc.io/paper/1.21.11/org/bukkit/plugin/ServicesManager.html).
your plugin can save and recall without copying HTTP code or receiving the bridge
token. the service uses the same configured world allowlist, worker, queue,
request limits and connection as staff commands.

build/install the source artifact into the Maven repository used by your build:

```sh
mvn -B -ntp -f integrations/minecraft/paper/pom.xml install
```

in your plugin's POM, use it as a **provided** dependency:

```xml
<dependency>
  <groupId>dev.engram</groupId>
  <artifactId>engram-paper</artifactId>
  <version>0.1.0</version>
  <scope>provided</scope>
</dependency>
```

in your plugin's `plugin.yml`, add:

```yaml
depend: [EngramMemory]
```

this is a locally built source dependency, not an artifact published to Maven
Central. keep the API classes out of your plugin's shaded JAR: both plugins need
the same service interface class. Paper documents the dependency/load-order
behavior in its [plugin.yml reference](https://docs.papermc.io/paper/dev/plugin-yml/#dependencies).

look up the service in your own plugin's `onEnable`:

```java
import dev.engram.paper.api.EngramMemoryService;

EngramMemoryService memory = getServer().getServicesManager()
    .load(EngramMemoryService.class);
if (memory == null) {
    getLogger().severe("EngramMemory service is unavailable");
    getServer().getPluginManager().disablePlugin(this);
    return;
}
```

the public [service source](https://github.com/raya-ac/engram/blob/main/integrations/minecraft/paper/src/main/java/dev/engram/paper/api/EngramMemoryService.java)
defines immutable records and two methods:

| method/type | contract |
| --- | --- |
| `save(Note)` | returns `CompletableFuture<SaveResult>` after the bridge confirms the write |
| `recall(Key)` | returns `CompletableFuture<Optional<Checkpoint>>`; an empty optional means no saved note |
| `Key(world, kind, key)` | exact identity inside the bridge's fixed server project |
| `Note(key, summary)` | a complete replacement snapshot; the longer constructor also accepts decisions, next steps and blockers |
| `Checkpoint.note()` / `.updatedAt()` | full bounded note and its saved timestamp, without chat-display truncation |
| `Failure.reason()` | distinguishes blocked world, busy queue, stopped service, bridge failure and invalid response |

for example, a town-management plugin can save a reviewed construction handoff:

```java
var key = new EngramMemoryService.Key("world", "handoff", "town-east-dock");
var note = new EngramMemoryService.Note(
    key, "Dock supports passed the builder's review; roof work remains.",
    java.util.List.of("Keep the public path open"),
    java.util.List.of("Bring twelve copper blocks"), java.util.List.of());
var saved = memory.save(note);
```

constructors reject invalid labels and oversized text. a note allows a
4,000-character summary and up to eight 500-character entries in each list;
the encoded request must still fit the bridge's 16 KiB limit. all lists are
copied. saving replaces omitted lists with empty ones. concurrent saves to the
same key are last-writer-wins, with no compare-and-swap or automatic merge: assign
one owner per checkpoint or serialize updates in your plugin.

calls enqueue I/O and are safe to submit from either thread. **never `join()` or
`get()` a pending future on the game thread.** completion callbacks may run on
any thread, including immediately for a rejected request. snapshot Bukkit
objects before submitting, then schedule player/world access back through
`getServer().getScheduler().runTask(yourPlugin, ...)`. check that your plugin and
player are still available, and recheck interaction permissions/current state
before displaying or acting on a delayed result. callbacks should be short.

the command permission nodes protect `/engram`; Java API callers are trusted
server plugins and must authorize their own players and events. the service
does not infer a player from a key. when EngramMemory stops, it unregisters the
service and fails waiting futures. retain no assumption that a cancelled or
failed in-flight save was rolled back; recall before retrying.

## recipes to build on this source

### player build continuity

use a stable build ID from your claim/build plugin as the key. a player finishing
a work session chooses “save progress”; your handler verifies ownership and
captures their world and coordinates on the main thread. save a complete
snapshot: completed work, materials still needed, and the next useful action.

when an authorized builder reopens the project, recall that key and display the
snapshot with its saved time. compare it with the current claim and world before
offering navigation or work suggestions. the note is not a teleport destination
authorization or proof that a structure is still present. the existing
`/engram save build` command implements the explicit save/coordinate path for
staff; a player-facing menu and claim checks belong to your integration.

### shared town lore

keep a small set of reviewed keys such as `rule/town-canon`, `note/town-history`
and `note/current-festival`. a town editor approves a replacement snapshot;
your NPC or quest plugin recalls those named keys when a conversation needs
them. write separate keys for separate topics instead of replacing one town
record with every conversation.

an approved lore entry can supply names and background for dialogue. a player
claim such as “the mayor gave me the vault” is a reported claim until the town
system confirms it; it should not overwrite canon or grant access. ownership,
economy and quest flags remain in their authoritative game systems.

### staff handoff

use `handoff/staff-current` for a reviewed shift summary, or a stable task key
for each ongoing incident. save observed state, decisions already made and the
next checks. example: “rail tunnel closed for repairs; west entrance marked;
next shift must inspect the support beams before reopening.”

the next shift recalls the note, checks the actual tunnel and records a new
snapshot after inspection. Engram does not close routes, ban players or infer
moderation actions. a historical audit trail requires a separate event log;
repeatedly saving this checkpoint keeps only its latest state.

### NPC memory from an existing plugin

the compiled [NpcMemoryHooks source](https://github.com/raya-ac/engram/blob/main/integrations/minecraft/paper/src/main/java/dev/engram/paper/recipes/NpcMemoryHooks.java)
adds explicit save/recall hooks to your NPC or quest implementation:

```java
import dev.engram.paper.recipes.NpcMemoryHooks;

var npcNotes = new NpcMemoryHooks(memory);

// Capture these from your authorized interaction handler, on the server thread.
String world = player.getWorld().getName();
java.util.UUID playerId = player.getUniqueId();
java.util.UUID npcId = persistentNpcId; // your NPC system's saved identity

var recalled = npcNotes.recall(world, npcId, playerId);

// Call only after your quest system verifies the outcome and reviews this snapshot.
var saved = npcNotes.saveReviewedState(world, npcId, playerId,
    "Bridge delivery accepted; reward recorded by quest system. Town tour remains available.");
```

these are calls from your existing handlers, not newly registered Paper or
Citizens events. the helper creates no NPC and generates no dialogue. it hashes
the stable NPC/player UUID pair into a 64-character key; the world and bridge
project provide the remaining scope. use persistent IDs, not display names or
a freshly generated UUID per interaction. hashing separates keys but is not
access control or anonymization.

wire a conversation feature through these steps:

1. **interaction:** your handler verifies the player can interact with that NPC
   and snapshots the IDs/world. request the saved pair-specific note.
2. **context:** an empty result means no remembered state. a failed future means
   memory is unavailable; do not turn either into a fabricated past encounter.
3. **prompt or script:** combine the note as labeled reference data with current
   quest facts and reviewed town lore. delimit remembered text separately from
   instructions. clip the supplied context to your dialogue budget and retain
   `updatedAt` so old context is visible.
4. **outcome:** the quest plugin checks inventory, quest flags or other current
   state before accepting delivery or granting a reward. model output and old
   notes do not authorize game actions.
5. **save:** after the verified event, replace the snapshot with the current
   relationship/quest summary. handle save failure visibly; do not append a raw
   chat transcript or repeat the write every game tick.

a useful NPC snapshot might say: “the player repaired the east bridge; the
quest system recorded the reward; the archivist offered a town tour.” on the
next visit, dialogue can acknowledge that history while the quest system prevents
a duplicate reward. Engram provides the stored context; your NPC framework,
dialogue renderer and optional model integration remain your code.

## failures and implementation details

`Engram request queued` is not a save confirmation. wait for `Saved…`, a returned
note, or an explicit error. one request per sender can be pending; the worker
has eight queue slots. requests have a full-response deadline, bounded request
and response bodies, and no automatic retries. if a save times out, recall its
key before deciding whether to retry: the write may already have completed.

for a refused request, check the token and both world allowlists. for connection
failure, check that the bridge is running on the configured host/port with the
right Engram config. missing notes require checking world, kind, key and project
scope; they do not require lowering retrieval confidence.

the Java tests cover HTTP framing/authentication, redirects, response limits,
timeouts, plain-text rendering, API scope, immutable records, queue saturation,
shutdown completion and NPC/player key separation. `ActualBridgeTest` exercises
Java → Python → SQLite when `ENGRAM_TEST_BRIDGE_URL` and
`ENGRAM_TEST_BRIDGE_TOKEN` point to an isolated test bridge allowing `world`.
its service check saves a typed snapshot and recalls it through a fresh service.
these checks do not establish that a live Paper server loaded the plugin or
that in-game permissions behaved correctly; use the server checks above.

network calls run off the game thread, and replies return through the scheduler.
the implementation follows Paper's [threading guidance](https://docs.papermc.io/paper/dev/scheduler/):
capture game state on the server thread and keep slow I/O outside it. to adapt
the same bounded HTTP contract to another platform, see
[game server plugins](game-server-plugins.md).
