# minecraft server integration

this tutorial connects real Paper commands to Engram. staff can save a server
rule, record a build at their current coordinates, or leave a handoff for the
next group. recall reads the named note for the current world.

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
timeouts and plain-text rendering. `ActualBridgeTest` additionally exercises
Java → Python → SQLite when `ENGRAM_TEST_BRIDGE_URL` and
`ENGRAM_TEST_BRIDGE_TOKEN` point to an isolated test bridge allowing `world`.
these checks do not establish that a live Paper server loaded the plugin or
that in-game permissions behaved correctly; use the server checks above.

network calls run off the game thread, and replies return through the scheduler.
the implementation follows Paper's [threading guidance](https://docs.papermc.io/paper/dev/scheduler/):
capture game state on the server thread and keep slow I/O outside it. to adapt
the same bounded HTTP contract to another platform, see
[game server plugins](game-server-plugins.md).
