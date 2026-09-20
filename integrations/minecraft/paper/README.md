# engram paper integration source

a small Paper plugin that connects explicit staff commands to the repository's
[local game bridge](../../game_server_bridge.py). full setup and scope:
[minecraft tutorial](../../../docs/tutorials/minecraft-server.md).

target: Paper **1.21.11**, Java **21**. no compatibility claim for Folia, Fabric,
Bedrock or newer Paper 26.x. this is source to use and adapt, not a published
plugin binary. Engram's Python package does not install it into a game server.

## build

from the repository root:

```sh
mvn -B -ntp -f integrations/minecraft/paper/pom.xml verify
```

output: `integrations/minecraft/paper/target/engram-paper-0.1.0.jar`.
Gson is bundled and relocated; the Paper API is provided by the server. the POM
uses Paper's upstream snapshot API. capture the resolved Maven dependency tree
with build evidence if you need to reproduce a specific compilation.

the build runs bounded HTTP/formatting unit tests. to include the real bridge
test, start the Python bridge against an isolated store allowing world `world`
and set `ENGRAM_TEST_BRIDGE_URL` / `ENGRAM_TEST_BRIDGE_TOKEN`. the test writes and
replaces `build/ci-checkpoint`; do not point it at a production bridge.

## configuration and commands

the default [config](src/main/resources/config.yml) expects a bridge on
`http://127.0.0.1:8422` and `GAME_MEMORY_TOKEN` in the Minecraft process environment.
a private `token` config value is available as a fallback. the token must be
32–256 letters, digits, `_` or `-`; Python's `secrets.token_urlsafe(32)` generates
a suitable value. both sides must allow the same actual world names.

```text
/engram status
/engram save rule harbor-builds Keep the public dock route open.
/engram save build east-lighthouse Stairs need another landing.
/engram recall build east-lighthouse
/engram save handoff weekend-build Bring copper and finish the roof.
/engram recall handoff weekend-build
```

permissions: `engram.status`, `engram.save`, `engram.recall`, or `engram.*`;
all default to operators. `build` saves require a player and capture coordinates
on the main thread. other commands support the console's configured world.
saves replace the same world/kind/key checkpoint. there are no chat listeners,
automatic transcript capture, RCON calls, or world-changing actions.

## implementation boundary

`EngramPlugin` checks permissions/world scope and snapshots player location before
submitting work. `BridgeClient` refuses external hosts, redirects and proxies,
authenticates every request, caps response bytes and enforces a whole-response
timeout. a single worker plus eight waiting slots bounds outgoing work.
`ContextView` renders bounded plain text; retrieved content is never parsed as a
command or MiniMessage. no response is broadcast to other players.

compile and bridge tests are separate from in-game acceptance. after building,
verify plugin loading, save/recall across a bridge restart, world isolation and
permission denial on your target server. the tutorial gives the exact sequence.
