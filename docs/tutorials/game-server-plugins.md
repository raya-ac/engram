# connect a game server

the repository includes a small
[HTTP bridge](https://github.com/raya-ac/engram/blob/main/integrations/game_server_bridge.py)
and [Paper plugin source](https://github.com/raya-ac/engram/tree/main/integrations/minecraft/paper).
the plugin turns explicit game commands into saved rules, build notes and
handoffs. the bridge keeps the database credentials and project path on the
server side. use the [Minecraft tutorial](minecraft-server.md) to build and try
the Paper integration.

this page covers the bridge contract for adapting another server plugin. the
repository does not include ready-made Rust, Valheim or other game plugins.

## run the bridge from source

use a Python environment with Engram installed, an initialized dedicated store,
and an existing directory representing this server. for a new store, follow
[installation](../getting-started/installation.md); for an existing one, use
`doctor` rather than running `init` over it.

from the repository root, in a POSIX shell:

```sh
# generate a private token for this local bridge process
export GAME_MEMORY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

python integrations/game_server_bridge.py \
  --config /absolute/path/to/game-engram.yaml \
  --project /absolute/path/to/server-project \
  --world survival --world creative \
  --port 8422
```

`python` must be the interpreter from the Engram environment. the token stays in
this shell's environment; give the trusted plugin the same value through its
server-side configuration. do not put it in player messages or committed files.
the bridge requires at least 32 visible ASCII characters without spaces; the
Paper plugin uses the URL-safe token alphabet produced by the command above.

the default host is `127.0.0.1`. keep the plugin and bridge on the same host for
the supplied Paper example. the bridge's `--host` option can change binding, but
it does not add TLS, per-player authentication or a remote hosting service. the
Paper plugin deliberately uses a loopback endpoint.

configuration and `ENGRAM_*` overrides work as in the main CLI. inspect the
effective config before starting. the bridge does not initialize a database,
load embedding models, capture chat or call an LLM. it opens and closes its own
native service for each request.

## check the actual store

every request, including health, needs the Bearer token. from a shell with the
same `GAME_MEMORY_TOKEN` value:

```sh
curl --fail-with-body --max-time 10 \
  -H "Authorization: Bearer $GAME_MEMORY_TOKEN" \
  http://127.0.0.1:8422/health
```

a successful response has `ok: true`, `status: "ok"` and `storage` containing actual native
store status. a listening HTTP process with an unavailable or uninitialized
store returns 503. use the in-game command afterward to verify the plugin too;
this request alone does not exercise its permissions or event handling.

## save and retrieve an exact note

```sh
curl --fail-with-body --max-time 10 \
  -H "Authorization: Bearer $GAME_MEMORY_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"world":"survival","kind":"rule","key":"spawn","summary":"Keep the marked spawn area clear of permanent builds.","decisions":["Builders agreed to use the east district"]}' \
  http://127.0.0.1:8422/v1/checkpoints

curl --fail-with-body --max-time 10 \
  -H "Authorization: Bearer $GAME_MEMORY_TOKEN" \
  'http://127.0.0.1:8422/v1/checkpoints?world=survival&kind=rule&key=spawn'
```

POST returns `status: "saved"`, the native task key, the configured project path
and `memory_reinforcement: false`. GET returns `found` and a `checkpoints` array.
when no note exists, it returns HTTP 200 with `found: false` and an empty array;
that is different from a failed request.

| field | contract |
| --- | --- |
| `world` | configured allowlisted label, 1–64 ASCII letters, numbers, `.`, `_` or `-` |
| `kind` | `rule`, `build`, `handoff` or `note` |
| `key` | stable note identifier using the same character/length limits as `world` |
| `summary` | POST only; nonempty text, at most 4,000 characters |
| `decisions`, `next_steps`, `blockers` | optional POST arrays, each up to 8 nonempty strings of at most 500 characters |

requests reject unknown fields, duplicate JSON fields and extra/repeated query
parameters. the complete UTF-8 request body must fit within 16 KiB, even if its
individual fields would otherwise be valid. send `Content-Type: application/json`
for POST.

the bridge maps the exact tuple to native task
`game:<world>:<kind>:<key>` under its fixed canonical project directory. saving
that tuple again **replaces its checkpoint**, including optional lists; it does
not append a history entry. choose separate keys for separate build sites or
handoffs. worlds with the same note key remain separate.

there is no semantic search, arbitrary native operation, project-path parameter,
delete endpoint or automatic transcript collection in this bridge. GET returns
only the selected checkpoint, excluding the ordinary memory context that the
underlying native resume operation can also read. these checkpoints live in the
native checkpoint store; they do not become semantic-search memories.

## wire another server's plugin

adapt the boundary first, then the platform API:

1. register explicit save and recall commands in the server plugin. define
   permissions separately for viewing and writing notes.
2. obtain the authenticated player/server identity and current world from the
   server runtime. map them to configured allowed labels. never accept a config
   path, project path or arbitrary bridge URL from command text.
3. validate the kind, key and text limits before sending a request. the bridge
   validates again, but early feedback is clearer for a player.
4. send HTTP off the gameplay thread with a finite timeout and bounded response.
   send the response back through the platform's permitted server-thread path.
   use its current threading contract when implementing the adapter.
5. render returned summaries as plain reference text. do not dispatch them as
   console commands, grant permissions or change inventory based on their text.
6. report “saved” only after a successful response. after a timeout, the write
   may already have completed; read the exact tuple before deciding to retry.

for a chat-driven NPC, retrieve a permitted note when the conversation starts,
then pass it as reference context to your own dialogue system. for a server
moderation helper, let an authorized moderator save an approved rule. neither
feature requires saving every player message. your game remains authoritative
for player permissions, quest completion and world state.

the shared token authorizes **all allowlisted worlds** on that bridge. world
labels partition checkpoints; they do not authenticate individual players.
private player notes need permission checks and trusted ID mapping in the plugin,
or separate bridge/store boundaries. public server lore and staff-only notes
should not share an unrestricted player lookup path.

## handle failures and verify the integration

| HTTP status | meaning |
| --- | --- |
| `400` | invalid world/kind/key, unexpected fields or malformed checkpoint input |
| `401` | missing or wrong token |
| `404` | unknown route; this is not the response for a missing note |
| `413` | request body exceeds 16 KiB |
| `415` | POST body is not declared as JSON |
| `503` | configured storage could not complete the operation |

validation, authentication and storage errors contain sanitized `error.code`
and `error.message` fields. do not log the token, full request bodies or raw
provider/database exceptions in your plugin.
the source bridge disables HTTP access logging so query keys are not logged by
default.

before using a real server, verify a save followed by recall, restart and recall,
replacement of the same key, separation across two allowed worlds, and rejection
of an unlisted world. test with a player who lacks the write permission, then
stop the bridge and confirm the game keeps running and reports the failure.
the repository's isolated SQLite tests cover the HTTP/storage boundary; the
plugin's actual command and permission path needs its own server check.

see [the integration hub](integration-patterns.md) for other app patterns and
[the native API](../native-api.md) for the checkpoint contract behind the bridge.
