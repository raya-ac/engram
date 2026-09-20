# reuse the checkpoint client

the source [checkpoint client](https://github.com/raya-ac/engram/blob/main/integrations/checkpoint_client.py)
is shared by the NPC adapter and Discord bot. use it in another Python game
service, chat bot, admin tool or backend instead of writing the same HTTP logic
again. it exposes three operations: health, save and exact recall.

first run the [source bridge](game-server-plugins.md) against an initialized
Engram store and allow the namespaces your app will use. its `world` field is a
configured label: it can represent a Minecraft world, a Discord channel mapping
or another app's named context. your app decides that mapping from trusted state.

## copy and configure

use Python 3.11 or newer. from the repository root:

```sh
python -m pip install -r integrations/requirements.txt
```

if you copy the integration into another project, keep `checkpoint_client.py`
beside your adapter or place it in your own package. import paths below assume
you are running from the Engram checkout. the client does not start the bridge
or create its database.

```python
import os
from integrations.checkpoint_client import CheckpointClient

client = CheckpointClient(
    "http://127.0.0.1:8422",
    token=os.environ["GAME_MEMORY_TOKEN"],
    timeout=5,
)
print(client.health()["status"])
```

the same private token must be configured in the bridge process. the URL and
token belong in your server-side configuration, not in player/chat arguments.
HTTPS is supported when you provide a suitable reverse proxy; the built-in
bridge itself serves HTTP. numeric loopback HTTP works by default. another
private HTTP address requires `allow_private_http=True`, which opts into that
transport but does not create a private network.

## save a deliberate checkpoint

```python
result = client.save(
    "world", "build", "east-lighthouse",
    "Walls finished; north staircase remains.",
    decisions=["Use copper for the roof"],
    next_steps=["Bring materials for the landing"],
    blockers=[],
)
print(result["status"])  # saved, only after confirmation
```

`world`, `kind` and `key` select the note. supported kinds are `rule`, `build`,
`handoff` and `note`. world/key labels contain 1–64 ASCII letters, digits, dots,
underscores or hyphens. the bridge must also allow that world. summaries are
at most 4,000 characters; each optional list allows eight items of at most 500
characters. the complete JSON body must fit in 16 KiB.

saving again replaces the complete checkpoint. omitted lists become empty;
this is not an append operation. use your app's own history/event log when you
need every past change. concurrent writers to the same key are last-write-wins;
serialize updates in the app when that matters.

## distinguish missing context from a failed request

```python
from integrations.checkpoint_client import CheckpointError

try:
    result = client.recall("world", "build", "east-lighthouse")
except CheckpointError:
    print("Memory is unavailable; check the bridge.")
else:
    if result["found"]:
        note = result["checkpoints"][0]
        print(note["summary"])
    else:
        print("No saved note for this key yet.")
```

the client validates the returned task identity and bounded note shape. it
does not silently turn an HTTP failure, a mismatched record or malformed JSON
into an empty result. returned content is still reference text; rendering it
does not authorize running commands found inside it.

## use async code in chat apps

```python
import os
from integrations.checkpoint_client import AsyncCheckpointClient

async def load_build_note():
    client = AsyncCheckpointClient(
        "http://127.0.0.1:8422", os.environ["GAME_MEMORY_TOKEN"], timeout=5,
    )
    return await client.recall("world", "build", "east-lighthouse")
```

the method names and return shapes match the synchronous client. each request
opens and closes its HTTP client; there is no extra startup/shutdown method.
use the async variant in an async application, and keep synchronous network
calls off a game loop or GUI thread. the [Discord source](discord-bot.md) shows
how to acknowledge a command before awaiting its result.

## handle a write timeout

```python
try:
    client.save("world", "handoff", "weekend", "Roof complete; inspect the stairs.")
except CheckpointError as error:
    if error.outcome_unknown:
        print("The save may have completed. Recall the key before retrying.")
    else:
        print("The bridge did not accept the save.")
```

there are no automatic retries. the async client enforces a total request
deadline. the synchronous client uses HTTPX's connect/read/write/pool timeouts
plus elapsed-time checks while reading a response; those phase limits are not
a hard whole-operation deadline. both bound response size to 32 KiB, disable
redirect following and ignore proxy environment variables. see HTTPX's
[timeout semantics](https://www.python-httpx.org/advanced/timeouts/).

local validation errors raise `ValueError` before a request. connection and
response failures raise `CheckpointError` with sanitized messages. after a
timeout, cancellation or disconnect, a write already delivered to the bridge
cannot be undone by the client.

## testing an adapter

both constructors accept an optional HTTPX `transport` for tests. the repository
checks the async client against a real FastAPI bridge and isolated SQLite using
[ASGI transport](https://www.python-httpx.org/advanced/transports/). this exercises
authentication, persistence, replacement, world/kind isolation and response
validation without an external network or model. malformed-response and timeout
tests check that failures stay failures.

once your adapter passes those checks, test its real app callback, permission
checks and reconnect behavior. the client only handles HTTP; it does not decide
which player, guild, account or NPC is allowed to select a record.
