# Give an NPC a memory of each player

Mara keeps the harbour charts. One player returned a lost chart and mentioned a
green light offshore. Another has just arrived looking for a sibling. Mara can
remember those encounters separately, keep her public knowledge of the harbour,
and pick up either conversation after your application restarts.

the repository includes a runnable, framework-neutral implementation in
[`integrations/npc_memory.py`](https://github.com/raya-ac/engram/blob/main/integrations/npc_memory.py).
it saves explicit reference snapshots through the existing
[checkpoint bridge](game-server-plugins.md). it does not provide a game engine
plugin, dialogue model, quest system or automatic conversation recorder.

## what the NPC remembers

each **world + NPC** has one public record containing its persona and permitted
lore. each **world + NPC + player** has one player-specific, latest encounter snapshot:

```text
harbour / Mara
  public persona + lore ──────┬─→ context for player 001
                             └─→ context for player 002
  player 001 snapshot ─────────→ context for player 001 only
  player 002 snapshot ─────────→ context for player 002 only

another NPC or world → different record keys
```

the adapter chooses these records from trusted IDs. the bridge token is still
shared server authority; the diagram describes selection, not a per-player login.

| field | meaning |
| --- | --- |
| `relationship` | a short description useful for dialogue, such as “recognizes the chart helper” |
| `observations` | assertions supplied by trusted game code or a reviewer |
| `player_claims` | what the player said; still unverified even when the encounter was confirmed |
| `handoff` | a suggested conversational thread for next time |
| `event_id` | the stable event reference supplied by the game or reviewer |
| `confirmation` | `reviewed` or `game_confirmed`, recording the caller's basis for this explicit save |

Engram does not independently verify these assertions. a player claiming to own
the ferry does not acquire the ferry, and a remembered delivery does not grant
a quest reward. your game's authoritative state still decides inventory,
currency, permissions and quest completion.

every save replaces its selected record. the event ID is stored inside the
snapshot; it does not create an append-only event history. keep only the still
useful prior details when constructing a replacement.

## 1. prepare a dedicated demo store

use a checkout of this repository and a Python environment with Engram
installed. follow [installation](../getting-started/installation.md) first if
needed, then install the source client's extra dependency:

```sh
python -m pip install -r integrations/requirements.txt
```

for a new, isolated store:

```sh
python -m engram --config /absolute/npc-demo.yaml init \
  --yes --preset light --db-path /absolute/npc-demo.db
```

choose paths that do not already exist. create an existing directory to identify
this demo server, then start the source bridge with that directory and an
allowlisted world. set a private `GAME_MEMORY_TOKEN` in the bridge environment:

```sh
export GAME_MEMORY_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"

python integrations/game_server_bridge.py \
  --config /absolute/npc-demo.yaml \
  --project /absolute/npc-demo-server \
  --world harbour --world mountain \
  --host 127.0.0.1 --port 8422
```

run the client commands in another terminal with the **same token** in its
environment. keep that token on the trusted server side, never in a game client,
player message or source repository. the bridge does not initialize storage or
load models; these checkpoints need no provider key.

## 2. run the two-player harbour example

from the repository root:

```sh
python integrations/npc_memory.py \
  --base-url http://127.0.0.1:8422 \
  --world harbour --npc-id demo.harbour-keeper \
  demo --yes
```

this explicitly saves Mara's public persona/lore and two fictional player
snapshots, then prints their separate contexts. demo writes require an NPC ID
starting with `demo.` and `--yes`. use the dedicated store above: rerunning the
command replaces those same three demo records.

the first player's context includes the returned chart and the unverified green
light sighting. the second includes the missing-sibling claim. both see the
public ferry and lighthouse lore. neither context includes the other player's
encounter.

now run a fresh process:

```sh
python integrations/npc_memory.py \
  --world harbour --npc-id demo.harbour-keeper \
  context --player-id demo.player.001
```

repeat with `demo.player.002`. restart the bridge and repeat again to check
persistence in your deployment. the same IDs in `--world mountain`, or a
different NPC ID, have no records until you explicitly save them there. an
unknown player in `harbour` receives Mara's public reference and
`player_reference: null`.

## 3. write your own persona and encounter

a reviewed `persona.json` file looks like:

```json
{
  "persona": "Mara keeps the harbour charts. She is curious and speaks plainly.",
  "lore": ["The public ferry leaves at sunrise.", "The east dock overlooks the lighthouse."],
  "confirmation": "reviewed"
}
```

save it explicitly:

```sh
python integrations/npc_memory.py \
  --world harbour --npc-id npc.harbour-keeper \
  persona persona.json --yes
```

put only knowledge this NPC is permitted to share in public lore. staff notes,
another NPC's secrets and private player details do not belong there.

after a reviewed or game-confirmed event, an `encounter.json` file can contain:

```json
{
  "event_id": "delivery.4821",
  "confirmation": "game_confirmed",
  "relationship": "Recognizes the player who returned the buoy chart.",
  "observations": ["The server recorded the buoy chart being returned."],
  "player_claims": ["The player says they saw a green light beyond the breakwater."],
  "handoff": "Ask about the sighting if the player wants to discuss it."
}
```

```sh
python integrations/npc_memory.py \
  --world harbour --npc-id npc.harbour-keeper \
  save-event --player-id account.001 encounter.json --yes
```

`game_confirmed` is a label supplied by trusted code, not proof generated by
Engram. use it after the game has checked the relevant event. a human reviewing a
note can use `reviewed`; player speech remains in `player_claims` either way.
unknown JSON fields, missing confirmation and oversized notes fail validation.

## 4. call it from your dialogue system

the source module depends on the adjacent
[`checkpoint_client.py`](https://github.com/raya-ac/engram/blob/main/integrations/checkpoint_client.py)
and `httpx`. import from the repository, or copy both files together. a server
worker can use the same interface without invoking the CLI:

```python
import os

from integrations.checkpoint_client import CheckpointClient
from integrations.npc_memory import NPCMemory

client = CheckpointClient("http://127.0.0.1:8422", os.environ["GAME_MEMORY_TOKEN"])
client.health()
mara = NPCMemory(client, world_id="harbour", npc_id="npc.harbour-keeper")

# Obtain this from the authenticated game session, never from dialogue text.
player_id = "account.001"
context = mara.dialogue_context(player_id)
dialogue_input = {
    "player_message": "Do you remember the light I mentioned?",
    "reference_data": context,
}
```

`dialogue_input` is plain data for your chosen dialogue engine. the module does
not call a model or insert retrieved text into a system instruction. preserve
the returned `boundary` and the distinction between observations and claims
when adapting it to your model or scripted dialogue.

when trusted game code chooses to save an event, call:

```python
mara.save_event(
    player_id,
    event_id="conversation.4822",
    confirmation="reviewed",
    relationship="Recognizes the chart helper.",
    observations=["The server recorded the earlier chart return."],
    player_claims=["The player now describes the light as three green flashes."],
    handoff="The player may return with a sketch of the light.",
)
```

this replaces the prior encounter snapshot. dialogue generation alone does not
call `save_event`. choose explicit save moments such as a confirmed interaction,
an approved conversation summary or a reviewed handoff; do not automatically
promote an LLM's account of events into observations.

the synchronous client performs HTTP work. run it off the gameplay/UI thread
and return the result through your engine's supported scheduling mechanism. if
several events can update the same NPC/player pair, serialize those writes in
your application. there is no revision check or merge: the last completed save
wins.

## identity and limits

use stable NPC and authenticated account IDs. display names can change and are
not identity. the module accepts 1–128 ASCII letters, digits, `.`, `_`, `:`, or
`-`, starting with a letter or digit. syntax validation cannot tell whether a
caller supplied a trustworthy ID; the server must make that decision.

worlds use the bridge's configured 1–64 character labels. each record key is
`npc-` plus 60 SHA-256 hex characters over a structured, versioned selection of
world, NPC, record type and player. this avoids delimiter collisions and keeps
keys within the bridge limit. hashing is not access control: anyone holding the
bridge token can access that bridge's allowlisted worlds. authenticate players
and select their IDs before calling this module; do not expose arbitrary ID
lookup to players.

| value | bound |
| --- | --- |
| persona | 700 characters |
| public lore | 6 entries, 250 characters each |
| relationship | 240 characters |
| observations / player claims | 4 entries each, 200 characters per entry |
| handoff | 300 characters |
| event ID | 128 characters |
| encoded snapshot | 4,000 characters, also subject to the bridge's 16 KiB request limit |

only the selected public and player records are read. there is no semantic
search, transcript scan, cross-NPC knowledge sharing or general memory import.
stored JSON must match the expected schema and exact scope; a corrupt or foreign
snapshot causes an error instead of being returned as dialogue context.

missing state returns `null`. authentication, storage and transport errors
raise `CheckpointError` and make the CLI fail; they are not empty-memory
successes. a timed-out save may have completed. inspect the same player's
context before deciding to retry; the client never retries a write automatically.

## what you can build on it

- a merchant who remembers a player's stated interests while prices and stock stay in the game's own systems.
- a town guide who recalls which public landmarks a player has already discussed.
- a companion who resumes a reviewed conversation thread without treating it as quest completion.
- several NPCs with deliberately different public knowledge and separate player relationships.

the isolated tests exercise the actual HTTP bridge, native checkpoint storage,
two-player/NPC/world/project separation, replacement, fresh-client persistence,
limits and malformed-state rejection. they use no dialogue model or network
listener. this verifies the source memory workflow; test your own engine's
authentication, scheduling, rendering and dialogue behavior before using it in
a live game.
