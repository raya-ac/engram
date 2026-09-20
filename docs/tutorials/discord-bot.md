# explicit channel notes in Discord

use two slash commands to keep a channel's current decisions and handoffs:

```text
/engram-save key:release summary:The staging checks passed; production approval is pending.
/engram-recall key:release
```

the bot saves only the supplied note. a configured moderator role is required
to save; users who can invoke the command in an allowed channel can recall its
notes. responses are ephemeral. it does not read message history, capture
transcripts, generate replies with a model or search the whole Engram store.

the source is
[`integrations/discord_bot.py`](https://github.com/raya-ac/engram/blob/main/integrations/discord_bot.py),
using the shared
[`checkpoint_client.py`](https://github.com/raya-ac/engram/blob/main/integrations/checkpoint_client.py).
keep these files together if you copy them out of the repository.

## 1. prepare the bridge

follow [the checkpoint bridge setup](game-server-plugins.md#run-the-bridge-from-source)
with a dedicated initialized Engram store. choose one world label per Discord
channel; here, `discord-project` and `discord-staff` are namespaces, not game worlds.

```sh
python integrations/game_server_bridge.py \
  --config /absolute/discord-memory/engram.yaml \
  --project /absolute/discord-memory/project \
  --world discord-project --world discord-staff \
  --port 8422
```

the config file and project directory must already exist. the bridge process
and bot process both need the same private `GAME_MEMORY_TOKEN`. check the
authenticated `/health` route before starting the bot. no embedding model or
LLM is needed for these checkpoint operations.

## 2. create a dedicated Discord application

create an application in the Discord Developer Portal, obtain its bot token,
and install it in a test server using **Guild Install** with the `bot` and
`applications.commands` scopes. Discord's
[official setup guide](https://docs.discord.com/developers/quick-start/getting-started)
explains application creation and installation.

give the bot access to the intended channels; do not grant it Administrator.
keep privileged Message Content, Server Members and Presence intents disabled.
this source uses slash-command options and the member information attached to
the interaction, with only the non-privileged guild intent enabled. see
[discord.py's intent guide](https://discordpy.readthedocs.io/en/stable/intents.html).

use a dedicated application for this source: startup synchronizes these two
commands into each configured guild. if adapting an existing bot, add the
commands to its existing command tree instead of replacing its registered set.

## 3. map channels and write roles

save this as a private local JSON file such as
`/absolute/discord-memory/channels.json`, replacing all IDs with your own:

```json
{
  "channels": [
    {
      "guild_id": "111111111111111111",
      "channel_id": "222222222222222222",
      "world": "discord-project",
      "save_role_id": "333333333333333333"
    },
    {
      "guild_id": "111111111111111111",
      "channel_id": "444444444444444444",
      "world": "discord-staff",
      "save_role_id": "555555555555555555"
    }
  ]
}
```

use numeric IDs, not display names. Discord's
[Developer Mode guide](https://support.discord.com/hc/en-us/articles/206346498-Where-can-I-find-my-User-Server-Message-ID)
explains copying server and channel IDs. select a dedicated moderator role for
`save_role_id` and assign it to the members who may replace notes.

every channel must map to a unique world label. the configured role is required
even when the caller has Administrator; the `@everyone` role is rejected as a
save role. DMs, other guilds and unlisted channels are denied before the bridge
is called. a thread has its own channel ID and needs its own explicit entry;
allowing its parent channel does not automatically allow the thread.

channel labels partition stored notes. the bridge token still grants access to
every allowlisted world, so keep it on the bot server. do not hand it to users
or use a shared public channel for staff-only notes.

## 4. run the source

in a Python 3.11+ environment, from the repository root:

```sh
python -m pip install -r integrations/requirements-discord.txt

# use your host's secret mechanism, or set these privately in the shell
export DISCORD_BOT_TOKEN='your-private-discord-bot-token'
export GAME_MEMORY_TOKEN='the-same-private-token-used-by-the-bridge'

python integrations/discord_bot.py \
  --channels /absolute/discord-memory/channels.json \
  --bridge-url http://127.0.0.1:8422 \
  --timeout 5
```

the bot can use a separate Python environment from Engram. the shared HTTP
client defaults to numeric loopback HTTP or HTTPS, follows no redirects, ignores
proxy environment settings, limits responses and enforces a total request
deadline. this bot's CLI does not enable unencrypted remote HTTP. use a deliberate
private deployment and HTTPS endpoint if the bridge is on another host.

startup checks bridge health and synchronizes guild commands. the source uses discord.py's
[command-tree setup pattern](https://github.com/Rapptz/discord.py/blob/master/examples/app_commands/basic.py).
neither token belongs in the channel configuration or a committed source file.

## 5. test both permissions and persistence

in a configured channel, use the two commands at the top of this page. then:

1. restart the bot and recall `release` again;
2. save the same key with a revised summary and confirm recall shows the revision;
3. recall that key in the second allowed channel and confirm it has a separate note;
4. try saving without the configured moderator role, including with an
   administrator account that lacks that role;
5. try a DM and an unlisted channel, then stop the bridge and try an allowed call.

an empty lookup reports “no saved note.” failures are reported separately. a
timed-out save may have completed, so recall the exact key before retrying.
there is no automatic write retry.

notes use bridge kind `note` and the supplied exact key. saving replaces the
current checkpoint, not an append-only history. keys accept 1–64 ASCII letters,
numbers, `.`, `_` or `-`; summaries accept up to 4,000 characters. displayed
references are bounded and marked when truncated. they do not become ordinary
semantic-search memories.

## adapt the bot

`CommandPolicy` owns the trusted guild/channel-to-world mapping and save-role
check. `NoteCommands` owns explicit save/recall behavior. `create_bot` registers
the Discord commands. keep those boundaries when adding a modal, a reviewed
handoff command or an event callback.

the handler defers ephemerally before HTTP work and sends an ephemeral followup.
Discord requires an initial interaction response within three seconds; deferral
acknowledges it while the bounded request runs. see
[Discord's interaction response contract](https://docs.discord.com/developers/interactions/receiving-and-responding).
mentions are disabled, Markdown is escaped and link previews are suppressed in
recalled text. a note that says “run this command” remains text.

for semantic question-answering, build a separate opted-in feature against a
dedicated store; an exact-key bot is not semantic search. do not route a query
to a mixed private database merely because it came from an allowed channel.
see [integration patterns](integration-patterns.md) for the scope choices.

## verification scope

the tests exercise command authorization and mock Discord interactions through
the real asynchronous HTTP client, bridge and isolated SQLite storage. a
dependency-enabled smoke test constructs the actual discord.py command tree
without logging in. no Discord account, bot login or live message delivery was
used to validate this source; follow the live checks above for your server.
