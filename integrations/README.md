# source integrations

code you can use and adapt to connect Engram to another application. these files
are kept in the repository; installing Engram does not add plugins to your apps.

| integration | source | walkthrough |
| --- | --- | --- |
| Open WebUI memory filter | [filter](open_webui/engram_filter.py) | [chat app filter](../docs/tutorials/open-webui-filter.md) |
| Minecraft Paper plugin | [Java project](minecraft/paper/) | [Minecraft](../docs/tutorials/minecraft-server.md) |
| game server checkpoint bridge | [Python HTTP bridge](game_server_bridge.py) | [other game servers](../docs/tutorials/game-server-plugins.md) |
| reusable sync/async HTTP client | [checkpoint client](checkpoint_client.py) | [client guide](../docs/tutorials/checkpoint-client.md) |
| NPC persona and player memory | [NPC adapter](npc_memory.py) | [NPC memory and dialogue](../docs/tutorials/npc-memory.md) |
| Discord channel notes | [Discord bot](discord_bot.py) | [Discord setup](../docs/tutorials/discord-bot.md) |
| task app assistant | [Python client and workflow](../examples/integrations/) | [task assistant](../docs/tutorials/task-assistant.md) |

start with an initialized, dedicated Engram config/store. each walkthrough
specifies the host app version, dependencies, configuration and verification
steps. the [integration guide](../docs/tutorials/integration-patterns.md) explains
which interface to choose and what changes when you adapt it.

the [recipe guide](../docs/tutorials/what-you-can-build.md) develops those pieces
into building companions, remembered NPC encounters, quest recaps, server lore
and chat workflows. the Paper source also registers `EngramMemoryService` for
other plugins and includes `NpcMemoryHooks` as a Java integration example.

the NPC Python adapter and Java hook use different record/key formats; they
are two starting points, not interchangeable readers for the same notes.
checkpoint saves replace the latest exact record. your application owns
authorization, game state, history and any dialogue model.

Python tests cover the source adapters with isolated storage. hosted Java tests
compile the Paper source and exercise its HTTP client against a real Engram
bridge. these checks do not stand in for testing your host app or live server.
