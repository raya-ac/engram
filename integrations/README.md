# source integrations

code you can use and adapt to connect Engram to another application. these files
are kept in the repository; installing Engram does not add plugins to your apps.

| integration | source | walkthrough |
| --- | --- | --- |
| Open WebUI memory filter | [filter](open_webui/engram_filter.py) | [chat app filter](../docs/tutorials/open-webui-filter.md) |
| Minecraft Paper plugin | [Java project](minecraft/paper/) | [Minecraft](../docs/tutorials/minecraft-server.md) |
| game server checkpoint bridge | [Python HTTP bridge](game_server_bridge.py) | [other game servers](../docs/tutorials/game-server-plugins.md) |
| task app assistant | [Python client and workflow](../examples/integrations/) | [task assistant](../docs/tutorials/task-assistant.md) |

start with an initialized, dedicated Engram config/store. each walkthrough
specifies the host app version, dependencies, configuration and verification
steps. the [integration guide](../docs/tutorials/integration-patterns.md) explains
which interface to choose and what changes when you adapt it.

Python tests cover the source adapters with isolated storage. hosted Java tests
compile the Paper source and exercise its HTTP client against a real Engram
bridge. these checks do not stand in for testing your host app or live server.
