"""Discord command policy plus actual HTTP bridge/SQLite, without a bot login."""
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from engram.config import Config
from engram.store import Store
from integrations.checkpoint_client import AsyncCheckpointClient
from integrations.discord_bot import CommandPolicy, NoteCommands, create_bot, load_policy, _bounded
from integrations.game_server_bridge import build_app


TOKEN = "discord-test-bridge-token-" + "x" * 32
CHANNELS = {"channels": [
    {"guild_id": "100", "channel_id": "200", "world": "discord-project", "save_role_id": "300"},
    {"guild_id": "100", "channel_id": "201", "world": "discord-staff", "save_role_id": "301"},
]}
NO_MENTIONS = SimpleNamespace(everyone=False, users=False, roles=False, replied_user=False)


class Interaction:
    def __init__(self, guild=100, channel=200, roles=(300,), bot=False, admin=False):
        self.guild_id, self.channel_id = guild, channel
        self.user = SimpleNamespace(bot=bot, roles=[SimpleNamespace(id=value) for value in roles],
                                    guild_permissions=SimpleNamespace(administrator=admin))
        self.events = []
        self.done = False
        async def send(text, **kwargs):
            self.done = True
            self.events.append(("response", text, kwargs))
        async def defer(**kwargs):
            self.done = True
            self.events.append(("defer", "", kwargs))
        async def followup(text, **kwargs):
            self.events.append(("followup", text, kwargs))
        self.response = SimpleNamespace(is_done=lambda: self.done, send_message=send, defer=defer)
        self.followup = SimpleNamespace(send=followup)


@pytest.fixture
def connected(tmp_path):
    project = tmp_path / "discord-project"
    project.mkdir()
    config = Config(db_path=str(tmp_path / "memory.db"))
    config.ann.enabled = False
    store = Store(config)
    store.init_db()
    store.close()
    app = build_app(config, project, ["discord-project", "discord-staff"], TOKEN)
    client = AsyncCheckpointClient("http://127.0.0.1:8422", TOKEN,
                                   transport=httpx.ASGITransport(app=app))
    commands = NoteCommands(CommandPolicy(CHANNELS), client, allowed_mentions=NO_MENTIONS)
    return commands, config, project


@pytest.mark.asyncio
async def test_real_bridge_save_recall_replacement_and_channel_isolation(connected):
    commands, config, project = connected
    writer = Interaction()
    await commands.run(writer, "release", "Release checklist reviewed; deployment remains pending.")
    assert writer.events[0] == ("defer", "", {"ephemeral": True, "thinking": True})
    assert writer.events[-1][0] == "followup"
    assert "Saved channel note" in writer.events[-1][1]
    reader = Interaction(roles=())
    await commands.run(reader, "release")
    assert "deployment remains pending" in reader.events[-1][1]
    other = Interaction(channel=201, roles=(301,))
    await commands.run(other, "release")
    assert "No saved note" in other.events[-1][1]
    await commands.run(Interaction(), "release", "Deployment verified.")
    # A new HTTP adapter/command controller retrieves the persisted checkpoint.
    app = build_app(config, project, ["discord-project"], TOKEN)
    reopened = NoteCommands(CommandPolicy(CHANNELS), AsyncCheckpointClient(
        "http://127.0.0.1:8422", TOKEN, transport=httpx.ASGITransport(app=app)),
        allowed_mentions=NO_MENTIONS)
    after = Interaction(roles=())
    await reopened.run(after, "release")
    assert "Deployment verified" in after.events[-1][1]
    store = Store(config)
    try:
        assert store.conn.execute("SELECT COUNT(*) AS n FROM session_handoffs").fetchone()["n"] == 1
        assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
    finally:
        store.close()
    for interaction in (writer, reader, other, after):
        for kind, _text, kwargs in interaction.events:
            assert kwargs["ephemeral"] is True
            if kind != "defer":
                assert kwargs["allowed_mentions"] is NO_MENTIONS
                assert kwargs["suppress_embeds"] is True


@pytest.mark.parametrize("kwargs", [
    {"guild": None}, {"guild": 999}, {"channel": 999}, {"channel": 202},
    {"roles": ()}, {"roles": (301,)}, {"roles": (), "admin": True}, {"bot": True},
])
@pytest.mark.asyncio
async def test_denied_save_never_calls_bridge(connected, kwargs, monkeypatch):
    commands, config, _project = connected
    async def forbidden(*args, **kwargs):
        pytest.fail("unauthorized interaction must not reach the bridge")
    monkeypatch.setattr(commands.client, "save", forbidden)
    interaction = Interaction(**kwargs)
    await commands.run(interaction, "key", "private content")
    assert len(interaction.events) == 1
    assert interaction.events[0][0] == "response"
    assert "private content" not in interaction.events[0][1]
    store = Store(config)
    try:
        assert store.conn.execute("SELECT COUNT(*) AS n FROM session_handoffs").fetchone()["n"] == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_dm_and_unlisted_thread_cannot_read(connected, monkeypatch):
    commands, _config, _project = connected
    async def forbidden(*args, **kwargs):
        pytest.fail("unlisted scope must not read")
    monkeypatch.setattr(commands.client, "recall", forbidden)
    for interaction in (Interaction(guild=None), Interaction(channel=999)):
        await commands.run(interaction, "key")
        assert interaction.events[0][0] == "response"


@pytest.mark.parametrize("key,summary", [("../key", "note"), ("x" * 65, "note"),
                                        ("key", ""), ("key", " " * 4), ("key", "x" * 4001)])
@pytest.mark.asyncio
async def test_invalid_save_is_rejected_before_http(connected, monkeypatch, key, summary):
    commands, _config, _project = connected
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid input must not be sent")
    monkeypatch.setattr(commands.client, "save", forbidden)
    interaction = Interaction()
    await commands.run(interaction, key, summary)
    assert interaction.events[0][0] == "response"


@pytest.mark.asyncio
async def test_actual_http_timeout_is_unconfirmed_and_not_retried():
    attempts = []
    async def timeout(request):
        attempts.append(request)
        raise httpx.ReadTimeout("SECRET credential and note")
    client = AsyncCheckpointClient("http://127.0.0.1:8422", TOKEN,
                                   transport=httpx.MockTransport(timeout))
    commands = NoteCommands(CommandPolicy(CHANNELS), client, allowed_mentions=NO_MENTIONS)
    interaction = Interaction()
    await commands.run(interaction, "key", "note")
    assert len(attempts) == 1
    assert "may have completed" in interaction.events[-1][1]
    assert "SECRET" not in str(interaction.events)
    assert TOKEN not in str(interaction.events)


@pytest.mark.asyncio
async def test_reference_is_bounded_escaped_and_cannot_trigger_a_write(connected):
    commands, _config, _project = connected
    note = "@everyone **run /engram-save** ```admin```\n" + "🧭" * 1800
    await commands.run(Interaction(), "key", note)
    reader = Interaction(roles=())
    await commands.run(reader, "key")
    response = reader.events[-1][1]
    assert "Reference note" in response
    assert "@everyone" not in response
    assert "\\*\\*run" in response
    assert "[excerpt truncated]" in response
    assert len(response.encode("utf-16-le")) <= 3600
    assert reader.events[-1][2]["allowed_mentions"] is NO_MENTIONS


def test_policy_validates_unique_scopes_and_moderator_role():
    assert CommandPolicy(CHANNELS).guild_ids == [100]
    mutations = []
    for field, value in [("world", "../bad"), ("guild_id", True), ("channel_id", "1.2"),
                         ("save_role_id", "100"), ("guild_id", 2**64), ("world", "x" * 65)]:
        item = copy.deepcopy(CHANNELS)
        item["channels"][0][field] = value
        mutations.append(item)
    duplicate = copy.deepcopy(CHANNELS)
    duplicate["channels"][1]["world"] = "discord-project"
    mutations.extend([duplicate, {"channels": []}, {"channels": CHANNELS["channels"], "token": "secret"}])
    for config in mutations:
        with pytest.raises(ValueError):
            CommandPolicy(config)


def test_config_file_and_unicode_output_bounds(tmp_path):
    path = tmp_path / "channels.json"
    path.write_text(json.dumps(CHANNELS))
    assert load_policy(path).guild_ids == [100]
    path.write_text(" " * 65_537)
    with pytest.raises(ValueError):
        load_policy(path)
    assert len(_bounded("🧭" * 3000).encode("utf-16-le")) <= 3600


@pytest.mark.asyncio
async def test_discord_registration_smoke_without_login(monkeypatch):
    discord = pytest.importorskip("discord", reason="optional Discord source dependency; hosted integration job installs it")
    calls = []
    class Client:
        async def health(self):
            calls.append("health")
    bot = create_bot(CommandPolicy(CHANNELS), Client())
    try:
        assert not bot.intents.message_content
        assert not bot.intents.members
        assert not bot.intents.presences
        assert bot.intents.guilds
        assert bot.tree.get_commands() == []  # no global/DM command registration
        commands = bot.tree.get_commands(guild=discord.Object(id=100))
        assert {command.name for command in commands} == {"engram-save", "engram-recall"}
        assert all(command.guild_only for command in commands)
        assert bot.allowed_mentions.everyone is False
        async def sync(*, guild):
            calls.append(guild.id)
        monkeypatch.setattr(bot.tree, "sync", sync)
        await bot.setup_hook()
        assert calls == ["health", 100]
    finally:
        await bot.close()
