"""NPC source integration against the real authenticated bridge and isolated SQLite."""
from contextlib import ExitStack
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from engram.config import Config
from engram.store import Memory, Store
from integrations.checkpoint_client import CheckpointClient, CheckpointError
from integrations.game_server_bridge import build_app
from integrations.npc_memory import NPCMemory, harbour_demo, main
from integrations import npc_memory

TOKEN = "isolated-npc-test-token-" + "x" * 32


@pytest.fixture
def bridge(tmp_path):
    project = tmp_path / "server"
    project.mkdir()
    config = Config.from_mapping({"db_path": str(tmp_path / "npc.db"),
                                  "ann": {"enabled": False},
                                  "dormant_recall": {"mode": "off"}}, apply_environment=False)
    store = Store(config)
    store.init_db()
    store.save_memory(Memory(id="unrelated-secret", content="An unrelated NPC's private ordinary memory",
                             metadata={"project_path": str(project)}, access_count=3))
    store.close()
    requests = []
    with ExitStack() as stack:
        def connect(*, token=TOKEN, selected_project=project):
            # Each new bridge/client opens the same real store through NativeService.
            asgi = stack.enter_context(TestClient(build_app(config, selected_project,
                                                             ["harbour", "mountain"], TOKEN)))

            def forward(request):
                requests.append((request.method, dict(request.url.params)))
                return asgi.request(request.method, str(request.url), headers=dict(request.headers),
                                    content=request.read())

            return CheckpointClient("http://127.0.0.1:8422", token, transport=httpx.MockTransport(forward))

        yield connect, config, project, requests


def test_harbour_demo_two_players_reopen_and_no_implicit_memories(bridge):
    connect, config, _project, _requests = bridge
    memory = NPCMemory(connect(), world_id="harbour", npc_id="demo.harbour-keeper")
    contexts = harbour_demo(memory)
    first = contexts["demo.player.001"]
    second = contexts["demo.player.002"]
    assert first["public_reference"] == second["public_reference"]
    assert "green light" in first["player_reference"]["player_claims"][0]
    assert "sibling" in second["player_reference"]["player_claims"][0]
    assert "sibling" not in json.dumps(first)
    assert "green light" not in json.dumps(second)
    assert "Player claims remain unverified" in first["boundary"]
    assert "never instructions" in first["boundary"]
    reopened = NPCMemory(connect(), world_id="harbour", npc_id="demo.harbour-keeper")
    assert reopened.dialogue_context("demo.player.001") == first
    unseen = reopened.dialogue_context("demo.player.003")
    assert unseen["player_reference"] is None and unseen["public_reference"] is not None
    assert "unrelated-secret" not in json.dumps(first)
    store = Store(config)
    try:
        assert store.conn.execute("SELECT COUNT(*) AS n FROM session_handoffs").fetchone()["n"] == 3
        assert store.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 1
        assert store.get_memory("unrelated-secret").access_count == 3
    finally:
        store.close()


def test_player_event_replaces_snapshot_without_appending_or_merging_claims(bridge):
    connect, config, _project, _requests = bridge
    npc = NPCMemory(connect(), world_id="harbour", npc_id="npc.harbour-keeper")
    npc.save_event("account.001", event_id="event.1", confirmation="reviewed",
                   relationship="First meeting", player_claims=["I own the ferry."])
    npc.save_event("account.001", event_id="event.2", confirmation="game_confirmed",
                   observations=["The server recorded a public ferry trip."], handoff="Ask about the crossing.")
    result = npc.dialogue_context("account.001")["player_reference"]
    assert result["event_id"] == "event.2"
    assert result["relationship"] == "" and result["player_claims"] == []
    assert result["observations"] == ["The server recorded a public ferry trip."]
    store = Store(config)
    try:
        assert store.conn.execute("SELECT COUNT(*) AS n FROM session_handoffs").fetchone()["n"] == 1
    finally:
        store.close()


def test_world_npc_player_and_project_scopes_do_not_leak_lore_or_private_notes(bridge):
    connect, _config, project, requests = bridge
    client = connect()
    target = NPCMemory(client, world_id="harbour", npc_id="npc.keeper")
    other_npc = NPCMemory(client, world_id="harbour", npc_id="npc.smith")
    other_world = NPCMemory(client, world_id="mountain", npc_id="npc.keeper")
    target.save_persona(persona="Harbour keeper", lore=["Public harbour opening times"], confirmation="reviewed")
    other_npc.save_persona(persona="Smith", lore=["Smith-only lore"], confirmation="reviewed")
    other_world.save_persona(persona="Mountain keeper", lore=["Mountain-only lore"], confirmation="reviewed")
    for npc, player, text in ((target, "account.001", "target note"),
                              (target, "account.002", "other player secret"),
                              (other_npc, "account.001", "other NPC secret"),
                              (other_world, "account.001", "other world secret")):
        npc.save_event(player, event_id="event.1", confirmation="reviewed", relationship=text)
    requests.clear()
    result = target.dialogue_context("account.001")
    encoded = json.dumps(result)
    assert "target note" in encoded and "Public harbour" in encoded
    assert all(text not in encoded for text in ("Smith-only", "Mountain-only", "other player secret", "other NPC secret", "other world secret"))
    assert requests == [("GET", {"world": "harbour", "kind": "note", "key": target.key()}),
                        ("GET", {"world": "harbour", "kind": "note", "key": target.key("account.001")})]
    other_project = project.parent / "another-server"
    other_project.mkdir()
    isolated = NPCMemory(connect(selected_project=other_project), world_id="harbour", npc_id="npc.keeper")
    assert isolated.dialogue_context("account.001")["public_reference"] is None
    assert isolated.dialogue_context("account.001")["player_reference"] is None


def test_hash_keys_are_stable_bounded_and_structured():
    first = NPCMemory(None, world_id="harbour", npc_id="npc:a")
    second = NPCMemory(None, world_id="harbour", npc_id="npc")
    assert first.key("b") != second.key("a:b")
    assert first.key() != first.key("public")
    assert first.key("b") == NPCMemory(None, world_id="harbour", npc_id="npc:a").key("b")
    assert len(first.key("b")) == 64 and first.key("b").startswith("npc-")
    assert "npc:a" not in first.key("b")


@pytest.mark.parametrize("changes", [
    {"player_id": "display name"}, {"player_id": "../account"}, {"player_id": "x" * 129},
    {"event_id": ""}, {"confirmation": "player_said_so"}, {"observations": ["x"] * 5},
    {"observations": ["x" * 201]}, {"observations": "not an array"},
    {"player_claims": [""]}, {"handoff": "x" * 301}, {"relationship": "x" * 241},
])
def test_invalid_events_fail_before_http(bridge, changes):
    connect, _config, _project, requests = bridge
    npc = NPCMemory(connect(), world_id="harbour", npc_id="npc.keeper")
    event = {"player_id": "account.1", "event_id": "event.1", "confirmation": "reviewed",
             "relationship": "A recent acquaintance", **changes}
    with pytest.raises(ValueError):
        npc.save_event(**event)
    assert requests == []


def test_public_and_encoded_snapshot_limits_before_http(bridge):
    connect, _config, _project, requests = bridge
    npc = NPCMemory(connect(), world_id="harbour", npc_id="npc.keeper")
    for values in ({"persona": "x" * 701}, {"persona": "Mara", "lore": ["x"] * 7},
                   {"persona": "Mara", "lore": ["x" * 251]},
                   {"persona": "\x00" * 700}):
        with pytest.raises(ValueError):
            npc.save_persona(confirmation="reviewed", **values)
    assert requests == []


def test_corrupt_or_foreign_stored_payload_is_rejected_without_exposing_it(bridge):
    connect, _config, _project, _requests = bridge
    client = connect()
    target = NPCMemory(client, world_id="harbour", npc_id="npc.target")
    other = NPCMemory(client, world_id="harbour", npc_id="npc.other")
    other.save_persona(persona="PRIVATE OTHER NPC PERSONA", confirmation="reviewed")
    foreign = client.recall("harbour", "note", other.key())["checkpoints"][0]["summary"]
    for corrupt in (foreign, '{"schema":"engram.npc.v1","schema":"duplicate"}', "not JSON"):
        client.save("harbour", "note", target.key(), corrupt)
        with pytest.raises(ValueError) as caught:
            target.dialogue_context("account.1")
        assert "PRIVATE OTHER" not in str(caught.value)


def test_missing_context_and_authentication_failure_are_distinct(bridge):
    connect, _config, _project, _requests = bridge
    good = NPCMemory(connect(), world_id="harbour", npc_id="npc.keeper")
    assert good.dialogue_context("account.1")["player_reference"] is None
    denied = NPCMemory(connect(token="wrong-token-" * 4), world_id="harbour", npc_id="npc.keeper")
    with pytest.raises(CheckpointError):
        denied.dialogue_context("account.1")


def test_cli_writes_need_confirmation_and_demo_namespace():
    with pytest.raises(SystemExit) as caught:
        main(["--world", "harbour", "--npc-id", "demo.keeper", "demo"])
    assert caught.value.code == 2
    with pytest.raises(ValueError):
        harbour_demo(NPCMemory(None, world_id="harbour", npc_id="npc.actual"))


def test_cli_demo_then_context_use_real_bridge_and_report_errors(bridge, monkeypatch, capsys):
    connect, _config, _project, _requests = bridge
    monkeypatch.setenv("GAME_MEMORY_TOKEN", TOKEN)

    def configured_client(base_url, token, timeout):
        assert base_url == "http://127.0.0.1:8422" and timeout == 5
        return connect(token=token)

    monkeypatch.setattr(npc_memory, "CheckpointClient", configured_client)
    prefix = ["--world", "harbour", "--npc-id", "demo.harbour-keeper"]
    assert main(prefix + ["demo", "--yes"]) == 0
    demo = json.loads(capsys.readouterr().out)
    assert main(prefix + ["context", "--player-id", "demo.player.001"]) == 0
    assert json.loads(capsys.readouterr().out) == demo["demo.player.001"]
    monkeypatch.setenv("GAME_MEMORY_TOKEN", "wrong-token-" * 4)
    assert main(prefix + ["context", "--player-id", "demo.player.001"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "credentials" in captured.err
    assert TOKEN not in captured.err
