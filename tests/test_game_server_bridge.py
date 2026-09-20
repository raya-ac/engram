"""Real isolated SQLite acceptance for the source game-server HTTP bridge."""
import importlib.util
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import pytest

from engram.config import Config
from engram.service import NativeService
from engram.store import Memory, Store


SOURCE = Path(__file__).resolve().parents[1] / "integrations" / "game_server_bridge.py"
spec = importlib.util.spec_from_file_location("game_server_bridge_example", SOURCE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
TOKEN = "fictional-game-test-token-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN}
SELECT = {"world": "survival", "kind": "rule", "key": "spawn"}


@pytest.fixture
def isolated(tmp_path):
    project = tmp_path / "server"
    project.mkdir()
    config = Config(db_path=str(tmp_path / "memory.db"))
    config.ann.enabled = False
    config.ann.index_path = str(tmp_path / "unused.index")
    store = Store(config)
    store.init_db()
    store.save_memory(Memory(id="private-reference", content="private ordinary context must not be returned",
                             metadata={"project_path": str(project)}, access_count=3, importance=.7))
    store.close()
    app = bridge.build_app(config, project, ["survival", "creative"], TOKEN)
    with TestClient(app) as client:
        yield client, config, project


def save(client, **changes):
    return client.post("/v1/checkpoints", headers=HEADERS,
                       json={**SELECT, "summary": "Spawn is a no-build area.", **changes})


def read(client, **changes):
    return client.get("/v1/checkpoints", headers=HEADERS, params={**SELECT, **changes})


def test_authentication_all_routes_and_no_generic_api(isolated):
    client, _config, _project = isolated
    for path in ("/health", "/v1/checkpoints", "/unknown", "/docs", "/openapi.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/checkpoints", json={}).status_code == 401
    for path in ("/unknown", "/docs", "/openapi.json", "/search", "/health/"):
        assert client.get(path, headers=HEADERS, follow_redirects=False).status_code == 404
    assert client.get("/health", headers=[("Authorization", "Bearer " + TOKEN),
                                         ("Authorization", "Bearer " + TOKEN)]).status_code == 401


def test_save_read_replace_reopen_and_exact_world_key_scope(isolated):
    client, config, project = isolated
    assert read(client).json() == {"found": False, "checkpoints": []}
    response = save(client, decisions=["Keep spawn clear"], next_steps=["Mark the boundary"])
    assert response.status_code == 200
    assert response.json() == {"status": "saved", "task": "game:survival:rule:spawn",
                               "project_path": str(project), "memory_reinforcement": False}
    item = read(client).json()
    assert item["found"] is True
    assert item["checkpoints"][0]["summary"] == "Spawn is a no-build area."
    assert "private ordinary context" not in json.dumps(item)
    assert read(client, world="creative").json()["found"] is False
    assert read(client, kind="build").json()["found"] is False
    assert read(client, key="different").json()["found"] is False
    assert save(client, summary="Spawn boundary marked.").status_code == 200
    with TestClient(bridge.build_app(config, project, ["survival"], TOKEN)) as reopened:
        result = read(reopened).json()
        assert len(result["checkpoints"]) == 1
        assert result["checkpoints"][0]["summary"] == "Spawn boundary marked."
        assert result["checkpoints"][0]["next_steps"] == []
    other_project = project.parent / "other-server"
    other_project.mkdir()
    with TestClient(bridge.build_app(config, other_project, ["survival"], TOKEN)) as other:
        assert read(other).json()["found"] is False


def test_reads_and_saves_do_not_reinforce_memories_or_initialize_schema(isolated, monkeypatch):
    client, config, _project = isolated
    monkeypatch.setattr(Store, "init_db", lambda *_: pytest.fail("bridge must never initialize storage"))
    store = Store(config)
    try:
        before = store.get_memory("private-reference")
        counts = {table: store.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                  for table in ("memories", "access_log", "events")}
        assert save(client).status_code == 200
        assert read(client).status_code == 200
        health = client.get("/health", headers=HEADERS)
        assert health.status_code == 200
        assert health.json()["ok"] is True
        assert health.json()["status"] == "ok"
        assert health.json()["storage"]["protocol"] == "engram-jsonl"
        after = store.get_memory("private-reference")
        assert (before.access_count, before.last_accessed, before.importance) == (
            after.access_count, after.last_accessed, after.importance)
        for table, count in counts.items():
            assert store.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"] == count
    finally:
        store.close()


@pytest.mark.parametrize("changes", [
    {"world": "unlisted"}, {"world": "../survival"}, {"world": "sürvival"},
    {"kind": "admin"}, {"key": "x" * 65}, {"key": "a:b"}, {"key": 1},
    {"summary": ""}, {"summary": " "}, {"summary": "x" * 4001}, {"summary": False},
    {"decisions": ["x"] * 9}, {"next_steps": ["x" * 501]}, {"blockers": [""]},
    {"blockers": "not a list"}, {"project_id": "/another/project"},
    {"operation": "search"}, {"action": "clear"},
])
def test_rejects_unscoped_or_unbounded_writes(isolated, changes):
    client, _config, _project = isolated
    assert save(client, **changes).status_code == 400
    assert read(client).json()["found"] is False


def test_get_rejects_unknown_missing_and_repeated_parameters(isolated):
    client, _config, _project = isolated
    for params in ({"world": "survival"}, {**SELECT, "project_id": "/secret"},
                   {**SELECT, "world": "unknown"}, [*SELECT.items(), ("world", "creative")]):
        assert client.get("/v1/checkpoints", headers=HEADERS, params=params).status_code == 400
    assert client.post("/v1/checkpoints?key=other", headers=HEADERS,
                       json={**SELECT, "summary": "Note"}).status_code == 400


@pytest.mark.parametrize("body", [b"not-json SECRET", b"[]", b"{\"summary\": NaN}",
                                  b'{"summary":"one","summary":"SECRET"}', b"\xff"])
def test_malformed_json_is_sanitized(isolated, body):
    client, _config, _project = isolated
    response = client.post("/v1/checkpoints", headers={**HEADERS, "Content-Type": "application/json"}, content=body)
    assert response.status_code == 400
    assert "SECRET" not in response.text
    assert TOKEN not in response.text


def test_body_bound_with_content_length_and_chunked_stream(isolated):
    client, _config, _project = isolated
    headers = {**HEADERS, "Content-Type": "application/json"}
    body = b"x" * (bridge.MAX_BODY_BYTES + 1)
    assert client.post("/v1/checkpoints", headers=headers, content=body).status_code == 413
    assert client.post("/v1/checkpoints", headers=headers,
                       content=iter([body[:9000], body[9000:]])).status_code == 413
    assert client.post("/v1/checkpoints", headers=HEADERS, content="{}").status_code == 415
    assert read(client).json()["found"] is False


def test_maximum_labels_and_text_fit_native_task_limit(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config(db_path=str(tmp_path / "memory.db"))
    store = Store(config)
    store.init_db()
    store.close()
    world = "w" * 64
    params = {"world": world, "kind": "handoff", "key": "k" * 64}
    with TestClient(bridge.build_app(config, project, [world], TOKEN)) as client:
        response = client.post("/v1/checkpoints", headers=HEADERS,
                               json={**params, "summary": "x" * 4000, "decisions": ["x" * 500] * 8})
        assert response.status_code == 200
        assert len(response.json()["task"]) <= 200
        assert client.get("/v1/checkpoints", headers=HEADERS, params=params).json()["found"] is True


def test_missing_store_fails_health_reads_writes_without_creating_files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = Config(db_path=str(tmp_path / "must-not-create" / "memory.db"))
    with TestClient(bridge.build_app(config, project, ["survival"], TOKEN)) as client:
        assert client.get("/health", headers=HEADERS).status_code == 503
        assert read(client).status_code == 503
        assert save(client).status_code == 503
    assert not (tmp_path / "must-not-create").exists()


def test_storage_failure_does_not_expose_error_or_credentials(isolated, monkeypatch, capsys):
    client, _config, _project = isolated
    def fail(_self, *args, **kwargs):
        raise RuntimeError("SECRET postgres://user:password@private/database")
    monkeypatch.setattr(NativeService, "status", fail)
    monkeypatch.setattr(NativeService, "session_checkpoint", fail)
    monkeypatch.setattr(NativeService, "session_resume", fail)
    for response in (client.get("/health", headers=HEADERS), save(client), read(client)):
        assert response.status_code == 503
        assert "SECRET" not in response.text
        assert "password" not in response.text
    output = capsys.readouterr()
    assert "SECRET" not in output.out + output.err


def test_service_is_per_request_and_closed_on_the_same_thread(isolated, monkeypatch):
    client, _config, _project = isolated
    lifecycle = []
    class TrackedService(NativeService):
        def __init__(self, config):
            super().__init__(config)
            self.origin_thread = threading.get_ident()
            lifecycle.append(self)
            self.was_closed = False
        def close(self):
            assert threading.get_ident() == self.origin_thread
            self.was_closed = True
            super().close()
    monkeypatch.setattr(bridge, "NativeService", TrackedService)
    assert save(client).status_code == 200
    assert read(client).status_code == 200
    assert client.get("/health", headers=HEADERS).status_code == 200
    assert len(lifecycle) == 3
    assert all(service.was_closed for service in lifecycle)


@pytest.mark.parametrize("token", ["", "x" * 31, " " * 32, "x" * 32 + "\n", "é" * 32])
def test_invalid_tokens_rejected_before_serving(tmp_path, token):
    with pytest.raises(ValueError):
        bridge.build_app(Config(), tmp_path, ["survival"], token)


def test_startup_validates_fixed_project_and_worlds(tmp_path):
    for project in ("relative", tmp_path / "missing"):
        with pytest.raises(ValueError):
            bridge.build_app(Config(), project, ["survival"], TOKEN)
    for worlds in ([], "survival", ["x" * 65], ["../world"], [None]):
        with pytest.raises(ValueError):
            bridge.build_app(Config(), tmp_path, worlds, TOKEN)


def test_cli_validates_config_before_startup_and_never_prints_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GAME_MEMORY_TOKEN", TOKEN)
    with pytest.raises(SystemExit) as error:
        bridge.main(["--config", "relative.yaml", "--project", str(tmp_path), "--world", "survival"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert TOKEN not in output.out + output.err
