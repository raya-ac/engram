"""Reusable adapters against actual authenticated bridge/SQLite and malformed peers."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from engram.config import Config
from engram.store import Store
from integrations.checkpoint_client import AsyncCheckpointClient, CheckpointClient, CheckpointError
from integrations.game_server_bridge import build_app

TOKEN = "test-only-token-" + "x" * 32


@pytest.fixture
def app(tmp_path):
    config = Config(db_path=str(tmp_path / "memory.db"))
    config.ann.enabled = False
    config.ann.index_path = str(tmp_path / "unused.index")
    store = Store(config)
    store.init_db()
    store.close()
    return build_app(config, tmp_path, ["world", "other"], TOKEN)


@pytest.mark.asyncio
async def test_async_round_trip_replacement_isolation_and_auth(app):
    transport = httpx.ASGITransport(app=app)
    client = AsyncCheckpointClient("http://127.0.0.1:8422", TOKEN, transport=transport)
    assert (await client.health())["ok"] is True
    assert await client.recall("world", "note", "harbor") == {"found": False, "checkpoints": []}
    await client.save("world", "note", "harbor", "The harbor is open.", next_steps=["Repair the pier"])
    first = await client.recall("world", "note", "harbor")
    assert first["checkpoints"][0]["next_steps"] == ["Repair the pier"]
    await client.save("world", "note", "harbor", "The pier is repaired.")
    assert (await client.recall("world", "note", "harbor"))["checkpoints"][0]["next_steps"] == []
    assert (await client.recall("other", "note", "harbor"))["found"] is False
    assert (await client.recall("world", "rule", "harbor"))["found"] is False
    bad = AsyncCheckpointClient("http://127.0.0.1:8422", "wrong-" + "z" * 32, transport=transport)
    with pytest.raises(CheckpointError) as failure:
        await bad.save("world", "note", "harbor", "rejected")
    assert not failure.value.outcome_unknown
    assert "rejected" not in (await client.recall("world", "note", "harbor"))["checkpoints"][0]["summary"]


def test_sync_round_trip_reopens_http_client_without_losing_storage(app):
    with TestClient(app) as target:
        def handler(request):
            response = target.request(request.method, str(request.url),
                                      headers=dict(request.headers), content=request.content)
            return httpx.Response(response.status_code, content=response.content)
        client = CheckpointClient("http://127.0.0.1:8422", TOKEN, transport=httpx.MockTransport(handler))
        assert client.health()["status"] == "ok"
        client.save("world", "build", "pier", "桟橋 repaired.")
        assert client.recall("world", "build", "pier")["checkpoints"][0]["summary"] == "桟橋 repaired."


@pytest.mark.parametrize("url", ["file:///tmp/store", "http://example.com", "http://localhost:8422",
                                   "https://name:secret@example.com", "https://example.com/path",
                                   "https://example.com?token=secret", "https://example.com/#fragment"])
def test_origin_must_be_explicit_and_credentials_stay_out_of_url(url):
    with pytest.raises(ValueError):
        CheckpointClient(url, TOKEN)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 121])
def test_timeout_is_finite_and_bounded(timeout):
    with pytest.raises(ValueError):
        CheckpointClient("http://127.0.0.1:8422", TOKEN, timeout=timeout)


def test_private_network_opt_in_and_validated_input_before_requests():
    calls = []
    client = CheckpointClient("http://private-bridge:8422", TOKEN, allow_private_http=True,
                              transport=httpx.MockTransport(lambda r: calls.append(r)))
    for args in [("bad world", "note", "key", "text"), ("world", "unknown", "key", "text"),
                 ("world", "note", "../key", ""), ("world", "note", "key", "x" * 4001)]:
        with pytest.raises(ValueError):
            client.save(*args)
    with pytest.raises(ValueError):
        client.save("world", "note", "key", "a", next_steps=["x"] * 9)
    with pytest.raises(ValueError):
        client.save("world", "note", "key", "🐍" * 4000, next_steps=["x" * 500] * 8)
    assert calls == []


@pytest.mark.parametrize("status,body,unknown", [
    (302, b"redirect", False), (401, b"private credentials", False),
    (500, b"private traceback", True), (200, b"not json", True),
    (200, b'{"status":"saved","task":"another-task"}', True),
    (200, b'{"status":"saved","status":"saved"}', True),
    (200, b"x" * 32769, True),
])
def test_unconfirmed_save_and_no_redirect_retry_or_private_error_leak(status, body, unknown):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(status, content=body, headers={"Location": "https://other.example/"})
    client = CheckpointClient("http://127.0.0.1:8422", TOKEN, transport=httpx.MockTransport(handler))
    with pytest.raises(CheckpointError) as failure:
        client.save("world", "note", "key", "private note")
    assert failure.value.outcome_unknown is unknown
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Bearer " + TOKEN
    assert TOKEN not in str(failure.value) and "private" not in str(failure.value)


@pytest.mark.asyncio
async def test_async_total_deadline_does_not_retry_a_write():
    calls = []
    async def delayed(request):
        calls.append(request)
        await asyncio.sleep(1)
        return httpx.Response(200, json={"status": "saved"})
    client = AsyncCheckpointClient("http://127.0.0.1:8422", TOKEN, timeout=.02,
                                   transport=httpx.MockTransport(delayed))
    with pytest.raises(CheckpointError) as failure:
        await client.save("world", "note", "key", "text")
    assert failure.value.outcome_unknown is True
    assert len(calls) == 1


def test_corrupt_scope_or_shape_never_becomes_context():
    for result in [{"found": True, "checkpoints": []}, {"found": 1, "checkpoints": []},
                   {"found": False, "checkpoints": [{"summary": "other user's note"}]},
                   {"found": True, "checkpoints": [{"task": "game:other:note:key", "summary": "private"}]}]:
        client = CheckpointClient("http://127.0.0.1:8422", TOKEN,
                                  transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=result)))
        with pytest.raises(CheckpointError):
            client.recall("world", "note", "key")
