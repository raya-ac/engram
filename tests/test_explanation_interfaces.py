"""The same diagnostic contract through public interfaces, with fake models."""

from argparse import Namespace
import copy
from io import StringIO
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from engram import embeddings, retrieval
from engram.cli import cmd_search
from engram.config import Config
from engram.mcp_server import MCPServer
from engram.service import NativeService, run_stdio
from engram.store import Memory, Store
from engram.web.app import create_app


QUERY = "Where is the maintenance key?"
TOKEN = "isolated-explanation-interface-token"


@pytest.fixture
def context(tmp_path, monkeypatch):
    config = Config(db_path=str(tmp_path / "interfaces.sqlite"))
    config.ann.enabled = False
    config.ann.index_path = str(tmp_path / "unused.index")
    config.web.auth_token = TOKEN
    config.retrieval.min_confidence = 0.6
    config.retrieval.rerank_passage_fallback = False
    config.dormant_recall.mode = "shadow"
    store = Store(config)
    store.init_db()
    state = SimpleNamespace(config=config, store=store, candidates=[], scores={}, calls=[], noisy=False)

    def add(mid, content, *, raw=-1.0, **kwargs):
        store.save_memory(Memory(id=mid, content=content, layer="semantic", fact_date="2026-01-01",
                                 memory_type=kwargs.pop("memory_type", "fact"), **kwargs))
        state.candidates.append((mid, 0.9 - 0.01 * len(state.candidates)))
        state.scores[content] = raw

    state.add = add
    add("key", "The maintenance key is in the amber drawer.")
    add("forgotten", "PRIVATE forgotten contents", forgotten=True)
    add("inactive", "PRIVATE superseded contents", status="superseded")
    monkeypatch.setattr(retrieval, "_dense_search", lambda *_: list(state.candidates))
    for name in ("_bm25_search", "_graph_search", "_hopfield_search"):
        monkeypatch.setattr(retrieval, name, lambda *_: [])

    def fake_rerank(query, docs, model):
        state.calls.append((query, list(docs), model))
        if state.noisy:
            print("test model diagnostic: warming up")
        return sorted(enumerate(state.scores[doc] for doc in docs), key=lambda pair: -pair[1])

    monkeypatch.setattr(retrieval, "cross_encoder_rerank", fake_rerank)
    monkeypatch.setattr(embeddings, "set_backend", Mock())
    monkeypatch.setattr(embeddings, "set_default_model", Mock())
    yield state
    store.close()


def rpc(server, **arguments):
    response = server.handle_request({
        "jsonrpc": "2.0", "id": "explain", "method": "tools/call",
        "params": {"name": "recall_explain", "arguments": arguments},
    })
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


@pytest.fixture(params=["cli", "native", "mcp", "web_debug", "web_explain"])
def interface(request, context, monkeypatch, capsys):
    state = context
    adapter = SimpleNamespace(name=request.param, close=lambda: None, stores=[state.store])
    if request.param == "cli":
        def invoke(query=QUERY, top_k=5):
            capsys.readouterr()
            cmd_search(Namespace(query=[query], top_k=top_k, debug=True, rerank=True,
                                 json_output=True), state.config)
            captured = capsys.readouterr()
            adapter.stderr = captured.err
            return json.loads(captured.out)
    elif request.param == "native":
        service = NativeService(state.config)
        adapter.stores.append(service.store)
        adapter.close = service.close

        def invoke(query=QUERY, top_k=5):
            result = service.handle_request({"id": 1, "operation": "search_explain",
                                             "params": {"query": query, "top_k": top_k}})
            assert "error" not in result, result
            return result["result"]
    elif request.param == "mcp":
        server = object.__new__(MCPServer)
        server.config, server.store, server._reranker = state.config, state.store, None
        server._sweep_working = Mock(side_effect=AssertionError("diagnostic swept working memory"))
        server._refresh_session_handoff = Mock(side_effect=AssertionError("diagnostic refreshed handoff"))
        adapter.server = server

        def invoke(query=QUERY, top_k=5):
            return rpc(server, query=query, top_k=top_k)
    else:
        # Exercise real app auth/routing/storage, without the model warmup thread.
        with patch("engram.web.app.threading.Thread.start"):
            app = create_app(state.config)
        client = TestClient(app)
        client.__enter__()
        adapter.client = client
        adapter.path = "/api/search" if request.param == "web_debug" else "/api/search/explain"
        adapter.stores.append(app.state.store)

        def close():
            client.__exit__(None, None, None)
            app.state.store.close()
        adapter.close = close

        def invoke(query=QUERY, top_k=5):
            response = client.get(adapter.path, params={"q": query, "top_k": top_k, "debug": "true"},
                                  headers={"Authorization": f"Bearer {TOKEN}"})
            assert response.status_code == 200, response.text
            return response.json()

    adapter.invoke = invoke
    # Construction is outside the read-only operation. A diagnostic must not
    # initialize schema, read/write result caches, record access, or emit events.
    adapter.before = list(state.store.conn.iterdump())
    adapter.cache_before = []
    for store in adapter.stores:
        store._search_cache[("sentinel",)] = (store._search_cache_version, [{"sentinel": True}])
        adapter.cache_before.append(copy.deepcopy(store._search_cache))
    for name in ("init_db", "record_search", "set_search_cache", "get_search_cache"):
        monkeypatch.setattr(Store, name, Mock(side_effect=AssertionError(f"diagnostic called {name}")))
    monkeypatch.setattr("engram.dormant.evaluate_shadow",
                        Mock(side_effect=AssertionError("diagnostic evaluated dormant memories")))
    monkeypatch.setattr("engram.web.routes.push_event",
                        Mock(side_effect=AssertionError("diagnostic pushed a web event")))
    yield adapter
    adapter.close()


def without_latency(report):
    result = copy.deepcopy(report)
    result.pop("latency_ms")
    return result


def test_rejected_candidates_share_core_report_and_leave_storage_unchanged(context, interface):
    expected_results, dbg = retrieval.search(QUERY, context.store, context.config, top_k=5, debug=True)
    assert expected_results == []
    payload = interface.invoke()
    assert payload["results"] == []
    assert without_latency(payload["explanation"]) == without_latency(dbg.to_dict())
    rows = {row["memory_id"]: row for row in payload["explanation"]["candidates"]}
    assert rows["key"]["outcome"] == "below_confidence"
    assert rows["key"]["confidence"]["threshold"] == 0.6
    assert rows["key"]["content"] == "The maintenance key is in the amber drawer."
    assert rows["forgotten"]["outcome"] == "forgotten"
    assert rows["inactive"]["outcome"] == "inactive"
    assert "PRIVATE" not in json.dumps(payload, allow_nan=False)
    assert TOKEN not in json.dumps(payload)
    assert list(context.store.conn.iterdump()) == interface.before
    for store, before in zip(interface.stores, interface.cache_before):
        assert store._search_cache == before
    if interface.name == "mcp":
        interface.server._sweep_working.assert_not_called()
        interface.server._refresh_session_handoff.assert_not_called()


def test_no_candidate_is_valid_json_with_empty_explanation(context, interface):
    context.candidates.clear()
    payload = interface.invoke()
    assert payload["results"] == []
    assert payload["explanation"]["candidates"] == []
    assert payload["explanation"]["final_ids"] == []
    assert payload["explanation"]["counts"]["considered"] == 0
    assert context.calls == []
    json.dumps(payload, allow_nan=False)


def test_returned_candidate_diagnostics_do_not_reinforce(context, interface):
    context.scores["The maintenance key is in the amber drawer."] = 3.0
    payload = interface.invoke()
    assert [row["id"] for row in payload["results"]] == ["key"]
    assert payload["explanation"]["final_ids"] == ["key"]
    assert list(context.store.conn.iterdump()) == interface.before
    assert context.store.get_memory("key").access_count == 0


def test_mcp_profile_filter_protects_content(context, monkeypatch):
    context.add("narrative", "PRIVATE profile-excluded narrative", memory_type="narrative", raw=5.0)
    server = object.__new__(MCPServer)
    server.config, server.store, server._reranker = context.config, context.store, None
    server._sweep_working = Mock(side_effect=AssertionError("sweep"))
    server._refresh_session_handoff = Mock(side_effect=AssertionError("handoff"))
    monkeypatch.setattr("engram.dormant.evaluate_shadow", Mock(side_effect=AssertionError("shadow")))
    before = list(context.store.conn.iterdump())
    payload = rpc(server, query=QUERY, mode="facts_only")
    rows = {row["memory_id"]: row for row in payload["explanation"]["candidates"]}
    assert rows["narrative"]["outcome"] == "profile_filtered"
    assert "content" not in rows["narrative"]
    assert "PRIVATE" not in json.dumps(payload)
    assert list(context.store.conn.iterdump()) == before


def test_cli_json_output_keeps_model_diagnostics_on_stderr(context, capsys):
    context.noisy = True
    cmd_search(Namespace(query=[QUERY], top_k=5, debug=True, rerank=True, json_output=True), context.config)
    captured = capsys.readouterr()
    assert json.loads(captured.out)["results"] == []
    assert "test model diagnostic" in captured.err


@pytest.mark.parametrize("flag", ["--explain", "--debug"])
def test_cli_explain_aliases_reach_the_same_core_contract(context, monkeypatch, capsys, flag):
    from engram.cli import main
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, *_: context.config))
    monkeypatch.setattr(sys, "argv", ["engram", "search", QUERY, flag, "--rerank", "--json"])
    before = list(context.store.conn.iterdump())
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"] == []
    assert payload["explanation"]["candidates"][0]["outcome"] == "below_confidence"
    assert list(context.store.conn.iterdump()) == before


@pytest.mark.parametrize("kind", ["cli", "native"])
def test_explanation_does_not_create_an_uninitialized_store(tmp_path, kind):
    path = tmp_path / "missing-directory" / "memory.sqlite"
    config = Config(db_path=str(path))
    if kind == "cli":
        with pytest.raises(ValueError, match="initialized store"):
            cmd_search(Namespace(query=[QUERY], top_k=5, debug=True, rerank=True,
                                 json_output=True), config)
    else:
        service = NativeService(config)
        try:
            response = service.handle_request({"id": 1, "operation": "search_explain",
                                               "params": {"query": QUERY}})
            assert response["error"]["code"] == "operation_failed"
        finally:
            service.close()
    assert not path.parent.exists()


def test_native_jsonl_explanation_keeps_model_diagnostics_off_protocol(context, capsys):
    context.noisy = True
    output = StringIO()
    frames = StringIO(json.dumps({"id": "explain", "operation": "search_explain",
                                 "params": {"query": QUERY}}) + "\n")
    run_stdio(context.config, frames, output)
    payload = json.loads(output.getvalue())
    assert payload["id"] == "explain"
    assert payload["result"]["results"] == []
    assert payload["result"]["explanation"]["candidates"][0]["outcome"] == "below_confidence"
    assert "test model diagnostic" in capsys.readouterr().err


@pytest.mark.parametrize("top_k", [0, -1, 21, True, 1.5])
def test_native_invalid_limits_are_structured_errors_without_model_calls(context, top_k):
    service = NativeService(context.config)
    try:
        response = service.handle_request({"id": 1, "operation": "search_explain",
                                           "params": {"query": QUERY, "top_k": top_k}})
        assert response["error"]["code"] == "invalid_params"
        assert context.calls == []
        assert service._store is None
    finally:
        service.close()


def test_native_uses_configured_limit_and_validates_it_before_loading(context):
    context.config.retrieval.top_k = 3
    service = NativeService(context.config)
    try:
        report = service.search_explain(QUERY)
        assert report["explanation"]["settings"]["top_k"] == 3
    finally:
        service.close()
    context.config.retrieval.top_k = 21
    service = NativeService(context.config)
    try:
        with pytest.raises(ValueError, match="1 to 20"):
            service.search_explain(QUERY)
        assert service._store is None
    finally:
        service.close()


@pytest.mark.parametrize("path", ["/api/search", "/api/search/explain"])
def test_web_explanation_requires_auth_and_rejects_invalid_inputs(context, path):
    with patch("engram.web.app.threading.Thread.start"):
        app = create_app(context.config)
    before = list(app.state.store.conn.iterdump())
    try:
        with TestClient(app) as client:
            params = {"q": QUERY, "debug": "true"}
            assert client.get(path, params=params).status_code == 401
            assert client.get(path, params=params, headers={"Authorization": "Bearer wrong"}).status_code == 401
            headers = {"Authorization": f"Bearer {TOKEN}"}
            for limit in (0, -1, 101, "invalid"):
                assert client.get(path, params={**params, "top_k": limit}, headers=headers).status_code == 422
            assert client.get(path, params={"q": "", "debug": "true"}, headers=headers).status_code == 422
            assert client.get(path, params={"q": " ", "debug": "true"}, headers=headers).status_code == 400
        assert context.calls == []
        assert list(app.state.store.conn.iterdump()) == before
    finally:
        app.state.store.close()
