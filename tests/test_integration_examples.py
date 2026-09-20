"""Example clients: bounded protocol, explicit writes and isolated native storage."""
import importlib
import json
import os
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "integrations"))
    return importlib.import_module("native_client"), importlib.import_module("task_assistant")


@pytest.fixture
def fake(examples, tmp_path, monkeypatch):
    package = tmp_path / "engram"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text('''
import json, os, sys, time
if os.environ.get("EXAMPLE_NO_READ"):
    time.sleep(20)
for raw in sys.stdin.buffer:
    request = json.loads(raw)
    operation, ident = request["operation"], request["id"]
    with open(os.environ["EXAMPLE_REQUEST_LOG"], "a") as log:
        log.write(json.dumps(request) + "\\n")
    if operation == "hang":
        time.sleep(20)
    if operation == "exit":
        sys.exit(0)
    if operation == "stderr":
        sys.stderr.write("diagnostic" * 20000)
        sys.stderr.flush()
    special = {
        "wrong_id": {"id": "wrong", "result": {}},
        "both": {"id": ident, "result": {}, "error": {}},
        "bad_error": {"id": ident, "error": {"code": 4, "message": "bad"}},
        "fail": {"id": ident, "error": {"code": "invalid_params", "message": "Rejected parameters"}},
        "oversize": {"id": ident, "result": "x" * 6000},
    }
    response = json.dumps(special.get(operation, {"id": ident, "result": {
        "params": request["params"], "argv": sys.argv[1:], "db_override": os.getenv("ENGRAM_DB_PATH")}}))
    if operation == "nan":
        response = '{"id":"' + ident + '","result":NaN}'
    if operation == "infinite":
        response = '{"id":"' + ident + '","result":1e400}'
    if operation == "duplicate":
        response = '{"id":"wrong","id":"' + ident + '","result":{}}'
    if operation == "invalid_utf8":
        sys.stdout.buffer.write(b"\\xff\\n")
        sys.stdout.buffer.flush()
    elif operation == "partial":
        sys.stdout.write(response)
        sys.stdout.flush()
        sys.exit(0)
    else:
        print(response, flush=True)
''')
    config = tmp_path / "config ; with spaces.json"
    config.write_text(json.dumps({"db_path": str(tmp_path / "unused.db")}))
    log = tmp_path / "requests.jsonl"
    environment = dict(os.environ, PYTHONPATH=str(tmp_path),
                       ENGRAM_DB_PATH=str(tmp_path / "explicit-override.db"),
                       EXAMPLE_REQUEST_LOG=str(log))
    monkeypatch.chdir(tmp_path)
    def create(**kwargs):
        return examples[0].NativeClient(config=config, python=sys.executable, env=environment, **kwargs)
    return create, log, config, environment


def test_native_client_preserves_paths_environment_and_sequential_mapping(fake):
    create, log, config, environment = fake
    with create() as client:
        first = client.call("echo", note="café\nsecond line")
        second = client.call("echo", value=2)
        assert first["argv"] == ["--config", str(config), "api"]
        assert first["db_override"] == environment["ENGRAM_DB_PATH"]
        assert first["params"]["note"] == "café\nsecond line"
        assert second["params"] == {"value": 2}
        process = client._process
    assert process.poll() == 0
    assert [json.loads(line)["id"] for line in log.read_text().splitlines()] == ["request-1", "request-2"]
    assert not Path(environment["ENGRAM_DB_PATH"]).exists()


@pytest.mark.parametrize("operation", ["wrong_id", "both", "bad_error", "nan", "infinite", "duplicate", "invalid_utf8", "partial", "oversize"])
def test_native_client_rejects_corrupt_responses_and_closes(fake, examples, operation):
    with fake[0](max_response_bytes=512) as client:
        with pytest.raises(examples[0].NativeClientError) as caught:
            client.call(operation)
        assert caught.value.code == "protocol_error"
        assert client._process.poll() is not None


def test_native_client_valid_api_error_is_not_success_and_next_call_works(fake, examples):
    with fake[0]() as client:
        with pytest.raises(examples[0].NativeClientError) as caught:
            client.call("fail")
        assert caught.value.code == "invalid_params"
        assert client.call("echo", ok=True)["params"] == {"ok": True}


def test_native_client_rejects_nonfinite_and_oversize_requests_before_send(fake, examples):
    with fake[0]() as client:
        for value, expected in ((float("nan"), "invalid_request"), ("private-marker" * 6000, "request_too_large")):
            with pytest.raises(examples[0].NativeClientError) as caught:
                client.call("echo", value=value)
            assert caught.value.code == expected
            assert "private-marker" not in str(caught.value)
        assert not fake[1].exists()
        assert client.call("echo", okay=1)["params"] == {"okay": 1}


@pytest.mark.parametrize("does_not_read", [False, True])
def test_native_client_times_out_writes_and_reads_and_reaps_child(fake, examples, does_not_read):
    if does_not_read:
        fake[3]["EXAMPLE_NO_READ"] = "1"
    started = time.monotonic()
    with fake[0](timeout=0.15) as client:
        with pytest.raises(examples[0].NativeClientError) as caught:
            client.call("hang", note="x" * 64000)
        assert caught.value.code == "timeout"
        assert client._process.poll() is not None
    assert time.monotonic() - started < 4


def test_native_client_stderr_does_not_deadlock_or_enter_results(fake):
    with fake[0](timeout=5) as client:
        assert client.call("stderr")["params"] == {}


def test_native_client_eof_and_caller_exception_cleanup(fake, examples):
    with fake[0]() as client:
        with pytest.raises(examples[0].NativeClientError) as caught:
            client.call("exit")
        assert caught.value.code == "process_exit"
    with pytest.raises(RuntimeError):
        with fake[0]() as interrupted:
            interrupted.call("echo")
            raise RuntimeError("caller failed")
    assert interrupted._process.poll() is not None


def test_task_example_requires_explicit_write_confirmation(examples, tmp_path):
    with pytest.raises(SystemExit) as caught:
        examples[1].main(["--config", str(tmp_path / "not-read"), "--python", sys.executable,
                          "--project", str(tmp_path), "save", "issue:DEMO-7",
                          "--summary-file", str(tmp_path / "not-read-either")])
    assert caught.value.code == 2


def test_task_example_real_scoped_checkpoint_survives_new_process(examples, tmp_path, monkeypatch, capsys):
    from engram.config import Config
    from engram.store import Store, Memory
    # Fully isolate the subprocess from inherited user storage/provider overrides.
    for key in list(os.environ):
        if key.startswith("ENGRAM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    project = tmp_path / "project"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    db = tmp_path / "example.db"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"db_path": str(db), "ann": {"enabled": False},
                                      "dormant_recall": {"mode": "off"}}))
    config = Config.from_mapping({"db_path": str(db), "ann": {"enabled": False}}, apply_environment=False)
    store = Store(config)
    store.init_db()
    store.save_memory(Memory(id="owned", content="Owned project fact", metadata={"project_path": str(project)}, access_count=2))
    store.save_memory(Memory(id="other", content="Other project fact", metadata={"project_path": str(other)}))
    before = store.get_memory("owned")
    summary = tmp_path / "reviewed.txt"
    summary.write_text("Task verification is pending.")
    prefix = ["--config", str(config_path), "--python", sys.executable, "--project", str(project)]
    try:
        assert examples[1].main(prefix + ["save", "issue:DEMO-7", "--summary-file", str(summary),
                                       "--next-step", "Run the documented check", "--yes"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "saved"
        assert examples[1].main(prefix + ["resume", "issue:DEMO-7"]) == 0
        resumed = json.loads(capsys.readouterr().out)
        assert [row["id"] for row in resumed["memories"]] == ["owned"]
        assert resumed["checkpoints"][0]["summary"] == "Task verification is pending."
        assert resumed["checkpoints"][0]["next_steps"] == ["Run the documented check"]
        assert examples[1].main(prefix + ["resume", "issue:ANOTHER"]) == 0
        assert json.loads(capsys.readouterr().out)["checkpoints"] == []
        after = store.get_memory("owned")
        assert (after.access_count, after.last_accessed, after.importance) == (before.access_count, before.last_accessed, before.importance)
        assert examples[1].main(prefix + ["clear", "issue:DEMO-7", "--yes"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "cleared"
        assert examples[1].main(prefix + ["resume", "issue:DEMO-7"]) == 0
        assert json.loads(capsys.readouterr().out)["checkpoints"] == []
    finally:
        store.close()
