"""Fictional demo flow with real isolated storage and model-free inference."""

import ast
import io
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from engram import cli, demo
from engram.config import Config


@pytest.fixture
def isolated_demo(tmp_path, monkeypatch):
    original = demo.tempfile.mkdtemp
    made = []
    def temporary(**kwargs):
        path = original(dir=tmp_path, **kwargs)
        made.append(Path(path))
        return path
    monkeypatch.setattr(demo.tempfile, "mkdtemp", temporary)
    return made


@pytest.fixture
def fake_models(monkeypatch):
    import engram.embeddings as embeddings
    import engram.retrieval as retrieval
    calls = []
    monkeypatch.setattr(embeddings, "set_backend", lambda backend: calls.append(("backend", backend)))
    monkeypatch.setattr(embeddings, "set_default_model", lambda model: calls.append(("model", model)))
    monkeypatch.setattr(embeddings, "embed_documents", lambda docs, model: np.ones((len(docs), 384), dtype=np.float32) / np.sqrt(384))
    monkeypatch.setattr(retrieval, "embed_query", lambda *args: np.ones(384, dtype=np.float32) / np.sqrt(384))
    def rerank(query, docs, model):
        def score(content):
            if "copper lockbox" in content:
                return -6.0 if len(content.split()) > 160 else 3.0
            return 1.0 if "inspected every Monday" in content else -8.0
        return sorted(enumerate(map(score, docs)), key=lambda pair: -pair[1])
    monkeypatch.setattr(retrieval, "cross_encoder_rerank", rerank)
    return calls


def test_demo_is_python311_syntax_and_uses_fictional_data():
    ast.parse(Path(demo.__file__).read_text(), feature_version=(3, 11))
    contents = " ".join(item[3] for item in demo.MEMORIES)
    assert "Ari" not in contents and "melee.garden" not in contents
    assert "fictional" in contents


def test_full_model_free_flow_keeps_real_confidence_and_readonly_explanation(isolated_demo, fake_models):
    output = []
    report = demo.run_demo(yes=True, keep_db=True, output_fn=output.append)
    directory = Path(report["directory"])
    config = json.loads((directory / "config.json").read_text())
    assert config["retrieval"]["min_confidence"] == Config().retrieval.min_confidence
    assert config["retrieval"]["rerank_passage_fallback"] is True
    assert report["explanation_unchanged"] is True
    assert report["checkpoint"]["task"] == "release rehearsal"
    explanation = json.loads((directory / "explanation.json").read_text())
    rows = {row["memory_id"]: row for row in explanation["candidates"]}
    assert rows["lunch"]["outcome"] == "below_confidence"
    assert rows["access-key"]["passage"]["excerpt_raw_score"] == 3.0
    assert report["reranked_ids"][0] == "access-key"
    assert ("backend", "sentence_transformers") in fake_models
    assert "<redacted>" in (directory / "config-report.json").read_text()
    for filename in ("config.json", "memory.db", "config-report.json", "explanation.json"):
        assert stat.S_IMODE((directory / filename).stat().st_mode) == 0o600
    assert any(line.startswith("Recall: ") and str(directory / "config.json") in line for line in output)
    assert any(line.startswith("Doctor: ") and "doctor --full" in line for line in output)
    assert any("engram init" in line for line in output)
    assert any("source excerpt:" in line and "copper lockbox" in line for line in output)
    assert any("deliberately padded" in line for line in output)


def test_hostile_environment_and_real_config_never_affect_demo(tmp_path, isolated_demo, fake_models, monkeypatch):
    real_db = tmp_path / "real-user.db"
    real_db.write_bytes(b"do not open or modify")
    real_config = tmp_path / "config.yaml"
    real_config.write_text("malformed: [never read")
    monkeypatch.chdir(tmp_path)
    env = {"ENGRAM_STORAGE_BACKEND": "postgres", "ENGRAM_POSTGRES_DSN": "private-secret-dsn",
           "ENGRAM_DB_PATH": str(real_db), "ENGRAM_ANN_INDEX_PATH": str(tmp_path / "real.index"),
           "ENGRAM_EMBEDDING_BACKEND": "voyage", "ENGRAM_RETRIEVAL_MIN_CONFIDENCE": "nan",
           "ENGRAM_WEB_HOST": "0.0.0.0", "ENGRAM_LLM_API_KEY": "private-secret-key"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(Config, "load", lambda *args: pytest.fail("demo read user configuration"))
    output = []
    report = demo.run_demo(yes=True, keep_db=True, output_fn=output.append)
    cfg = json.loads(Path(report["config_file"]).read_text())
    assert cfg["storage_backend"] == "sqlite"
    assert cfg["web"]["host"] == "127.0.0.1"
    assert cfg["embedding_backend"] == "sentence_transformers"
    assert cfg["dormant_recall"]["mode"] == "off"
    assert real_db.read_bytes() == b"do not open or modify"
    assert real_config.read_text() == "malformed: [never read"
    assert not (tmp_path / "real.index").exists()
    assert "private-secret" not in "\n".join(output)
    assert all(os.environ[key] == value for key, value in env.items())
    command = shlex.split(next(line.removeprefix("Inspect: ") for line in output if line.startswith("Inspect: ")))
    assert command[:3] == ["env", "-u", sorted(env)[0]]
    assert "ENGRAM_DB_PATH" in command


def test_completion_and_failure_cleanup(isolated_demo, fake_models, monkeypatch):
    demo.run_demo(yes=True, output_fn=lambda line: None)
    assert not isolated_demo[-1].exists()
    import engram.embeddings as embeddings
    def broken(*args):
        raise RuntimeError("private-secret-provider-error")
    monkeypatch.setattr(embeddings, "embed_documents", broken)
    with pytest.raises(demo.DemoError) as error:
        demo.run_demo(yes=True, output_fn=lambda line: None)
    assert "private-secret" not in str(error.value)
    assert not isolated_demo[-1].exists()


def test_mutating_explanation_fails_and_cleans_up(isolated_demo, fake_models, monkeypatch):
    import engram.retrieval as retrieval
    original = retrieval.search
    def mutate(*args, **kwargs):
        answer = original(*args, **kwargs)
        if kwargs.get("debug"):
            args[1].record_access("access-key", "bad diagnostic write")
        return answer
    monkeypatch.setattr(retrieval, "search", mutate)
    with pytest.raises(demo.DemoError, match="explanation changed"):
        demo.run_demo(yes=True, output_fn=lambda line: None)
    assert not isolated_demo[-1].exists()


def test_interrupt_cleans_up_and_keep_retains_partial_files(isolated_demo):
    def interrupt(prompt):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        demo.run_demo(input_fn=interrupt, output_fn=lambda line: None)
    assert not isolated_demo[-1].exists()
    output = []
    with pytest.raises(KeyboardInterrupt):
        demo.run_demo(keep_db=True, input_fn=interrupt, output_fn=output.append)
    assert (isolated_demo[-1] / "config.json").is_file()
    assert any("Kept partial" in line for line in output)


def test_noninteractive_requires_yes_before_creating_directory(isolated_demo, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(demo.DemoError, match="--yes"):
        demo.run_demo()
    assert isolated_demo == []


class Child:
    def __init__(self):
        self.running = True
        self.terminated = False
        self.killed = False
        self.waits = []
    def poll(self):
        return None if self.running else 0
    def terminate(self):
        self.terminated = True
    def kill(self):
        self.killed = True
    def wait(self, timeout):
        self.waits.append(timeout)
        self.running = False
        return 0


class PortProbe:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def bind(self, address):
        assert address[0] == "127.0.0.1"


def test_web_child_uses_isolated_config_current_package_and_scrubbed_environment(tmp_path, monkeypatch):
    cfg = Config(db_path=str(tmp_path / "demo.db"))
    cfg.web.auth_token = "synthetic-demo-token"
    cfg.web.port = 8421
    for key in ("ENGRAM_DB_PATH", "ENGRAM_STORAGE_BACKEND", "ENGRAM_WEB_HOST", "OPENAI_API_KEY", "PGPASSWORD", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.setenv(key, "private-secret")
    calls = []
    child = Child()
    monkeypatch.setattr(demo.subprocess, "Popen", lambda command, **kwargs: calls.append((command, kwargs)) or child)
    monkeypatch.setattr(demo.socket, "socket", PortProbe)
    monkeypatch.setattr(demo, "_web_ready", lambda *args: True)
    config_path = tmp_path / "config.json"
    process, url = demo._start_web(cfg, config_path, tmp_path)
    command, options = calls[0]
    assert command[:5] == [sys.executable, "-m", "engram", "--config", str(config_path)]
    assert options["cwd"] == Path(demo.__file__).resolve().parents[1]
    assert not any(key.startswith("ENGRAM_") or key in demo._PROVIDER_ENV for key in options["env"])
    assert process is child and url == "http://127.0.0.1:8421/?token=synthetic-demo-token"
    assert os.environ["ENGRAM_DB_PATH"] == "private-secret"


def test_web_startup_timeout_stops_child(tmp_path, monkeypatch):
    child = Child()
    monkeypatch.setattr(demo.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(demo.socket, "socket", PortProbe)
    cfg = Config(db_path=str(tmp_path / "demo.db"))
    with pytest.raises(demo.DemoError, match="timed out"):
        demo._start_web(cfg, tmp_path / "config.json", tmp_path, timeout=0)
    assert child.terminated and child.waits == [5]


def test_stubborn_web_child_is_killed_with_bounded_wait():
    child = Child()
    def wait(timeout):
        child.waits.append(timeout)
        if not child.killed:
            raise subprocess.TimeoutExpired("demo", timeout)
        child.running = False
    child.wait = wait
    demo._stop_web(child)
    assert child.terminated and child.killed and child.waits == [5, 5]


def test_web_readiness_checks_exact_demo_identity_and_authentication(tmp_path, monkeypatch):
    config_path, db_path = tmp_path / "config.json", tmp_path / "memory.db"
    report = {"config_file": str(config_path), "values": {"db_path": str(db_path), "storage_backend": "sqlite"}}
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return io.StringIO(json.dumps(report))
    monkeypatch.setattr(demo, "build_opener", lambda *args: Opener())
    assert demo._web_ready("http://127.0.0.1:8421/", "demo-token", config_path, db_path)
    assert requests[0][0].get_header("Authorization") == "Bearer demo-token"
    assert requests[0][1] == 0.5
    report["values"]["db_path"] = "/different/store.db"
    assert not demo._web_ready("http://127.0.0.1:8421/", "demo-token", config_path, db_path)


@pytest.mark.parametrize("yes", [False, True])
def test_web_final_prompt_and_cleanup_even_when_kept(isolated_demo, fake_models, monkeypatch, yes):
    child = Child()
    monkeypatch.setattr(demo, "_start_web", lambda *args: (child, "http://127.0.0.1:8421/?token=demo"))
    prompts = []
    report = demo.run_demo(yes=yes, keep_db=True, start_web=True,
                           input_fn=lambda prompt: prompts.append(prompt), output_fn=lambda line: None)
    assert child.terminated
    assert Path(report["directory"]).exists()
    assert (not prompts) if yes else ("inspect the demo" in prompts[-1])


def test_interrupt_at_web_inspection_stops_child_and_cleans_files(isolated_demo, fake_models, monkeypatch):
    child = Child()
    monkeypatch.setattr(demo, "_start_web", lambda *args: (child, "http://127.0.0.1:8421/"))
    def pause(prompt):
        if "inspect the demo" in prompt:
            raise KeyboardInterrupt
        return ""
    with pytest.raises(KeyboardInterrupt):
        demo.run_demo(start_web=True, input_fn=pause, output_fn=lambda line: None)
    assert child.terminated
    assert not isolated_demo[-1].exists()


def test_cli_demo_bypasses_config_loading(monkeypatch):
    calls = []
    monkeypatch.setattr(Config, "load", lambda *args: pytest.fail("CLI loaded user config"))
    monkeypatch.setattr(demo, "run_demo", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["engram", "demo", "--yes", "--keep", "--web", "--port", "8847"])
    cli.main()
    assert calls == [{"keep_db": True, "start_web": True, "web_port": 8847, "yes": True}]


def test_cli_rejects_explicit_config_without_reading_it(monkeypatch):
    monkeypatch.setattr(Config, "load", lambda *args: pytest.fail("CLI loaded user config"))
    monkeypatch.setattr(sys, "argv", ["engram", "--config", "private-missing.yaml", "demo", "--yes"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_interrupt_exits_without_traceback(monkeypatch):
    def interrupted(**kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(demo, "run_demo", interrupted)
    monkeypatch.setattr(sys, "argv", ["engram", "demo", "--yes"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 130
