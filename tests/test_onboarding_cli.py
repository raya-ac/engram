"""Public CLI setup/diagnostics across real process and SQLite boundaries."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from engram.config import Config


ROOT = Path(__file__).resolve().parents[1]


def cli(directory, *arguments, extra_env=None):
    aliases = {alias for field in Config.schema()["fields"].values() for alias in field["aliases"]}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ENGRAM_") and key not in aliases}
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TMPDIR=str(directory))
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", "engram", *arguments], cwd=directory,
                          input="", env=env, text=True, capture_output=True, timeout=30)


def test_new_install_json_then_real_local_mcp_check(tmp_path):
    config = tmp_path / "settings.yaml"
    database = tmp_path / "memory.db"
    result = cli(tmp_path, "--config", str(config), "init", "--yes", "--preset", "portable",
                 "--db-path", str(database), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["database_initialized"] is True
    assert report["config_file"] == str(config)
    assert report["agent_setup"]["mcp_config"]["mcpServers"]["engram"]["args"][-2:] == ["serve", "--mcp"]
    from engram.store import Store, Memory
    store = Store(Config.from_mapping({"db_path": str(database)}, apply_environment=False))
    store.save_memory(Memory(id="kept", content="existing note stays intact"))
    store.close()
    with sqlite3.connect(database) as connection:
        before = list(connection.iterdump())
    original_config = config.read_bytes()
    checked = cli(tmp_path, "--config", str(config), "doctor", "--check-connection", "--json")
    assert checked.returncode == 0, checked.stderr
    checks = {item["name"]: item for item in json.loads(checked.stdout)["checks"]}
    assert checks["storage"]["status"] == "pass"
    assert checks["mcp_connection"]["status"] == "pass"
    assert checks["mcp_connection"]["details"]["external_agent_connection_verified"] is False
    assert checks["embedding_inference"]["status"] == "skipped"
    with sqlite3.connect(database) as connection:
        assert list(connection.iterdump()) == before
    assert config.read_bytes() == original_config
    repeated = cli(tmp_path, "--config", str(config), "init", "--yes", "--json")
    assert repeated.returncode == 2
    assert json.loads(repeated.stdout)["ok"] is False
    assert config.read_bytes() == original_config


def test_doctor_invalid_configuration_has_structured_failure(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("retrieval:\n  top_k: 'private-invalid-value'\n")
    result = cli(tmp_path, "--config", str(config), "doctor", "--json")
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["checks"][0]["name"] == "configuration"
    assert "private-invalid-value" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


def test_missing_storage_doctor_fails_without_initializing_it(tmp_path):
    config = tmp_path / "missing-store.yaml"
    absent = tmp_path / "absent" / "memory.db"
    config.write_text(f"db_path: {absent}\nembedding_backend: sentence_transformers\n")
    result = cli(tmp_path, "--config", str(config), "doctor", "--json")
    assert result.returncode == 1
    checks = {item["name"]: item for item in json.loads(result.stdout)["checks"]}
    assert checks["storage"]["status"] == "fail"
    assert not absent.parent.exists()


def test_init_noninteractive_requires_explicit_defaults(tmp_path):
    config = tmp_path / "new.yaml"
    result = cli(tmp_path, "--config", str(config), "init")
    assert result.returncode == 2
    assert "--yes" in result.stderr
    assert not config.exists()
