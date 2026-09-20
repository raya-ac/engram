"""Configuration inspection remains usable before storage or models exist."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import patch

from fastapi.testclient import TestClient

from engram import __version__
from engram.config import Config
from engram.service import NativeService, operations

ROOT = Path(__file__).resolve().parents[1]


def cli(tmp_path, *args, extra_env=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ENGRAM_") and k not in
           {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.update(extra_env or {})
    # A real CLI process must not import any inference runtime for these commands.
    command = ("import sys; from engram.cli import main; "
               "sys.argv=['engram', *sys.argv[1:]]; main(); "
               "assert not any(n in sys.modules for n in "
               "['torch','transformers','sentence_transformers','mlx.core'])")
    return subprocess.run([sys.executable, "-c", command, *args], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=15)


def test_effective_settings_are_redacted_with_provenance_and_no_database(tmp_path):
    config = tmp_path / "chosen.yaml"
    database = tmp_path / "not-created" / "memory.db"
    config.write_text(f"db_path: {database}\nhf_token: private-hf\n"
                      "web:\n  auth_token: private-web\nretrieval:\n  top_k: 7\n")
    result = cli(tmp_path, "--config", str(config), "config", "show", "--json",
                 extra_env={"ENGRAM_RETRIEVAL_TOP_K": "4"})
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["values"]["retrieval"]["top_k"] == 4
    assert data["sources"]["retrieval.top_k"] == "env:ENGRAM_RETRIEVAL_TOP_K"
    assert data["config_file"] == str(config)
    assert data["values"]["hf_token"] == "<redacted>"
    assert "private-hf" not in result.stdout + result.stderr
    assert "private-web" not in result.stdout + result.stderr
    assert not database.parent.exists()


def test_defaults_and_schema_work_despite_broken_local_config_and_environment(tmp_path):
    (tmp_path / "config.yaml").write_text("retrieval: [invalid]\n")
    env = {"ENGRAM_RETRIEVAL_TOP_K": "invalid", "ENGRAM_HF_TOKEN": "do-not-print"}
    result = cli(tmp_path, "config", "show", "--defaults", "--json", extra_env=env)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["values"]["retrieval"]["top_k"] == Config().retrieval.top_k
    assert data["sources"]["retrieval.top_k"] == "default"
    assert "do-not-print" not in result.stdout
    result = cli(tmp_path, "config", "schema", extra_env=env)
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)["fields"]
    assert schema["retrieval.min_confidence"]["constraints"]["maximum"] == 1
    assert schema["web.auth_token"]["secret"] is True


def test_invalid_and_missing_files_fail_as_clean_json_without_values(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text("web:\n  port: secret-value\n")
    for path in (config, tmp_path / "missing.yaml"):
        result = cli(tmp_path, "--config", str(path), "config", "check", "--json")
        assert result.returncode == 2
        assert json.loads(result.stdout)["valid"] is False
        assert "secret-value" not in result.stdout + result.stderr
        assert "Traceback" not in result.stderr


def test_native_inspection_does_not_open_missing_storage(tmp_path):
    cfg = Config(db_path=str(tmp_path / "absent" / "memory.db"))
    cfg.postgres_dsn = "postgres://private-user:private-password@host/database"
    cfg.llm.api_key = "private-key"
    service = NativeService(cfg)
    result = service.handle_request({"id": 1, "operation": "config_show"})
    assert result["result"]["version"] == __version__
    assert service._store is None
    assert not (tmp_path / "absent").exists()
    assert "private-" not in json.dumps(result)
    assert next(o for o in operations()["operations"] if o["name"] == "config_show")["writes"] is False


def test_mcp_and_web_inspection_use_the_same_secret_boundary(tmp_path):
    from engram.mcp_server import MCPServer
    from engram.web.app import create_app
    cfg = Config(db_path=str(tmp_path / "web.db"))
    cfg.ann.enabled = False
    cfg.web.auth_token = "private-token"
    cfg.hf_token = "private-hf"
    server = MCPServer.__new__(MCPServer)
    server.config = cfg
    report = server._call_tool("config_show", {})
    assert report["version"] == __version__
    assert "private-" not in json.dumps(report)
    with patch("engram.web.app.threading.Thread.start"):
        app = create_app(cfg)
    try:
        with TestClient(app) as client:
            assert client.get("/api/config").status_code == 401
            response = client.get("/api/config", headers={"Authorization": "Bearer private-token"})
            assert response.status_code == 200
            assert response.json()["values"] == report["values"]
            assert "private-" not in response.text
            assert app.version == __version__
    finally:
        app.state.store.close()


def test_version_is_consistent_across_package_cli_and_protocols(tmp_path):
    from engram.mcp_server import MCPServer
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == __version__
    assert cli(tmp_path, "--version").stdout.strip() == f"engram {__version__}"
    server = MCPServer.__new__(MCPServer)
    reply = server.handle_request({"id": 1, "method": "initialize"})
    assert reply["result"]["serverInfo"]["version"] == __version__
    assert operations()["engram_version"] == __version__
    assert operations()["version"] == 1  # Framing protocol stays compatible.
