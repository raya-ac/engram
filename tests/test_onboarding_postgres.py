"""Optional PostgreSQL onboarding acceptance against an explicitly supplied DB.

ENGRAM_TEST_POSTGRES_DSN must point to a disposable test database. Each test owns
one random schema and removes only that schema, even when an assertion fails.
No model is loaded or downloaded. The option-preservation unit checks are offline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from engram.config import Config
from engram import doctor as diagnostic


@pytest.mark.parametrize("dsn_options,environment_options,expected", [
    ("-c search_path=chosen_schema -c application_name=doctor_probe", "-c search_path=wrong_schema",
     "-c search_path=chosen_schema -c application_name=doctor_probe"),
    (None, "-c search_path=environment_schema", "-c search_path=environment_schema"),
    ("", "-c search_path=must_not_apply", ""),
])
def test_postgres_preserves_effective_options_before_diagnostic_limits(monkeypatch, dsn_options, environment_options, expected):
    captured = {}
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, sql): return SimpleNamespace(fetchall=lambda: [])
    def connect(dsn, **kwargs):
        captured.update(kwargs)
        return Connection()
    parsed = {} if dsn_options is None else {"options": dsn_options}
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(
        connect=connect, conninfo=SimpleNamespace(conninfo_to_dict=lambda dsn: parsed)))
    monkeypatch.setenv("PGOPTIONS", environment_options)
    config = Config(storage_backend="postgres", postgres_dsn="postgresql://example.invalid/test")
    # Missing schema is expected in this fake; the check still must retain all options.
    assert diagnostic._storage_check(config, True)["status"] == "fail"
    assert captured["options"] == (expected + " -c default_transaction_read_only=on -c statement_timeout=3000").strip()
    assert captured["connect_timeout"] == 3


def _assert_no_credentials(text, secrets):
    if any(secret and secret in text for secret in secrets):
        pytest.fail("a credential appeared in onboarding output", pytrace=False)


def _cli(config_path, arguments, environment, secrets, expected=0):
    process = subprocess.run(
        [sys.executable, "-m", "engram", "--config", str(config_path), *arguments],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        text=True, capture_output=True, timeout=45,
    )
    _assert_no_credentials(process.stdout + process.stderr, secrets)
    if process.returncode != expected:
        pytest.fail(f"onboarding CLI exited with {process.returncode}; expected {expected}", pytrace=False)
    try:
        return json.loads(process.stdout)
    except ValueError:
        pytest.fail("onboarding CLI did not return clean JSON", pytrace=False)


@pytest.fixture
def postgres_schema(tmp_path):
    base_dsn = os.environ.get("ENGRAM_TEST_POSTGRES_DSN")
    if not base_dsn:
        pytest.skip("set ENGRAM_TEST_POSTGRES_DSN for the isolated PostgreSQL onboarding test")
    psycopg = pytest.importorskip("psycopg")
    schema = "engram_onboarding_" + uuid.uuid4().hex
    try:
        admin = psycopg.connect(base_dsn, autocommit=True, connect_timeout=5)
    except Exception:
        pytest.fail("could not connect to the dedicated PostgreSQL test database", pytrace=False)
    created = False
    try:
        admin.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
        created = True
        yield psycopg, admin, schema, base_dsn
    finally:
        try:
            if created:
                admin.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))
        finally:
            admin.close()


@pytest.mark.parametrize("option_source", ["dsn", "environment"])
def test_postgres_init_doctor_and_reinit_protect_existing_records(postgres_schema, tmp_path, monkeypatch, option_source):
    psycopg, admin, schema, base_dsn = postgres_schema
    from engram.store import Store, Memory
    options = psycopg.conninfo.conninfo_to_dict(base_dsn)
    original_options = options.pop("options", os.environ.get("PGOPTIONS", ""))
    schema_options = (original_options + f" -c search_path={schema}").strip()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ENGRAM_")}
    if option_source == "dsn":
        scoped_dsn = psycopg.conninfo.make_conninfo(**options, options=schema_options)
        environment["PGOPTIONS"] = "-c search_path=pg_catalog"
    else:
        scoped_dsn = psycopg.conninfo.make_conninfo(**options)
        environment["PGOPTIONS"] = schema_options
    environment.update(ENGRAM_POSTGRES_DSN=scoped_dsn, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path))
    monkeypatch.setenv("PGOPTIONS", environment["PGOPTIONS"])
    secrets = [scoped_dsn, options.get("password"), os.environ.get("PGPASSWORD")]
    config_path = tmp_path / "new-config.yaml"
    report = _cli(config_path, ["init", "--yes", "--storage", "postgres", "--preset", "portable", "--json"], environment, secrets)
    assert report["database_initialized"] is True and report["storage"] == "postgres"
    _assert_no_credentials(config_path.read_text(), secrets)
    assert config_path.stat().st_mode & 0o077 == 0

    result = _cli(config_path, ["doctor", "--check-connection", "--json"], environment, secrets)
    storage = next(item for item in result["checks"] if item["name"] == "storage")
    assert storage["status"] == "pass"
    assert storage["details"]["counts"]["memories"] == 0
    assert result["status"] == "incomplete"  # Model inference was deliberately not requested.
    assert next(item for item in result["checks"] if item["name"] == "mcp_connection")["status"] == "pass"
    assert result["configuration"]["values"]["postgres_dsn"] == "<redacted>"

    config = Config(storage_backend="postgres", postgres_dsn=scoped_dsn)
    store = Store(config)
    try:
        store.save_memory(Memory(id="onboarding-existing-fact", content="the synthetic onboarding marker is copper", memory_type="fact"))
        before = store.conn.execute("SELECT content, access_count, last_accessed, status FROM memories WHERE id = 'onboarding-existing-fact'").fetchone()
        before_access = store.conn.execute("SELECT COUNT(*) AS count FROM access_log").fetchone()["count"]
    finally:
        store.close()

    inspect = diagnostic._schema_and_counts
    observed = {}
    def verify_transaction(connection, columns):
        observed["schema"] = connection.execute("SELECT current_schema()").fetchone()[0]
        observed["readonly"] = connection.execute("SHOW transaction_read_only").fetchone()[0]
        observed["timeout"] = connection.execute("SHOW statement_timeout").fetchone()[0]
        return inspect(connection, columns)
    monkeypatch.setattr(diagnostic, "_schema_and_counts", verify_transaction)
    direct = diagnostic._storage_check(config, True)
    assert direct["status"] == "pass", direct
    assert observed == {"schema": schema, "readonly": "on", "timeout": "3s"}
    assert direct["details"]["counts"]["memories"] == 1
    assert direct["details"]["counts"]["active"] == 1

    populated = _cli(config_path, ["doctor", "--check-connection", "--json"], environment, secrets)
    assert next(item for item in populated["checks"] if item["name"] == "storage")["details"]["counts"]["memories"] == 1
    fresh_config = tmp_path / "refused-config.yaml"
    rejected = _cli(fresh_config, ["init", "--yes", "--storage", "postgres", "--preset", "portable", "--json"], environment, secrets, expected=2)
    assert "empty current schema" in rejected["error"]
    assert not fresh_config.exists()
    store = Store(config)
    try:
        after = store.conn.execute("SELECT content, access_count, last_accessed, status FROM memories WHERE id = 'onboarding-existing-fact'").fetchone()
        after_access = store.conn.execute("SELECT COUNT(*) AS count FROM access_log").fetchone()["count"]
        assert after == before and after_access == before_access
        assert store.conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"] == 1
    finally:
        store.close()
