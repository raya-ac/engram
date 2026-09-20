"""Bounded setup diagnostics. Basic checks never initialize storage or load models."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

from engram import __version__
from engram.config import Config, ConfigError

_CONNECTION_TIMEOUT = 20
_INFERENCE_TIMEOUT = 180
_REQUIRED_COLUMNS = {
    "memories": {"id", "content", "embedding", "memory_type", "status", "forgotten", "access_count"},
    "memories_fts": {"content"},
    "entities": {"id", "canonical_name"},
    "relationships": {"source_entity_id", "target_entity_id", "relation_type"},
    "access_log": {"memory_id", "accessed_at"},
}
_PROVIDER_PACKAGES = {"voyage": "voyageai", "openai": "openai", "gemini": "google-genai"}
_PROVIDER_KEYS = {"voyage": ("VOYAGE_API_KEY",), "openai": ("OPENAI_API_KEY",),
                  "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def _check(name, status, message, **details):
    result = {"name": name, "status": status, "message": message}
    if details:
        result["details"] = details
    return result


def _failure(name, exc, action):
    # Driver/model exceptions may contain a DSN, API key, URL, or response body.
    # Classifications are useful without copying those messages into the report.
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        message = "check timed out; " + action
    elif isinstance(exc, ImportError):
        message = "a required Python package could not be imported; " + action
    elif "out of memory" in str(exc).lower():
        message = "model runtime ran out of memory; use a smaller model or another backend"
    else:
        message = action
    return _check(name, "fail", message, error_type=type(exc).__name__)


def _schema_and_counts(connection, columns):
    missing = {table: sorted(required - columns.get(table, set()))
               for table, required in _REQUIRED_COLUMNS.items()
               if required - columns.get(table, set())}
    if missing:
        return _check("storage", "fail", "existing storage lacks required schema; initialize or migrate it deliberately",
                      missing_columns=missing)
    row = connection.execute(
        "SELECT COUNT(*), COUNT(embedding), "
        "COALESCE(SUM(CASE WHEN forgotten = 0 AND status = 'active' THEN 1 ELSE 0 END), 0) FROM memories"
    ).fetchone()
    counts = {"memories": int(row[0]), "with_embeddings": int(row[1]), "active": int(row[2])}
    for table in ("entities", "relationships", "access_log"):
        counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return _check("storage", "pass", "existing storage is readable and its required schema is present",
                  counts=counts, inspection="read-only transaction; no initialization or migration")


def _storage_check(config, check_connection):
    if config.normalized_storage_backend == "postgres":
        if not check_connection:
            return _check("storage", "skipped", "PostgreSQL reachability needs --check-connection; no network connection attempted")
        try:
            import psycopg
            # libpq uses explicit DSN options ahead of PGOPTIONS. Preserve that
            # selection (including search_path), then enforce diagnostic limits.
            options = psycopg.conninfo.conninfo_to_dict(config.postgres_dsn).get(
                "options", os.environ.get("PGOPTIONS", ""),
            )
            options = (options + " -c default_transaction_read_only=on -c statement_timeout=3000").strip()
            with psycopg.connect(config.postgres_dsn, connect_timeout=3, options=options) as connection:
                connection.execute("SET TRANSACTION READ ONLY")
                columns = {}
                for table, column in connection.execute(
                    "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = current_schema()"
                ).fetchall():
                    columns.setdefault(table, set()).add(column)
                return _schema_and_counts(connection, columns)
        except Exception as exc:
            return _failure("storage", exc, "PostgreSQL read-only inspection failed; check the driver, host, credentials, and schema")
    # Do not use Config.resolved_db_path or Store: both can create paths.
    try:
        path = Path(config.db_path).expanduser().absolute()
        if not path.is_file():
            return _check("storage", "fail", "SQLite database does not exist; doctor did not create it", path=str(path))
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
        try:
            deadline = time.monotonic() + 3
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            columns = {table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                       for table in _REQUIRED_COLUMNS}
            return _schema_and_counts(connection, columns)
        finally:
            connection.close()
    except Exception as exc:
        return _failure("storage", exc, "SQLite read-only inspection failed; check path permissions, database integrity, and locks")


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cached_model(model):
    local = Path(model).expanduser()
    if local.is_dir():
        candidates = [local]
        source = "local model directory"
    else:
        hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if not hub:
            hf_home = os.environ.get("HF_HOME")
            cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            hub = str(Path(hf_home).expanduser() / "hub") if hf_home else str(cache_home / "huggingface/hub")
        snapshots = Path(hub).expanduser() / ("models--" + model.replace("/", "--")) / "snapshots"
        candidates = sorted(snapshots.iterdir()) if snapshots.is_dir() else []
        source = "Hugging Face cache"
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        weights = any((candidate / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "weights.safetensors"))
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = candidate / index_name
            if index.is_file():
                try:
                    shards = set(json.loads(index.read_text())["weight_map"].values())
                    weights = weights or bool(shards) and all((candidate / shard).is_file() for shard in shards)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        tokenizer = (candidate / "tokenizer.json").is_file() or (candidate / "vocab.txt").is_file() or (candidate / "tokenizer.model").is_file()
        if weights and tokenizer:
            return {"files_present": True, "source": source, "runtime_load_verified": False}
    return {"files_present": False, "source": source, "runtime_load_verified": False}


def _model_readiness(config):
    from engram.embeddings import MODEL_BACKENDS, RERANKER_BACKENDS
    checks = []
    backend = MODEL_BACKENDS.get(config.embedding_model, config.embedding_backend)
    if backend == "auto":
        backend = "mlx" if _package_version("mlx") else "sentence_transformers"
    for role, model, selected in (
        ("embedding", config.embedding_model, backend),
        ("reranker", config.cross_encoder_model, "voyage" if config.cross_encoder_model in RERANKER_BACKENDS else "sentence_transformers"),
    ):
        if selected in _PROVIDER_PACKAGES:
            package = _PROVIDER_PACKAGES[selected]
            version = _package_version(package)
            present = any(bool(os.environ.get(key)) for key in _PROVIDER_KEYS[selected])
            checks.append(_check(role + "_readiness", "pass" if version and present else "fail",
                                 "provider package and credential presence checked; no request or credential validation performed",
                                 backend=selected, package=package, package_version=version,
                                 credential_present=present, credential_environment=list(_PROVIDER_KEYS[selected])))
        else:
            names = ("mlx", "mlx-embeddings") if selected == "mlx" else ("sentence-transformers", "torch")
            packages = {name: _package_version(name) for name in names}
            try:
                cache = _cached_model(model)
            except (OSError, ValueError) as exc:
                checks.append(_failure(role + "_readiness", exc, "could not inspect the model cache; check the configured path and permissions"))
                continue
            ready = all(packages.values()) and cache["files_present"]
            checks.append(_check(role + "_readiness", "pass" if ready else "warning",
                                 "package metadata and model files checked; runtime loading and inference remain unverified" if ready
                                 else "model packages or cached files are missing; an explicit --check-models can load/download the configured models",
                                 backend=selected, configured_backend=config.embedding_backend if role == "embedding" else selected,
                                 packages=packages, cache=cache))
    return checks


def _isolated_config(config, directory):
    isolated = copy.deepcopy(config)
    isolated.storage_backend = "sqlite"
    isolated.db_path = str(Path(directory) / "memory.db")
    isolated.postgres_dsn = ""
    isolated.ann.enabled = False
    isolated.ann.index_path = str(Path(directory) / "hnsw.index")
    isolated.dormant_recall.mode = "off"
    return isolated


def _connection_check(config):
    try:
        with tempfile.TemporaryDirectory(prefix="engram-doctor-mcp-") as directory:
            isolated = _isolated_config(config, directory)
            isolated.hf_token = isolated.llm.api_key = isolated.web.auth_token = ""
            config_path = Path(directory) / "config.json"
            fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(asdict(isolated), stream)
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "engram-doctor", "version": __version__}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "config_show", "arguments": {}}},
            ]
            removed = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                       "VOYAGE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"}
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("ENGRAM_") and key not in removed}
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            process = subprocess.run(
                [sys.executable, "-m", "engram", "--config", str(config_path), "serve", "--mcp", "--no-warmup"],
                input="".join(json.dumps(item) + "\n" for item in requests), text=True, capture_output=True,
                cwd=Path(__file__).resolve().parents[1], env=env, timeout=_CONNECTION_TIMEOUT,
            )
            if process.returncode:
                return _check("mcp_connection", "fail", "isolated Engram stdio process exited before a successful handshake",
                              returncode=process.returncode)
            responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
            if len(responses) != 3 or {item.get("id") for item in responses} != {1, 2, 3}:
                raise ValueError("unexpected MCP responses")
            replies = {item["id"]: item for item in responses}
            if any(item.get("jsonrpc") != "2.0" or "error" in item for item in replies.values()):
                raise ValueError("MCP protocol or tool failure")
            if replies[1]["result"]["protocolVersion"] != "2024-11-05" or replies[3]["result"].get("isError"):
                raise ValueError("MCP negotiation or tool failure")
            server = replies[1]["result"]["serverInfo"]
            tools = replies[2]["result"]["tools"]
            if server["name"] != "engram" or server["version"] != __version__ or not any(tool["name"] == "config_show" for tool in tools):
                raise ValueError("MCP endpoint mismatch")
            settings = json.loads(replies[3]["result"]["content"][0]["text"])
            values = settings["values"]
            if values["db_path"] != isolated.db_path or values["storage_backend"] != "sqlite" or values["ann"]["enabled"]:
                raise ValueError("MCP storage isolation mismatch")
            if values["embedding_model"] != config.embedding_model or values["cross_encoder_model"] != config.cross_encoder_model:
                raise ValueError("MCP configuration mismatch")
            return _check("mcp_connection", "pass", "local Engram stdio initialize, tools/list, and config_show succeeded",
                          tool_count=len(tools), server_version=server["version"],
                          scope="isolated SQLite configuration; credentials omitted; model warmup disabled",
                          external_agent_connection_verified=False)
    except Exception as exc:
        return _failure("mcp_connection", exc, "isolated MCP handshake failed; check the installed executable, dependencies, and protocol support")


def _inference_checks(config, smoke):
    """Runs only in the bounded worker; model/provider calls use synthetic text."""
    import numpy as np
    from engram.embeddings import embed_documents, embed_query, cross_encoder_rerank, set_backend, set_default_model
    set_backend(config.embedding_backend)
    set_default_model(config.embedding_model)
    if config.hf_token:
        os.environ["HF_TOKEN"] = config.hf_token
    documents = ["The doctor smoke-test launch marker is copper.",
                 "The doctor smoke-test lunch is vegetable soup.",
                 "The doctor smoke-test office opens at nine."]
    query = "What color is the doctor smoke-test launch marker?"
    checks = []
    vectors = None
    try:
        vectors = embed_documents(documents, config.embedding_model)
        query_vector = embed_query(query, config.embedding_model)
        if vectors.shape != (len(documents), config.embedding_dim) or query_vector.shape != (config.embedding_dim,):
            checks.append(_check("embedding_inference", "fail", "model output dimension differs from embedding_dim",
                                 expected_dimension=config.embedding_dim, document_shape=list(vectors.shape), query_shape=list(query_vector.shape)))
            vectors = None
        elif not np.isfinite(vectors).all() or not np.isfinite(query_vector).all():
            checks.append(_check("embedding_inference", "fail", "embedding model produced non-finite values"))
            vectors = None
        else:
            checks.append(_check("embedding_inference", "pass", "configured document and query embeddings ran on synthetic text",
                                 dimension=config.embedding_dim, documents=len(documents)))
    except Exception as exc:
        checks.append(_failure("embedding_inference", exc, "configured embedding inference failed; check package installation, model availability, and provider credentials"))
    rerank_ok = False
    try:
        ranked = cross_encoder_rerank(query, documents, config.cross_encoder_model)
        rerank_ok = len(ranked) == len(documents) and {index for index, score in ranked} == set(range(len(documents))) and all(math.isfinite(score) for _, score in ranked)
        checks.append(_check("reranker_inference", "pass" if rerank_ok else "fail",
                             "configured reranker ran on synthetic text" if rerank_ok else "reranker returned invalid indices or scores"))
    except Exception as exc:
        checks.append(_failure("reranker_inference", exc, "configured reranker inference failed; check package installation, model availability, and provider credentials"))
    if not smoke:
        return checks
    if vectors is None or not rerank_ok:
        checks.append(_check("isolated_retrieval", "skipped", "save/retrieve smoke requires successful configured embedding and reranker checks"))
        return checks
    try:
        from engram.store import Store, Memory
        from engram.retrieval import search
        with tempfile.TemporaryDirectory(prefix="engram-doctor-retrieval-") as directory:
            isolated = _isolated_config(config, directory)
            store = Store(isolated)
            try:
                store.init_db()
                ids = [f"doctor-synthetic-{index}" for index in range(len(documents))]
                for identifier, content, vector in zip(ids, documents, vectors):
                    store.save_memory(Memory(id=identifier, content=content, embedding=vector,
                                             memory_type="fact", source_type="remember:human", source_file="doctor:synthetic"))
                stored = all(store.get_memory(identifier) is not None for identifier in ids)
                results, _ = search(query, store, isolated, rerank=True, debug=True)
                retrieved = [result.memory.id for result in results]
                matched = stored and ids[0] in retrieved
                checks.append(_check("isolated_retrieval", "pass" if matched else "fail",
                                     "real save and retrieval recovered the synthetic fact through configured ranking and relevance gates" if matched
                                     else "synthetic fact was stored but not returned; inspect retrieval settings and relevance gates",
                                     stored_count=len(ids) if stored else 0, returned_count=len(results),
                                     expected_rank=retrieved.index(ids[0]) + 1 if ids[0] in retrieved else None,
                                     scope="temporary SQLite; ANN and dormant recall disabled; user's memory store untouched",
                                     production_storage_backend_verified=False))
            finally:
                store.close()
    except Exception as exc:
        checks.append(_failure("isolated_retrieval", exc, "isolated save/retrieve failed; check model and storage dependencies"))
    return checks


def _run_inference(config, smoke):
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([sys.executable, "-m", "engram.doctor", "--worker"],
                                 input=json.dumps({"config": asdict(config), "smoke": smoke}),
                                 text=True, capture_output=True, timeout=_INFERENCE_TIMEOUT,
                                 cwd=Path(__file__).resolve().parents[1], env=env)
        if process.returncode:
            return [_check("model_process", "fail", "model diagnostic process exited unsuccessfully; check runtime/backend compatibility",
                           returncode=process.returncode)]
        results = json.loads(process.stdout)
        if not isinstance(results, list) or not results or any(item.get("status") not in {"pass", "fail", "skipped"} for item in results):
            raise ValueError("invalid worker report")
        return results
    except Exception as exc:
        return [_failure("model_process", exc, "configured model checks did not complete; try a smaller model or an explicit backend")]


def doctor(config, config_path=None, check_models=False, check_connection=False, smoke=False) -> dict:
    """Return redacted diagnostics; explicit flags opt into network/model/temp writes.

    The existing store is only read. --smoke also exercises both models. Optional
    model calls may download configured weights or call a configured provider;
    only fixed synthetic text is sent. No agent client configuration is changed.
    """
    report = {"version": __version__, "configuration": None, "checks": []}
    checks = report["checks"]
    try:
        config.validate()
        report["configuration"] = config.describe()
        if config_path is not None:
            report["configuration"]["requested_config_file"] = str(config_path)
        warnings = report["configuration"]["warnings"]
        checks.append(_check("configuration", "warning" if warnings else "pass",
                             "effective configuration validates with warnings; credential fields are redacted" if warnings
                             else "effective configuration validates; credential fields are redacted", warnings=warnings))
    except ConfigError as exc:
        checks.append(_check("configuration", "fail", str(exc)))
        report.update(ok=False, status="fail")
        return report
    checks.append(_storage_check(config, check_connection))
    checks.extend(_model_readiness(config))
    if check_connection:
        checks.append(_connection_check(config))
    else:
        checks.append(_check("mcp_connection", "skipped", "use --check-connection to test a spawned local MCP endpoint; external agent attachment is not tested"))
    if check_models or smoke:
        inference = _run_inference(config, smoke)
        checks.extend(inference)
        passed = {item["name"] for item in inference if item["status"] == "pass"}
        for item in checks:
            if item["name"] in {"embedding_readiness", "reranker_readiness"} and item["name"].replace("_readiness", "_inference") in passed:
                item.setdefault("details", {})["runtime_load_verified"] = True
                if "cache" in item["details"]:
                    item["details"]["cache"]["runtime_load_verified"] = True
                item["status"] = "pass"
                item["message"] = "configured model loading and inference verified on synthetic text"
    else:
        checks.append(_check("embedding_inference", "skipped", "use --check-models for configured model inference on synthetic text"))
        checks.append(_check("reranker_inference", "skipped", "use --check-models for configured reranker inference on synthetic text"))
    if not smoke:
        checks.append(_check("isolated_retrieval", "skipped", "use --smoke for real save/retrieve in a temporary SQLite database"))
    failed = any(item["status"] == "fail" for item in checks)
    incomplete = any(item["status"] in {"skipped", "warning"} for item in checks)
    report.update(ok=not failed, status="fail" if failed else "incomplete" if incomplete else "pass")
    return report


def _worker():
    try:
        payload = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(sys.stderr):
            config = Config.from_mapping(payload["config"], apply_environment=False)
            checks = _inference_checks(config, payload["smoke"])
    except Exception as exc:
        checks = [_failure("model_process", exc, "model diagnostic initialization failed")]
    sys.stdout.write(json.dumps(checks) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("use engram doctor")
    _worker()
