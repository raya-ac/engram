"""A focused fictional-project walkthrough, isolated from the user's store."""

from __future__ import annotations

import copy
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from engram.config import Config


class DemoError(ValueError):
    """A demo failure that can be reported without a traceback or credentials."""


_BOILERPLATE = "The equipment inventory lists spare cables, folding chairs and blank forms. "
MEMORIES = (
    ("overview", "semantic", "fact", "Lantern is a fictional community observatory scheduling project."),
    ("rollback", "procedural", "procedure", "For a Lantern release, save the previous signed build before deployment. If the health check fails, restore that build and repeat the health check."),
    ("access-key", "semantic", "fact", _BOILERPLATE * 60 + "The Lantern project's emergency access key is stored in the copper lockbox. " + _BOILERPLATE * 10),
    ("inspection", "procedural", "procedure", "The Lantern project's emergency access key is inspected every Monday. Mira records the inspection date."),
    ("decision", "episodic", "narrative", "The Lantern team chose a manual approval before each public release. The next task is to rehearse rollback."),
    ("lunch", "semantic", "fact", "The Lantern office serves vegetable soup for lunch on Thursdays."),
)
RECALL_QUERY = "Where is the Lantern project's emergency access key stored?"
_PROVIDER_ENV = {
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
    "PGPASSWORD", "PGPASSFILE", "PGSERVICE", "PGSERVICEFILE", "PGOPTIONS",
}


def _child_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("ENGRAM_") and key not in _PROVIDER_ENV}


def _command(config_path, *args):
    # Config.load gives env precedence. Unset current overrides when reusing this store.
    prefix = []
    for name in sorted(key for key in os.environ if key.startswith("ENGRAM_")):
        prefix.extend(["-u", name])
    return shlex.join([*(["env", *prefix] if prefix else []), sys.executable,
                       "-m", "engram", "--config", str(config_path), *args])


def _write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _state(store):
    return tuple(store.conn.iterdump()), copy.deepcopy(store._search_cache)


def _stop_web(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _web_ready(url, token, config_path, db_path):
    request = Request(url + "api/config", headers={"Authorization": "Bearer " + token})
    try:
        # Local readiness must not pass through an inherited HTTP proxy.
        with build_opener(ProxyHandler({})).open(request, timeout=0.5) as response:
            report = json.load(response)
        return (report.get("config_file") == str(config_path)
                and report.get("values", {}).get("db_path") == str(db_path)
                and report.get("values", {}).get("storage_backend") == "sqlite")
    except (OSError, URLError, ValueError):
        return False


def _start_web(config, config_path, directory, *, timeout=20):
    url = f"http://127.0.0.1:{config.web.port}/"
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", config.web.port))
        except OSError:
            raise DemoError("demo web port is unavailable; choose another --port") from None
    fd = os.open(directory / "web.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "engram", "--config", str(config_path),
             "serve", "--web", "--port", str(config.web.port)],
            env=_child_environment(), cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        )
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise DemoError("demo web process exited during startup; its log is in the demo directory")
            if _web_ready(url, config.web.auth_token, config_path, config.db_path):
                return process, url + "?token=" + config.web.auth_token
            time.sleep(0.1)
        raise DemoError("demo web startup timed out; its log is in the demo directory")
    except BaseException:
        _stop_web(process)
        raise


def _show_results(output, label, results):
    output(label)
    if not results:
        output("  no results passed this search's eligibility and relevance rules")
    for rank, result in enumerate(results, 1):
        output(f"  {rank}. {result.memory.id}  score={result.score:.4f}")
        if "excerpt_raw_score" in result.sources:
            start, end = int(result.sources["source_start"]), int(result.sources["source_end"])
            output("    source excerpt: " + result.memory.content[start:end])
        else:
            output("    " + result.memory.content[:120])


def _walkthrough(config, config_path, directory, *, yes, input_fn, output):
    from engram.embeddings import embed_documents, set_backend, set_default_model
    from engram.project_context import ProjectContext
    from engram.retrieval import search
    from engram.store import Memory, Store

    def pause():
        if not yes:
            input_fn("Press Enter to continue: ")

    project = directory / "lantern-project"
    project.mkdir()
    report = config.describe()
    _write_json(directory / "config-report.json", report)
    output("1. Isolated setup and effective configuration")
    output(f"  config: {config_path}")
    output(f"  SQLite: {config.db_path}")
    output(f"  local models: {config.embedding_model}; {config.cross_encoder_model}")
    output(f"  final confidence cutoff: {config.retrieval.min_confidence}; excerpt retries enabled")
    output("  inherited Engram settings are ignored; ANN and dormant recall are off for this small demo")
    pause()

    set_backend(config.embedding_backend)
    set_default_model(config.embedding_model)
    store = Store(config)
    try:
        store.init_db()
        output("2. Saving six fictional project memories with local embeddings")
        output("  first use may download local model weights; no LLM or hosted inference is used")
        output("  the long access-key note is deliberately padded to illustrate excerpt retries, not to measure accuracy")
        vectors = embed_documents([item[3] for item in MEMORIES], config.embedding_model)
        if len(vectors) != len(MEMORIES):
            raise DemoError("local embeddings did not return one vector per demo memory")
        for (identifier, layer, memory_type, content), vector in zip(MEMORIES, vectors):
            store.save_memory(Memory(
                id=identifier, content=content, layer=layer, memory_type=memory_type,
                embedding=vector, importance=0.7, source_type="remember:human",
                source_file="demo:fictional", metadata={"project_path": str(project)},
            ))
        hybrid = search("Lantern release rollback procedure", store, config, top_k=3, rerank=False)
        _show_results(output, "3. Ordinary hybrid recall (records accesses in this demo store)", hybrid)

        before = _state(store)
        reranked, debug = search(RECALL_QUERY, store, config, top_k=3, rerank=True, debug=True)
        explanation = debug.to_dict()
        if before != _state(store):
            raise DemoError("explanation changed stored records or the result cache")
        _write_json(directory / "explanation.json", explanation)
        _show_results(output, "4. Reranked recall and its read-only explanation", reranked)
        output(f"  query: {RECALL_QUERY}")
        for candidate in explanation["candidates"]:
            output(f"  {candidate['memory_id']}: {candidate['outcome']} — {candidate['reason']}")
            passage = candidate.get("passage", {})
            if "excerpt_raw_score" in passage:
                output(f"    excerpt {passage['source_start']}:{passage['source_end']}; full logit={passage['base_raw_score']:.4f}, excerpt logit={passage['excerpt_raw_score']:.4f}")
        output("  actual model scores decide acceptance; the production cutoff has not been lowered")
        output("  explanation left stored memories, access history and the result cache unchanged")
        pause()
    finally:
        store.close()

    output("5. Save an explicit project checkpoint, then reopen it")
    context = ProjectContext(config, str(project))
    try:
        context.checkpoint("release rehearsal", summary="Lantern's rollback procedure is recorded; a rehearsal is the next task.",
                           decisions=["Keep manual release approval."],
                           next_steps=["Rehearse restoring the previous signed build."])
    finally:
        context.close()
    context = ProjectContext(config, str(project))
    try:
        before = _state(context.store)
        resumed = context.context(task="release rehearsal", limit=3)
        if before != _state(context.store):
            raise DemoError("resuming the checkpoint changed stored records")
        output("  " + resumed["checkpoints"][0]["summary"])
        output("  next: " + resumed["checkpoints"][0]["next_steps"][0])
    finally:
        context.close()
    output("This fictional workflow illustrates the APIs; its rankings are not a general accuracy or cross-agent continuity measurement.")
    return {"hybrid_ids": [row.memory.id for row in hybrid],
            "reranked_ids": [row.memory.id for row in reranked],
            "explanation_unchanged": True, "checkpoint": resumed["checkpoints"][0]}


def run_demo(keep_db=False, start_web=False, web_port=8421, yes=False, *, input_fn=None, output_fn=print):
    """Run locally; --keep retains files, never an orphan web process."""
    if type(web_port) is not int or not 1 <= web_port <= 65535:
        raise DemoError("demo port must be an integer from 1 to 65535")
    if not yes and input_fn is None and not sys.stdin.isatty():
        raise DemoError("interactive demo requires a terminal; use --yes for an unattended run")
    input_fn = input_fn or input
    directory = Path(tempfile.mkdtemp(prefix="engram-demo-")).resolve()
    config_path = directory / "config.json"
    process = None
    complete = False
    try:
        config = Config.from_mapping({
            "storage_backend": "sqlite", "db_path": str(directory / "memory.db"),
            "embedding_backend": "sentence_transformers",
            "ann": {"enabled": False, "index_path": str(directory / "hnsw.index")},
            "dormant_recall": {"mode": "off"},
            "web": {"host": "127.0.0.1", "port": web_port, "auth_token": secrets.token_urlsafe(24)},
        }, apply_environment=False)
        from dataclasses import asdict
        _write_json(config_path, asdict(config))
        fd = os.open(config.db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        output_fn("Engram demo: the fictional Lantern project")
        summary = _walkthrough(config, config_path, directory, yes=yes, input_fn=input_fn, output=output_fn)
        if start_web:
            process, url = _start_web(config, config_path, directory)
            output_fn("Demo workspace is ready after the walkthrough: " + url)
            if yes:
                output_fn("Readiness checked; --yes stops the web process when the demo finishes.")
            else:
                input_fn("Open the URL to inspect the demo. Press Enter when finished to stop the web process: ")
        complete = True
        output_fn("For your own store, run: engram init")
        output_fn("Then run the doctor --full command printed by setup.")
        return {"directory": str(directory), "config_file": str(config_path),
                "kept": bool(keep_db), "complete": True, **summary}
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt from None
    except DemoError:
        raise
    except Exception as exc:
        raise DemoError(f"demo could not complete ({type(exc).__name__}); check local model availability and runtime dependencies") from None
    finally:
        try:
            _stop_web(process)
        finally:
            if keep_db:
                output_fn(f"Kept {'completed' if complete else 'partial'} demo files: {directory}")
                output_fn("Inspect: " + _command(config_path, "config", "show"))
                output_fn("Doctor: " + _command(config_path, "doctor", "--full"))
                output_fn("Recall: " + _command(config_path, "search", RECALL_QUERY, "--rerank", "--explain"))
                if start_web:
                    output_fn("Restart workspace: " + _command(config_path, "serve", "--web", "--port", str(web_port)))
            else:
                shutil.rmtree(directory)
                output_fn("Temporary demo files removed; any demo web process is stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    try:
        run_demo(keep_db=args.keep, start_web=args.web, web_port=args.port, yes=args.yes)
    except DemoError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
