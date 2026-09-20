"""Engram's local JSONL API; no host protocol, network listener, or implicit capture.

Start an initialized store with ``engram --config /absolute/config.yaml api``.
One request and response per line; operations are explicit and sequential.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import json
import os
from pathlib import Path
import sys
import time

from engram import __version__
from engram.config import Config

VERSION = 1
MAX_REQUEST_BYTES = 65536
_PROJECT = {"type": "string", "minLength": 1, "maxLength": 4096,
            "description": "Canonical absolute project directory, not a project name."}
_TASK = {"type": "string", "minLength": 1, "maxLength": 200}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 20, "default": 8}
_ITEMS = {"type": "array", "maxItems": 8,
          "items": {"type": "string", "minLength": 1, "maxLength": 500}}


def _operation(name, description, properties=None, required=None, writes=False):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties or {},
                            "required": required or [], "additionalProperties": False},
            "writes": writes}


def operations():
    """Discovery is available even when storage is unavailable."""
    from engram.evidence import TOOLS as evidence_schemas
    result = [
        _operation("operations", "Discover this native API's operations and argument schemas."),
        _operation("status", "Read core storage counts and this process's non-secret runtime status."),
        _operation("config_show", "Inspect effective settings and their sources, with secrets redacted. Does not open storage or load models."),
        _operation("recall", "Read the most recently created active memories explicitly owned by a project. No semantic query or access reinforcement.",
                   {"project_id": _PROJECT, "limit": _LIMIT}, ["project_id"]),
        _operation("search", "Run ordinary semantic/hybrid Engram retrieval across the whole configured store. NOT project scoped; records ordinary accesses and configured shadow evaluations. Models load lazily.",
                   {"query": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Defaults to retrieval.top_k (must be at most 20)"}}, ["query"], True),
        _operation("search_explain", "Explain returned and rejected retrieval candidates across the whole configured store. NOT project scoped. Models load lazily; access history, result cache and dormant evaluations are unchanged.",
                   {"query": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Defaults to retrieval.top_k (must be at most 20)"}}, ["query"]),
        _operation("dormant_review", "List metadata-only dormant evaluations across the whole configured store. NOT project scoped; listing does not expose candidate content.",
                   {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}}),
        _operation("dormant_inspect", "Explicitly open a store-wide dormant candidate after rechecking active eligibility. Records separate exposure, never ordinary reinforcement.",
                   {"event_id": {"type": "string", "minLength": 1, "maxLength": 200}}, ["event_id"], True),
        _operation("dormant_feedback", "Record explicit store-wide useful/irrelevant/dismissed feedback after inspection. Useful means actual use, not automatic relevance judgment.",
                   {"event_id": {"type": "string", "minLength": 1, "maxLength": 200},
                    "category": {"type": "string", "enum": ["useful", "irrelevant", "dismissed"]}},
                   ["event_id", "category"], True),
        _operation("session_resume", "Read this project's active context and explicit native task checkpoints without reinforcing memories.",
                   {"project_id": _PROJECT, "task": _TASK, "limit": _LIMIT}, ["project_id"]),
        _operation("session_checkpoint", "Explicitly save or clear a bounded project/task checkpoint. No automatic transcript capture or memory reinforcement.",
                   {"project_id": _PROJECT, "task": _TASK,
                    "summary": {"type": "string", "maxLength": 4000},
                    "action": {"type": "string", "enum": ["save", "clear"], "default": "save"},
                    "decisions": _ITEMS, "next_steps": _ITEMS, "blockers": _ITEMS},
                   ["project_id", "task"], True),
    ]
    for alias, name in (("checkpoint", "session_checkpoint"), ("resume", "session_resume")):
        item = copy.deepcopy(next(item for item in result if item["name"] == name))
        item.update(name=alias, alias_for=name)
        result.append(item)
    result.extend({"name": item["name"], "description": item["description"],
                   "inputSchema": item["inputSchema"],
                   "writes": not item["annotations"]["readOnlyHint"]}
                  for item in evidence_schemas)
    return {"protocol": "engram-jsonl", "version": VERSION, "engram_version": __version__,
            "max_request_bytes": MAX_REQUEST_BYTES, "operations": copy.deepcopy(result)}


class NativeService:
    def __init__(self, config: Config):
        config.validate()
        self.config = config
        self.started_at = time.time()
        self._store = None
        self._search_configured = False

    @property
    def store(self):
        if self._store is None:
            if self.config.normalized_storage_backend == "sqlite":
                # Do not create directories, an empty DB, or legacy schema backfills.
                if not Path(self.config.db_path).expanduser().is_file():
                    raise RuntimeError("initialized store required")
            from engram.store import Store
            candidate = Store(self.config)
            try:
                candidate.conn.execute("SELECT id, status, forgotten FROM memories LIMIT 0")
                candidate.conn.execute("SELECT session_id FROM session_handoffs LIMIT 0")
            except Exception:
                candidate.close()
                raise
            self._store = candidate
        return self._store

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    def status(self):
        return {**self.store.get_stats(), "protocol": "engram-jsonl", "version": VERSION,
                "engram_version": __version__,
                "pid": os.getpid(), "started_at": self.started_at,
                "storage_backend": self.config.normalized_storage_backend,
                "dormant_mode": self.config.dormant_recall.mode,
                "automatic_capture": False,
                "connection_scope": "this local process only"}

    def config_show(self):
        return {"version": __version__, **self.config.describe()}

    def _context(self, project_id, task=None, limit=8):
        self.store  # validate initialized schema before ProjectContext opens it
        from engram.project_context import ProjectContext
        context = ProjectContext(self.config, project_id)
        try:
            return context.context(task=task, limit=limit)
        finally:
            context.close()

    def recall(self, project_id, limit=8):
        return self._context(project_id, limit=limit)

    def session_resume(self, project_id, task=None, limit=8):
        return self._context(project_id, task, limit)

    def session_checkpoint(self, project_id, task, summary="", decisions=None,
                           next_steps=None, blockers=None, action="save"):
        self.store
        from engram.project_context import ProjectContext
        context = ProjectContext(self.config, project_id)
        try:
            return context.checkpoint(task, action=action, summary=summary,
                                      decisions=decisions, next_steps=next_steps, blockers=blockers)
        finally:
            context.close()

    def search(self, query, top_k=None):
        return self._search(query, top_k)

    def search_explain(self, query, top_k=None):
        return self._search(query, top_k, explain=True)

    def _search(self, query, top_k, explain=False):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("query must be nonempty text of at most 4000 characters")
        if top_k is None:
            top_k = self.config.retrieval.top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")
        store = self.store
        # Search and its explicit diagnostic counterpart load models lazily.
        from engram.embeddings import set_backend, set_default_model
        from engram.retrieval import search
        if not self._search_configured:
            set_backend(self.config.embedding_backend)
            set_default_model(self.config.embedding_model)
            self._search_configured = True
        result = (search(query, store, self.config, top_k=top_k, debug=True) if explain
                  else search(query, store, self.config, top_k=top_k))
        if explain:
            results, debug = result
        else:
            results = result
        memories = [{"id": r.memory.id, "content": r.memory.content, "score": round(r.score, 4),
                 "layer": r.memory.layer, "memory_type": r.memory.memory_type,
                 "importance": r.memory.importance} for r in results]
        return {"results": memories, "explanation": debug.to_dict()} if explain else memories

    def dormant_review(self, limit=20):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        self.store
        from engram.dormant import review
        return review(self.config, limit)

    def dormant_inspect(self, event_id):
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
            raise ValueError("event_id must be nonempty text of at most 200 characters")
        self.store
        from engram.dormant import inspect_event
        return inspect_event(self.config, event_id)

    def dormant_feedback(self, event_id, category):
        if not isinstance(event_id, str) or not event_id.strip() or len(event_id) > 200:
            raise ValueError("event_id must be nonempty text of at most 200 characters")
        self.store
        from engram.dormant import feedback
        return feedback(self.config, event_id, category)

    def handle_request(self, request):
        ident = request.get("id") if isinstance(request, dict) else None
        if isinstance(ident, bool) or not isinstance(ident, (str, int, type(None))) or (isinstance(ident, str) and len(ident) > 200):
            ident = None
            return _error(ident, "invalid_request", "id must be null, an integer or a string of at most 200 characters")
        if (not isinstance(request, dict) or "id" not in request
                or set(request) - {"id", "operation", "params"}
                or not isinstance(request.get("operation"), str)
                or not isinstance(request.get("params", {}), dict)):
            return _error(ident, "invalid_request", "Expected {id, operation, params}; params must be an object")
        name, params = request["operation"], request.get("params", {})
        handlers = {"operations": operations, "status": self.status, "recall": self.recall,
                    "config_show": self.config_show,
                    "search": self.search, "search_explain": self.search_explain, "session_resume": self.session_resume,
                    "session_checkpoint": self.session_checkpoint,
                    "checkpoint": self.session_checkpoint, "resume": self.session_resume,
                    "dormant_review": self.dormant_review, "dormant_inspect": self.dormant_inspect,
                    "dormant_feedback": self.dormant_feedback}
        try:
            if name in ("evidence_put", "evidence_get", "evidence_list"):
                from engram import evidence
                result = getattr(evidence, name)(self.store, **params)
            elif name in handlers:
                result = handlers[name](**params)
            else:
                return _error(ident, "unknown_operation", "Unknown operation; call operations for supported names")
            return {"id": ident, "result": result}
        except TypeError:
            return _error(ident, "invalid_params", "Unexpected, missing or invalid operation parameters")
        except ValueError:
            return _error(ident, "invalid_params", "Parameters or stored data were rejected; check the operation schema and current record state")
        except Exception:
            # Driver errors may include credentials/query contents. Discard failed
            # connections so a later request can recover after transient failures.
            self.close()
            return _error(ident, "operation_failed", "Operation failed; check configured storage and required local models. No automatic initialization or restart was attempted")


def _error(ident, code, message):
    return {"id": ident, "error": {"code": code, "message": message}}


def _reject_constant(_value):
    raise ValueError("non-finite JSON")


def run_stdio(config: Config, input_stream=None, output_stream=None):
    """Serve bounded JSONL until EOF. Diagnostics cannot enter protocol stdout."""
    source = input_stream if input_stream is not None else getattr(sys.stdin, "buffer", sys.stdin)
    output = output_stream if output_stream is not None else sys.stdout
    service = NativeService(config)
    try:
        while True:
            raw = source.readline(MAX_REQUEST_BYTES + 1)
            if not raw:
                break
            newline = b"\n" if isinstance(raw, bytes) else "\n"
            try:
                size = len(raw) if isinstance(raw, bytes) else len(raw.encode("utf-8"))
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                invalid_utf8 = False
            except UnicodeError:
                size = len(raw)
                line = ""
                invalid_utf8 = True
            if size > MAX_REQUEST_BYTES:
                while not raw.endswith(newline):
                    raw = source.readline(MAX_REQUEST_BYTES + 1)
                    if not raw:
                        break
                response = _error(None, "request_too_large", "Request exceeds 65536 UTF-8 bytes")
            elif invalid_utf8:
                response = _error(None, "invalid_json", "Request must be valid UTF-8 JSON")
            elif not line.strip():
                continue
            else:
                try:
                    request = json.loads(line, parse_constant=_reject_constant)
                except (ValueError, RecursionError):
                    response = _error(None, "invalid_json", "Expected one finite JSON object per line")
                else:
                    with redirect_stdout(sys.stderr):
                        response = service.handle_request(request)
            try:
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False)
            except (TypeError, ValueError, RecursionError):
                encoded = json.dumps(_error(response.get("id"), "operation_failed", "Operation returned an invalid JSON result"))
            output.write(encoded + "\n")
            output.flush()
    finally:
        service.close()
