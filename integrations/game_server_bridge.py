"""Source integration: a narrow authenticated HTTP bridge for game checkpoints.

Run with an installed Engram environment and an already initialized store.
No models, transcript capture, arbitrary operations or player-selected paths.
"""
from __future__ import annotations

import argparse
import copy
import hmac
import json
import os
from pathlib import Path
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from engram.config import Config, ConfigError
from engram.service import NativeService

MAX_BODY_BYTES = 16_384
KINDS = frozenset({"rule", "build", "handoff", "note"})
LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_FIELDS = frozenset({"world", "kind", "key", "summary", "decisions", "next_steps", "blockers"})


def _error(status, code, message):
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class _BoundaryMiddleware:
    """Authenticate before routing; bound streamed bodies before JSON decoding."""

    def __init__(self, app, *, token):
        self.app = app
        self.authorization = ("Bearer " + token).encode("ascii")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        values = [value for name, value in headers if name.lower() == b"authorization"]
        if len(values) != 1 or not hmac.compare_digest(values[0], self.authorization):
            await _error(401, "unauthorized", "A valid Bearer token is required")(scope, receive, send)
            return
        lengths = [value for name, value in headers if name.lower() == b"content-length"]
        if lengths:
            try:
                if len(lengths) != 1 or int(lengths[0]) < 0:
                    raise ValueError
                too_large = int(lengths[0]) > MAX_BODY_BYTES
            except ValueError:
                await _error(400, "invalid_request", "Invalid request framing")(scope, receive, send)
                return
            if too_large:
                await _error(413, "request_too_large", "Request body exceeds 16384 bytes")(scope, receive, send)
                return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await _error(413, "request_too_large", "Request body exceeds 16384 bytes")(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        scope["game_bridge_body"] = bytes(body)

        async def replay():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite number")


def _selection(values, worlds):
    world, kind, key = (values.get(name) for name in ("world", "kind", "key"))
    if (not isinstance(world, str) or not LABEL.fullmatch(world)
            or world not in worlds or not isinstance(kind, str) or kind not in KINDS
            or not isinstance(key, str) or not LABEL.fullmatch(key)):
        raise ValueError("invalid selection")
    # Maximum length: 5 + 64 + 1 + 7 + 1 + 64 = 142, below native's 200.
    return f"game:{world}:{kind}:{key}"


def _save_params(payload, worlds):
    if not isinstance(payload, dict) or set(payload) - _FIELDS:
        raise ValueError("unknown fields")
    task = _selection(payload, worlds)
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
        raise ValueError("invalid summary")
    result = {"task": task, "summary": summary}
    for name in ("decisions", "next_steps", "blockers"):
        if name not in payload:
            continue
        values = payload[name]
        if (not isinstance(values, list) or len(values) > 8
                or any(not isinstance(value, str) or not value.strip() or len(value) > 500
                       for value in values)):
            raise ValueError("invalid list")
        result[name] = values
    return result


def build_app(config: Config, project: str | Path, worlds, token: str) -> FastAPI:
    """Build an app fixed to one project and an explicit set of world labels.

    Each request creates and closes its own NativeService on one worker thread.
    The token authorizes every allowed world; per-player permissions belong to
    the trusted game-server plugin. Config validation does not initialize storage.
    """
    config.validate()
    cfg = copy.deepcopy(config)
    path = Path(project).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("project must be an existing absolute directory")
    project_path = str(path.resolve())
    if isinstance(worlds, (str, bytes)):
        raise ValueError("worlds must be a nonempty collection of labels")
    try:
        allowed_worlds = frozenset(worlds)
    except TypeError:
        raise ValueError("worlds must be a nonempty collection of labels") from None
    if not allowed_worlds or any(not isinstance(world, str) or not LABEL.fullmatch(world)
                                 for world in allowed_worlds):
        raise ValueError("world labels must be 1-64 ASCII letters, numbers, dot, underscore or hyphen")
    if (not isinstance(token, str) or len(token) < 32
            or any(ord(char) < 33 or ord(char) > 126 for char in token)):
        raise ValueError("token must contain at least 32 printable ASCII characters without spaces")

    app = FastAPI(title="Engram game checkpoint bridge", docs_url=None,
                  redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.add_middleware(_BoundaryMiddleware, token=token)

    def invoke(operation, **params):
        service = None
        try:
            service = NativeService(cfg)
            response = service.handle_request({"id": "bridge", "operation": operation, "params": params})
            if "error" in response:
                return None
            return response["result"]
        except Exception:
            # No driver errors, connection strings or memory contents in HTTP/logs.
            return None
        finally:
            if service is not None:
                try:
                    service.close()
                except Exception:
                    pass

    def unavailable():
        return _error(503, "storage_unavailable", "Engram storage is unavailable; check the initialized store and configuration")

    @app.get("/health")
    def health():
        result = invoke("status")
        if result is None:
            return unavailable()
        return {"ok": True, "status": "ok", "storage": result}

    @app.get("/v1/checkpoints")
    def get_checkpoint(request: Request):
        pairs = list(request.query_params.multi_items())
        if len(pairs) != 3 or {name for name, _value in pairs} != {"world", "kind", "key"}:
            return _error(400, "invalid_params", "Provide exactly world, kind and key")
        try:
            task = _selection(dict(pairs), allowed_worlds)
        except ValueError:
            return _error(400, "invalid_params", "World, kind or key is not allowed")
        result = invoke("session_resume", project_id=project_path, task=task, limit=1)
        if result is None:
            return unavailable()
        # Never return the general memories/context that native resume also reads.
        checkpoints = result["checkpoints"]
        return {"found": bool(checkpoints), "checkpoints": checkpoints}

    @app.post("/v1/checkpoints")
    def save_checkpoint(request: Request):
        if request.query_params:
            return _error(400, "invalid_params", "Save requests do not accept query parameters")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return _error(415, "unsupported_media_type", "Use application/json")
        try:
            payload = json.loads(request.scope["game_bridge_body"].decode("utf-8"),
                                 parse_constant=_reject_constant, object_pairs_hook=_unique_object)
            params = _save_params(payload, allowed_worlds)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return _error(400, "invalid_params", "Invalid checkpoint fields; use an allowed world/kind/key and bounded text")
        result = invoke("session_checkpoint", project_id=project_path, **params)
        return unavailable() if result is None else result

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve explicit game checkpoints from an initialized Engram store")
    parser.add_argument("--config", required=True, help="existing absolute Engram config file")
    parser.add_argument("--project", required=True, help="existing absolute project directory fixed for this bridge")
    parser.add_argument("--world", action="append", required=True, help="allowed world label; repeat for more worlds")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: loopback)")
    parser.add_argument("--port", default=8422, type=int)
    args = parser.parse_args(argv)
    path = Path(args.config).expanduser()
    if not path.is_absolute() or not path.is_file():
        parser.error("--config must be an existing absolute file path")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        config = Config.load(str(path))
        app = build_app(config, args.project, args.world, os.environ.get("GAME_MEMORY_TOKEN", ""))
    except ConfigError:
        parser.error("invalid Engram configuration; run engram config check with the selected file")
    except (ValueError, OSError):
        parser.error("invalid project/world settings or GAME_MEMORY_TOKEN; use an existing absolute project, valid labels and a 32+ character token")
    import uvicorn
    # Query strings contain app keys. Keep the HTTP access log off by default.
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
