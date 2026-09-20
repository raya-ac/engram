"""Reusable sync/async HTTP client for the source game-server checkpoint bridge.

Copy this beside an adapter or import it from the repository. Dependencies:
httpx>=0.27,<1. No retry, model call, app identity inference or implicit save.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from urllib.parse import urlsplit

import httpx

LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
KINDS = frozenset({"rule", "build", "handoff", "note"})
MAX_REQUEST_BYTES = 16_384
MAX_RESPONSE_BYTES = 32_768


class CheckpointError(RuntimeError):
    """Sanitized failure; outcome_unknown warns against blindly replaying writes."""

    def __init__(self, message, *, outcome_unknown=False):
        super().__init__(message)
        self.outcome_unknown = outcome_unknown


def _select(world, kind, key):
    if (not isinstance(world, str) or not LABEL.fullmatch(world)
            or not isinstance(key, str) or not LABEL.fullmatch(key)
            or not isinstance(kind, str) or kind not in KINDS):
        raise ValueError("Use an allowed kind and 1-64 character world/key labels (ASCII letters, digits, ._-)")
    return {"world": world, "kind": kind, "key": key}


def _payload(world, kind, key, summary, decisions, next_steps, blockers):
    data = _select(world, kind, key)
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
        raise ValueError("summary must contain 1-4000 characters")
    data["summary"] = summary
    for field, values in (("decisions", decisions), ("next_steps", next_steps), ("blockers", blockers)):
        if values is not None:
            if (not isinstance(values, list) or len(values) > 8
                    or any(not isinstance(value, str) or not value.strip() or len(value) > 500 for value in values)):
                raise ValueError("checkpoint lists accept up to eight nonempty strings of 500 characters")
            data[field] = values
    try:
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeError:
        raise ValueError("checkpoint text must be valid UTF-8") from None
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("checkpoint request exceeds 16384 UTF-8 bytes")
    return encoded


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("non-finite JSON number")


def _decode(response, data, writing):
    if response.status_code != 200:
        unknown = writing and response.status_code >= 500
        message = ("Bridge rejected the credentials." if response.status_code in {401, 403}
                   else "Bridge returned HTTP " + str(response.status_code) + "; no successful result was confirmed.")
        raise CheckpointError(message, outcome_unknown=unknown)
    try:
        result = json.loads(data, parse_constant=_constant, object_pairs_hook=_unique_object)
        if not isinstance(result, dict):
            raise ValueError("expected object")
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise CheckpointError("Bridge returned invalid JSON; no result was confirmed.",
                              outcome_unknown=writing) from None


def _checked(result, operation, selection=None):
    writing = operation == "save"
    task = None if selection is None else "game:{world}:{kind}:{key}".format(**selection)
    valid = False
    if operation == "health":
        valid = result.get("ok") is True and result.get("status") == "ok"
    elif writing:
        valid = result.get("status") == "saved" and result.get("task") == task
    elif type(result.get("found")) is bool and isinstance(result.get("checkpoints"), list):
        rows = result["checkpoints"]
        valid = not result["found"] and rows == []
        if result["found"] and len(rows) == 1 and isinstance(rows[0], dict):
            item = rows[0]
            valid = (item.get("task") == task and isinstance(item.get("summary"), str)
                     and len(item["summary"]) <= 4000)
            for field in ("decisions", "next_steps", "blockers"):
                values = item.get(field, [])
                valid = valid and isinstance(values, list) and len(values) <= 8 and all(
                    isinstance(value, str) and len(value) <= 500 for value in values)
    if not valid:
        raise CheckpointError("Bridge response did not match the requested operation or note.",
                              outcome_unknown=writing)
    return result


class _Settings:
    def __init__(self, base_url, token, timeout=5, transport=None, *, allow_private_http=False):
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTP(S) origin")
        parts = urlsplit(base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError("base_url must be an HTTP(S) origin without credentials, query or path")
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError("invalid bridge port")
        if (parts.scheme == "http" and parts.hostname not in {"127.0.0.1", "::1"}
                and not allow_private_http):
            raise ValueError("Use numeric loopback HTTP, HTTPS, or explicitly allow private-network HTTP")
        if (not isinstance(token, str) or not 32 <= len(token) <= 256
                or any(ord(char) < 33 or ord(char) > 126 for char in token)):
            raise ValueError("token must contain 32-256 visible ASCII characters without spaces")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be finite and between 0 and 120 seconds")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = float(timeout)
        self.transport = transport

    def _options(self):
        return {"timeout": self.timeout, "follow_redirects": False, "trust_env": False,
                "transport": self.transport, "headers": {
                    "Authorization": "Bearer " + self._token,
                    "Accept": "application/json", "Content-Type": "application/json",
                }}


class CheckpointClient(_Settings):
    """Synchronous adapter for background workers and scripts; never a game/UI thread."""

    def _request(self, method, path, **kwargs):
        writing = method == "POST"
        started = time.monotonic()
        try:
            with httpx.Client(**self._options()) as client:
                with client.stream(method, self.base_url + path, **kwargs) as response:
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        if time.monotonic() - started > self.timeout:
                            raise CheckpointError("Bridge response deadline exceeded.", outcome_unknown=writing)
                        if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise CheckpointError("Bridge response exceeds 32768 bytes.", outcome_unknown=writing)
                        data.extend(chunk)
                    return _decode(response, data, writing)
        except httpx.HTTPError:
            raise CheckpointError("Bridge connection failed; a write may have completed. Read before retrying."
                                  if writing else "Bridge connection failed; context is unavailable.",
                                  outcome_unknown=writing) from None

    def health(self):
        return _checked(self._request("GET", "/health"), "health")

    def recall(self, world, kind, key):
        selection = _select(world, kind, key)
        return _checked(self._request("GET", "/v1/checkpoints", params=selection), "recall", selection)

    def save(self, world, kind, key, summary, *, decisions=None, next_steps=None, blockers=None):
        payload = _payload(world, kind, key, summary, decisions, next_steps, blockers)
        return _checked(self._request("POST", "/v1/checkpoints", content=payload), "save", _select(world, kind, key))


class AsyncCheckpointClient(_Settings):
    """Async adapter for chat bots and application servers, with a total deadline."""

    async def _request(self, method, path, **kwargs):
        writing = method == "POST"
        try:
            async with asyncio.timeout(self.timeout):
                async with httpx.AsyncClient(**self._options()) as client:
                    async with client.stream(method, self.base_url + path, **kwargs) as response:
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
                                raise CheckpointError("Bridge response exceeds 32768 bytes.", outcome_unknown=writing)
                            data.extend(chunk)
                        return _decode(response, data, writing)
        except (httpx.HTTPError, TimeoutError):
            raise CheckpointError("Bridge connection failed; a write may have completed. Read before retrying."
                                  if writing else "Bridge connection failed; context is unavailable.",
                                  outcome_unknown=writing) from None

    async def health(self):
        return _checked(await self._request("GET", "/health"), "health")

    async def recall(self, world, kind, key):
        selection = _select(world, kind, key)
        return _checked(await self._request("GET", "/v1/checkpoints", params=selection), "recall", selection)

    async def save(self, world, kind, key, summary, *, decisions=None, next_steps=None, blockers=None):
        payload = _payload(world, kind, key, summary, decisions, next_steps, blockers)
        return _checked(await self._request("POST", "/v1/checkpoints", content=payload), "save", _select(world, kind, key))
