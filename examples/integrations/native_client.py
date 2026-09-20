"""Small stdlib client for Engram's sequential native JSONL API.

Use an explicit installed Python and existing config. Environment overrides are
inherited normally. This is a local subprocess transport, not an MCP client.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import subprocess
import threading
from typing import Mapping

MAX_REQUEST_BYTES = 65_536
DEFAULT_RESPONSE_BYTES = 2 * 1024 * 1024


class NativeClientError(RuntimeError):
    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


def _reject_constant(_value):
    raise ValueError("non-finite JSON")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class NativeClient:
    """One context-managed process, with sequential, bounded request/response calls.

    A timeout or protocol failure closes the process. A timed-out write may have
    completed: this client never retries operations automatically. Stderr is
    drained and discarded so diagnostics cannot block or contaminate JSONL.
    """

    def __init__(self, *, config: str | Path, python: str | Path,
                 timeout: float = 30.0, max_response_bytes: int = DEFAULT_RESPONSE_BYTES,
                 env: Mapping[str, str] | None = None):
        config_path = Path(config).expanduser()
        python_path = Path(python).expanduser()
        if not config_path.is_absolute() or not config_path.is_file():
            raise ValueError("config must be an existing absolute file path")
        if not python_path.is_absolute() or not python_path.is_file():
            raise ValueError("python must be an existing absolute interpreter path")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes < 256:
            raise ValueError("max_response_bytes must be an integer of at least 256")
        # Preserve the interpreter's venv path: resolving its symlink can select
        # a different Python environment.
        self.command = [str(python_path), "-m", "engram", "--config", str(config_path), "api"]
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self._env = None if env is None else dict(env)
        self._process = None
        self._stderr_thread = None
        self._io_thread = None
        self._closed = False
        self._counter = 0
        self._lock = threading.Lock()

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    def __enter__(self):
        if self._closed or self._process is not None:
            raise NativeClientError("Create a new client for each process lifetime", code="closed")
        try:
            self._process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=self._env, shell=False,
            )
        except OSError:
            raise NativeClientError("Could not start the configured Python interpreter", code="startup_failed") from None
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        return self

    def __exit__(self, *_exc):
        self.close()

    def _drain_stderr(self):
        try:
            while self._process.stderr.read1(4096):
                pass
        except (OSError, ValueError):
            pass

    def _exchange(self, frame, completion):
        try:
            self._process.stdin.write(frame)
            self._process.stdin.flush()
            raw = self._process.stdout.readline(self.max_response_bytes + 1)
            completion.put((True, raw))
        except (OSError, ValueError):
            completion.put((False, None))

    def call(self, operation: str, **params):
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be nonempty text")
        if not self._lock.acquire(blocking=False):
            raise NativeClientError("Calls on one client must be sequential", code="busy")
        try:
            if self._closed or self._process is None:
                raise NativeClientError("Use NativeClient inside a with block", code="closed")
            self._counter += 1
            ident = f"request-{self._counter}"
            try:
                frame = (json.dumps({"id": ident, "operation": operation, "params": params},
                                    ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            except (TypeError, ValueError, UnicodeError, RecursionError):
                raise NativeClientError("Request parameters must be finite UTF-8 JSON", code="invalid_request") from None
            if len(frame) > MAX_REQUEST_BYTES:
                raise NativeClientError("Request exceeds 65,536 bytes including newline", code="request_too_large")
            completion = queue.Queue(maxsize=1)
            self._io_thread = threading.Thread(target=self._exchange, args=(frame, completion), daemon=True)
            self._io_thread.start()
            try:
                ok, raw = completion.get(timeout=self.timeout)
            except queue.Empty:
                self.close()
                raise NativeClientError("Engram timed out; a write may have completed. No retry was attempted", code="timeout") from None
            if not ok or not raw:
                self.close()
                raise NativeClientError("Engram exited or closed its response stream; check config and interpreter", code="process_exit")
            try:
                if len(raw) > self.max_response_bytes or not raw.endswith(b"\n"):
                    raise ValueError("invalid frame")
                response = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant,
                                      parse_float=_finite_float, object_pairs_hook=_unique_object)
                if not isinstance(response, dict) or response.get("id") != ident:
                    raise ValueError("invalid response id")
                if set(response) == {"id", "result"}:
                    return response["result"]
                if set(response) != {"id", "error"}:
                    raise ValueError("invalid response envelope")
                error = response["error"]
                if (not isinstance(error, dict) or set(error) != {"code", "message"}
                        or not isinstance(error["code"], str) or not error["code"]
                        or not isinstance(error["message"], str) or not error["message"]):
                    raise ValueError("invalid error envelope")
            except (ValueError, UnicodeError, RecursionError):
                self.close()
                raise NativeClientError("Invalid or oversized Engram response; process closed", code="protocol_error") from None
            # Native API errors are sanitized by Engram. Do not add raw frames,
            # stderr, request parameters or subprocess exceptions to this error.
            raise NativeClientError(error["message"], code=error["code"])
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        pending = self._io_thread is not None and self._io_thread.is_alive()
        if pending and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        elif not pending:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                process.wait(timeout=1.0)
        for thread in (self._io_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=0.5)
        # Never block closing a buffered stream that a still-running reader owns.
        if self._io_thread is None or not self._io_thread.is_alive():
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        if self._stderr_thread is None or not self._stderr_thread.is_alive():
            process.stderr.close()
