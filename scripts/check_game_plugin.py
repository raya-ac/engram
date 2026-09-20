"""Run the Java source integration tests against a real isolated Engram bridge.

This checks transport/storage, not a running Minecraft server. Requires Java 21
and Maven; intended for the hosted source-integrations workflow.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from engram.config import Config
from engram.store import Store


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="engram-java-bridge-") as directory:
        work = Path(directory)
        project = work / "server"
        project.mkdir()
        settings = {"db_path": str(work / "memory.db"),
                    "embedding_dim": 4, "ann": {"enabled": False,
                    "index_path": str(work / "unused.index")}}
        config_path = work / "config.json"
        config_path.write_text(json.dumps(settings))
        config_path.chmod(0o600)
        store = Store(Config.from_mapping(settings, apply_environment=False))
        try:
            store.init_db()
        finally:
            store.close()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        token = secrets.token_urlsafe(32)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("ENGRAM_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
        env.update(GAME_MEMORY_TOKEN=token, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        url = f"http://127.0.0.1:{port}"
        opener = build_opener(ProxyHandler({}))
        with (work / "bridge.log").open("wb") as log:
            process = subprocess.Popen([
                sys.executable, str(root / "integrations/game_server_bridge.py"),
                "--config", str(config_path), "--project", str(project),
                "--world", "world", "--port", str(port),
            ], cwd=root, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("isolated bridge exited before readiness")
                    try:
                        with opener.open(Request(url + "/health", headers={
                            "Authorization": "Bearer " + token}), timeout=1) as response:
                            health = json.load(response)
                        if health.get("status") == "ok" or health.get("ok") is True:
                            break
                    except (URLError, TimeoutError, OSError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("isolated bridge did not become ready")
                    time.sleep(0.1)
                env.update(ENGRAM_TEST_BRIDGE_URL=url, ENGRAM_TEST_BRIDGE_TOKEN=token)
                subprocess.run([
                    "mvn", "-B", "-ntp", "-f", str(root / "integrations/minecraft/paper/pom.xml"),
                    "verify",
                ], cwd=root, env=env, check=True, timeout=600)
                print("Java HTTP client and real Engram storage verified; Minecraft host acceptance remains separate.")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
