"""Explicit local task notes and project context; no task-service or LLM calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from native_client import NativeClient, NativeClientError


def _project(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("project must be an existing absolute directory")
    return str(path.resolve())


def _summary(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(16_001)
    if len(raw) > 16_000:
        raise ValueError("summary must contain at most 4,000 characters")
    text = raw.decode("utf-8").strip()
    if not text or len(text) > 4000:
        raise ValueError("summary must contain 1–4,000 characters")
    return text


def _items(values, name):
    if len(values) > 8 or any(not item.strip() or len(item.strip()) > 500 for item in values):
        raise ValueError(f"{name} accepts at most eight nonempty strings of 500 characters")
    return [item.strip() for item in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing absolute Engram config path")
    parser.add_argument("--python", required=True, help="Absolute Python interpreter with Engram installed")
    parser.add_argument("--project", required=True, help="Absolute task project directory")
    parser.add_argument("--timeout", type=float, default=30.0)
    commands = parser.add_subparsers(dest="command", required=True)
    context = commands.add_parser("context", help="Read recent explicitly project-owned context")
    context.add_argument("--limit", type=int, choices=range(1, 21), default=8)
    resume = commands.add_parser("resume", help="Read project context and one exact task checkpoint")
    resume.add_argument("task", help="Stable task key, for example issue:APP-42")
    resume.add_argument("--limit", type=int, choices=range(1, 21), default=8)
    save = commands.add_parser("save", help="Save an explicitly reviewed task summary")
    save.add_argument("task")
    save.add_argument("--summary-file", required=True, help="UTF-8 text already reviewed for saving")
    save.add_argument("--decision", action="append", default=[])
    save.add_argument("--next-step", action="append", default=[])
    save.add_argument("--blocker", action="append", default=[])
    save.add_argument("--yes", action="store_true", help="Confirm saving these supplied notes")
    clear = commands.add_parser("clear", help="Delete only this project's exact task checkpoint")
    clear.add_argument("task")
    clear.add_argument("--yes", action="store_true", help="Confirm deleting this task checkpoint")
    args = parser.parse_args(argv)
    if args.command in {"save", "clear"} and not args.yes:
        parser.error("save and clear require --yes after reviewing the supplied task and notes")
    try:
        project = _project(args.project)
        params = {"project_id": project}
        if args.command == "context":
            operation = "recall"
            params["limit"] = args.limit
        else:
            task = args.task.strip()
            if not task or len(task) > 200:
                raise ValueError("task must contain 1–200 characters")
            params["task"] = task
            operation = "session_resume" if args.command == "resume" else "session_checkpoint"
            if args.command == "resume":
                params["limit"] = args.limit
            elif args.command == "clear":
                params["action"] = "clear"
            else:
                params.update(action="save", summary=_summary(args.summary_file),
                              decisions=_items(args.decision, "--decision"),
                              next_steps=_items(args.next_step, "--next-step"),
                              blockers=_items(args.blocker, "--blocker"))
        with NativeClient(config=args.config, python=args.python, timeout=args.timeout) as client:
            discovery = client.call("operations")
            if (not isinstance(discovery, dict) or discovery.get("protocol") != "engram-jsonl"
                    or discovery.get("version") != 1 or not isinstance(discovery.get("operations"), list)):
                raise NativeClientError("Installed Engram does not advertise the required native operation", code="unsupported")
            available = [item.get("name") for item in discovery["operations"] if isinstance(item, dict)]
            if operation not in available:
                raise NativeClientError("Installed Engram does not advertise the required native operation", code="unsupported")
            client.call("status")  # A spawned process alone does not prove storage readiness.
            result = client.call(operation, **params)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (OSError, UnicodeError):
        print("Could not read the config, interpreter or UTF-8 summary file", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except NativeClientError as exc:
        print(f"Engram {exc.code}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; no automatic retry was attempted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
