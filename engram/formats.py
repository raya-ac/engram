"""Chat format parsers: Claude Code JSONL, Claude.ai JSON, ChatGPT, Slack, plain text, markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "claude_code"
    elif suffix == ".json":
        content = path.read_text(errors="replace")[:500]
        if '"mapping"' in content:
            return "chatgpt"
        if '"chat_messages"' in content or '"uuid"' in content:
            return "claude_ai"
        if '"messages"' in content and '"channel"' in content:
            return "slack"
        return "json_generic"
    elif suffix == ".md":
        return "markdown"
    elif suffix == ".txt":
        return "plaintext"
    elif suffix == ".pdf":
        return "pdf"
    elif suffix in (".mbox", ".eml"):
        return "email"
    return "plaintext"


def parse_file(path: Path) -> list[dict]:
    """Parse a file into exchange units. Returns list of {role, content, timestamp?}."""
    fmt = detect_format(path)
    parsers = {
        "claude_code": _parse_claude_code,
        "claude_ai": _parse_claude_ai,
        "chatgpt": _parse_chatgpt,
        "slack": _parse_slack,
        "markdown": _parse_markdown,
        "plaintext": _parse_plaintext,
        "pdf": _parse_pdf,
        "json_generic": _parse_json_generic,
    }
    parser = parsers.get(fmt, _parse_plaintext)
    return parser(path)


def _parse_claude_code(path: Path) -> list[dict]:
    exchanges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                role = obj.get("role", "unknown")
                content = obj.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if content.strip():
                    exchanges.append({"role": role, "content": content.strip()})
            except json.JSONDecodeError:
                continue
    return exchanges


def _parse_claude_ai(path: Path) -> list[dict]:
    data = json.loads(path.read_text(errors="replace"))
    exchanges = []
    messages = data if isinstance(data, list) else data.get("chat_messages", [])
    for msg in messages:
        role = msg.get("sender", msg.get("role", "unknown"))
        content = msg.get("text", msg.get("content", ""))
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        if content.strip():
            exchanges.append({"role": role, "content": content.strip()})
    return exchanges


def _parse_chatgpt(path: Path) -> list[dict]:
    data = json.loads(path.read_text(errors="replace"))
    exchanges = []
    mapping = data.get("mapping", {})
    # build ordered list from tree
    nodes = sorted(mapping.values(), key=lambda n: n.get("create_time", 0) or 0)
    for node in nodes:
        msg = node.get("message")
        if not msg:
            continue
        role = msg.get("author", {}).get("role", "unknown")
        content = msg.get("content", {})
        if isinstance(content, dict):
            parts = content.get("parts", [])
            text = " ".join(str(p) for p in parts)
        else:
            text = str(content)
        if text.strip():
            exchanges.append({"role": role, "content": text.strip()})
    return exchanges


def _parse_slack(path: Path) -> list[dict]:
    data = json.loads(path.read_text(errors="replace"))
    messages = data if isinstance(data, list) else data.get("messages", [])
    exchanges = []
    for msg in messages:
        user = msg.get("user", msg.get("username", "unknown"))
        text = msg.get("text", "")
        if text.strip():
            exchanges.append({
                "role": user,
                "content": text.strip(),
                "timestamp": msg.get("ts"),
            })
    return exchanges


def _parse_markdown(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    # split by headers
    sections = re.split(r"^(#{1,3}\s+.+)$", text, flags=re.MULTILINE)
    exchanges = []
    current_header = None
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,3}\s+", part):
            current_header = part
        else:
            content = part
            if current_header:
                content = f"{current_header}\n{content}"
            if content.strip():
                exchanges.append({"role": "document", "content": content.strip()})
    if not exchanges and text.strip():
        exchanges.append({"role": "document", "content": text.strip()})
    return exchanges


def _parse_plaintext(path: Path) -> list[dict]:
    text = path.read_text(errors="replace").strip()
    if not text:
        return []
    return [{"role": "document", "content": text}]


def _parse_pdf(path: Path) -> list[dict]:
    try:
        import pymupdf
        doc = pymupdf.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return [{"role": "document", "content": text.strip()}]
    except ImportError:
        pass
    return []


def _parse_json_generic(path: Path) -> list[dict]:
    data = json.loads(path.read_text(errors="replace"))
    if isinstance(data, list):
        return [{"role": "document", "content": json.dumps(item, indent=2)} for item in data[:50]]
    return [{"role": "document", "content": json.dumps(data, indent=2)}]


def group_exchanges(exchanges: list[dict]) -> list[str]:
    """Group consecutive user/assistant exchanges into Q+A pairs."""
    groups = []
    i = 0
    while i < len(exchanges):
        ex = exchanges[i]
        if ex["role"] in ("human", "user") and i + 1 < len(exchanges):
            next_ex = exchanges[i + 1]
            if next_ex["role"] in ("assistant", "bot"):
                groups.append(f"Q: {ex['content']}\nA: {next_ex['content']}")
                i += 2
                continue
        groups.append(ex["content"])
        i += 1
    return groups
