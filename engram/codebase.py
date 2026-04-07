"""Codebase layer — compressed code knowledge extraction.

Scans a project directory and extracts:
- File tree structure
- Function/class signatures (not full bodies)
- Import/dependency graph
- Config files and entry points
- Architecture patterns

Stores as compact memories in the codebase layer, using way less
tokens than raw code while keeping the information needed to
work with the project.
"""

from __future__ import annotations

import os
import re
import uuid
import time
from pathlib import Path
from collections import defaultdict

from engram.store import Store, Memory, MemoryLayer, SourceType

# file extensions to scan
CODE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".swift", ".rs", ".go",
    ".java", ".kt", ".rb", ".php", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".vue", ".svelte",
}
CONFIG_EXTS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".lock", ".xml",
}
DOC_EXTS = {".md", ".txt", ".rst"}

IGNORE_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "dist",
    "build", ".next", ".nuxt", "target", ".gradle", ".idea",
    "Pods", "DerivedData", ".build",
}

# patterns for extracting signatures
PY_CLASS = re.compile(r"^class\s+(\w+)(?:\(([^)]*)\))?:", re.MULTILINE)
PY_FUNC = re.compile(r"^(?:async\s+)?def\s+(\w+)\(([^)]*)\)(?:\s*->\s*(\S+))?:", re.MULTILINE)
PY_IMPORT = re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)$", re.MULTILINE)

JS_FUNC = re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\(([^)]*)\)", re.MULTILINE)
JS_CLASS = re.compile(r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE)
JS_ARROW = re.compile(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>", re.MULTILINE)
JS_IMPORT = re.compile(r"import\s+(?:{([^}]+)}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)

SWIFT_FUNC = re.compile(r"(?:public\s+|private\s+|internal\s+)?(?:static\s+)?func\s+(\w+)\(([^)]*)\)(?:\s*->\s*(\S+))?", re.MULTILINE)
SWIFT_CLASS = re.compile(r"(?:public\s+|private\s+)?(?:class|struct|enum|protocol)\s+(\w+)(?:\s*:\s*([^{]+))?", re.MULTILINE)


def scan_codebase(root: str | Path, store: Store, project_name: str | None = None) -> dict:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return {"error": f"Not a directory: {root}"}

    project_name = project_name or root.name
    stats = {"files_scanned": 0, "memories_created": 0, "functions": 0, "classes": 0}

    # 1. File tree
    tree = _build_tree(root)
    tree_mem = Memory(
        id=str(uuid.uuid4()),
        content=f"[Codebase: {project_name}] File tree:\n{tree}",
        source_file=str(root),
        source_type=SourceType.CODE_SCAN,
        layer=MemoryLayer.CODEBASE,
        importance=0.6,
        metadata={"project": project_name, "type": "file_tree"},
    )
    _save_code_memory(store, tree_mem)
    stats["memories_created"] += 1

    # 2. Scan each file for signatures
    all_signatures = []
    all_imports = defaultdict(list)

    for fpath in _iter_files(root, CODE_EXTS):
        rel = fpath.relative_to(root)
        try:
            content = fpath.read_text(errors="replace")
        except Exception:
            continue

        stats["files_scanned"] += 1
        sigs = _extract_signatures(content, fpath.suffix)

        if sigs["classes"] or sigs["functions"]:
            parts = [f"[{project_name}] {rel}"]
            for cls in sigs["classes"]:
                parts.append(f"  class {cls['name']}" + (f"({cls['extends']})" if cls.get("extends") else ""))
                stats["classes"] += 1
            for fn in sigs["functions"]:
                ret = f" -> {fn['returns']}" if fn.get("returns") else ""
                parts.append(f"  {'async ' if fn.get('async') else ''}def {fn['name']}({fn['params']}){ret}")
                stats["functions"] += 1

            sig_mem = Memory(
                id=str(uuid.uuid4()),
                content="\n".join(parts),
                source_file=str(fpath),
                source_type=SourceType.CODE_SCAN,
                layer=MemoryLayer.CODEBASE,
                importance=0.5,
                metadata={"project": project_name, "type": "signatures", "file": str(rel)},
            )
            _save_code_memory(store, sig_mem)
            stats["memories_created"] += 1

        for imp in sigs["imports"]:
            all_imports[str(rel)].append(imp)

    # 3. Dependency summary
    if all_imports:
        dep_parts = [f"[{project_name}] Dependencies:"]
        # count external imports
        ext_deps = defaultdict(int)
        for file, imports in all_imports.items():
            for imp in imports:
                if imp.get("module") and not imp["module"].startswith("."):
                    top = imp["module"].split(".")[0]
                    ext_deps[top] += 1
        for dep, count in sorted(ext_deps.items(), key=lambda x: -x[1])[:30]:
            dep_parts.append(f"  {dep} ({count} imports)")

        dep_mem = Memory(
            id=str(uuid.uuid4()),
            content="\n".join(dep_parts),
            source_file=str(root),
            source_type=SourceType.CODE_SCAN,
            layer=MemoryLayer.CODEBASE,
            importance=0.5,
            metadata={"project": project_name, "type": "dependencies"},
        )
        _save_code_memory(store, dep_mem)
        stats["memories_created"] += 1

    # 4. Config/entry point files (store content compressed)
    for fpath in _iter_files(root, CONFIG_EXTS):
        rel = fpath.relative_to(root)
        if rel.parts[0] in IGNORE_DIRS:
            continue
        # only top-level config files
        if len(rel.parts) > 2:
            continue
        try:
            content = fpath.read_text(errors="replace")[:2000]
        except Exception:
            continue
        if len(content.strip()) < 10:
            continue

        cfg_mem = Memory(
            id=str(uuid.uuid4()),
            content=f"[{project_name}] Config {rel}:\n{content}",
            source_file=str(fpath),
            source_type=SourceType.CODE_SCAN,
            layer=MemoryLayer.CODEBASE,
            importance=0.4,
            metadata={"project": project_name, "type": "config", "file": str(rel)},
        )
        _save_code_memory(store, cfg_mem)
        stats["memories_created"] += 1

    return stats


def _save_code_memory(store: Store, mem: Memory):
    from engram.embeddings import embed_documents
    emb = embed_documents([mem.content])
    if emb.size > 0:
        mem.embedding = emb[0]
    store.save_memory(mem)


def _build_tree(root: Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> str:
    if depth >= max_depth:
        return prefix + "...\n"

    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    for i, entry in enumerate(entries):
        if entry.name.startswith(".") and entry.name not in (".env", ".gitignore"):
            continue
        if entry.name in IGNORE_DIRS:
            continue
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            child_count = sum(1 for _ in entry.iterdir() if not _.name.startswith("."))
            lines.append(f"{prefix}{connector}{entry.name}/ ({child_count})")
            extension = "    " if is_last else "│   "
            sub = _build_tree(entry, prefix + extension, max_depth, depth + 1)
            if sub.strip():
                lines.append(sub.rstrip())
        else:
            size = entry.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size // (1024*1024)}MB"
            elif size > 1024:
                size_str = f"{size // 1024}KB"
            else:
                size_str = f"{size}B"
            lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

    return "\n".join(lines)


def _iter_files(root: Path, exts: set) -> list[Path]:
    files = []
    for fpath in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in fpath.parts):
            continue
        if fpath.is_file() and fpath.suffix in exts:
            files.append(fpath)
    return sorted(files)


def _extract_signatures(content: str, suffix: str) -> dict:
    result = {"classes": [], "functions": [], "imports": []}

    if suffix == ".py":
        for m in PY_CLASS.finditer(content):
            result["classes"].append({"name": m.group(1), "extends": m.group(2) or ""})
        for m in PY_FUNC.finditer(content):
            # skip private methods for compression
            if m.group(1).startswith("__") and m.group(1) != "__init__":
                continue
            result["functions"].append({
                "name": m.group(1),
                "params": _compress_params(m.group(2)),
                "returns": m.group(3) or "",
                "async": "async" in content[max(0, m.start()-6):m.start()],
            })
        for m in PY_IMPORT.finditer(content):
            result["imports"].append({"module": m.group(1) or "", "names": m.group(2).strip()})

    elif suffix in (".js", ".ts", ".tsx", ".jsx"):
        for m in JS_CLASS.finditer(content):
            result["classes"].append({"name": m.group(1), "extends": m.group(2) or ""})
        for m in JS_FUNC.finditer(content):
            result["functions"].append({
                "name": m.group(1),
                "params": _compress_params(m.group(2)),
                "async": "async" in content[max(0, m.start()-6):m.start()],
            })
        for m in JS_ARROW.finditer(content):
            result["functions"].append({
                "name": m.group(1),
                "params": _compress_params(m.group(2)),
                "async": "async" in content[max(0, m.start()-6):m.start()],
            })
        for m in JS_IMPORT.finditer(content):
            names = m.group(1) or m.group(2) or ""
            result["imports"].append({"module": m.group(3), "names": names.strip()})

    elif suffix == ".swift":
        for m in SWIFT_CLASS.finditer(content):
            result["classes"].append({"name": m.group(1), "extends": (m.group(2) or "").strip()})
        for m in SWIFT_FUNC.finditer(content):
            result["functions"].append({
                "name": m.group(1),
                "params": _compress_params(m.group(2)),
                "returns": m.group(3) or "",
            })

    return result


def _compress_params(params: str) -> str:
    """Compress parameter lists — keep names and types, drop defaults."""
    if not params or not params.strip():
        return ""
    # remove default values
    params = re.sub(r"\s*=\s*[^,)]+", "", params)
    # compress whitespace
    params = re.sub(r"\s+", " ", params).strip()
    # truncate if very long
    if len(params) > 100:
        params = params[:97] + "..."
    return params
