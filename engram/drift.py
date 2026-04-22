"""Memory drift detection — verify claims in memories against filesystem reality.

Inspired by mex's drift checkers: memories that reference files, functions,
commands, or dependencies can become stale when the codebase changes. This
module extracts claims from memory content, validates them against the current
filesystem, and scores overall memory system health.

Claim types:
- path: file/directory paths referenced in memory content
- function: function/class names referenced in memory content
- command: shell commands referenced in memory content
- dependency: package/library names referenced in memory content
"""

from __future__ import annotations

import os
import re
import subprocess
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram.store import Store, Memory


# --- Claim extraction patterns ---

# File paths: things like ~/foo/bar.py, /Users/x/y, src/main.ts, ./config.json
PATH_PATTERN = re.compile(
    r'(?:^|[\s`\'"(])('
    r'(?:~/|/[\w.])'  # starts with ~/ or /word
    r'[\w./\-]+'      # path chars
    r'(?:\.\w{1,10})?' # optional extension
    r')',
    re.MULTILINE,
)

# Also catch relative paths with extensions
RELATIVE_PATH_PATTERN = re.compile(
    r'(?:^|[\s`\'"(])'
    r'((?:[\w\-]+/)+[\w\-]+\.\w{1,10})',  # dir/dir/file.ext
    re.MULTILINE,
)

# Function/class references: FunctionName, ClassName, function_name
# Look for patterns like "function X", "class X", "def X", or backtick-wrapped identifiers
FUNC_PATTERN = re.compile(
    r'(?:function|class|def|method|fn)\s+[`]?(\w{2,60})[`]?',
    re.IGNORECASE,
)

# Backtick-wrapped identifiers that look like code: `someFunction`, `MyClass`
BACKTICK_CODE = re.compile(
    r'`([A-Za-z_]\w{2,60}(?:\.\w+)?(?:\(\))?)`'
)

# Shell commands: npm run X, python X, cargo X, make X, etc.
COMMAND_PATTERN = re.compile(
    r'(?:^|[\s`])((?:npm|yarn|pnpm|bun|make|cargo|python|pip|go|node|npx|tsx|pytest|flask|django|uvicorn|gunicorn)\s+\S+)',
    re.MULTILINE | re.IGNORECASE,
)

# Package/dependency names in context: "uses React", "depends on Flask", etc.
DEPENDENCY_CONTEXT = re.compile(
    r'(?:uses?|depends?\s+on|requires?|imports?|installed?)\s+[`]?(\w[\w\-\.]{1,40})[`]?',
    re.IGNORECASE,
)

# Known file extensions for path validation
KNOWN_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb', '.java',
    '.json', '.yaml', '.yml', '.toml', '.md', '.css', '.scss', '.html',
    '.vue', '.svelte', '.sh', '.bash', '.zsh', '.sql', '.graphql',
    '.proto', '.c', '.cpp', '.h', '.hpp', '.swift', '.kt', '.dart',
}


@dataclass
class Claim:
    """A verifiable claim extracted from a memory."""
    kind: str           # path, function, command, dependency
    value: str          # the actual claimed value
    memory_id: str      # which memory contains this claim
    memory_content: str # first 200 chars for context
    layer: str          # memory layer


@dataclass
class DriftIssue:
    """A single drift issue found during verification."""
    code: str           # DEAD_PATH, MISSING_FUNCTION, STALE_MEMORY, etc.
    severity: str       # error, warning, info
    memory_id: str
    claim: str          # what was claimed
    message: str        # human-readable description
    memory_preview: str # first 150 chars of memory


@dataclass
class DriftReport:
    """Full drift analysis report."""
    score: int                        # 0-100, starts at 100, deducts per issue
    issues: list[DriftIssue]
    memories_checked: int
    claims_extracted: int
    claims_verified: int
    claims_valid: int
    stale_memories: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "issues": [
                {"code": i.code, "severity": i.severity, "memory_id": i.memory_id,
                 "claim": i.claim, "message": i.message, "memory_preview": i.memory_preview}
                for i in self.issues
            ],
            "memories_checked": self.memories_checked,
            "claims_extracted": self.claims_extracted,
            "claims_verified": self.claims_verified,
            "claims_valid": self.claims_valid,
            "stale_memories": self.stale_memories,
            "timestamp": self.timestamp,
        }


# --- Claim extraction ---

def extract_claims(memory: Memory) -> list[Claim]:
    """Extract verifiable claims from a single memory."""
    # skip codebase-layer memories — their paths are relative to the scanned
    # project root, not to cwd, so we can't verify them without that context
    if memory.layer == "codebase":
        return []

    # skip codebase scan artifacts that got promoted to other layers
    # they start with "[project-name] relative/path"
    content = memory.content
    if re.match(r'^\[[\w\-]+\]\s+\S+', content):
        return []
    claims = []
    seen = set()

    def _add(kind: str, value: str):
        key = (kind, value)
        if key not in seen:
            seen.add(key)
            claims.append(Claim(
                kind=kind,
                value=value.strip(),
                memory_id=memory.id,
                memory_content=content[:200],
                layer=memory.layer,
            ))

    # Path claims
    for m in PATH_PATTERN.finditer(content):
        path = m.group(1).strip().rstrip('.,;:)')
        if any(path.endswith(ext) for ext in KNOWN_EXTENSIONS) or '/' in path:
            # skip URLs
            if not path.startswith('http') and not path.startswith('//'):
                _add("path", path)

    for m in RELATIVE_PATH_PATTERN.finditer(content):
        path = m.group(1).strip().rstrip('.,;:)')
        if any(path.endswith(ext) for ext in KNOWN_EXTENSIONS):
            _add("path", path)

    # Function/class claims
    for m in FUNC_PATTERN.finditer(content):
        _add("function", m.group(1))

    # Backtick code references (only if they look like code identifiers)
    FUNC_NOISE = {
        'data', 'staging', 'true', 'false', 'null', 'none', 'undefined',
        'string', 'number', 'boolean', 'object', 'array', 'int', 'float',
        'self', 'this', 'super', 'return', 'import', 'from', 'class',
        'function', 'const', 'let', 'var', 'def', 'async', 'await',
    }
    for m in BACKTICK_CODE.finditer(content):
        val = m.group(1).rstrip('()')
        if ('/' not in val and '.' not in val and not val.startswith('-')
                and val.lower() not in FUNC_NOISE
                and len(val) > 3):
            _add("function", val)

    # Command claims
    for m in COMMAND_PATTERN.finditer(content):
        _add("command", m.group(1).strip())

    # Dependency claims
    DEPENDENCY_NOISE = {
        'it', 'this', 'that', 'the', 'a', 'an', 'is', 'to', 'on', 'in',
        'for', 'with', 'not', 'but', 'and', 'or', 'so', 'if', 'then',
        'be', 'do', 'no', 'yes', 'all', 'some', 'any', 'one', 'two',
        'data', 'file', 'code', 'text', 'name', 'type', 'value', 'key',
        'state', 'each', 'them', 'you', 'your', 'its', 'own', 'new',
        'old', 'set', 'get', 'use', 'run', 'add', 'try', 'can',
        'reconstruction', 'persistsave', 'staging', 'something',
    }
    for m in DEPENDENCY_CONTEXT.finditer(content):
        dep = m.group(1).strip()
        if dep.lower() not in DEPENDENCY_NOISE and len(dep) > 2:
            _add("dependency", dep)

    return claims


def extract_all_claims(store: Store, layers: list[str] | None = None) -> list[Claim]:
    """Extract claims from all non-forgotten memories."""
    all_claims = []

    query = "SELECT * FROM memories WHERE forgotten = 0"
    params = []
    if layers:
        placeholders = ",".join("?" * len(layers))
        query += f" AND layer IN ({placeholders})"
        params = layers

    rows = store.conn.execute(query, params).fetchall()

    for row in rows:
        mem = store._row_to_memory(row)
        claims = extract_claims(mem)
        all_claims.extend(claims)

    return all_claims


# --- Claim verification ---

def _expand_path(path: str) -> str:
    """Expand ~ and resolve path."""
    return os.path.expanduser(path)


def _is_likely_filesystem_path(path: str) -> bool:
    """Distinguish real filesystem paths from URL routes, API paths, etc."""
    has_extension = any(path.endswith(ext) for ext in KNOWN_EXTENSIONS)
    has_relative_root = path.startswith('./')

    # /home/ is a filesystem root on Linux but often appears as a URL route
    # (/home/credits, /home/earn). Only treat as filesystem if deep or has extension.
    if path.startswith('/home/'):
        segments = [s for s in path.split('/') if s]
        if len(segments) <= 2 and not has_extension:
            return False

    has_root = path.startswith(('/Users/', '/home/', '/tmp/', '/var/', '/etc/',
                                '/opt/', '~/'))

    # Web paths that look like file paths but aren't filesystem
    WEB_PATH_PREFIXES = ('/css/', '/js/', '/static/', '/assets/', '/swagger',
                          '/sslvpn/', '/actuator/', '/api/', '/graphql')
    if any(path.startswith(p) for p in WEB_PATH_PREFIXES):
        return False

    # URL routes: /api/foo, /agents/spawn, /overlay — short, no extension, no deep path
    if path.startswith('/') and not has_root:
        # if it has a known extension, it's probably a file
        if has_extension:
            return True
        # if it has 3+ path segments and looks filesystem-y, maybe
        segments = [s for s in path.split('/') if s]
        if len(segments) < 3:
            return False
        # short single-word segments = likely URL route
        if all(len(s) < 20 and '.' not in s for s in segments):
            return False

    return has_root or has_relative_root or has_extension


# Placeholder patterns in paths — these aren't real paths
_TEMPLATE_VARS = re.compile(r'(USERNAME|YYYY|MM|DD|<[^>]+>|\{[^}]+\}|\[.*\])')


def verify_path_claim(claim: Claim) -> DriftIssue | None:
    """Check if a claimed file path exists."""
    path = _expand_path(claim.value)

    # skip template-like paths
    if any(c in path for c in '<>[]{}'):
        return None

    # skip paths with template variables (USERNAME, YYYY, etc.)
    if _TEMPLATE_VARS.search(claim.value):
        return None

    # skip things that aren't actually filesystem paths
    if not _is_likely_filesystem_path(claim.value):
        return None

    # handle paths that might be truncated at a space boundary
    # e.g. "~/Library/Application" is really "~/Library/Application Support/..."
    if not os.path.exists(path):
        # check if a path starting with this prefix exists (space-truncated)
        parent = os.path.dirname(path)
        basename = os.path.basename(path)
        if os.path.isdir(parent):
            try:
                entries = os.listdir(parent)
                if any(e.startswith(basename + ' ') for e in entries):
                    return None  # truncated at space, real path exists
            except OSError:
                pass

        # relative paths (no ~ or / prefix) might be valid in their project
        # context — demote to warning since we can't resolve them
        is_relative = not claim.value.startswith(('/', '~'))
        return DriftIssue(
            code="DEAD_PATH",
            severity="warning" if is_relative else "error",
            memory_id=claim.memory_id,
            claim=claim.value,
            message=f"Path does not exist: {claim.value}",
            memory_preview=claim.memory_content[:150],
        )
    return None


def verify_function_claim(claim: Claim, search_roots: list[str] | None = None) -> DriftIssue | None:
    """Check if a claimed function/class name exists in the codebase."""
    if not search_roots:
        return None  # can't verify without a search root

    name = claim.value
    # skip very common names that would match everywhere
    if name.lower() in ('main', 'init', 'test', 'run', 'start', 'stop', 'get', 'set', 'new'):
        return None

    for root in search_roots:
        root_path = _expand_path(root)
        if not os.path.isdir(root_path):
            continue
        try:
            result = subprocess.run(
                ['grep', '-rl', '--include=*.py', '--include=*.ts', '--include=*.js',
                 '--include=*.go', '--include=*.rs', '--include=*.java',
                 '--include=*.tsx', '--include=*.jsx', '--include=*.rb',
                 '-m', '1',  # stop after first match
                 name, root_path],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return None  # found it
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return DriftIssue(
        code="MISSING_FUNCTION",
        severity="warning",
        memory_id=claim.memory_id,
        claim=name,
        message=f"Function/class '{name}' not found in searched directories",
        memory_preview=claim.memory_content[:150],
    )


def verify_command_claim(claim: Claim, project_root: str | None = None) -> DriftIssue | None:
    """Check if a claimed command's script exists (for npm/yarn/make)."""
    cmd = claim.value
    parts = cmd.split()
    if len(parts) < 2:
        return None

    runner = parts[0].lower()
    script = parts[-1]  # the script name is usually the last part

    if runner in ('npm', 'yarn', 'pnpm', 'bun') and 'run' in parts:
        # check package.json for the script
        if project_root:
            pkg_path = os.path.join(_expand_path(project_root), 'package.json')
            if os.path.exists(pkg_path):
                import json
                try:
                    with open(pkg_path) as f:
                        pkg = json.load(f)
                    scripts = pkg.get('scripts', {})
                    if script not in scripts:
                        return DriftIssue(
                            code="DEAD_COMMAND",
                            severity="warning",
                            memory_id=claim.memory_id,
                            claim=cmd,
                            message=f"Script '{script}' not found in package.json",
                            memory_preview=claim.memory_content[:150],
                        )
                except (json.JSONDecodeError, OSError):
                    pass

    elif runner == 'make':
        if project_root:
            makefile = os.path.join(_expand_path(project_root), 'Makefile')
            if os.path.exists(makefile):
                try:
                    with open(makefile) as f:
                        content = f.read()
                    if f'{script}:' not in content:
                        return DriftIssue(
                            code="DEAD_COMMAND",
                            severity="warning",
                            memory_id=claim.memory_id,
                            claim=cmd,
                            message=f"Make target '{script}' not found in Makefile",
                            memory_preview=claim.memory_content[:150],
                        )
                except OSError:
                    pass

    return None


def check_staleness(memory: Memory, days_warn: int = 60, days_error: int = 180) -> DriftIssue | None:
    """Check if a memory is stale based on age and access patterns."""
    age_days = (time.time() - memory.created_at) / 86400
    last_access_days = (time.time() - memory.last_accessed) / 86400

    # only flag memories that are both old AND unaccessed
    if last_access_days >= days_error and age_days >= days_error:
        return DriftIssue(
            code="STALE_MEMORY",
            severity="error",
            memory_id=memory.id,
            claim=f"age={int(age_days)}d, last_access={int(last_access_days)}d",
            message=f"Memory hasn't been accessed in {int(last_access_days)} days",
            memory_preview=memory.content[:150],
        )
    elif last_access_days >= days_warn and age_days >= days_warn:
        return DriftIssue(
            code="STALE_MEMORY",
            severity="warning",
            memory_id=memory.id,
            claim=f"age={int(age_days)}d, last_access={int(last_access_days)}d",
            message=f"Memory hasn't been accessed in {int(last_access_days)} days",
            memory_preview=memory.content[:150],
        )
    return None


def check_invalidated_still_referenced(store: Store) -> list[DriftIssue]:
    """Find memories that reference entities also referenced by invalidated memories."""
    issues = []
    # find invalidated memory IDs
    inv_rows = store.conn.execute(
        "SELECT id, content FROM memories WHERE forgotten = 0 AND json_extract(metadata, '$.invalidated') = 1"
    ).fetchall()

    for row in inv_rows:
        issues.append(DriftIssue(
            code="INVALIDATED_ACTIVE",
            severity="info",
            memory_id=row["id"],
            claim="invalidated=true but not forgotten",
            message="Memory is marked invalidated but still active — consider forgetting it",
            memory_preview=row["content"][:150],
        ))
    return issues


# --- Scoring ---

SEVERITY_COST = {
    "error": 10,
    "warning": 3,
    "info": 1,
}


def compute_drift_score(issues: list[DriftIssue]) -> int:
    """Compute drift score from 0-100. Starts at 100, deducts per issue."""
    score = 100
    for issue in issues:
        score -= SEVERITY_COST.get(issue.severity, 1)
    return max(0, min(100, score))


# --- Main entry points ---

def run_drift_check(
    store: Store,
    search_roots: list[str] | None = None,
    project_root: str | None = None,
    layers: list[str] | None = None,
    check_functions: bool = True,
) -> DriftReport:
    """Run full drift detection across all memories.

    Args:
        store: Memory store
        search_roots: Directories to search for function/class verification
        project_root: Project root for command verification (package.json, Makefile)
        layers: Memory layers to check (default: all)
        check_functions: Whether to grep for function names (slower)
    """
    all_issues: list[DriftIssue] = []
    all_claims: list[Claim] = []
    claims_verified = 0
    claims_valid = 0
    stale_count = 0

    # Extract claims from all memories
    claims = extract_all_claims(store, layers)
    all_claims = claims

    # Verify path claims
    path_claims = [c for c in claims if c.kind == "path"]
    for claim in path_claims:
        issue = verify_path_claim(claim)
        claims_verified += 1
        if issue:
            all_issues.append(issue)
        else:
            claims_valid += 1

    # Verify function claims (optional, slower)
    if check_functions and search_roots:
        func_claims = [c for c in claims if c.kind == "function"]
        # limit to avoid long grep sessions
        for claim in func_claims[:50]:
            issue = verify_function_claim(claim, search_roots)
            claims_verified += 1
            if issue:
                all_issues.append(issue)
            else:
                claims_valid += 1

    # Verify command claims
    cmd_claims = [c for c in claims if c.kind == "command"]
    for claim in cmd_claims:
        issue = verify_command_claim(claim, project_root)
        claims_verified += 1
        if issue:
            all_issues.append(issue)
        else:
            claims_valid += 1

    # Check staleness
    query = "SELECT * FROM memories WHERE forgotten = 0"
    params = []
    if layers:
        placeholders = ",".join("?" * len(layers))
        query += f" AND layer IN ({placeholders})"
        params = layers

    rows = store.conn.execute(query, params).fetchall()
    memories_checked = len(rows)

    for row in rows:
        mem = store._row_to_memory(row)
        issue = check_staleness(mem)
        if issue:
            all_issues.append(issue)
            stale_count += 1

    # Check invalidated-but-active
    all_issues.extend(check_invalidated_still_referenced(store))

    score = compute_drift_score(all_issues)

    return DriftReport(
        score=score,
        issues=all_issues,
        memories_checked=memories_checked,
        claims_extracted=len(all_claims),
        claims_verified=claims_verified,
        claims_valid=claims_valid,
        stale_memories=stale_count,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def auto_fix_drift(store: Store, report: DriftReport, dry_run: bool = True) -> dict:
    """Auto-fix drift issues by invalidating or forgetting drifted memories.

    Args:
        store: Memory store
        report: Drift report from run_drift_check
        dry_run: If True, only report what would be done

    Returns:
        Dict with fix actions taken/planned
    """
    actions = []

    def _begin_write_transaction() -> None:
        attempts = 40
        for attempt in range(attempts):
            try:
                store.conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt == attempts - 1:
                    raise
                time.sleep(0.25)

    def _execute_write(sql: str, params: tuple[Any, ...]) -> None:
        store.conn.execute(sql, params)

    if not dry_run:
        _begin_write_transaction()

    for issue in report.issues:
        if issue.code == "DEAD_PATH":
            action = {
                "memory_id": issue.memory_id,
                "action": "invalidate",
                "reason": f"Referenced path no longer exists: {issue.claim}",
                "preview": issue.memory_preview,
            }
            if not dry_run:
                # Mark as invalidated with drift reason
                mem = store.get_memory(issue.memory_id)
                if mem:
                    mem.metadata["invalidated"] = True
                    mem.metadata["invalidation_reason"] = f"drift:dead_path:{issue.claim}"
                    mem.metadata["invalidated_at"] = time.time()
                    import json
                    _execute_write("UPDATE memories SET metadata = ? WHERE id = ?",
                                   (json.dumps(mem.metadata), mem.id))
            actions.append(action)

        elif issue.code == "STALE_MEMORY" and issue.severity == "error":
            action = {
                "memory_id": issue.memory_id,
                "action": "flag_stale",
                "reason": issue.message,
                "preview": issue.memory_preview,
            }
            if not dry_run:
                mem = store.get_memory(issue.memory_id)
                if mem:
                    mem.metadata["drift_flagged"] = True
                    mem.metadata["drift_flagged_at"] = time.time()
                    mem.metadata["drift_reason"] = issue.message
                    import json
                    _execute_write("UPDATE memories SET metadata = ? WHERE id = ?",
                                   (json.dumps(mem.metadata), mem.id))
            actions.append(action)

        elif issue.code == "INVALIDATED_ACTIVE":
            action = {
                "memory_id": issue.memory_id,
                "action": "forget",
                "reason": "Invalidated memory still active",
                "preview": issue.memory_preview,
            }
            if not dry_run:
                _execute_write("UPDATE memories SET forgotten = 1 WHERE id = ?", (issue.memory_id,))
            actions.append(action)

    if not dry_run:
        store.conn.commit()
        store.invalidate_embedding_cache()
        store.invalidate_search_cache()

    return {
        "dry_run": dry_run,
        "total_actions": len(actions),
        "actions": actions,
        "invalidated": sum(1 for a in actions if a["action"] == "invalidate"),
        "flagged_stale": sum(1 for a in actions if a["action"] == "flag_stale"),
        "forgotten": sum(1 for a in actions if a["action"] == "forget"),
    }
