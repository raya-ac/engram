#!/usr/bin/env python3
"""Check release archives without extracting or importing project code.

The optional installed-wheel smoke runs in a separate environment and directory.
This script uses only the Python standard library; build and twine stay in CI.
"""
from __future__ import annotations

import argparse
import ast
import configparser
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


class DistributionError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionError(message)


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name, f"unsafe archive path: {name!r}")
    forbidden = {"scratch", "artifacts", "private", "tmp", "cache", "__pycache__",
                 ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache"}
    require(not any(p.lower() in forbidden or p.startswith("pytest-of-")
                    or p.startswith(".env") for p in path.parts),
            f"private or generated archive path: {name}")
    require(path.name not in {"config.yaml", "config.yml"},
            f"private configuration in archive: {name}")
    return path


def source_files(root: Path, directory: str, pattern: str) -> dict[str, bytes]:
    files = {}
    for path in (root / directory).glob(pattern):
        require(not path.is_symlink(), f"source symlink is not allowed: {path.relative_to(root)}")
        name = path.relative_to(root).as_posix()
        safe_path(name)
        files[name] = path.read_bytes()
    return files


def module_version(content: bytes) -> str:
    for statement in ast.parse(content).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            require(isinstance(value, str), "__version__ must be a literal string")
            return value
    raise DistributionError("missing __version__")


def metadata_version(content: bytes, version: str, label: str) -> None:
    metadata = BytesParser().parsebytes(content)
    require(metadata["Name"] == "engram-memory-system", f"{label}: wrong project name")
    require(metadata["Version"] == version, f"{label}: version differs from {version}")


def same_files(actual: dict[str, bytes], expected: dict[str, bytes], label: str) -> None:
    unexpected = sorted(actual.keys() - expected.keys())
    missing = sorted(expected.keys() - actual.keys())
    require(not unexpected, f"{label}: unexpected files: {unexpected}")
    require(not missing, f"{label}: missing files: {missing}")
    changed = sorted(name for name in expected if actual[name] != expected[name])
    require(not changed, f"{label}: files differ from checkout: {changed}")


def check_archives(root: Path, dist: Path, expected_tag: str | None) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = project["version"]
    require(module_version((root / "engram/__init__.py").read_bytes()) == version,
            "pyproject.toml and engram.__version__ disagree")
    if expected_tag:
        require(expected_tag == f"v{version}", f"release tag must be v{version}")
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    require(len(wheels) == 1 and len(sdists) == 1,
            "expected exactly one wheel and one .tar.gz sdist in the distribution directory")
    require(wheels[0].name.startswith(f"engram_memory_system-{version}-"), "wheel filename version differs")
    require(sdists[0].name == f"engram_memory_system-{version}.tar.gz", "sdist filename version differs")
    package = source_files(root, "engram", "**/*.py")
    package.update(source_files(root, "engram/web/templates", "**/*.html"))
    require("engram/web/templates/index.html" in package, "web workspace template is missing")
    info = f"engram_memory_system-{version}.dist-info"
    metadata_names = {f"{info}/{name}" for name in (
        "METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE", "LICENSE",
    )}
    wheel_files = {}
    with zipfile.ZipFile(wheels[0]) as archive:
        for member in archive.infolist():
            safe_path(member.filename)
            require(not stat.S_ISLNK(member.external_attr >> 16),
                    f"wheel symlink: {member.filename}")
            if member.is_dir():
                continue
            require(member.filename not in wheel_files, f"duplicate wheel member: {member.filename}")
            wheel_files[member.filename] = archive.read(member)
    require(set(wheel_files) <= set(package) | metadata_names,
            f"wheel: unintended files: {sorted(set(wheel_files) - set(package) - metadata_names)}")
    for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt"):
        require(f"{info}/{name}" in wheel_files, f"wheel: missing {name}")
    require(any(name in wheel_files for name in (f"{info}/LICENSE", f"{info}/licenses/LICENSE")),
            "wheel: license is missing")
    metadata_version(wheel_files[f"{info}/METADATA"], version, "wheel metadata")
    entrypoints = configparser.ConfigParser()
    entrypoints.read_string(wheel_files[f"{info}/entry_points.txt"].decode())
    require(entrypoints.get("console_scripts", "engram", fallback="") == "engram.cli:main",
            "wheel: engram console entry point is missing or changed")
    same_files({name: data for name, data in wheel_files.items() if name in package}, package, "wheel")

    sdist_files = {}
    prefix = f"engram_memory_system-{version}"
    with tarfile.open(sdists[0], "r:gz") as archive:
        for member in archive.getmembers():
            path = safe_path(member.name)
            require(path.parts[0] == prefix, "sdist: unexpected root directory")
            require(member.isdir() or member.isfile(), f"sdist: non-regular entry: {member.name}")
            if member.isdir():
                continue
            name = PurePosixPath(*path.parts[1:]).as_posix()
            require(name not in sdist_files, f"duplicate sdist member: {name}")
            stream = archive.extractfile(member)
            require(stream is not None, f"cannot read sdist member: {name}")
            sdist_files[name] = stream.read()
    require("PKG-INFO" in sdist_files, "sdist: PKG-INFO is missing")
    metadata_version(sdist_files.pop("PKG-INFO"), version, "sdist metadata")
    expected = dict(package)
    expected.update(source_files(root, "tests", "**/*.py"))
    expected.update(source_files(root, "benchmarks/longmemeval", "*.py"))
    expected.update(source_files(root, "benchmarks/production_retrieval", "*.py"))
    for name in ("dev.json", "holdout.json", "validation_v2.json"):
        path = root / "benchmarks/production_retrieval" / name
        if path.is_file():
            expected[path.relative_to(root).as_posix()] = path.read_bytes()
    for name in ("README.md", "LICENSE", "pyproject.toml", "config.example.yaml", ".gitignore"):
        expected[name] = (root / name).read_bytes()
    same_files(sdist_files, expected, "sdist")
    for artifact in (wheels[0], sdists[0]):
        print(f"verified {artifact.name} sha256={hashlib.sha256(artifact.read_bytes()).hexdigest()}")
    print(f"version {version}; {len(package)} package files; {len(expected)} source files")
    return version


def smoke_installed(python: Path, root: Path, version: str) -> None:
    python = python.absolute()  # keep the venv path; resolving its symlink loses the environment
    command = python.parent / ("engram.exe" if os.name == "nt" else "engram")
    require(python.is_file() and command.is_file(), "installed Python and engram command are required")
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="engram-wheel-smoke-") as directory:
        workdir = Path(directory).resolve()
        require(not workdir.is_relative_to(root), "smoke directory must be outside the checkout")
        def run(arguments: list[str]) -> str:
            result = subprocess.run(arguments, cwd=workdir, env=env, text=True,
                                    capture_output=True, timeout=30, check=True)
            return result.stdout.strip()
        probe = run([str(python), "-I", "-c",
                     "import engram, json, sys; print(json.dumps({'file': engram.__file__, "
                     "'version': engram.__version__, 'prefix': sys.prefix}))"])
        installed = json.loads(probe)
        location = Path(installed["file"]).resolve()
        require(not location.is_relative_to(root), "smoke imported Engram from the source checkout")
        require(location.is_relative_to(Path(installed["prefix"]).resolve()),
                "smoke imported Engram from outside the installed environment")
        require(installed["version"] == version, "installed module version differs from archives")
        require(run([str(command), "--version"]) == f"engram {version}", "installed CLI version differs")
        report = json.loads(run([str(command), "config", "show", "--defaults", "--json"]))
        require(report["config_file"] is None and report["values"]["storage_backend"] == "sqlite",
                "installed config inspection did not use package defaults")
        require(report["sources"] and set(report["sources"].values()) == {"default"},
                "installed default configuration has unexpected provenance")
        require(not list(workdir.iterdir()), "read-only installed commands created files in their working directory")
    print("installed wheel: --version and config show --defaults --json passed outside the checkout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-tag")
    parser.add_argument("--smoke-python", type=Path)
    args = parser.parse_args()
    try:
        root = args.source_root.resolve()
        version = check_archives(root, args.dist_dir.resolve(), args.expected_tag)
        if args.smoke_python:
            smoke_installed(args.smoke_python, root, version)
    except (DistributionError, OSError, ValueError, KeyError, SyntaxError,
            subprocess.SubprocessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"distribution check failed: {exc}\n")


if __name__ == "__main__":
    main()
