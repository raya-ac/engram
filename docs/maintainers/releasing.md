# release process

publish a checked commit, then the exact distributions built from its release tag.
normal pushes run tests and package checks; PyPI publishing runs only when a
GitHub release is published.

## prepare the release

1. update `pyproject.toml` and `engram/__init__.py` to the same version. CLI, web,
   and MCP version reporting use `engram.__version__`.
2. update `docs/changelog.md`, installation notes, and any changed public
   commands. keep benchmark conditions attached to their numbers.
3. run the relevant tests and `python -m mkdocs build --strict`.
4. commit the reviewed changes with a short, lowercase message and push `main`.
5. wait for the Linux/macOS Python test matrix, **package checks** and
   **postgres onboarding** jobs to pass for that commit.

## what the package gate checks

`scripts/check_distribution.py` reads the archives without extracting them or
importing project code. it requires one wheel and one source archive, and checks:

- matching versions in the source, wheel metadata, and source-archive metadata;
- the `engram` CLI entry point and the web workspace HTML template;
- package files identical to the checked-out source;
- only Python/package templates in the wheel, plus package metadata and license;
- only the declared package, tests, benchmark Python scripts and synthetic
  production-retrieval fixtures, README, license,
  package metadata, example configuration, and `.gitignore` in the source archive;
- no scratch files, artifacts, private configuration, `.env`, databases, model
  caches, or raw benchmark results.

`twine check --strict` validates both distributions' package metadata and README.
CI then installs the wheel in a fresh virtual environment with only PyYAML for
these lightweight commands, and runs them from outside the source checkout:

```sh
engram --version
engram config show --defaults --json
```

that smoke check confirms the installed package imports from its environment,
reports the expected version, and inspects defaults without creating working
directory files. it does not exercise model inference or the complete runtime;
the normal test matrix covers the project's dependencies separately.

for a local archive check, with build tools already installed:

```sh
python -m build
python scripts/check_distribution.py
python -m twine check --strict dist/*
```

build from a clean checkout. the source-archive include list in `pyproject.toml`
and the checker intentionally agree; update both when adding a new public file
type or package asset. do not widen them to include a local diagnostic directory.

## publish

create a GitHub release targeting the tested commit with its matching version
tag, for example `v0.8.0`. the publishing workflow checks that the tag equals
`v` plus the package version, rebuilds the distributions, and repeats the package
and installed-wheel checks.

the verification job retains the checked distributions for 90 days. the separate
`pypi` environment job downloads those exact artifacts and uses trusted
publishing to upload them. only that job receives the OIDC publishing permission.
a failed check prevents the upload. ordinary CI retains checked distributions
for 14 days.

## verify the published result

check that the release, PyPI version, and publish workflow all refer to the
intended version. download the published wheel and source archive and compare
their SHA-256 hashes with the checker output in the release verification job.
verify the docs deployment and its `build.json` source revision if docs changed.

PyPI versions cannot be overwritten. if a published artifact is wrong, preserve
the evidence, prepare a corrected version, and decide whether the broken release
needs to be yanked.
