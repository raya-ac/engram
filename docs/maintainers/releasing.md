# Release Process

Use this checklist when cutting a new release.

## Before tagging

1. update the package version in:
   - `pyproject.toml`
   - `engram/__init__.py`
2. add a changelog entry in:
   - `docs/changelog.md`
3. update any stale tool counts, feature references, or screenshots if needed
4. rebuild the docs site locally:
   - `python -m mkdocs build`
5. run tests:
   - `pytest tests/ -v`

## Publish

1. commit the release prep
2. push `main`
3. create a GitHub release with the matching tag, for example:
   - `v0.5.0`

Engram publishes to PyPI on `release.published`, not on normal pushes.

## After publish

Verify:

- GitHub release exists
- publish workflow completed successfully
- PyPI shows the new version
- docs deployment completed if docs changed
