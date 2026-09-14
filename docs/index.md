<p align="center">
  <img src="assets/logo-192.png" alt="engram" width="130">
</p>

# engram

memory for work that continues after the conversation ends.

engram keeps decisions, procedures, project notes and their connections in sqlite
or postgres. search them through a CLI or local API, connect an MCP client, or inspect the
store in the web workspace. records retain their source, lifecycle and access
history so retrieval can be checked against what actually happened.

## find context, keep its boundaries

ordinary retrieval combines embeddings, full-text search, entity relationships
and associative candidates, then applies its ranking and relevance gates.
memories can be edited, annotated, challenged, superseded or forgotten. a recalled
note is historical context, not proof that its claim is still current.

[retrieval pipeline](guides/retrieval-pipeline.md) ·
[memory lifecycle](guides/memory-lifecycle.md) ·
[storage migration](guides/postgres-migration.md)

## a workspace for the whole store

the redesigned web interface starts with the memory list. search, inspect, edit,
and navigate connected entities without losing your place. continuity, timelines,
analytics, graphs and maintenance tools share the same navigation and inspector.
mobile layouts retain the menu and activity drawer instead of hiding them.

all eighteen existing views remain available, alongside a new dormant review
view. it shows evaluation metadata first; opening a candidate and recording
feedback are explicit actions.

[web workspace and controls](guides/web-workspace.md)

## revisit a neglected connection

dormant recall is an optional shadow experiment. it checks current active,
non-forgotten database embeddings independently of the ordinary ANN cache, then
requires a separate query/content relevance check. relevance gates come before
a bounded dormancy bonus. it records at most one candidate per search, or none,
and leaves ordinary answers unchanged.

an investigation found that an old persisted index omitted a rarely accessed
release note. at the unchanged 0.75 cosine threshold, the repaired path recovered
it from an isolated snapshot and added release-packaging context absent from
ordinary results. a generic high-similarity database fact was rejected because it
did not answer the actual question. these are targeted evaluator observations;
usefulness across real work remains unproven.

computing or inspecting a candidate does not reinforce the memory. feedback is
explicit, cooldown survives log rotation, and silence never means useful.

[dormant recall: setup, evidence and limits](dormant-recall.md)

## a native boundary for harnesses

the local JSONL API gives applications a direct Engram-owned process for
project context, semantic search, deliberate checkpoints and structured evidence.
it does not require MCP framing or a Codex client. Kiln owns its native
application experience; Mythic owns checking and planning; Engram persists the
context and observations they can inspect.

evidence is an immutable caller-supplied observation with a project, source,
observation time and expiry. Engram does not execute its contents or certify a
reported result. stale or unavailable evidence cannot silently become a verified
premise. scoped context reads do not reinforce memories; ordinary semantic
search retains its existing access accounting.

[native API and evidence contract](native-api.md) ·
[optional Codex adapter](codex-adapter.md) ·
[other MCP clients](guides/client-configs.md)

## start from source

```sh
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
```

review the storage and model configuration before the first write. source
checkout features described here may be newer than the published package.

[installation](getting-started/installation.md) ·
[quick start](getting-started/quickstart.md) ·
[configuration](reference/config.md) ·
[GitHub README](https://github.com/raya-ac/engram#readme)

## inspect before relying on it

contract tests cover isolated SQLite/PostgreSQL storage, eligibility, independent
candidate retrieval, feedback/cooldown, project boundaries and absence of
reinforcement. fresh CLI/MCP process checks cover the public interfaces. these
checks establish behavior; they do not turn similarity into truth or replace
actual user feedback.

this documentation is published from the repository's `docs/` directory through
its GitHub Pages workflow. [build metadata](build.json) records the source revision
of the published site.
