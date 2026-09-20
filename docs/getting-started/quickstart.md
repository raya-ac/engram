# quick start

save one useful note, retrieve it, then connect the agent you want to use.
these commands assume [Engram is installed](installation.md) in your active
Python environment.

## 1. create a store, or keep the one you have

for a new installation:

```sh
engram init --yes --preset portable
```

setup prints the new config path and agent connection settings. it refuses
existing files and does not load models. if you already have a store, skip
`init` and use its config path.

replace `/absolute/path/to/config.yaml` in the commands below with that path.
use the same path in your shell, web process and agent connection.

```sh
engram --config /absolute/path/to/config.yaml config show
engram --config /absolute/path/to/config.yaml doctor
```

`config show` tells you which file and environment settings are effective.
`doctor` checks existing storage and model readiness. an `incomplete` result is
expected when optional checks have not run.

## 2. check the complete local path

```sh
engram --config /absolute/path/to/config.yaml doctor --full
```

this checks model inference, the local MCP endpoint and a synthetic save/retrieve
cycle in temporary storage. local weights may download on first use. no
Claude account or generative LLM is required; a hosted embedding/reranking model
uses its own configured provider. your existing memory records are unchanged.

[doctor statuses and fixes](troubleshooting.md#understand-doctor-results)
explain skipped checks, failures and exit codes.

## 3. remember something

```sh
engram --config /absolute/path/to/config.yaml remember "The release checklist lives in docs/release-checklist.md." --importance 0.8
```

the command prints `Remembered:` and a memory ID. this is a deliberate write to
your selected store. embeddings use the configured model.

manual remembering works without a working Claude installation or LLM account.
it also attempts optional hypothetical-query enrichment through `llm.backend`;
when that is unavailable, the note is saved without generated queries. an
available configured LLM may therefore be called. `doctor` and `demo` provide
an explicitly LLM-free synthetic walkthrough.

## 4. retrieve and inspect it

```sh
engram --config /absolute/path/to/config.yaml search "Where is the release checklist?"
engram --config /absolute/path/to/config.yaml search "Where is the release checklist?" --rerank
```

the first command uses ordinary hybrid retrieval. `--rerank` adds the configured
cross-encoder and its final relevance gate. use an explanation to see both
returned and rejected candidates:

```sh
engram --config /absolute/path/to/config.yaml search "Where is the release checklist?" --rerank --explain
```

explanation mode leaves memory access history, dormant evaluations and the
result cache unchanged. `--debug` is an alias for `--explain`. for structured
output, add `--json`; the explained response contains `results` and
`explanation`.

an empty result can be correct. inspect the explanation and the selected store
before changing a threshold. the [retrieval guide](../guides/retrieval-pipeline.md)
covers the stages and their limits.

## 5. connect your agent

open the [agent setup hub](../guides/client-configs.md), choose your client, and
use the interpreter/config paths printed by `init`.

stdio clients launch Engram as a child process. the underlying command is:

```sh
/absolute/path/to/python -m engram --config /absolute/path/to/config.yaml serve --mcp
```

this command normally waits for protocol input; it is not an interactive chat.
your client starts and manages it. after registration, reconnect the client and
ask it to inspect Engram's `config_show` tool before writing or recalling data.
a successful `doctor` self-test does not prove that the external client attached.

## try the isolated demo

Engram 0.8.1 includes a walkthrough using the fictional Lantern project:

```sh
engram demo
```

it shows saved memories, ordinary and reranked recall, read-only explanations,
and a saved project checkpoint in a temporary store. it uses real local models
without a generative LLM. inherited Engram configuration overrides are ignored;
local weights may still download. omit `--config` because the demo owns its
isolated configuration.

| command | behavior |
| --- | --- |
| `engram demo --yes` | run without prompts, then clean up |
| `engram demo --keep` | keep the printed demo directory for later inspection |
| `engram demo --web` | finish the walkthrough, then keep the local demo workspace open until you press Enter |
| `engram demo --web --port 8422` | use a different local web port |

use the complete URL printed for the demo workspace, including its generated
token. `--yes --web` only checks web readiness and then stops the process; omit
`--yes` when you want to browse. `--keep` retains files, not a running server.

## inspect your own store in the browser

```sh
engram --config /absolute/path/to/config.yaml serve --web
```

open `http://127.0.0.1:8420/` unless your `web.host`, `web.port` or `--port`
selects another address. this workspace uses your real store and its edit/write
controls affect that store. the server also warms models and the ANN index.

if `web.auth_token` is configured, use the token-authenticated browser URL;
[web troubleshooting](troubleshooting.md#web-workspace-and-ports) explains the
flow. the [web workspace guide](../guides/web-workspace.md) covers its controls.

## optional file ingestion

file extraction is separate from storing and retrieving a manual note. it uses
`llm.backend`, which defaults to `claude_cli`. available choices are the Claude
CLI, Anthropic API, OpenAI API and local MLX generation. configure that backend
and its authentication before relying on extracted facts.

```sh
engram --config /absolute/path/to/config.yaml ingest ./notes/
engram --config /absolute/path/to/config.yaml ingest ./notes/ --no-queries
```

`--no-queries` skips hypothetical-query generation; it **does not disable fact
extraction's LLM call**. if extraction fails, the current implementation can
store the text chunk as a fallback, so an ingested record alone does not prove
that LLM extraction succeeded. provider-backed extraction sends the source text
to that configured provider.

see [LLM configuration](../reference/config.md#llm-backends),
[export and import](../guides/export-import.md), and
[troubleshooting](troubleshooting.md) for the next steps.
