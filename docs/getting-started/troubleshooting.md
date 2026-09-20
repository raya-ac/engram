# troubleshooting

start with the same Python executable and absolute config path your agent uses.
these checks separate an installation problem from a wrong store, an unavailable
model or a client that has not connected.

```sh
python -m engram --version
python -m engram --config /absolute/path/to/config.yaml config check
python -m engram --config /absolute/path/to/config.yaml config show
python -m engram --config /absolute/path/to/config.yaml doctor --json
```

`config show` and doctor reports redact credential fields. avoid sharing your
raw config, DSN, API keys, tokens or exported memory contents when reporting a
problem.

## Python, environment and command paths

| symptom | check or fix |
| --- | --- |
| `engram: command not found` | activate the environment, or use its full Python path followed by `-m engram` |
| `No module named engram` | run that interpreter's `-m pip show engram-memory-system`; install into that environment if absent |
| the version is older than expected | compare `python -m engram --version` with the exact executable in your client config; update that environment |
| Python version is rejected | Engram requires Python 3.11+; create the environment with a suitable interpreter |
| PowerShell will not run `Activate.ps1` | call `.\.venv\Scripts\python.exe -m engram` directly; activation is optional |
| it worked before the environment moved | recreate the environment at its final location and update the client's absolute interpreter path |

these environment behaviors follow Python's
[virtual environment documentation](https://docs.python.org/3/library/venv.html#how-venvs-work).
use `python -m pip` so the installer and interpreter are the same environment.

if installation fails while building a dependency, keep the first compiler/build
error. Engram includes native dependencies such as
[hnswlib](https://github.com/nmslib/hnswlib); changing a memory setting cannot fix
a failed package build. verify the Python version and the build tools requested
by that dependency before retrying.

## find the configuration actually being used

an explicit `--config` path must exist for normal commands. without it, Engram
uses the first existing file in this order:

1. `config.yaml` in the process's current working directory;
2. `config.yaml` in the directory containing the installed `engram` package
   directory — the repository root in a source checkout;
3. `~/.config/engram/config.yaml`.

if none exists, package defaults apply. an agent can start in a different working
directory from your shell, so give it an absolute config path.

```sh
engram --config /absolute/path/to/config.yaml config show --json
engram config show --defaults --json
```

the first command shows the selected file, effective values and their sources.
the second intentionally ignores files and environment overrides. do not combine
`--defaults` with `--config`.

normal precedence is **environment → selected file → defaults**. each setting
has an `ENGRAM_` environment name; for example:

| config field | environment override |
| --- | --- |
| `db_path` | `ENGRAM_DB_PATH` |
| `storage_backend` | `ENGRAM_STORAGE_BACKEND` |
| `postgres_dsn` | `ENGRAM_POSTGRES_DSN` |
| `embedding_backend` | `ENGRAM_EMBEDDING_BACKEND` |
| `retrieval.min_confidence` | `ENGRAM_RETRIEVAL_MIN_CONFIDENCE` |
| `web.auth_token` | `ENGRAM_WEB_AUTH_TOKEN` |

an invalid file value is rejected even when an environment override would
replace it. fix the file rather than trying to mask it. boolean environment
values accept `true`, `false`, `1` and `0`. use `engram config schema` for supported
fields, types and constraints.

a setting change affects new processes. restart the relevant web/MCP process
and reconnect the client; editing YAML does not change a running server.

## understand doctor results

| result | CLI exit code | meaning |
| --- | --- | --- |
| `pass` | `0` | the checks requested for this run passed |
| `incomplete` | `0` | optional checks were skipped or readiness has warnings; this is not full verification |
| `fail` | `1` | one or more diagnostic checks failed |
| invalid configuration | `2` | configuration could not be loaded or validated before diagnosis |

plain `doctor` does not load model runtimes, download weights or make a Postgres
connection. it inspects effective config, local package/cache readiness and
existing SQLite storage. a missing database is reported, not created.

choose additional checks deliberately:

```sh
engram --config /absolute/path/to/config.yaml doctor --check-models
engram --config /absolute/path/to/config.yaml doctor --check-connection
engram --config /absolute/path/to/config.yaml doctor --smoke
engram --config /absolute/path/to/config.yaml doctor --full
```

- `--check-models` runs configured embeddings and reranking on synthetic text.
  it can download model weights or contact a configured hosted provider.
- `--check-connection` spawns an isolated local stdio MCP process and checks
  `initialize`, `tools/list` and `config_show`. for Postgres it also permits a
  separate read-only database connection.
- `--smoke` runs models and saves/retrieves synthetic records in temporary
  SQLite. it keeps the configured relevance gate; it does not lower the cutoff
  to force a pass.
- `--full` combines all of those checks.

the temporary MCP/smoke checks disable ANN and dormant recall. they do not
initialize or migrate your existing store, write memory/access history there,
change agent settings or prove an external agent connection. SQLite reads can
still perform normal lock and WAL/SHM bookkeeping.

in 0.8.1, the model worker is bounded at 180 seconds and the local MCP check at
20 seconds. the PostgreSQL probe uses a 3-second connection timeout and statement
timeout. a timeout is a failed check, not evidence that setup completed.

## models are missing, slow or fail to load

first inspect `embedding_model`, `embedding_backend`, `embedding_dim` and
`cross_encoder_model` in `config show`. doctor reports package presence and
cached-file presence separately from real inference.

| observation | next step |
| --- | --- |
| missing cached files on a new install | run `doctor --check-models` with network access for the initial download |
| a required model/provider package is missing | install the relevant package or Engram extra into the same environment your agent uses |
| offline mode prevents a download | check `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`; weights must already be cached when those are enabled |
| MLX cannot load in this runtime | select `embedding_backend: sentence_transformers` in the intended config and check again |
| a device runs out of memory | close competing workloads or choose a smaller configured model; `init --preset light` is available for a new store |
| embedding dimensions disagree | inspect `embedding_dim` and any environment override; changing models for an existing store requires re-embedding |
| a hosted check fails | check that the provider package and API-key environment variable reach this process; package/key presence alone does not validate credentials |

`portable` selects sentence-transformers, not a guaranteed CPU device. changing
the embedding backend while keeping the same model differs from changing the
model itself. see [embedding backends](../guides/embedding-backends.md) before
re-embedding an existing store.

if a first download exceeds the doctor's timeout, check network/cache access
and retry after resolving the download problem. avoid launching many identical
model-loading processes while diagnosing it.

## the agent says the server is unavailable

use the client-specific steps in the [agent setup hub](../guides/client-configs.md).
check these values against the snippet printed by `init`:

1. the executable is the full path to the environment's Python;
2. arguments include `-m`, `engram`, `--config`, the absolute config path,
   `serve`, and `--mcp`, as separate arguments;
3. required environment overrides and provider credentials reach the agent's
   child process;
4. the client has reconnected or restarted since its configuration changed.

run the same interpreter and config with `doctor --check-connection`. passing
that check isolates the remaining issue to something the self-test did not
exercise, such as client registration, process permissions or the client's own
timeout. inspect its MCP logs next.

`serve --mcp` waits for JSON-RPC input on stdin. silence after starting it manually
can be normal. keep stdout reserved for protocol messages; do not wrap it in a
shell command that prints banners. clients should launch the command directly.

for a connection diagnosis that avoids eager model loading, add `--no-warmup`
after `--mcp` in the client arguments. a later tool call can still load models.
`--no-warmup` applies to stdio MCP, not the web or SSE server. increase a client's
startup/tool timeout only through that client's supported settings; see its
setup page for the exact option.

## a saved note does not appear in search

check the store identity first: the writing process and reading process must use
the same effective backend, config and database. then inspect retrieval:

```sh
engram --config /absolute/path/to/config.yaml search "your actual question" --rerank --explain --json
```

look for rejected candidates, their eligibility/relevance reasons, and whether
the note entered the candidate pool. forgotten or ineligible records and a
relevance cutoff can legitimately produce an empty result. ordinary search
without `--rerank` uses a different ranking path, so compare the same flags when
investigating a result.

`--explain` requires an initialized store and leaves access history unchanged.
use [retrieval explanations](../guides/retrieval-pipeline.md) to interpret the
output; do not treat a high similarity score as proof that a note is current.

## ingestion or remembering mentions an LLM

embeddings/reranking and generative extraction are separate configurations.
`llm.backend` defaults to `claude_cli`; its alternatives are `anthropic`,
`openai` and `mlx`.

manual `remember` can save without a working LLM, but attempts optional query
enrichment before saving. an installed yet unresponsive Claude CLI can delay
that attempt; the CLI backend has a 120-second subprocess timeout. ordinary
`search` does not need generative extraction.

`ingest --no-queries` disables query enrichment, not extraction. extraction errors
can fall back to stored text chunks. use the configured backend's diagnostics
if you expected structured facts. the [LLM settings](../reference/config.md#llm-backends)
cover credentials and optional packages. `llm.api_key` does not configure
embedding-provider credentials.

## web workspace and ports

```sh
engram --config /absolute/path/to/config.yaml serve --web --port 8420
```

the default bind address is `127.0.0.1`. `--port` overrides `web.port`; if the port
is occupied, choose another, for example `--port 8422`, and open that address.
`serve --mcp` uses stdio and has no TCP port. the SSE server and the isolated web
demo default to port `8421`.

when `web.auth_token` is set, open
`http://127.0.0.1:8420/?token=YOUR_TOKEN`, using the actual configured token and
port. the workspace reads that URL token and attaches it to same-origin API
requests. keep the token in the workspace URL while browsing; removing it can
make later requests fail with `401`. API callers can use an
`Authorization: Bearer ...` header. token-bearing URLs are credentials, so do
not put them in shared screenshots or issue reports.

the demo prints its own token-authenticated URL after its walkthrough. with
`engram demo --web`, press Enter only when finished browsing. with
`engram demo --yes --web`, the web process stops after its readiness check.
`--keep` retains demo files; use the printed restart command to reopen them.

## update or recover without replacing a store

if `init` reports that a target already exists, use the existing config with
`doctor` or choose genuinely new paths. do not delete the original store to get
past the refusal. Postgres setup also requires an empty selected schema; it is
not the migration command.

upgrade the package inside its current environment, retain your configuration,
then restart running Engram processes. follow the
[update steps](installation.md#update-an-existing-installation) and
[Postgres migration guide](../guides/postgres-migration.md) for the corresponding
operation.
