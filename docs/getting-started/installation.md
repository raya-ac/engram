# installation

install Engram in its own Python environment, then either create a new store or
point it at the configuration you already use. installing or upgrading the
package does not run `init` or replace your configuration.

these instructions cover Engram 0.8.1. check the installed version with
`engram --version`.

## install the package

use Python 3.11 or newer. the test matrix covers Python 3.11 and 3.12 on Linux and
macOS. keep the environment in a stable directory: agent connections will use
its absolute Python path.

=== "macOS / Linux"

    ```sh
    python3 --version
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade engram-memory-system
    source .venv/bin/activate
    python -m engram --version
    ```

=== "Windows PowerShell"

    ```powershell
    python --version
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade engram-memory-system
    .\.venv\Scripts\python.exe -m engram --version
    ```

    optional activation:

    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

    if activation is unavailable, keep using
    `.\.venv\Scripts\python.exe -m engram` in place of `engram` below.

activation puts this environment's commands on your shell's path. using its
Python executable directly also works; see the
[Python virtual environment guide](https://docs.python.org/3/library/venv.html#how-venvs-work).

local models download on first use and reuse cached weights afterward. storage
setup itself does not download or load them. model size, hardware and the chosen
backend affect first-run time and memory use.

## choose new or existing storage

### new installation

```sh
engram init
```

follow the prompts for a model preset and new config/database paths. for the
local defaults without prompts:

```sh
engram init --yes
```

setup creates the config and an initialized store, then prints the Python
command and MCP connection snippet to use with your agent. it does not install
an agent, change client settings or download model weights.

| default created by `init` | location |
| --- | --- |
| configuration | `~/.config/engram/config.yaml` |
| SQLite database | `~/.local/share/engram/memory.db` |
| configured ANN index path | `~/.local/share/engram/memory.hnsw.index` |

the index file is built when needed; `init` does not build it. `~` means your
home directory, including on Windows.

for a separate store, choose paths that do not already exist:

```sh
engram --config /absolute/new/config.yaml init --yes --db-path /absolute/new/memory.db --preset portable
```

`--config` is a global option: put it **before** `init`, `doctor`, `search` or
another command. during `init`, it names the new file to create. during ordinary
commands, it names an existing file to load.

setup refuses to overwrite existing config/database files or reuse an existing
enabled ANN index. inherited `ENGRAM_*` overrides still apply; setup lists their
names and explains which variables the agent process must also receive.

### existing installation

keep your current config and store. inspect them before changing anything:

```sh
engram --config /absolute/path/to/config.yaml config show
engram --config /absolute/path/to/config.yaml doctor
```

`config show` includes effective settings and their sources, with credentials
redacted. `doctor` reads existing storage without initializing or migrating it.
use [the update steps](#update-an-existing-installation) for an upgrade;
`init` is for a new store.

## choose a local model preset

| preset | embedding runtime | embedding model | reranker |
| --- | --- | --- | --- |
| `local` | automatic local selection | `BAAI/bge-small-en-v1.5` | `BAAI/bge-reranker-base` |
| `portable` | `sentence_transformers` | `BAAI/bge-small-en-v1.5` | `BAAI/bge-reranker-base` |
| `light` | `sentence_transformers` | `BAAI/bge-small-en-v1.5` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

choose a preset when creating the store, for example `engram init --preset
light`. automatic selection can use an available MLX runtime or fall back to
sentence-transformers. `portable` selects a library; it does not force CPU
execution. the `light` preset changes the reranker and its results may differ.

embeddings and reranking do not require a Claude account or a generative LLM.
optional file extraction and query enrichment use a separate `llm` configuration;
the [quick start](quickstart.md#optional-file-ingestion) explains that boundary.

## verify and connect

use the path printed by setup:

```sh
engram --config /absolute/path/to/config.yaml doctor --full
```

this runs configured embedding/reranker models on synthetic text, checks a local
MCP process and saves/retrieves fictional data in temporary SQLite storage.
first use can download weights; configured hosted models can call their provider.
your existing memory records and agent settings are not changed. normal SQLite
read locking and WAL/SHM bookkeeping can still occur.

a passing local MCP check verifies Engram's endpoint. connect and verify your
actual client next in the [agent setup hub](../guides/client-configs.md).
plain `doctor` skips expensive checks and often reports `incomplete`; see
[statuses and troubleshooting](troubleshooting.md#understand-doctor-results).

## Postgres instead of SQLite

SQLite needs no database service. choose Postgres when your installation needs
that backend, such as concurrent web and MCP processes.

for a **new** Postgres store, provide `ENGRAM_POSTGRES_DSN` in the process
environment, then run:

```sh
engram --config /absolute/new/postgres.yaml init --storage postgres --preset portable --yes
```

the database must already exist, and its selected schema must be empty. setup
creates Engram's tables without clearing or migrating existing ones. it does not
copy an environment-supplied DSN into the generated file or client snippet;
provide that variable to each client-launched process too.

`doctor --check-connection` permits a read-only Postgres connection and separately
tests an isolated local MCP endpoint. moving existing SQLite data is a different
operation: use the [Postgres migration guide](../guides/postgres-migration.md).

## optional hosted model packages

install the extra for your chosen provider into the same environment:

```sh
python -m pip install 'engram-memory-system[voyage]'
python -m pip install 'engram-memory-system[openai]'
python -m pip install 'engram-memory-system[gemini]'
```

you only need the extra you will use. set its provider credentials in the process
environment and choose the model in your existing config. `init` presets are
local; hosted model configuration is a separate step. see
[embedding backends](../guides/embedding-backends.md) and
[configuration](../reference/config.md). changing the embedding model for an
existing store also requires re-embedding its records.

## update an existing installation

keep a backup appropriate to your storage backend before updating. a
[record export](../guides/export-import.md) is useful for portability; use a
SQLite or Postgres database backup when you need a full database rollback.

with the existing environment active:

```sh
python -m pip install --upgrade engram-memory-system
python -m engram --version
engram --config /absolute/path/to/config.yaml config check
engram --config /absolute/path/to/config.yaml doctor
```

keep the same config path, database and selected models. do not run `init` over
them or replace your config with an example file. restart running Engram web/MCP
processes and reconnect the agent so they load the updated package and settings.
review the [changelog](../changelog.md) for version-specific changes.

## work from source

```sh
git clone https://github.com/raya-ac/engram.git
cd engram
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m engram --version
```

on Windows, use the environment creation and activation commands above. source
checkout features may be newer than the published package. use `init` only for
a new store; otherwise pass your existing config explicitly. contributors who
need the test tools can install `python -m pip install -e '.[dev]'`.

for containers, follow the [Docker guide](../guides/docker.md). for install,
model or connection failures, use [troubleshooting](troubleshooting.md).
