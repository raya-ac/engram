# A task assistant that remembers the handoff

keep the current state of a task beside your existing issue tracker or notes app.
the example reads project context, saves a note you reviewed, and resumes the
same task in a fresh process. it does not connect to an issue service, change
remote tickets, call an LLM, or capture conversations.

## prepare the example

complete [installation](../getting-started/installation.md) and initialize a
store, or use an existing config. check it with:

```sh
/absolute/venv/bin/python -m engram --config /absolute/engram.yaml config check
```

get both files from the [integration examples](https://github.com/raya-ac/engram/tree/main/examples/integrations):

- `native_client.py`, the shared stdlib JSONL client
- `task_assistant.py`, the command-line task workflow

keep them beside each other. the commands below run from the repository root;
adjust the script path if you copied them elsewhere. `--python` must point to
the Python environment where Engram is installed. `--project` must be an
existing absolute directory for this work.

## read project context

```sh
python examples/integrations/task_assistant.py \
  --python /absolute/venv/bin/python \
  --config /absolute/engram.yaml \
  --project /absolute/projects/lantern \
  context --limit 5
```

the script discovers the native operations, checks storage, then calls `recall`
with that project path. this native operation returns recent, explicitly owned
project memories and recent checkpoints. it is not the full MCP server's
semantic `recall`. an empty project result is valid even when the store has
unrelated memories.

## save a reviewed task note

write a small UTF-8 file such as `/absolute/reviewed-task.txt`:

```text
Lantern issue APP-42: the retry loop now stops after three attempts. The unit
check passes. The real network-failure check remains pending; do not mark the
issue complete until that check is recorded.
```

choose a stable task key, then explicitly save the note:

```sh
python examples/integrations/task_assistant.py \
  --python /absolute/venv/bin/python \
  --config /absolute/engram.yaml \
  --project /absolute/projects/lantern \
  save issue:APP-42 \
  --summary-file /absolute/reviewed-task.txt \
  --decision "Cap the retry loop at three attempts" \
  --next-step "Run the real network-failure check" \
  --yes
```

`--yes` confirms saving these supplied notes; it is required for writes. review
the file and omit secrets before running it. a successful response has
`status: "saved"`. saving again replaces this exact project's task checkpoint.
it does not add a semantic memory or update the issue tracker.

summaries accept up to 4,000 characters. repeat `--decision`, `--next-step` or
`--blocker` for up to eight entries each, with at most 500 characters per entry.

## resume in another process

```sh
python examples/integrations/task_assistant.py \
  --python /absolute/venv/bin/python \
  --config /absolute/engram.yaml \
  --project /absolute/projects/lantern \
  resume issue:APP-42
```

the response contains `memories` and `checkpoints`. the checkpoint must match
both the canonical project path and the task key. another task can have its own
checkpoint while sharing the same project memories. inspect the stored summary
and current evidence before deciding what to do next.

to delete only this task's checkpoint, use the same prefix with:

```sh
clear issue:APP-42 --yes
```

## connect it to your application

a task app can map “save handoff” to `session_checkpoint` and “open task” to
`session_resume`. keep the stable issue ID in `task`, and use the actual project
directory for `project_id`. for multiple machines, those paths must resolve to
the same stored project identity; matching display names alone is insufficient.

show checkpoint text as reference context. the example never executes a stored
next step or treats a note as proof that a check passed. provider-neutral agent
wiring is covered in [build an agent](build-an-agent.md).

the child inherits its environment, including `ENGRAM_*` config overrides. use
`config_show` or `engram config show` to verify the effective store. configured
PostgreSQL can be remote; the example itself adds no network service. these
context/checkpoint operations do not load embedding models or reinforce memories.

the client serializes requests, limits frames and closes the process on timeout
or a malformed response. `--timeout` sets the per-request timeout in seconds.
a timed-out save may have completed: resume the task to inspect it before
retrying. there is no automatic retry. errors go to stderr and return a nonzero
exit status. see the [native API](../native-api.md) for exact schemas and filters.
