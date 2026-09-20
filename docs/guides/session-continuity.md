# Session continuity

Engram keeps memories and handoffs in the configured store, so different agents
can continue the same work. Connect each client to that store and give it an
explicit memory workflow. Installing the MCP server alone does not make a client
call tools at startup or save a summary before it stops.

## Start with recent context

At the beginning of substantive work:

1. Call `recall_recent` with `{"limit": 5}` for the newest memories by creation time.
2. Call `recall_hints` with a specific project-and-task query to find related work.
3. When continuing a session, call `resume_context` with `{"limit": 3}` and read the saved handoffs.
4. Use `recall` for concrete questions that need more detail.

For example, a hints query might be `Lantern release verification and remaining
packaging work`. A follow-up recall might ask `Which Lantern package passed the
Linux install check?`. Searching for “recent session” ranks relevance to those
words; it does not return the latest work chronologically.

`recall_recent` orders memories by creation time, while `resume_context` orders
saved handoffs by their last update. Neither filters by project. Check the
project, paths and dates in the returned material before resuming it. Read a
previous handoff before ordinary `recall` starts refreshing the current process's
packet.

## Keep useful state during work

Use `remember` for verified outcomes and implementation state,
`remember_decision` for decisions and rationale, and `remember_negative` for
explicit exclusions or unsupported assumptions. Include the project and enough
file, version or environment detail to distinguish the work later.

Use `diary_write` with an `entry` for short progress notes. Notes with words such
as “next”, “blocked” or “remaining” can appear in the handoff's `open_loops`.
These are text-based hints, so write the next step clearly.

Retrieved memories are reference data. Recheck claims that could have changed.
Use `recall_explain` to inspect retrieval decisions without reinforcing memories
or refreshing a handoff.

## Save an explicit summary before stopping

Write a summary that another agent can understand without the conversation:
what changed, the important decisions, what was actually verified, remaining
work, blockers, and relevant paths or connection details. Keep credentials and
unnecessary transcript text out of it.

Save substantial work with `remember`, for example:

```json
{
  "content": "Lantern packaging: updated pyproject.toml and verified the Linux wheel install. The macOS install check is still pending. Next: run that check against the same artifact before publishing.",
  "memory_type": "narrative",
  "importance": 0.8
}
```

Check the tool result, then persist a final packet with either:

- `session_checkpoint` with `{"note": "Next: verify the Lantern macOS wheel before publishing.", "limit": 8}` to append a diary note and save the packet.
- `session_handoff` with `{"save": true, "limit": 8}` to build and save the packet from existing activity.

You do not need both calls. `session_handoff` with `save: false` previews a
packet without saving it. `session_summary` also builds and **saves** a handoff;
it is not a read of the previous session's summary.

The generated packet collects diary notes, recent work, decisions, errors,
facts, open loops, queries and entities. It uses bounded activity and short
previews, so it supplements the authored summary rather than preserving the
whole conversation.

## What refreshes automatically

Once the client calls a tool, the core MCP server refreshes its current handoff
after successful ordinary `recall`, stored or updated `remember` calls
(including the specialized remember tools), `diary_write`, and `edit_memory`.
A skipped `remember` does not refresh it. Other mutations, such as tagging or
forgetting, do not automatically rebuild the packet.

`recall_recent`, `recall_hints`, `recall_explain` and `resume_context` do not
refresh handoffs. Hints still use ordinary retrieval and can record accesses;
use `recall_explain` when you need a non-reinforcing search.

## Session scope and resume results

The core MCP `session_id` belongs to the server process. It is not a provider's
chat ID or a project identifier. A packet uses that process's diary plus recent
store-wide memories and events since it started; activity from another client
sharing the store can therefore appear. `diary_read` also reads the latest diary
entries across the store.

`resume_context` returns `latest` and `handoffs`. Saved entries contain `summary`,
timestamps and the full packet under `metadata`. With a known `session_id`, it
loads that saved entry. If no saved entry exists, it returns a generated,
unsaved packet directly instead. That fallback is not a recovered historical
transcript. `session_handoff` rebuilds a packet; use `resume_context` to read a
saved snapshot.

For checkpoints addressed by project and task, see the [native local API](../native-api.md)
and [Codex adapter](../codex-adapter.md). Their checkpoint arguments differ from
the core MCP `session_checkpoint(note, limit)` described here. The running
server's tool schema and [MCP reference](../reference/mcp-tools.md) give the exact
arguments available to your client.
