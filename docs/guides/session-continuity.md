# Session Continuity

Engram now supports a cleaner resume flow for agents that work across multiple turns or multiple sessions.

The core idea is simple:

1. load the latest structured handoff
2. work normally while writing meaningful state into Engram
3. leave behind a handoff packet the next session can resume from

## Why this exists

Raw diary entries and broad recalls are useful, but they are a bad default resume protocol. They make new sessions reconstruct context from noise.

The continuity flow gives agents a smaller, more intentional packet:

- current state
- recent work
- decisions
- open loops
- recent queries
- touched entities

## Startup pattern

At the start of a session:

```text
resume_context
```

If the latest handoff is too thin, then expand with:

- `session_summary`
- `recall`
- `recall_hints`

But start with `resume_context` first.

## During work

Use the normal write path:

- `remember` for verified facts and implementation state
- `remember_decision` for tradeoffs
- `remember_negative` for things that are not true or should not be assumed
- `diary_write` for concise progress notes that would matter later

The active MCP session refreshes its structured handoff automatically after:

- recalls
- memory writes
- diary writes
- memory edits

## Before stopping

Near a natural stop point:

```text
session_handoff
```

Use this when you want to explicitly persist the current packet before handing work to another session or another agent.

## Suggested agent workflow

```text
start:
- resume_context

during work:
- remember / remember_decision / remember_negative
- diary_write when the progress note matters later

before stopping:
- session_handoff
```

## Repo-local skill

Engram ships with a continuity skill example here:

- [`examples/skills/session-continuity/SKILL.md`](https://github.com/raya-ac/engram/blob/main/examples/skills/session-continuity/SKILL.md)

Use that as a starting point for Codex, Claude Code, or any other MCP-capable agent setup.
