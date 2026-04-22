---
name: session-continuity
description: Use when an agent should pick work up from Engram quickly and leave behind a clean structured handoff for the next session. Best for multi-step coding, investigations, releases, debugging, and any task where recent decisions and open loops matter.
---

# Session Continuity

Use Engram as an active resume layer, not just a memory dump.

The goal is simple:

- start by loading the latest structured handoff
- work normally while keeping important state in Engram
- end with a handoff the next session can use immediately

## When to use this

Use this skill when:

- the task spans more than one turn or session
- a second agent may continue the work later
- there are important decisions, blockers, or open loops
- the repo or environment takes time to reconstruct from scratch

## Startup flow

At the start of a session:

1. Call `resume_context` first.
2. If the result is thin, call `session_summary` or a targeted `recall`.
3. Treat the returned `open_loops`, `decisions`, and `recent_work` as the default resume packet.

This is better than trying to rebuild context from raw diary entries or a large recall dump.

## During work

Keep the handoff current while you work:

- use `diary_write` when a short progress note will matter later
- use `remember` for verified facts and implementation state
- use `remember_decision` for tradeoffs and rationale
- use `remember_negative` for limitations, unsupported paths, and things that should not be assumed

Engram refreshes the active session handoff automatically after diary writes and important memory writes, so you do not need to manually rebuild the handoff every few minutes.

## Before stopping

Near a natural stop point:

1. Call `session_handoff`.
2. Save it unless you have a reason not to.
3. Check that it captures:
   - current state
   - key decisions
   - recent work
   - open loops

If the next agent should resume the same thread of work, the handoff should make sense without the full chat log.

## Suggested pattern

```text
start:
- resume_context

during work:
- diary_write for meaningful progress
- remember / remember_decision / remember_negative as needed

before stopping:
- session_handoff
```

## Quality bar

Good continuity state is:

- specific
- recent
- actionable
- easy to resume from

Bad continuity state is:

- vague summaries
- repeated chatter
- giant raw dumps with no prioritization
- facts with no repo, file, version, or environment context

## Example uses

- resume a half-finished bug fix without rereading the full thread
- pick up release work after a failed deployment
- continue an investigation with the prior session's open loops and touched entities
- hand a coding task from one agent to another without losing the real blocker
