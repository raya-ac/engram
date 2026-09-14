# check evidence

Engram can retain the observations behind an assumption check without becoming
the checker. Kiln, Mythic or another harness decides what to check, runs an
authorized check, and decides whether the result supports proceeding. Engram
stores a bounded, immutable observation and makes its age and provenance visible.

The primary local integration is the [native API](native-api.md). The same
library functions are available in `engram.evidence`; generic MCP tools forward
to them for compatibility. The [Codex adapter](codex-adapter.md) can optionally
read the same result. No host owns the evidence schema, and none of these
interfaces installs an automatic checking hook.

## record an observation

The neutral Python contract is:

```python
from engram.evidence import evidence_put, evidence_get, evidence_list

stored = evidence_put(
    store,
    project_id="/absolute/project",
    session_id="release-check",
    assumption_id="deployment-capability-available",
    evidence_id="a-stable-caller-generated-uuid",
    outcome="contradicted",
    observed_at=observed_at,
    expires_at=observed_at + 3600,
    observation={"capability": "deploy", "available": False},
    provenance={
        "producer": "mythic",
        "check_type": "engram_tool_available",
        "transport": "native-jsonl",
    },
    source_refs=[],
)
current = evidence_get(store, project_id="/absolute/project", id=stored["id"])
history = evidence_list(store, project_id="/absolute/project", assumption_id="deployment-capability-available", limit=20)
```

`store` is an existing initialized Engram `Store`. These functions do not create
a database, run migrations, load a model, generate an explanation or execute the
observation's contents. The native operations and generic MCP tools use the
same names and fields, without the Python `store` argument.

`project_id` is an absolute directory path, resolved canonically. It is never a
project name guessed from prose. Session, assumption and evidence IDs are
nonempty strings up to 200 characters. The caller supplies observation and
expiry times as finite Unix timestamps. Expiry must follow observation by no
more than 365 days; an observation more than five minutes in the future is
rejected. An already expired observation may be stored as historical evidence.

An observation is at most 4,096 encoded JSON bytes. Provenance is at most 2,048
bytes and requires `producer` and `check_type` strings. Source references allow
eight strings of up to 256 characters. Omit raw transcripts, secrets and personal
contents unnecessary to the check.

## read the state, not just the claim

| Result | Meaning |
|---|---|
| `state: supported` | The caller reported support, and the record is current and eligible. |
| `state: contradicted` | The caller reported contradiction, and the record is current and eligible. |
| `state: unknown`, `eligible: true` | The check produced no conclusion. The observation itself is current and available. |
| `state: stale`, `eligible: false` | The observation expired. `outcome` becomes `unknown`; `reported_outcome` preserves the historical report. |
| `state: unknown`, `eligible: false` | No usable record exists, or the record/source was forgotten, removed or made inactive. `reason` explains the distinction without returning the observation. |

`eligible` describes availability and freshness, not truth. Every result states
`verified_by_engram: false`. Returned provenance is labelled `caller_supplied`.
A caller cannot make Engram certify a claim by submitting `outcome: supported`,
naming a verifier in provenance, or adding a `verified` field. Verification and
the policy to proceed, hold or revise belong to the calling checker and harness.

Reads do not increment access count, update last access or increase importance.
Neither an observation nor its instructions authorize executing a command or
fetching a URL. Engram never executes or follows source references.

## immutable retries and forgetting

`evidence_id` is immutable within a canonical project. Replaying the same payload
returns the same Engram ID and stored timestamp. A different payload under that
ID is rejected; use a new evidence ID for another observation. Concurrent
identical submissions cannot overwrite one another.

Each record uses an inert memory anchor with `source_type: check:evidence` and
structured metadata `kind: check_evidence`. This reuses Engram's existing
explicit forgetting and status lifecycle without a new schema migration. There
is no embedding, full-text-search entry, generated prose or reinforcement. An
ordinary memory-list view may show the neutral anchor label; scoped project
context excludes it. Read the actual observation through the evidence API.

To forget evidence, pass the returned Engram `id` to the ordinary `forget` tool
or memory lifecycle API. A retry of the original payload cannot reactivate that
anchor. `evidence_get` reports `unknown/forgotten` with no observation, and
`evidence_list` omits forgotten/inactive anchors. Directly deleting database rows
also removes their idempotency history; explicit forgetting is the supported
lifecycle path.

`memory:<id>` source references check the referenced memory's current lifecycle.
Missing, forgotten or inactive sources make the dependent observation unknown.
An expired evidence source returns `unknown/source_stale`. Other reference
strings are opaque provenance only: a URL does not establish that it was fetched
or that its contents remain valid. No referenced memory contents are copied into
the result. A caller should supply a short observation from its authorized check
and retain only necessary source references.

Listing filters by exact project, optional session and optional assumption
before applying the limit (default 20, maximum 50), ordered by storage time.
History may contain conflicting observations. Engram does not resolve them by
silently choosing a convenient result or using an older record when a particular
requested ID is unavailable; that decision belongs to the harness.

## verified contract and remaining limits

Isolated SQLite and PostgreSQL tests cover persistence, project/session
isolation, immutable/concurrent retries, explicit forgetting, inactive and
missing sources, expiration, payload bounds, finite timestamps and unchanged
memory access fields. The generic MCP route and optional adapter use the same
functions. Native API tests exercise separate processes and restart/resume.

These checks establish storage and transport behavior. They do not establish
that a supplied observation is true, that a check was authorized, or that a
harness actually consulted it before acting. This version has no automatic
check scheduler, cryptographic verifier attestation or universal host hook.

## native integration check

a disposable SQLite integration test exercised a generic client through Mythic
and the native Engram API. a registered file-existence check observed a missing
project file, changed an unknown premise from hold to a contradicted premise
with a revise decision, and persisted the observation in Engram. after both
owned processes restarted, Mythic resumed the revised decision and retrieved
the same evidence record. it remained a caller observation:
`verified_by_engram` was false.

this verifies the persistence and revision path with a controlled check. it does
not establish the accuracy of arbitrary checks or the finished Kiln interface.
