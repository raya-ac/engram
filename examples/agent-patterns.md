# agent patterns

common patterns for integrating engram into AI agent workflows.

## session start: orientation

at the beginning of each session, orient the agent with relevant context.

```python
# option 1: lightweight hints (preferred — least context used)
hints = recall_hints(query="current projects and priorities")
# returns truncated snippets + entity names — enough to trigger recognition
# only pull full memories for items that need more detail

# option 2: graduated layers (best for system prompts)
context = layers(query="user preferences", max_tokens=2000)
# returns L0 (identity), L1 (core facts), L2 (recent), L3 (query-specific)

# option 3: task-aware skill selection
skills = get_skills(query="implementing authentication")
# returns 2-3 focused procedural memories only when injection helps
# skips when the task is well-covered by model pretraining

# option 4: broad recall (uses more context)
results = recall(query="recent work and active projects")
```

## learning from corrections

when the user corrects the agent, store the pattern:

```python
# user: "don't use mocks for database tests"
remember_error(
    error="Used mock database in integration tests",
    prevention="Always use real database connections for integration tests. "
               "Mock/prod divergence caused a broken migration to pass tests "
               "but fail in production."
)
```

## recording decisions

when a non-obvious choice is made, capture the rationale:

```python
remember_decision(
    decision="Use SQLite instead of PostgreSQL for the config service",
    rationale="Single-node deployment, no concurrent writes, "
              "eliminates ops complexity. Revisit if we need replication."
)
```

## negative knowledge

when something is deliberately excluded or doesn't exist:

```python
remember_negative(
    content="There is no caching layer in this project",
    context="Evaluated Redis and Memcached but SQLite WAL mode is sufficient. "
            "Prevents future recommendations to add caching.",
    scope="myproject"
)
```

this surfaces when someone searches for "caching" — preventing bad recommendations.

## check-before-you-store pattern

use `recall_hints` to check if you already know something before storing:

```python
# before storing a new memory
hints = recall_hints(query="database testing approach")
# if hints come back with relevant results, skip the store
# if empty or unrelated, go ahead and remember

# the surprise score also catches this automatically —
# redundant memories get low surprise and reduced importance
result = remember(content="Always use real DB for integration tests")
if result.get("warning") == "near-duplicate detected":
    # already knew this, memory was stored but with low importance
    pass
```

## entity-driven recall

when working with a specific person, project, or tool:

```python
# everything about a project
project = recall_entity(name="melee.garden")
# returns: memories, relationships, aliases, type

# timeline of events
timeline = recall_timeline(start="2026-03", end="2026-04")

# graph traversal — what's connected?
related = recall_related(name="Ari", max_hops=2)

# search codebase layer specifically
code = recall_code(query="auth middleware", project="myapp")
```

## periodic maintenance

schedule these to keep the memory system healthy:

```python
# retrain the retrieval model on accumulated usage patterns
train_reranker(epochs=50)

# run the dream cycle — consolidate, deduplicate, bridge cross-domain
consolidate()

# check system health
health()

# find and merge near-duplicates
dedup(threshold=0.92)

# detect stale memories referencing dead paths/functions
drift_check()

# extract reusable patterns from recent session activity
extract_patterns(hours=24)

# check memory quality metrics
quality_metrics()
```

## multi-agent setup

if multiple agents share the same engram database, they automatically
share memories. the access_log tracks which memories get used, so the
deep reranker learns from all agents' patterns.

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram/.venv/bin/python",
      "args": ["-m", "engram", "serve", "--mcp"],
      "env": {
        "ENGRAM_DB_PATH": "/shared/path/memory.db"
      }
    }
  }
}
```

all agents read/write to the same SQLite database (WAL mode handles
concurrent access). the web dashboard at `:8420` shows activity from
all processes in real time.

see [`multi-agent.py`](multi-agent.py) for a runnable experiment with 3 specialized agents.

## cognitive scaffolding vs full recall

the `recall_hints` tool exists for a reason — dumping full memory content
into every prompt replaces cognition instead of supporting it.

**use `recall_hints` when:**
- checking if you know something before looking it up
- browsing what's relevant without committing context tokens
- the agent needs to decide whether to dig deeper

**use `recall` when:**
- you need the actual content to act on
- the user asked a specific question that needs a detailed answer
- you're making a decision that requires full context

**use `get_skills` when:**
- starting a task that might benefit from procedural guidance
- the task is in an unfamiliar domain
- you want focused help, not a context dump

```python
# step 1: do i know anything about this?
hints = recall_hints(query="nginx rate limiting config")

# step 2: looks relevant, pull the full memory
if hints["hints"]:
    full = recall(query="nginx rate limiting config")
```

## embedding model selection

engram auto-detects the backend from the model name. switch by changing config:

```yaml
# config.yaml
embedding_model: voyage-3.5        # best quality ($0.18/1M tokens)
# embedding_model: voyage-3.5-lite  # budget ($0.02/1M tokens)
# embedding_model: text-embedding-3-small  # openai ($0.02/1M tokens)
# embedding_model: BAAI/bge-small-en-v1.5  # local, free
```

after switching models, re-embed all memories:

```bash
engram reembed
```

## backup and migration

export memories before major changes:

```bash
# full backup with embeddings (can restore without re-embedding)
engram export backup.json --include-embeddings

# import on a new machine
engram import backup.json --skip-duplicates

# layer-specific export
engram export procedural.json --layer procedural
```
