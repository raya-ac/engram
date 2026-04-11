# examples

ready-to-run examples covering setup, integration patterns, and advanced features.

## setup guides

| file | what it covers |
|------|---------------|
| [claude-code-setup.md](claude-code-setup.md) | full walkthrough: install, wire into claude code as MCP server, add CLAUDE.md instructions, seed memories |
| [hooks-setup.md](hooks-setup.md) | auto-extract memories from conversations via claude code hooks — no manual `remember()` calls needed |
| [agent-patterns.md](agent-patterns.md) | common patterns: session orientation, learning from corrections, check-before-store, cognitive scaffolding, multi-agent coordination |

## python examples

### getting started

| file | what it does | run it |
|------|-------------|--------|
| [python-client.py](python-client.py) | standalone usage without MCP — store, search, surprise scoring, reranker training, lifecycle sweep | `python examples/python-client.py` |
| [custom-agent.py](custom-agent.py) | conversational agent with engram memory using the Anthropic SDK — recall before responding, auto-store exchanges | `python examples/custom-agent.py` |
| [openai-compatible.py](openai-compatible.py) | same agent pattern for any OpenAI-compatible API — works with OpenAI, Ollama, vLLM, llama.cpp | `python examples/openai-compatible.py` |

### advanced

| file | what it does | run it |
|------|-------------|--------|
| [multi-agent.py](multi-agent.py) | 3 specialized agents sharing one engram database — cross-domain recall, surprise scoring, dream cycle synthesis. optional live dashboard | `python examples/multi-agent.py --web` |
| [api-embeddings.py](api-embeddings.py) | switch between local and cloud embedding backends (Voyage, OpenAI, Gemini) — shows auto-detection, dim handling, model comparison | `python examples/api-embeddings.py` |
| [entity-graph.py](entity-graph.py) | build and traverse the entity relationship graph — extract entities from text, query relationships, multi-hop traversal | `python examples/entity-graph.py` |
| [negative-knowledge.py](negative-knowledge.py) | store explicit "what does NOT exist" claims — prevents hallucinated recommendations for things that were deliberately excluded | `python examples/negative-knowledge.py` |
| [drift-detection.py](drift-detection.py) | detect stale memories that reference dead file paths, missing functions, or outdated commands — with auto-fix | `python examples/drift-detection.py` |
| [export-import.py](export-import.py) | export memories to portable JSON (with optional embeddings), import into a fresh database — for backup, migration, sharing | `python examples/export-import.py` |
| [codebase-scan.py](codebase-scan.py) | scan a project directory and extract compressed code knowledge — file trees, function signatures, import graphs | `python examples/codebase-scan.py ~/project` |

## patterns at a glance

### session startup

```python
# lightweight — just check if you know anything relevant
hints = recall_hints(query="current project context", top_k=5)

# full context — graduated layers for system prompt injection
context = layers(query="user preferences and project state", max_tokens=2000)
```

### storing with surprise

```python
result = remember(content="the API rate limit is 100 req/min")
# result includes: surprise score, adjusted importance, duplicate warning
```

### searching

```python
# basic hybrid search (HNSW + BM25 + entity graph + Hopfield + RRF)
results = recall(query="rate limiting", top_k=10)

# entity-focused
entity = recall_entity(name="Ari")  # everything about a person/project

# temporal
timeline = recall_timeline(start="2026-03", end="2026-04")
```

### maintenance

```python
# run periodically
consolidate()            # dream cycle — cluster, summarize, archive
train_reranker()         # learn from access patterns
drift_check()            # verify memories against filesystem
extract_patterns()       # distill procedural knowledge from session
dedup(threshold=0.92)    # merge near-duplicate memories
```

### switching embedding models

```yaml
# config.yaml — just change the model name
embedding_model: voyage-3.5
```

```bash
# re-embed all memories with the new model
engram reembed
```
