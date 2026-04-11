# Negative Knowledge

store explicit "what does NOT exist" claims. prevents AI from recommending things that were deliberately excluded.

## the problem

an AI agent searches your memory for "caching" and finds nothing. it concludes there's no information and recommends adding Redis. but you already evaluated Redis and rejected it — that decision just wasn't stored.

negative knowledge fills this gap.

## storing negatives

via MCP:

```
remember_negative(
    content="There is no caching layer in this project",
    context="Evaluated Redis and Memcached but SQLite WAL mode is sufficient",
    scope="myproject"
)
```

via CLI:

```bash
engram remember "NEGATIVE KNOWLEDGE: We do NOT use Redux. React context covers our needs." --layer semantic
```

via Python:

```python
from engram.store import Store, Memory
from engram.embeddings import embed_documents

mem = Memory(
    id="...",
    content="NEGATIVE KNOWLEDGE: The /admin endpoint was removed in v2.0. "
            "Admin operations go through the CLI.",
    layer="semantic",
    importance=0.75,
)
emb = embed_documents([mem.content])
mem.embedding = emb[0]
store.save_memory(mem)
```

## how it works

negative knowledge is stored in the semantic layer with a `NEGATIVE KNOWLEDGE` prefix. when someone searches for the thing that doesn't exist, the negative memory surfaces:

```
search("caching layer")
→ [semantic] NEGATIVE KNOWLEDGE: There is no caching layer in this project.
   Evaluated Redis and Memcached but SQLite WAL mode is sufficient.
```

the prefix ensures the memory is clearly marked as an absence, not a presence.

## examples

```
NEGATIVE KNOWLEDGE: There is no caching layer in this project
NEGATIVE KNOWLEDGE: We deliberately do not use Redux
NEGATIVE KNOWLEDGE: The /admin endpoint was removed in v2
NEGATIVE KNOWLEDGE: Do not use mocks for database tests
NEGATIVE KNOWLEDGE: The team decided against microservices
```

## when to store negatives

- after explicitly rejecting a technology or approach
- when removing a feature
- when a common recommendation doesn't apply to your project
- when you want to prevent a specific mistake from recurring

see also: [`examples/negative-knowledge.py`](https://github.com/raya-ac/engram/blob/main/examples/negative-knowledge.py) for a runnable example.
