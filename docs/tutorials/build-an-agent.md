# Build an Agent with Memory

step-by-step: a conversational agent that remembers across sessions using engram + the Anthropic SDK.

## prerequisites

```bash
pip install engram-memory-system anthropic
```

## the agent

```python
import uuid
import anthropic

from engram.config import Config
from engram.store import Store, Memory, SourceType
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search
from engram.surprise import compute_surprise, adjust_importance
from engram.entities import process_entities_for_memory

class MemoryAgent:
    def __init__(self):
        self.config = Config.load()
        self.store = Store(self.config)
        self.store.init_db()
        self.store.init_ann_index(background=True)
        self.client = anthropic.Anthropic()
        self.history = []

    def recall(self, query, top_k=5):
        """search memory for relevant context."""
        results = hybrid_search(query, self.store, self.config, top_k=top_k)
        if not results:
            return ""
        parts = [f"[{r.memory.layer}] {r.memory.content}" for r in results]
        return "\n---\n".join(parts)

    def remember(self, content, layer="episodic", importance=0.7):
        """store a memory with surprise scoring."""
        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            source_type=SourceType.AI,
            layer=layer,
            importance=importance,
        )
        emb = embed_documents([content], self.config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]
            surprise = compute_surprise(mem.embedding, self.store)
            mem.importance = adjust_importance(mem.importance, surprise)

        self.store.save_memory(mem)
        process_entities_for_memory(self.store, mem.id, content)
        return mem

    def chat(self, user_message):
        """process a message with memory-augmented context."""
        # recall relevant memories
        context = self.recall(user_message)

        system = "You are a helpful assistant with persistent memory."
        if context:
            system += f"\n\nRelevant memories:\n{context}"

        self.history.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=self.history,
        )

        reply = response.content[0].text
        self.history.append({"role": "assistant", "content": reply})

        # auto-store the exchange
        self.remember(f"User: {user_message}\nAssistant: {reply}", importance=0.5)

        return reply
```

## using it

```python
agent = MemoryAgent()

# first session
agent.chat("My name is Alex and I'm building a weather app")
agent.chat("I decided to use FastAPI for the backend")

# ... close and reopen later ...

# second session — agent remembers
agent.chat("What framework did I choose for the backend?")
# → "You decided to use FastAPI for the backend"
```

## what's happening

1. every `chat()` call searches memory for relevant context
2. context gets injected into the system prompt
3. after responding, the exchange is auto-stored as a memory
4. surprise scoring prevents redundant storage
5. entity extraction builds a relationship graph from the conversation

## next steps

- use `remember_decision()` for explicit decisions with rationale
- use `remember_error()` for error patterns with prevention
- use `recall_hints()` before `recall()` to save context tokens
- run `consolidate()` periodically to clean up and bridge domains
- train the reranker after a few days of usage

see also: [`examples/custom-agent.py`](https://github.com/raya-ac/engram/blob/main/examples/custom-agent.py) and [`examples/openai-compatible.py`](https://github.com/raya-ac/engram/blob/main/examples/openai-compatible.py) for complete runnable examples.
