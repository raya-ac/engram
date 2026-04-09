"""Engram with any OpenAI-compatible API.

Works with OpenAI, local LLMs (Ollama, vLLM, llama.cpp server),
or any provider that speaks the OpenAI chat completions format.
Engram handles the memory layer — the LLM just needs to generate text.
"""

import os
import uuid
from openai import OpenAI

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search
from engram.surprise import compute_surprise, adjust_importance
from engram.entities import process_entities_for_memory
from engram.deep_retrieval import DeepReranker


# --- configure your LLM provider ---

# OpenAI
# client = OpenAI()
# MODEL = "gpt-4o"

# Ollama (local)
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# MODEL = "llama3.2"

# vLLM / llama.cpp server
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
# MODEL = "meta-llama/Llama-3-8b"

# default: whatever OPENAI_BASE_URL and OPENAI_API_KEY are set to
client = OpenAI()
MODEL = os.environ.get("MODEL", "gpt-4o")


def setup():
    config = Config.load()
    store = Store(config)
    store.init_db()
    reranker_path = config.resolved_db_path.parent / "reranker.npz"
    reranker = DeepReranker(model_path=reranker_path)
    return config, store, reranker


def recall(query: str, store: Store, config: Config, reranker: DeepReranker,
           top_k: int = 5) -> list[dict]:
    results = hybrid_search(query, store, config, top_k=top_k,
                            deep_reranker=reranker)
    return [{"content": r.memory.content, "layer": r.memory.layer,
             "score": r.score} for r in results]


def remember(content: str, store: Store, config: Config,
             layer: str = "episodic", importance: float = 0.7) -> dict:
    mem = Memory(
        id=str(uuid.uuid4()),
        content=content,
        source_type=SourceType.AI,
        layer=layer,
        importance=importance,
    )
    emb = embed_documents([content], config.embedding_model)
    if emb.size > 0:
        mem.embedding = emb[0]
        surprise = compute_surprise(mem.embedding, store)
        mem.importance = adjust_importance(mem.importance, surprise)
        mem.metadata["surprise"] = surprise["surprise"]

    store.save_memory(mem)
    process_entities_for_memory(store, mem.id, content)
    return {"id": mem.id, "surprise": mem.metadata.get("surprise", 1.0),
            "importance": mem.importance}


def chat(user_msg: str, history: list, store: Store, config: Config,
         reranker: DeepReranker) -> str:
    # recall relevant context
    memories = recall(user_msg, store, config, reranker)
    memory_context = ""
    if memories:
        memory_context = "\n\nRelevant memories:\n" + "\n---\n".join(
            f"[{m['layer']}] {m['content']}" for m in memories
        )

    system_msg = (
        "You are a helpful assistant with persistent memory across sessions."
        + memory_context
    )

    history.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_msg}] + history,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    # store the exchange
    remember(f"User: {user_msg}\nAssistant: {reply}", store, config,
             importance=0.5)

    return reply


def main():
    config, store, reranker = setup()
    history = []

    stats = store.count_memories()
    print(f"Memory agent ready ({stats['total']} memories). Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("you: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            response = chat(user_input, history, store, config, reranker)
            print(f"agent: {response}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        store.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
