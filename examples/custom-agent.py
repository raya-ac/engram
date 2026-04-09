"""Minimal agent with engram memory.

Shows how to build a simple conversational agent that remembers
across sessions. Uses engram for storage and retrieval, Claude API
for generation.
"""

import os
import uuid
import anthropic

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search
from engram.surprise import compute_surprise, adjust_importance
from engram.entities import process_entities_for_memory
from engram.deep_retrieval import DeepReranker


class MemoryAgent:
    def __init__(self):
        self.config = Config.load()
        self.store = Store(self.config)
        self.store.init_db()

        reranker_path = self.config.resolved_db_path.parent / "reranker.npz"
        self.reranker = DeepReranker(model_path=reranker_path)

        self.client = anthropic.Anthropic()
        self.history = []

    def recall(self, query: str, top_k: int = 5) -> str:
        """Search memory and format as context."""
        results = hybrid_search(
            query, self.store, self.config,
            top_k=top_k,
            deep_reranker=self.reranker,
        )
        if not results:
            return ""

        parts = []
        for r in results:
            parts.append(f"[{r.memory.layer}, imp={r.memory.importance:.2f}] {r.memory.content}")
        return "\n---\n".join(parts)

    def remember(self, content: str, layer: str = "episodic",
                 importance: float = 0.7) -> dict:
        """Store a memory with surprise scoring."""
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
            mem.metadata["surprise"] = surprise["surprise"]

        self.store.save_memory(mem)
        process_entities_for_memory(self.store, mem.id, content)

        return {
            "id": mem.id,
            "surprise": mem.metadata.get("surprise", 1.0),
            "importance": mem.importance,
        }

    def chat(self, user_message: str) -> str:
        """Process a message with memory-augmented context."""
        # recall relevant memories
        context = self.recall(user_message)

        # build system prompt with memory context
        system = "You are a helpful assistant with persistent memory."
        if context:
            system += f"\n\nRelevant memories:\n{context}"

        # add to conversation history
        self.history.append({"role": "user", "content": user_message})

        # generate response
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=self.history,
        )

        assistant_msg = response.content[0].text
        self.history.append({"role": "assistant", "content": assistant_msg})

        # auto-extract memories from the exchange
        exchange = f"User: {user_message}\nAssistant: {assistant_msg}"
        self.remember(exchange, importance=0.5)

        return assistant_msg

    def close(self):
        self.store.close()


def main():
    agent = MemoryAgent()
    print("Memory agent ready. Type 'quit' to exit.")
    print(f"Memories: {agent.store.count_memories()['total']}")
    print()

    try:
        while True:
            user_input = input("you: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            response = agent.chat(user_input)
            print(f"agent: {response}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        agent.close()
        print("\nSession ended.")


if __name__ == "__main__":
    main()
