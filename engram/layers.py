"""L0-L3 graduated retrieval for context injection."""

from __future__ import annotations

from engram.config import Config
from engram.store import Store, MemoryLayer
from engram.retrieval import search


def get_context_layers(store: Store, query: str | None = None,
                       config: Config | None = None,
                       max_tokens: int = 4000) -> dict[str, str]:
    """Get graduated memory context for system prompt injection.

    Returns dict with keys l0, l1, l2, l3 — each a text block.
    l0: identity (~100 tokens, always present)
    l1: essential story (~500 tokens, top semantic memories)
    l2: filtered relevant (~1000 tokens, wing/room filtered)
    l3: deep search (rest of budget, full hybrid search)
    """
    if config is None:
        config = Config.load()

    layers = {}

    # L0: Identity — highest-importance semantic memories
    identity_mems = store.get_memories_by_layer(MemoryLayer.SEMANTIC, limit=5)
    identity_mems = sorted(identity_mems, key=lambda m: m.importance, reverse=True)
    l0_parts = []
    token_count = 0
    for m in identity_mems:
        est_tokens = len(m.content.split()) * 1.3
        if token_count + est_tokens > 150:
            break
        l0_parts.append(m.content)
        token_count += est_tokens
    layers["l0"] = "\n".join(l0_parts)

    # L1: Essential story — top episodic + semantic by importance
    all_important = []
    for layer_name in (MemoryLayer.SEMANTIC, MemoryLayer.EPISODIC):
        mems = store.get_memories_by_layer(layer_name, limit=20)
        all_important.extend(mems)
    all_important.sort(key=lambda m: m.importance, reverse=True)

    l1_parts = []
    token_count = 0
    seen_ids = {m.id for m in identity_mems[:len(l0_parts)]}
    for m in all_important:
        if m.id in seen_ids:
            continue
        est_tokens = len(m.content.split()) * 1.3
        if token_count + est_tokens > 600:
            break
        l1_parts.append(m.content)
        token_count += est_tokens
        seen_ids.add(m.id)
    layers["l1"] = "\n".join(l1_parts)

    # L2: Recent + procedural
    recent = store.get_recent_memories(limit=10)
    procedural = store.get_memories_by_layer(MemoryLayer.PROCEDURAL, limit=10)
    l2_mems = [m for m in recent + procedural if m.id not in seen_ids]
    l2_mems.sort(key=lambda m: m.created_at, reverse=True)

    l2_parts = []
    token_count = 0
    for m in l2_mems:
        est_tokens = len(m.content.split()) * 1.3
        if token_count + est_tokens > 1000:
            break
        l2_parts.append(m.content)
        token_count += est_tokens
        seen_ids.add(m.id)
    layers["l2"] = "\n".join(l2_parts)

    # L3: Query-driven deep search with salience-based token budgeting (MAGMA)
    if query:
        results = search(query, store, config, top_k=20)
        remaining = max_tokens - sum(len(layers[lk].split()) * 1.3 for lk in layers)

        # salience budgeting: high-score results get full content,
        # low-score results get compressed to one-liners
        l3_parts = []
        token_count = 0
        if results:
            max_score = results[0].score if results else 1.0
            for r in results:
                if r.memory.id in seen_ids:
                    continue
                salience = r.score / max(0.01, max_score)
                content = r.memory.content

                if salience < 0.3:
                    # low salience — compress to first line only
                    first_line = content.split('\n')[0][:120]
                    content = f"[{r.memory.layer}] {first_line}"
                elif salience < 0.6:
                    # medium salience — truncate to 200 chars
                    content = content[:200] + ("..." if len(content) > 200 else "")

                est_tokens = len(content.split()) * 1.3
                if token_count + est_tokens > remaining:
                    break
                l3_parts.append(content)
                token_count += est_tokens
                seen_ids.add(r.memory.id)
        layers["l3"] = "\n".join(l3_parts)
    else:
        layers["l3"] = ""

    return layers


def format_context(layers: dict[str, str], include: str = "all") -> str:
    """Format layers into a single context string."""
    parts = []

    if include in ("all", "l0") and layers.get("l0"):
        parts.append(f"[Identity]\n{layers['l0']}")
    if include in ("all", "l1") and layers.get("l1"):
        parts.append(f"[Core Knowledge]\n{layers['l1']}")
    if include in ("all", "l2") and layers.get("l2"):
        parts.append(f"[Recent & Procedural]\n{layers['l2']}")
    if include in ("all", "l3") and layers.get("l3"):
        parts.append(f"[Search Results]\n{layers['l3']}")

    return "\n\n".join(parts)
