"""AAAK-style compression for context windows."""

from __future__ import annotations

import re

# 3-char entity codes
_entity_codes: dict[str, str] = {}
_next_code_idx = 0


def _get_code(name: str) -> str:
    global _next_code_idx
    key = name.lower()
    if key not in _entity_codes:
        # generate 3-char uppercase code
        if len(name) >= 3:
            code = name[:3].upper()
        else:
            code = name.upper().ljust(3, "X")
        # deduplicate
        base = code
        i = 0
        while code in _entity_codes.values():
            i += 1
            code = base[:2] + str(i)
        _entity_codes[key] = code
        _next_code_idx += 1
    return _entity_codes[key]


def compress(text: str, entities: list[str] | None = None,
             aggressive: bool = False) -> str:
    result = text

    # entity substitution
    if entities:
        for name in sorted(entities, key=len, reverse=True):
            code = _get_code(name)
            result = re.sub(rf"\b{re.escape(name)}\b", code, result, flags=re.IGNORECASE)

    # strip filler
    filler = [
        r"\b(?:basically|essentially|fundamentally|importantly|interestingly|notably)\b",
        r"\b(?:in order to|for the purpose of|with regard to|in terms of)\b",
        r"\b(?:it is worth noting that|it should be noted that|it is important to note)\b",
        r"\b(?:as a matter of fact|as mentioned (?:earlier|above|before))\b",
    ]
    for pat in filler:
        result = re.sub(pat, "", result, flags=re.IGNORECASE)

    if aggressive:
        # strip articles and some conjunctions
        result = re.sub(r"\b(?:the|a|an)\b", "", result, flags=re.IGNORECASE)
        result = re.sub(r"\b(?:that|which|who)\b", "", result, flags=re.IGNORECASE)

    # collapse whitespace
    result = re.sub(r"  +", " ", result)
    result = re.sub(r"\n\s*\n\s*\n", "\n\n", result)

    return result.strip()


def compress_memories(memories: list, entities: list[str] | None = None,
                      max_tokens: int = 2000) -> str:
    parts = []
    token_count = 0

    for mem in memories:
        content = mem.content if hasattr(mem, "content") else str(mem)
        compressed = compress(content, entities)
        est_tokens = len(compressed.split()) * 1.3

        if token_count + est_tokens > max_tokens:
            # try aggressive compression
            compressed = compress(content, entities, aggressive=True)
            est_tokens = len(compressed.split()) * 1.3
            if token_count + est_tokens > max_tokens:
                break

        parts.append(compressed)
        token_count += est_tokens

    # prepend legend if codes were used
    if _entity_codes and entities:
        used_codes = {k: v for k, v in _entity_codes.items() if k in {e.lower() for e in (entities or [])}}
        if used_codes:
            legend = " | ".join(f"{v}={k}" for k, v in sorted(used_codes.items(), key=lambda x: x[1]))
            return f"[{legend}]\n\n" + "\n---\n".join(parts)

    return "\n---\n".join(parts)
