"""LLM-powered fact extraction with hypothetical query generation (docTTTTTquery)."""

from __future__ import annotations

import hashlib
import re
import uuid
import time
from pathlib import Path

from engram.config import Config
from engram.llm import query_llm, extract_json_from_response
from engram.store import Memory, MemoryLayer, MemoryType, SourceType

EXTRACTION_SYSTEM = """You are a memory extraction system. Your job is to distill raw text into atomic, self-contained factual statements.

Rules:
1. Each fact must be independently comprehensible — no pronouns, no "this", no "it"
2. Each fact must name its subject explicitly
3. Prefix each fact with a date in brackets if one can be inferred: [2026-03-28]
4. Include the source context (who said it, where it came from) when available
5. One fact per line
6. Classify each fact: factual, experiential, or procedural
7. Rate importance 0.0-1.0 (identity/values=0.9+, preferences=0.7, events=0.5, trivia=0.3)
8. If the text contains nothing worth remembering, return NONE

Respond with a JSON array of objects:
[
  {
    "content": "the atomic fact",
    "fact_date": "2026-03-28" or null,
    "type": "factual" | "experiential" | "procedural",
    "importance": 0.7,
    "entities": ["entity1", "entity2"],
    "relationships": [{"source": "entity1", "target": "entity2", "type": "WORKS_WITH"}]
  }
]"""

QUERY_GEN_SYSTEM = """Generate 2-3 short questions that this memory would answer. These are hypothetical search queries someone might type to find this information. Be specific and natural.

Respond with a JSON array of strings."""


def extract_facts(text: str, source_file: str | None = None,
                  config: Config | None = None) -> list[dict]:
    if config is None:
        config = Config.load()

    if len(text.strip()) < 20:
        return []

    # chunk long texts
    chunks = _chunk_text(text, max_chars=3000)
    all_facts = []

    for chunk in chunks:
        prompt = f"Extract atomic facts from this text:\n\n{chunk}"
        try:
            response = query_llm(prompt, system=EXTRACTION_SYSTEM, config=config)
            if "NONE" in response.upper() and len(response) < 20:
                continue
            facts = extract_json_from_response(response)
            if isinstance(facts, list):
                for f in facts:
                    f["source_file"] = source_file
                all_facts.extend(facts)
        except Exception as e:
            # fallback: treat the chunk as a single fact
            all_facts.append({
                "content": chunk.strip(),
                "fact_date": _extract_date(chunk),
                "type": "factual",
                "importance": 0.5,
                "entities": [],
                "relationships": [],
                "source_file": source_file,
            })

    return all_facts


def generate_hypothetical_queries(content: str, config: Config | None = None) -> list[str]:
    if config is None:
        config = Config.load()

    prompt = f"Memory content:\n{content}"
    try:
        response = query_llm(prompt, system=QUERY_GEN_SYSTEM, config=config)
        queries = extract_json_from_response(response)
        if isinstance(queries, list):
            return [str(q) for q in queries[:5]]
    except Exception:
        pass
    return []


def facts_to_memories(facts: list[dict], source_file: str | None = None) -> list[tuple[Memory, list[str]]]:
    results = []
    for fact in facts:
        content = fact.get("content", "").strip()
        if not content:
            continue

        layer_map = {
            "factual": MemoryLayer.EPISODIC,
            "experiential": MemoryLayer.EPISODIC,
            "procedural": MemoryLayer.PROCEDURAL,
        }
        type_map = {
            "factual": MemoryType.FACT,
            "experiential": MemoryType.NARRATIVE,
            "procedural": MemoryType.PROCEDURE,
        }
        layer = layer_map.get(fact.get("type", "factual"), MemoryLayer.EPISODIC)
        memory_type = type_map.get(fact.get("type", "factual"), MemoryType.NARRATIVE)

        mem = Memory(
            id=str(uuid.uuid4()),
            content=content,
            source_file=fact.get("source_file", source_file),
            source_type=SourceType.INGEST,
            layer=layer,
            memory_type=memory_type,
            importance=fact.get("importance", 0.5),
            fact_date=fact.get("fact_date"),
            metadata={
                "entities": fact.get("entities", []),
                "relationships": fact.get("relationships", []),
                "type": fact.get("type", "factual"),
            },
        )
        results.append((mem, []))  # queries will be generated in batch later
    return results


def _chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # try to break at paragraph or sentence boundary
            for sep in ("\n\n", "\n", ". ", " "):
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _extract_date(text: str) -> str | None:
    patterns = [
        r"\[(\d{4}-\d{2}-\d{2})\]",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(20\d{2}-\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None
