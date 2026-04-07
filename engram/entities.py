"""Entity extraction (regex + heuristics), registry, and relationship graph."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections import Counter

from engram.store import Store, Entity, Relationship

# known tech terms that shouldn't be treated as person names
TECH_TERMS = {
    "python", "javascript", "typescript", "react", "vue", "angular", "node",
    "redis", "sqlite", "postgres", "mongodb", "neo4j", "docker", "kubernetes",
    "linux", "macos", "windows", "git", "github", "gitlab", "vscode",
    "claude", "gpt", "openai", "anthropic", "llm", "api", "rest", "graphql",
    "swift", "swiftui", "electron", "rust", "golang", "java", "kotlin",
    "fastapi", "flask", "django", "express", "nextjs", "vite", "webpack",
    "chromadb", "faiss", "numpy", "pandas", "pytorch", "tensorflow",
    "supabase", "firebase", "aws", "gcp", "azure", "vercel", "netlify",
    "nginx", "apache", "tailscale", "wireguard", "ssl", "tls", "ssh",
    "homebrew", "pip", "npm", "yarn", "pnpm", "cargo", "gradle", "maven",
    "bert", "t5", "qwen", "llama", "mistral", "gemini", "sonnet", "opus", "haiku",
    "slippi", "melee", "minecraft", "modforge",
}

PERSON_PATTERNS = [
    r"\b([A-Z][a-z]+)\s+(?:said|told|asked|mentioned|explained|noted|wrote|built|created|made)\b",
    r"\b(?:with|from|by|for)\s+([A-Z][a-z]+)\b",
    r"\b([A-Z][a-z]+)'s\b",
]

RELATIONSHIP_PATTERNS = [
    (r"(\w+)\s+(?:works?\s+(?:with|on|at))\s+(\w+)", "WORKS_WITH"),
    (r"(\w+)\s+(?:uses?|using)\s+(\w+)", "USES"),
    (r"(\w+)\s+(?:built|created|made|wrote)\s+(\w+)", "CREATED"),
    (r"(\w+)\s+(?:depends?\s+on)\s+(\w+)", "DEPENDS_ON"),
    (r"(\w+)\s+(?:manages?|leads?|runs?)\s+(\w+)", "MANAGES"),
]


def extract_entities(text: str) -> list[dict]:
    entities = []
    seen_names = set()

    # emails
    for m in re.finditer(r"\b[\w.+-]+@[\w.-]+\.\w+\b", text):
        name = m.group(0)
        if name not in seen_names:
            entities.append({"name": name, "type": "email"})
            seen_names.add(name)

    # URLs
    for m in re.finditer(r"https?://[\w./\-?=&#]+", text):
        name = m.group(0)
        if name not in seen_names:
            entities.append({"name": name, "type": "url"})
            seen_names.add(name)

    # file paths
    for m in re.finditer(r"(?:~/|/[\w]+/)[\w./\-]+", text):
        name = m.group(0)
        if name not in seen_names:
            entities.append({"name": name, "type": "path"})
            seen_names.add(name)

    # dates
    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        name = m.group(0)
        if name not in seen_names:
            entities.append({"name": name, "type": "date"})
            seen_names.add(name)

    # person names (contextual patterns)
    for pat in PERSON_PATTERNS:
        for m in re.finditer(pat, text):
            name = m.group(1)
            if name.lower() not in TECH_TERMS and name not in seen_names and len(name) > 1:
                entities.append({"name": name, "type": "person"})
                seen_names.add(name)

    # capitalized words that appear multiple times (potential entities)
    caps = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
    cap_counts = Counter(caps)
    for name, count in cap_counts.items():
        if count >= 2 and name not in seen_names and name.lower() not in TECH_TERMS:
            entities.append({"name": name, "type": "concept"})
            seen_names.add(name)

    # tech terms mentioned
    text_lower = text.lower()
    for term in TECH_TERMS:
        if term in text_lower and term not in seen_names:
            if re.search(rf"\b{re.escape(term)}\b", text_lower):
                entities.append({"name": term, "type": "tool"})
                seen_names.add(term)

    return entities


def extract_relationships(text: str, entities: list[dict]) -> list[dict]:
    relationships = []
    entity_names = {e["name"].lower() for e in entities}

    for pat, rel_type in RELATIONSHIP_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            src = m.group(1)
            tgt = m.group(2)
            if src.lower() in entity_names or tgt.lower() in entity_names:
                relationships.append({
                    "source": src,
                    "target": tgt,
                    "type": rel_type,
                })

    # co-occurrence: entities in same sentence
    sentences = re.split(r"[.!?\n]+", text)
    for sent in sentences:
        sent_entities = []
        for e in entities:
            if e["name"].lower() in sent.lower():
                sent_entities.append(e["name"])
        for i, e1 in enumerate(sent_entities):
            for e2 in sent_entities[i+1:]:
                relationships.append({
                    "source": e1,
                    "target": e2,
                    "type": "CO_OCCURS",
                })

    return relationships


def ensure_entity(store: Store, name: str, entity_type: str = "concept") -> Entity:
    existing = store.find_entity_by_name(name)
    if existing:
        # update last_seen
        now = time.time()
        store.conn.execute("UPDATE entities SET last_seen = ? WHERE id = ?", (now, existing.id))
        store.conn.commit()
        return existing

    entity = Entity(
        id=str(uuid.uuid4()),
        canonical_name=name,
        entity_type=entity_type,
        first_seen=time.time(),
        last_seen=time.time(),
    )
    store.save_entity(entity)
    return entity


def process_entities_for_memory(store: Store, memory_id: str, text: str):
    raw_entities = extract_entities(text)
    raw_rels = extract_relationships(text, raw_entities)

    entity_map = {}
    for raw in raw_entities:
        entity = ensure_entity(store, raw["name"], raw["type"])
        entity_map[raw["name"].lower()] = entity
        store.link_entity_memory(entity.id, memory_id)

    now = time.time()
    for raw_rel in raw_rels:
        src_name = raw_rel["source"].lower()
        tgt_name = raw_rel["target"].lower()
        src = entity_map.get(src_name) or store.find_entity_by_name(raw_rel["source"])
        tgt = entity_map.get(tgt_name) or store.find_entity_by_name(raw_rel["target"])
        if src and tgt and src.id != tgt.id:
            rel = Relationship(
                source_entity_id=src.id,
                target_entity_id=tgt.id,
                relation_type=raw_rel["type"],
                created_at=now,
                last_seen=now,
            )
            store.save_relationship(rel)
