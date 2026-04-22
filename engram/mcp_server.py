"""MCP tool server — JSON-RPC over stdio with 66 tools."""

from __future__ import annotations

import json
import sys
import time
import uuid
from typing import Any

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, MemoryType, MemoryStatus, SourceType
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search
from engram.entities import process_entities_for_memory
from engram.extractor import generate_hypothetical_queries
from engram.layers import get_context_layers, format_context
from engram.compress import compress_memories
from engram.lifecycle import compute_importance
from engram.consolidator import consolidate
from engram.surprise import compute_surprise, adjust_importance
from engram.deep_retrieval import DeepReranker
from engram.skill_select import select_skills, format_skills
from engram.drift import run_drift_check, auto_fix_drift
from engram.patterns import extract_patterns_from_session, store_patterns
from engram.evolution import (enrich_memory, evolve_neighbors, check_confirmation,
                              get_source_trust, classify_write_operation,
                              annotate_causal_parent, canonicalize_content)

TOOLS = [
    # Read tools
    {"name": "recall", "description": "Search memories using hybrid retrieval (dense + BM25 + graph + cross-encoder). Use mode to filter by memory type: facts_only (structured knowledge), facts_plus_rules (+ procedures), full_context (everything).", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}, "mode": {"type": "string", "enum": ["facts_only", "facts_plus_rules", "full_context"], "default": "full_context", "description": "Retrieval profile — facts_only for statuses/states, facts_plus_rules for methodology, full_context for exhaustive recall"}}, "required": ["query"]}},
    {"name": "recall_explain", "description": "Search memories and include retrieval intent, expansions, cache status, candidate counts, and score breakdowns for debugging retrieval quality.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}, "mode": {"type": "string", "enum": ["facts_only", "facts_plus_rules", "full_context"], "default": "full_context"}}, "required": ["query"]}},
    {"name": "recall_entity", "description": "Get everything about a specific entity — facts, relationships, timeline", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "recall_timeline", "description": "Query memories by date range", "inputSchema": {"type": "object", "properties": {"start": {"type": "string", "description": "Start date YYYY-MM-DD or YYYY-MM"}, "end": {"type": "string"}}, "required": ["start"]}},
    {"name": "recall_related", "description": "Multi-hop graph traversal from an entity", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "max_hops": {"type": "integer", "default": 2}}, "required": ["name"]}},
    {"name": "recall_recent", "description": "Get last N memories by creation time", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
    {"name": "recall_layer", "description": "Search within a specific memory layer", "inputSchema": {"type": "object", "properties": {"layer": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]}, "limit": {"type": "integer", "default": 20}}, "required": ["layer"]}},
    {"name": "status", "description": "System stats — memory counts, entities, storage size", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "layers", "description": "Get L0-L3 graduated context for system prompt injection", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "max_tokens": {"type": "integer", "default": 4000}}}},
    {"name": "entity_graph", "description": "Get entity relationship subgraph as JSON", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "access_patterns", "description": "What memories are recalled most, hit rates", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
    # Write tools
    {"name": "remember", "description": "Store a memory", "inputSchema": {"type": "object", "properties": {"content": {"type": "string"}, "source_type": {"type": "string", "default": "remember:human"}, "layer": {"type": "string", "default": "episodic"}, "memory_type": {"type": "string", "enum": ["fact", "procedure", "narrative"], "default": "narrative", "description": "Semantic type: fact (structured knowledge), procedure (how-to/rules), narrative (session logs)"}, "importance": {"type": "number", "default": 0.7}}, "required": ["content"]}},
    {"name": "remember_interaction", "description": "Store a Q+A exchange pair", "inputSchema": {"type": "object", "properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "importance": {"type": "number", "default": 0.5}}, "required": ["question", "answer"]}},
    {"name": "remember_decision", "description": "Store a decision with rationale → procedural", "inputSchema": {"type": "object", "properties": {"decision": {"type": "string"}, "rationale": {"type": "string"}, "importance": {"type": "number", "default": 0.8}}, "required": ["decision"]}},
    {"name": "remember_error", "description": "Store an error pattern with prevention → procedural", "inputSchema": {"type": "object", "properties": {"error": {"type": "string"}, "prevention": {"type": "string"}, "importance": {"type": "number", "default": 0.7}}, "required": ["error"]}},
    {"name": "forget", "description": "Soft-delete a memory", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
    {"name": "update_entity", "description": "Add alias or update entity metadata", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "alias": {"type": "string"}, "metadata": {"type": "object"}}, "required": ["name"]}},
    {"name": "invalidate", "description": "Mark a fact as no longer true", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["memory_id"]}},
    # Lifecycle tools
    {"name": "consolidate", "description": "Run dream cycle — cluster, summarize, generate peer cards", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "promote", "description": "Promote a memory to a higher layer", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "target_layer": {"type": "string"}}, "required": ["memory_id", "target_layer"]}},
    {"name": "demote", "description": "Demote a memory to a lower layer", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "target_layer": {"type": "string"}}, "required": ["memory_id", "target_layer"]}},
    {"name": "compress", "description": "Get compressed version of retrieved memories", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "max_tokens": {"type": "integer", "default": 2000}}, "required": ["query"]}},
    {"name": "ingest", "description": "Ingest a file or directory into memory", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    # Diary tools
    {"name": "diary_write", "description": "Append to session diary", "inputSchema": {"type": "object", "properties": {"entry": {"type": "string"}}, "required": ["entry"]}},
    {"name": "diary_read", "description": "Read current session diary", "inputSchema": {"type": "object", "properties": {}}},
    # Extended tools
    {"name": "find_similar", "description": "Find memories most similar to a given memory by embedding distance", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["memory_id"]}},
    {"name": "merge_entities", "description": "Merge two entities that are the same thing — moves all relationships and memory links to the target", "inputSchema": {"type": "object", "properties": {"source_name": {"type": "string", "description": "Entity to merge FROM (will be deleted)"}, "target_name": {"type": "string", "description": "Entity to merge INTO (will be kept)"}}, "required": ["source_name", "target_name"]}},
    {"name": "remember_project", "description": "Store a structured project memory — name, status, location, notes → semantic layer", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "status": {"type": "string", "description": "e.g. active, paused, completed, abandoned"}, "location": {"type": "string", "description": "file path or URL"}, "notes": {"type": "string"}}, "required": ["name"]}},
    {"name": "recall_context", "description": "Search and return a formatted context block ready for prompt injection, with token budget", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "max_tokens": {"type": "integer", "default": 2000}}, "required": ["query"]}},
    {"name": "count_by", "description": "Count memories grouped by a field — layer, source_type, entity, or month", "inputSchema": {"type": "object", "properties": {"group_by": {"type": "string", "enum": ["layer", "source_type", "entity", "month"]}}, "required": ["group_by"]}},
    {"name": "bulk_forget", "description": "Forget all memories matching criteria — by source_file, layer, or older than a date", "inputSchema": {"type": "object", "properties": {"source_file": {"type": "string"}, "layer": {"type": "string"}, "older_than": {"type": "string", "description": "YYYY-MM-DD — forget memories created before this date"}, "confirm": {"type": "boolean", "description": "Must be true to execute"}}, "required": ["confirm"]}},
    {"name": "export", "description": "Export memories as markdown or JSON", "inputSchema": {"type": "object", "properties": {"format": {"type": "string", "enum": ["markdown", "json"], "default": "markdown"}, "layer": {"type": "string"}, "limit": {"type": "integer", "default": 100}}}},
    {"name": "health", "description": "System health check — embedding cache, FTS index, orphaned entities, stale working memories", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "tag", "description": "Add or remove tags on a memory", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "add": {"type": "array", "items": {"type": "string"}}, "remove": {"type": "array", "items": {"type": "string"}}}, "required": ["memory_id"]}},
    {"name": "link_memories", "description": "Manually link two memories as related via their entities", "inputSchema": {"type": "object", "properties": {"memory_id_1": {"type": "string"}, "memory_id_2": {"type": "string"}, "relation": {"type": "string", "default": "RELATED_TO"}}, "required": ["memory_id_1", "memory_id_2"]}},
    # Codebase tools
    {"name": "scan_codebase", "description": "Scan a project directory and extract compressed code knowledge — file tree, function signatures, dependencies. Uses ~10x fewer tokens than raw code.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "project_name": {"type": "string"}}, "required": ["path"]}},
    {"name": "recall_code", "description": "Search the codebase layer specifically — find functions, classes, files, dependencies across scanned projects", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "project": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["query"]}},
    {"name": "list_projects", "description": "List all scanned codebase projects with file counts and memory counts", "inputSchema": {"type": "object", "properties": {}}},
    # Dedup & maintenance
    {"name": "dedup", "description": "Find and merge near-duplicate memories by embedding similarity", "inputSchema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.92, "description": "Cosine similarity threshold (0.0-1.0)"}, "max_merges": {"type": "integer", "default": 50}}}},
    {"name": "find_duplicates", "description": "Find near-duplicate memory pairs without merging them", "inputSchema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.92}, "limit": {"type": "integer", "default": 20}}}},
    # Conversation tools
    {"name": "ingest_sessions", "description": "Ingest recent Claude Code conversation sessions into memory", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}},
    # Session tools
    {"name": "session_summary", "description": "Generate a summary of the current session based on diary and recent events", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "session_handoff", "description": "Build a structured handoff snapshot for the current session or a specific session. Can also persist the snapshot for later resume.", "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string", "description": "Optional specific session id. Defaults to the current MCP session."}, "save": {"type": "boolean", "default": True, "description": "Persist the generated handoff snapshot."}, "limit": {"type": "integer", "default": 8, "description": "Max items returned per section."}}}},
    {"name": "session_checkpoint", "description": "Write an optional checkpoint note and persist a richer handoff packet for the current session so another session can resume from a clean stop point.", "inputSchema": {"type": "object", "properties": {"note": {"type": "string", "description": "Optional checkpoint note to store in the session diary before saving the handoff."}, "limit": {"type": "integer", "default": 8, "description": "Max items returned per section."}}}},
    {"name": "resume_context", "description": "Get the latest structured handoff snapshot plus recent related activity so a new agent session can resume quickly.", "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string", "description": "Optional exact session id to resume."}, "limit": {"type": "integer", "default": 3, "description": "How many recent handoffs to include if session_id is omitted."}}}},
    # Backlinks
    {"name": "backlinks", "description": "Find all memories that reference or are linked to a specific memory", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
    # Batch operations
    {"name": "batch_tag", "description": "Add tags to all memories matching a search query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "top_k": {"type": "integer", "default": 10}}, "required": ["query", "tags"]}},
    {"name": "recompute_importance", "description": "Recalculate importance scores for all memories using the 9-factor formula", "inputSchema": {"type": "object", "properties": {}}},
    # Edit & annotate
    {"name": "edit_memory", "description": "Edit the content of an existing memory. Re-embeds automatically.", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "new_content": {"type": "string"}}, "required": ["memory_id", "new_content"]}},
    {"name": "annotate", "description": "Add a note/annotation to a memory without changing its content", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "note": {"type": "string"}}, "required": ["memory_id", "note"]}},
    {"name": "pin", "description": "Pin a memory so it never gets forgotten or archived by the dream cycle", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
    {"name": "unpin", "description": "Unpin a previously pinned memory", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
    # Entity tools
    {"name": "search_entities", "description": "Fuzzy search for entities by name — finds partial matches unlike recall_entity which needs exact name", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 20}}, "required": ["query"]}},
    {"name": "entity_timeline", "description": "Get an entity's memories ordered by date — see its story chronologically", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    # Explain
    {"name": "explain_importance", "description": "Break down a memory's importance score into its 9 component factors", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
    # Visualization data
    {"name": "memory_map", "description": "Get a high-level map of the entire memory system — layer counts, top entities per layer, recent activity, oldest/newest memories", "inputSchema": {"type": "object", "properties": {}}},
    # Deep retrieval
    {"name": "train_reranker", "description": "Train the deep retrieval MLP reranker on access patterns — learns which memories are actually useful from historical recall data", "inputSchema": {"type": "object", "properties": {"epochs": {"type": "integer", "default": 50}, "learning_rate": {"type": "number", "default": 0.01}}}},
    {"name": "reranker_status", "description": "Check if the deep reranker is trained and its model path", "inputSchema": {"type": "object", "properties": {}}},
    # Cognitive scaffolding
    {"name": "recall_hints", "description": "Search memories but return only hints (truncated snippets + entity names) to trigger recognition without replacing cognition. Use when you want to check if you know something before pulling full context. Returns memory IDs you can fetch with recall if needed.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}, "hint_length": {"type": "integer", "default": 60, "description": "Max characters per hint snippet"}}, "required": ["query"]}},
    # Skill selection
    {"name": "get_skills", "description": "Task-aware skill selection — get focused procedural guidance for a task. Returns 2-3 relevant skills only when injection would help. Skips when the task is well-covered by model pretraining or no relevant procedures exist. Based on SkillsBench finding that focused skills (+16.2pp) beat comprehensive docs (-2.9pp).", "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "What task or problem you need procedural guidance for"}, "max_skills": {"type": "integer", "default": 3, "description": "Maximum number of skills to return (2-3 is optimal)"}, "format": {"type": "boolean", "default": True, "description": "Return formatted context block (true) or raw selection data (false)"}}, "required": ["query"]}},
    # Drift detection
    {"name": "drift_check", "description": "Verify memories against filesystem reality — find dead paths, missing functions, stale memories. Returns a drift score (0-100) and list of issues. Zero AI cost, pure filesystem checks.", "inputSchema": {"type": "object", "properties": {"search_roots": {"type": "array", "items": {"type": "string"}, "description": "Directories to search for function/class verification (e.g. ['~/project/src'])"}, "project_root": {"type": "string", "description": "Project root for command verification (package.json, Makefile)"}, "layers": {"type": "array", "items": {"type": "string"}, "description": "Memory layers to check (default: all)"}, "check_functions": {"type": "boolean", "default": True, "description": "Whether to grep for function names (slower but more thorough)"}}}},
    {"name": "drift_fix", "description": "Auto-fix drift issues — invalidate memories with dead paths, flag stale memories, forget invalidated-but-active memories. Use dry_run=true first to preview changes.", "inputSchema": {"type": "object", "properties": {"search_roots": {"type": "array", "items": {"type": "string"}, "description": "Directories to search for function/class verification"}, "project_root": {"type": "string", "description": "Project root for command verification"}, "dry_run": {"type": "boolean", "default": True, "description": "If true, only report what would be done"}}}},
    # Pattern extraction
    {"name": "extract_patterns", "description": "Extract reusable procedural patterns from recent session activity — diary entries, new memories, errors, decisions. Checks novelty against existing procedural memories and stores only what's genuinely new.", "inputSchema": {"type": "object", "properties": {"hours": {"type": "number", "default": 4.0, "description": "How far back to look for session activity"}, "novelty_threshold": {"type": "number", "default": 0.25, "description": "Minimum novelty score (0-1) to store a pattern"}, "dry_run": {"type": "boolean", "default": False, "description": "If true, extract and score patterns but don't store them"}}}},
    # Negative knowledge
    {"name": "compress_embeddings", "description": "Compress old memory embeddings based on retention — active=32bit, warm=8bit, cold=4bit, archive=2bit. Uses FRQAD for mixed-precision comparison. Saves storage without losing retrievability.", "inputSchema": {"type": "object", "properties": {"dry_run": {"type": "boolean", "default": True, "description": "Preview what would be compressed"}}}},
    {"name": "detect_communities", "description": "Run label propagation over the entity graph to discover clusters. Generates community summaries for higher-level retrieval.", "inputSchema": {"type": "object", "properties": {"min_size": {"type": "integer", "default": 3, "description": "Minimum community size"}, "generate_summaries": {"type": "boolean", "default": False, "description": "Generate LLM summaries for communities"}}}},
    {"name": "quality_metrics", "description": "Memory system quality metrics — storage quality ratio (what % of stored memories get recalled), curation ratio (active maintenance vs passive accumulation), retrieval relevance. Based on AgeMem reward decomposition.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "remember_negative", "description": "Store explicit negative knowledge — what does NOT exist, what was deliberately excluded, what should NOT be done. Prevents future hallucinated recommendations. Examples: 'There is no caching layer in this project', 'We deliberately do not use Redux', 'The /admin endpoint was removed in v2'.", "inputSchema": {"type": "object", "properties": {"content": {"type": "string", "description": "What does NOT exist or should NOT be done"}, "context": {"type": "string", "description": "Why this negative fact matters — what mistake it prevents"}, "scope": {"type": "string", "description": "What project/system this applies to"}, "importance": {"type": "number", "default": 0.75}}, "required": ["content"]}},
    # Status & type tools (v0.3.0)
    {"name": "update_status", "description": "Transition a memory's lifecycle status (active → challenged → invalidated/merged/superseded) with audit trail", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}, "new_status": {"type": "string", "enum": ["active", "challenged", "invalidated", "merged", "superseded"]}, "reason": {"type": "string"}}, "required": ["memory_id", "new_status"]}},
    {"name": "recall_by_type", "description": "Get memories filtered by semantic type — fact (structured knowledge), procedure (how-to/rules), narrative (session logs)", "inputSchema": {"type": "object", "properties": {"memory_type": {"type": "string", "enum": ["fact", "procedure", "narrative"]}, "limit": {"type": "integer", "default": 20}}, "required": ["memory_type"]}},
    {"name": "status_history", "description": "Get the full status transition history for a memory — what changed, when, and why", "inputSchema": {"type": "object", "properties": {"memory_id": {"type": "string"}}, "required": ["memory_id"]}},
]

_session_diary: list[str] = []


def _suggest_resume_queries(open_loops: list[str], touched_entities: set[str], recall_queries: list[str], limit: int) -> list[str]:
    suggestions: list[str] = []
    for loop in open_loops[:limit]:
        suggestions.append(f"status of {loop[:80]}")
    for entity in sorted(touched_entities)[:limit]:
        suggestions.append(f"recent work involving {entity}")
    for query in recall_queries[:limit]:
        suggestions.append(query)
    deduped = []
    seen = set()
    for item in suggestions:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped[:limit]


class MCPServer:
    def __init__(self, config: Config):
        self.config = config
        # set embedding backend + default model from config
        from engram.embeddings import set_backend, set_default_model
        if config.embedding_backend and config.embedding_backend != "auto":
            set_backend(config.embedding_backend)
        set_default_model(config.embedding_model)
        self.store = Store(config)
        self.store.init_db()
        self.store.init_ann_index(background=True)
        self._session_id = str(uuid.uuid4())[:8]
        self._session_started_at = time.time()
        # deep reranker — persists model next to db
        model_path = config.resolved_db_path.parent / "reranker.npz"
        self._reranker = DeepReranker(model_path=model_path)

    def handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self._response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "engram", "version": "0.1.0"},
            })
        elif method == "tools/list":
            return self._response(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            try:
                result = self._call_tool(tool_name, tool_args)
                return self._response(req_id, {"content": [{"type": "text", "text": json.dumps(result, default=str)}]})
            except Exception as e:
                return self._error(req_id, str(e))
        elif method == "notifications/initialized":
            return None  # no response needed
        else:
            return self._error(req_id, f"Unknown method: {method}")

    def _call_tool(self, name: str, args: dict) -> Any:
        handlers = {
            "recall": self._recall,
            "recall_explain": self._recall_explain,
            "recall_entity": self._recall_entity,
            "recall_timeline": self._recall_timeline,
            "recall_related": self._recall_related,
            "recall_recent": self._recall_recent,
            "recall_layer": self._recall_layer,
            "status": self._status,
            "layers": self._layers,
            "entity_graph": self._entity_graph,
            "access_patterns": self._access_patterns,
            "remember": self._remember,
            "remember_interaction": self._remember_interaction,
            "remember_decision": self._remember_decision,
            "remember_error": self._remember_error,
            "forget": self._forget,
            "update_entity": self._update_entity,
            "invalidate": self._invalidate,
            "consolidate": self._consolidate,
            "promote": self._promote,
            "demote": self._demote,
            "compress": self._compress,
            "ingest": self._ingest,
            "diary_write": self._diary_write,
            "diary_read": self._diary_read,
            "find_similar": self._find_similar,
            "merge_entities": self._merge_entities,
            "remember_project": self._remember_project,
            "recall_context": self._recall_context,
            "count_by": self._count_by,
            "bulk_forget": self._bulk_forget,
            "export": self._export,
            "health": self._health,
            "tag": self._tag,
            "link_memories": self._link_memories,
            "scan_codebase": self._scan_codebase,
            "recall_code": self._recall_code,
            "list_projects": self._list_projects,
            "edit_memory": self._edit_memory,
            "annotate": self._annotate,
            "pin": self._pin,
            "unpin": self._unpin,
            "search_entities": self._search_entities,
            "entity_timeline": self._entity_timeline,
            "explain_importance": self._explain_importance,
            "memory_map": self._memory_map,
            "dedup": self._dedup,
            "find_duplicates": self._find_duplicates_tool,
            "ingest_sessions": self._ingest_sessions,
            "session_summary": self._session_summary,
            "session_handoff": self._session_handoff,
            "session_checkpoint": self._session_checkpoint,
            "resume_context": self._resume_context,
            "backlinks": self._backlinks,
            "batch_tag": self._batch_tag,
            "recompute_importance": self._recompute_importance,
            "train_reranker": self._train_reranker,
            "reranker_status": self._reranker_status,
            "recall_hints": self._recall_hints,
            "get_skills": self._get_skills,
            "drift_check": self._drift_check,
            "drift_fix": self._drift_fix,
            "extract_patterns": self._extract_patterns,
            "remember_negative": self._remember_negative,
            "quality_metrics": self._quality_metrics,
            "compress_embeddings": self._compress_embeddings,
            "detect_communities": self._detect_communities,
            "update_status": self._update_status,
            "recall_by_type": self._recall_by_type,
            "status_history": self._status_history,
        }
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")
        return handler(args)

    # --- Read tools ---

    def _recall(self, args: dict):
        self._sweep_working()
        results = hybrid_search(args["query"], self.store, self.config,
                                top_k=args.get("top_k", 10),
                                deep_reranker=self._reranker,
                                mode=args.get("mode", "full_context"))
        self._refresh_session_handoff()
        return [{"id": r.memory.id, "content": r.memory.content, "score": round(r.score, 4),
                 "layer": r.memory.layer, "memory_type": r.memory.memory_type,
                 "status": r.memory.status, "fact_date": r.memory.fact_date,
                 "importance": r.memory.importance} for r in results]

    def _recall_explain(self, args: dict):
        self._sweep_working()
        results, dbg = hybrid_search(
            args["query"],
            self.store,
            self.config,
            top_k=args.get("top_k", 10),
            debug=True,
            deep_reranker=self._reranker,
            mode=args.get("mode", "full_context"),
        )
        self._refresh_session_handoff()
        return {
            "query": dbg.query,
            "intent": dbg.intent,
            "expanded_terms": dbg.expanded_terms,
            "phrase_terms": dbg.phrase_terms,
            "cache_hit": dbg.cache_hit,
            "latency_ms": round(dbg.latency_ms, 1),
            "candidate_counts": {
                "dense": len(dbg.dense_candidates),
                "bm25": len(dbg.bm25_candidates),
                "graph": len(dbg.graph_candidates),
                "rrf": len(dbg.rrf_scores),
            },
            "results": [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "score": round(r.score, 4),
                    "sources": {k: round(v, 4) for k, v in r.sources.items()},
                    "layer": r.memory.layer,
                    "memory_type": r.memory.memory_type,
                    "status": r.memory.status,
                }
                for r in results
            ],
        }

    def _sweep_working(self):
        """Auto-promote old working memories to episodic."""
        cutoff = time.time() - 1800  # 30 minutes
        rows = self.store.conn.execute(
            "SELECT id FROM memories WHERE layer = 'working' AND created_at < ? AND forgotten = 0",
            (cutoff,),
        ).fetchall()
        for row in rows:
            self.store.update_layer(row["id"], "episodic")

    def _recall_entity(self, args: dict):
        entity = self.store.find_entity_by_name(args["name"])
        if not entity:
            return {"error": f"Entity '{args['name']}' not found"}
        memories = self.store.get_entity_memories(entity.id)
        rels = self.store.get_entity_relationships(entity.id)
        return {
            "entity": {"id": entity.id, "name": entity.canonical_name, "type": entity.entity_type,
                       "aliases": entity.aliases},
            "memories": [{"content": m.content, "date": m.fact_date, "layer": m.layer} for m in memories],
            "relationships": [dict(r) for r in rels],
        }

    def _recall_timeline(self, args: dict):
        mems = self.store.get_memories_by_date_range(args["start"], args.get("end"))
        return [{"content": m.content, "date": m.fact_date, "layer": m.layer, "importance": m.importance} for m in mems]

    def _recall_related(self, args: dict):
        entity = self.store.find_entity_by_name(args["name"])
        if not entity:
            return {"error": f"Entity '{args['name']}' not found"}
        return self.store.get_related_entities(entity.id, args.get("max_hops", 2))

    def _recall_recent(self, args: dict):
        mems = self.store.get_recent_memories(args.get("limit", 20))
        return [{"id": m.id, "content": m.content, "layer": m.layer, "created_at": m.created_at} for m in mems]

    def _recall_layer(self, args: dict):
        mems = self.store.get_memories_by_layer(args["layer"], args.get("limit", 20))
        return [{"id": m.id, "content": m.content, "importance": m.importance} for m in mems]

    def _status(self, args: dict):
        return self.store.get_stats()

    def _layers(self, args: dict):
        layers = get_context_layers(self.store, args.get("query"), self.config, args.get("max_tokens", 4000))
        return {k: v for k, v in layers.items() if v}

    def _entity_graph(self, args: dict):
        entity = self.store.find_entity_by_name(args["name"])
        if not entity:
            return {"error": f"Entity '{args['name']}' not found"}
        rels = self.store.get_entity_relationships(entity.id)
        related = self.store.get_related_entities(entity.id, max_hops=2)
        nodes = [{"id": entity.id, "name": entity.canonical_name, "type": entity.entity_type}]
        for r in related:
            nodes.append({"id": r["eid"], "name": r["canonical_name"], "type": r["entity_type"]})
        return {"nodes": nodes, "edges": [dict(r) for r in rels]}

    def _access_patterns(self, args: dict):
        rows = self.store.conn.execute(
            """SELECT m.id, m.content, m.access_count, m.last_accessed
            FROM memories m WHERE m.forgotten = 0
            ORDER BY m.access_count DESC LIMIT ?""",
            (args.get("limit", 20),),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Write tools ---

    def _remember(self, args: dict):
        # canonicalize content (normalize dates, strip whitespace)
        content = canonicalize_content(args["content"])

        mem = Memory(
            id=str(uuid.uuid4()), content=content,
            source_type=args.get("source_type", SourceType.HUMAN),
            layer=args.get("layer", MemoryLayer.EPISODIC),
            memory_type=args.get("memory_type", MemoryType.NARRATIVE),
            importance=args.get("importance", 0.7),
        )

        # enriched embeddings (A-Mem): generate keywords+tags+summary, embed concatenation
        enrichment = {}
        try:
            enrichment = enrich_memory(mem.content, self.config)
            embed_text = enrichment.get("enriched_text", mem.content)
            if enrichment.get("keywords"):
                mem.metadata["keywords"] = enrichment["keywords"]
            if enrichment.get("tags"):
                mem.metadata["tags"] = enrichment["tags"]
            if enrichment.get("summary"):
                mem.metadata["enrichment_summary"] = enrichment["summary"]
        except Exception:
            embed_text = mem.content

        emb = embed_documents([embed_text], self.config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]

        # surprise gate: compute novelty before storing
        surprise_info = {}
        evolved_ids = []
        if mem.embedding is not None:
            surprise_info = compute_surprise(mem.embedding, self.store)
            mem.importance = adjust_importance(mem.importance, surprise_info)
            mem.metadata["surprise"] = surprise_info["surprise"]
            if surprise_info["is_duplicate"] and surprise_info["nearest_id"]:
                mem.metadata["duplicate_of"] = surprise_info["nearest_id"]
                # check confirmation (SuperLocalMemory) — if near-duplicate, confirm the neighbor
                neighbor = self.store.get_memory(surprise_info["nearest_id"])
                if neighbor:
                    check_confirmation(mem.content, neighbor, self.store)

            # memory evolution (A-Mem): check if neighbors should be updated
            if surprise_info.get("nearest_ids"):
                neighbors = [self.store.get_memory(nid) for nid in surprise_info["nearest_ids"][:3]]
                neighbors = [n for n in neighbors if n is not None]
                if neighbors:
                    try:
                        evolved_ids = evolve_neighbors(mem, neighbors, self.store, self.config)
                    except Exception:
                        pass

        # CRUD classification (Mem0-inspired): decide ADD/UPDATE/NOOP
        crud_op = "ADD"
        if surprise_info and surprise_info.get("nearest_id") and not surprise_info.get("is_duplicate"):
            nearest = self.store.get_memory(surprise_info["nearest_id"])
            if nearest and surprise_info.get("nearest_distance", 1.0) < 0.4:
                similarity = 1.0 - surprise_info["nearest_distance"]
                try:
                    crud_result = classify_write_operation(mem.content, nearest, similarity, self.config)
                    crud_op = crud_result["operation"]
                    if crud_op == "UPDATE" and crud_result.get("merged_content"):
                        # update existing memory instead of adding new one
                        from engram.embeddings import embed_documents as _embed
                        merged_emb = _embed([crud_result["merged_content"]], self.config.embedding_model)
                        emb_blob = merged_emb[0].astype('float32').tobytes() if merged_emb.size > 0 else None
                        self.store.conn.execute(
                            "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                            (crud_result["merged_content"], emb_blob, nearest.id),
                        )
                        self.store.conn.commit()
                        self.store.invalidate_embedding_cache()
                        self.store.invalidate_search_cache()
                        self._refresh_session_handoff()
                        return {"id": nearest.id, "status": "updated", "operation": "UPDATE"}
                    elif crud_op == "NOOP":
                        return {"id": surprise_info["nearest_id"], "status": "skipped", "operation": "NOOP",
                                "reason": "redundant with existing memory"}
                except Exception:
                    crud_op = "ADD"

        # source trust tracking
        mem.metadata["source_trust"] = get_source_trust(mem.source_type)

        hqs = []
        try:
            hqs = generate_hypothetical_queries(mem.content, self.config)
        except Exception:
            pass
        self.store.save_memory(mem, hypothetical_queries=hqs)
        process_entities_for_memory(self.store, mem.id, mem.content)

        # causal parent annotation
        try:
            annotate_causal_parent(mem, self.store)
        except Exception:
            pass
        result = {"id": mem.id, "status": "stored"}
        if surprise_info:
            result["surprise"] = surprise_info["surprise"]
            result["importance"] = mem.importance
            if surprise_info["is_duplicate"]:
                result["warning"] = "near-duplicate detected"
                result["duplicate_of"] = surprise_info["nearest_id"]
        if evolved_ids:
            result["evolved"] = evolved_ids
        if enrichment.get("tags"):
            result["tags"] = enrichment["tags"]
        self._refresh_session_handoff()
        return result

    def _remember_interaction(self, args: dict):
        content = f"Q: {args['question']}\nA: {args['answer']}"
        return self._remember({"content": content, "source_type": SourceType.INTERACTION,
                               "importance": args.get("importance", 0.5)})

    def _remember_decision(self, args: dict):
        content = f"Decision: {args['decision']}\nRationale: {args.get('rationale', '')}"
        return self._remember({"content": content, "source_type": SourceType.AI,
                               "layer": MemoryLayer.PROCEDURAL,
                               "memory_type": MemoryType.PROCEDURE,
                               "importance": args.get("importance", 0.8)})

    def _remember_error(self, args: dict):
        content = f"Error: {args['error']}\nPrevention: {args.get('prevention', '')}"
        return self._remember({"content": content, "source_type": SourceType.AI,
                               "layer": MemoryLayer.PROCEDURAL,
                               "memory_type": MemoryType.PROCEDURE,
                               "importance": args.get("importance", 0.7)})

    def _forget(self, args: dict):
        self.store.forget_memory(args["memory_id"])
        return {"status": "forgotten"}

    def _update_entity(self, args: dict):
        entity = self.store.find_entity_by_name(args["name"])
        if not entity:
            return {"error": f"Entity '{args['name']}' not found"}
        if "alias" in args:
            aliases = entity.aliases + [args["alias"]]
            self.store.conn.execute("UPDATE entities SET aliases = ? WHERE id = ?",
                                   (json.dumps(aliases), entity.id))
            self.store.conn.commit()
        return {"status": "updated", "entity_id": entity.id}

    def _invalidate(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        reason = args.get("reason", "invalidated")
        mem.metadata["invalidated"] = True
        mem.metadata["invalidation_reason"] = reason
        mem.metadata["invalidated_at"] = time.time()
        self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                               (json.dumps(mem.metadata), mem.id))
        self.store.conn.commit()
        return {"status": "invalidated"}

    # --- Lifecycle tools ---

    def _consolidate(self, args: dict):
        return consolidate(self.store, self.config)

    def _promote(self, args: dict):
        self.store.update_layer(args["memory_id"], args["target_layer"])
        return {"status": "promoted"}

    def _demote(self, args: dict):
        self.store.update_layer(args["memory_id"], args["target_layer"])
        return {"status": "demoted"}

    def _compress(self, args: dict):
        results = hybrid_search(args["query"], self.store, self.config, top_k=20)
        memories = [r.memory for r in results]
        compressed = compress_memories(memories, max_tokens=args.get("max_tokens", 2000))
        return {"compressed": compressed, "original_count": len(memories)}

    def _ingest(self, args: dict):
        from engram.cli import cmd_ingest
        import argparse
        fake_args = argparse.Namespace(paths=[args["path"]], jobs=1, no_queries=False)
        cmd_ingest(fake_args, self.config)
        return {"status": "ingested"}

    # --- Diary tools (persistent) ---

    def _diary_write(self, args: dict):
        entry = f"[{time.strftime('%H:%M:%S')}] {args['entry']}"
        _session_diary.append(entry)  # keep in-memory for session_summary
        self.store.write_diary(entry, session_id=self._session_id)
        self._refresh_session_handoff()
        return {"status": "written", "entries": len(_session_diary)}

    def _diary_read(self, args: dict):
        # return persistent diary, fallback to session
        entries = self.store.get_diary(limit=50)
        texts = [e["text"] for e in entries]
        return {"diary": texts if texts else _session_diary}

    # --- Extended tools ---

    def _find_similar(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem or mem.embedding is None:
            return {"error": "Memory not found or has no embedding"}
        from engram.embeddings import cosine_similarity_search
        ids, vecs = self.store.get_all_embeddings()
        if not ids:
            return {"similar": []}
        hits = cosine_similarity_search(mem.embedding, vecs, top_k=args.get("top_k", 5) + 1)
        results = []
        for idx, score in hits:
            if ids[idx] == mem.id:
                continue  # skip self
            other = self.store.get_memory(ids[idx])
            if other:
                results.append({"id": other.id, "content": other.content[:200],
                                "similarity": round(score, 4), "layer": other.layer})
        return {"similar": results[:args.get("top_k", 5)]}

    def _merge_entities(self, args: dict):
        source = self.store.find_entity_by_name(args["source_name"])
        target = self.store.find_entity_by_name(args["target_name"])
        if not source:
            return {"error": f"Source entity '{args['source_name']}' not found"}
        if not target:
            return {"error": f"Target entity '{args['target_name']}' not found"}
        if source.id == target.id:
            return {"error": "Source and target are the same entity"}

        # move all memory links
        self.store.conn.execute(
            "UPDATE OR IGNORE entity_mentions SET entity_id = ? WHERE entity_id = ?",
            (target.id, source.id),
        )
        # move relationships
        self.store.conn.execute(
            "UPDATE OR IGNORE relationships SET source_entity_id = ? WHERE source_entity_id = ?",
            (target.id, source.id),
        )
        self.store.conn.execute(
            "UPDATE OR IGNORE relationships SET target_entity_id = ? WHERE target_entity_id = ?",
            (target.id, source.id),
        )
        # add source name as alias on target
        aliases = target.aliases + [source.canonical_name] + source.aliases
        aliases = list(set(aliases))
        self.store.conn.execute("UPDATE entities SET aliases = ? WHERE id = ?",
                               (json.dumps(aliases), target.id))
        # delete source entity
        self.store.conn.execute("DELETE FROM entity_mentions WHERE entity_id = ?", (source.id,))
        self.store.conn.execute("DELETE FROM relationships WHERE source_entity_id = ? OR target_entity_id = ?",
                               (source.id, source.id))
        self.store.conn.execute("DELETE FROM entities WHERE id = ?", (source.id,))
        self.store.conn.commit()
        return {"status": "merged", "kept": target.canonical_name, "deleted": source.canonical_name,
                "aliases": aliases}

    def _remember_project(self, args: dict):
        parts = [f"Project: {args['name']}"]
        if args.get("status"):
            parts.append(f"Status: {args['status']}")
        if args.get("location"):
            parts.append(f"Location: {args['location']}")
        if args.get("notes"):
            parts.append(f"Notes: {args['notes']}")
        content = "\n".join(parts)
        return self._remember({"content": content, "layer": MemoryLayer.SEMANTIC,
                               "memory_type": MemoryType.FACT,
                               "importance": 0.7, "source_type": SourceType.HUMAN})

    def _recall_context(self, args: dict):
        max_tokens = args.get("max_tokens", 2000)
        results = hybrid_search(args["query"], self.store, self.config, top_k=15)
        parts = []
        token_count = 0
        for r in results:
            est = int(len(r.memory.content.split()) * 1.3)
            if token_count + est > max_tokens:
                break
            parts.append(r.memory.content)
            token_count += est
        context = "\n---\n".join(parts)
        return {"context": context, "tokens": token_count, "memories_used": len(parts)}

    def _count_by(self, args: dict):
        group = args["group_by"]
        if group == "layer":
            rows = self.store.conn.execute(
                "SELECT layer as key, COUNT(*) as count FROM memories WHERE forgotten=0 GROUP BY layer ORDER BY count DESC"
            ).fetchall()
        elif group == "source_type":
            rows = self.store.conn.execute(
                "SELECT source_type as key, COUNT(*) as count FROM memories WHERE forgotten=0 GROUP BY source_type ORDER BY count DESC"
            ).fetchall()
        elif group == "month":
            rows = self.store.conn.execute(
                "SELECT SUBSTR(fact_date, 1, 7) as key, COUNT(*) as count FROM memories WHERE forgotten=0 AND fact_date IS NOT NULL GROUP BY key ORDER BY key"
            ).fetchall()
        elif group == "entity":
            rows = self.store.conn.execute(
                """SELECT e.canonical_name as key, COUNT(em.memory_id) as count
                   FROM entities e JOIN entity_mentions em ON em.entity_id = e.id
                   GROUP BY e.id ORDER BY count DESC LIMIT 30"""
            ).fetchall()
        else:
            return {"error": f"Unknown group_by: {group}"}
        return {"counts": [dict(r) for r in rows]}

    def _bulk_forget(self, args: dict):
        if not args.get("confirm"):
            return {"error": "Set confirm=true to execute bulk forget"}
        conditions = ["forgotten = 0"]
        params = []
        if args.get("source_file"):
            conditions.append("source_file = ?")
            params.append(args["source_file"])
        if args.get("layer"):
            conditions.append("layer = ?")
            params.append(args["layer"])
        if args.get("older_than"):
            # parse date to timestamp
            import datetime
            dt = datetime.datetime.strptime(args["older_than"], "%Y-%m-%d")
            conditions.append("created_at < ?")
            params.append(dt.timestamp())
        if len(conditions) <= 1:
            return {"error": "Must specify at least one filter (source_file, layer, or older_than)"}
        where = " AND ".join(conditions)
        count = self.store.conn.execute(f"SELECT COUNT(*) as cnt FROM memories WHERE {where}", params).fetchone()["cnt"]
        self.store.conn.execute(f"UPDATE memories SET forgotten = 1 WHERE {where}", params)
        self.store.conn.commit()
        self.store.invalidate_embedding_cache()
        return {"status": "forgotten", "count": count}

    def _export(self, args: dict):
        fmt = args.get("format", "markdown")
        layer = args.get("layer")
        limit = args.get("limit", 100)
        if layer:
            mems = self.store.get_memories_by_layer(layer, limit=limit)
        else:
            mems = self.store.get_recent_memories(limit=limit)
        if fmt == "json":
            return {"memories": [{"id": m.id, "content": m.content, "layer": m.layer,
                                  "importance": m.importance, "fact_date": m.fact_date,
                                  "source_type": m.source_type, "created_at": m.created_at}
                                 for m in mems]}
        else:
            lines = []
            for m in mems:
                date = m.fact_date or time.strftime("%Y-%m-%d", time.localtime(m.created_at))
                lines.append(f"## [{date}] {m.layer} (imp={m.importance:.2f})\n\n{m.content}\n")
            return {"markdown": "\n---\n\n".join(lines), "count": len(mems)}

    def _health(self, args: dict):
        stats = self.store.get_stats()
        # check embedding cache
        cache_loaded = self.store._embedding_cache is not None
        cache_size = len(self.store._embedding_cache[0]) if cache_loaded else 0
        # orphaned entities (no memory links)
        orphaned = self.store.conn.execute(
            """SELECT COUNT(*) as cnt FROM entities e
               WHERE NOT EXISTS (SELECT 1 FROM entity_mentions em WHERE em.entity_id = e.id)"""
        ).fetchone()["cnt"]
        # stale working memories
        stale_working = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE layer='working' AND forgotten=0 AND created_at < ?",
            (time.time() - 1800,),
        ).fetchone()["cnt"]
        # FTS row count
        fts_count = self.store.conn.execute("SELECT COUNT(*) as cnt FROM memories_fts").fetchone()["cnt"]
        # memories without embeddings
        no_embedding = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE embedding IS NULL AND forgotten=0"
        ).fetchone()["cnt"]
        return {
            **stats,
            "embedding_cache_loaded": cache_loaded,
            "embedding_cache_size": cache_size,
            "orphaned_entities": orphaned,
            "stale_working_memories": stale_working,
            "fts_indexed": fts_count,
            "memories_without_embeddings": no_embedding,
        }

    def _tag(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        tags = set(mem.metadata.get("tags", []))
        for t in args.get("add", []):
            tags.add(t)
        for t in args.get("remove", []):
            tags.discard(t)
        mem.metadata["tags"] = sorted(tags)
        self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                               (json.dumps(mem.metadata), mem.id))
        self.store.conn.commit()
        return {"tags": sorted(tags)}

    def _link_memories(self, args: dict):
        from engram.store import Relationship
        # find entities linked to each memory
        e1_rows = self.store.conn.execute(
            "SELECT entity_id FROM entity_mentions WHERE memory_id = ? LIMIT 1",
            (args["memory_id_1"],),
        ).fetchall()
        e2_rows = self.store.conn.execute(
            "SELECT entity_id FROM entity_mentions WHERE memory_id = ? LIMIT 1",
            (args["memory_id_2"],),
        ).fetchall()
        if not e1_rows or not e2_rows:
            return {"error": "One or both memories have no linked entities"}
        rel = Relationship(
            source_entity_id=e1_rows[0]["entity_id"],
            target_entity_id=e2_rows[0]["entity_id"],
            relation_type=args.get("relation", "RELATED_TO"),
            created_at=time.time(),
            last_seen=time.time(),
        )
        self.store.save_relationship(rel)
        return {"status": "linked", "relation": args.get("relation", "RELATED_TO")}

    # --- Edit & annotate ---

    def _edit_memory(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        new_content = args["new_content"]
        # re-embed
        emb = embed_documents([new_content], self.config.embedding_model)
        emb_blob = emb[0].astype('float32').tobytes() if emb.size > 0 else None
        # update content + embedding + FTS
        self.store.conn.execute(
            "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
            (new_content, emb_blob, mem.id),
        )
        # rebuild FTS entry
        row = self.store.conn.execute("SELECT rowid FROM memories WHERE id = ?", (mem.id,)).fetchone()
        if row:
            self.store.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (row[0],))
            hqs = self.store.conn.execute(
                "SELECT query_text FROM hypothetical_queries WHERE memory_id = ?", (mem.id,)
            ).fetchall()
            hq_text = " ".join(r["query_text"] for r in hqs)
            self.store.conn.execute(
                "INSERT INTO memories_fts (rowid, content, hypothetical_queries) VALUES (?, ?, ?)",
                (row[0], new_content, hq_text),
            )
        self.store.invalidate_embedding_cache()
        self.store.invalidate_search_cache()
        self.store._emit_event("memory_edit", memory_id=mem.id, detail=f"content updated ({len(new_content)} chars)")
        self.store.conn.commit()
        self._refresh_session_handoff()
        return {"status": "edited", "id": mem.id}

    def _annotate(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        annotations = mem.metadata.get("annotations", [])
        annotations.append({"note": args["note"], "timestamp": time.time()})
        mem.metadata["annotations"] = annotations
        self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                               (json.dumps(mem.metadata), mem.id))
        self.store.conn.commit()
        return {"status": "annotated", "total_annotations": len(annotations)}

    def _pin(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        mem.metadata["pinned"] = True
        self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                               (json.dumps(mem.metadata), mem.id))
        self.store.conn.commit()
        return {"status": "pinned"}

    def _unpin(self, args: dict):
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}
        mem.metadata.pop("pinned", None)
        self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                               (json.dumps(mem.metadata), mem.id))
        self.store.conn.commit()
        return {"status": "unpinned"}

    # --- Entity tools ---

    def _search_entities(self, args: dict):
        query = args["query"].lower()
        rows = self.store.conn.execute(
            """SELECT e.id, e.canonical_name, e.entity_type, e.aliases,
                      COUNT(em.memory_id) as mem_count
               FROM entities e
               LEFT JOIN entity_mentions em ON em.entity_id = e.id
               WHERE LOWER(e.canonical_name) LIKE ?
                  OR LOWER(e.aliases) LIKE ?
               GROUP BY e.id
               ORDER BY mem_count DESC
               LIMIT ?""",
            (f"%{query}%", f"%{query}%", args.get("limit", 20)),
        ).fetchall()
        return {"entities": [dict(r) for r in rows]}

    def _entity_timeline(self, args: dict):
        entity = self.store.find_entity_by_name(args["name"])
        if not entity:
            return {"error": f"Entity '{args['name']}' not found"}
        memories = self.store.get_entity_memories(entity.id, limit=100)
        # sort by fact_date first, then created_at
        def sort_key(m):
            if m.fact_date:
                return m.fact_date
            return time.strftime("%Y-%m-%d", time.localtime(m.created_at))
        memories.sort(key=sort_key)
        return {"entity": entity.canonical_name, "timeline": [
            {"date": m.fact_date or time.strftime("%Y-%m-%d", time.localtime(m.created_at)),
             "content": m.content[:200], "layer": m.layer, "importance": m.importance}
            for m in memories
        ]}

    # --- Explain ---

    def _explain_importance(self, args: dict):
        import math
        mem = self.store.get_memory(args["memory_id"])
        if not mem:
            return {"error": "Memory not found"}

        age_days = (time.time() - mem.last_accessed) / 86400
        recency = math.exp(-0.693 * age_days / 30)
        access_factor = min(1.0, 0.1 * math.log(1 + mem.access_count))
        emotion = abs(mem.emotional_valence) * 0.3
        if mem.access_count > 0:
            span = max(1, (mem.last_accessed - mem.created_at) / 86400)
            stability = min(1.0, mem.access_count / (span + 1))
        else:
            stability = 0.0
        layer_boost = {"working": 0.0, "episodic": 0.1, "semantic": 0.3, "procedural": 0.2, "codebase": 0.15}.get(mem.layer, 0.0)

        factors = {
            "base_importance": {"value": round(mem.importance, 3), "weight": 0.30},
            "access_frequency": {"value": round(access_factor, 3), "weight": 0.15, "raw_count": mem.access_count},
            "recency": {"value": round(recency, 3), "weight": 0.15, "age_days": round(age_days, 1)},
            "emotional_valence": {"value": round(emotion, 3), "weight": 0.10, "raw_valence": mem.emotional_valence},
            "stability": {"value": round(stability, 3), "weight": 0.10},
            "layer_boost": {"value": round(layer_boost, 3), "weight": 0.20, "layer": mem.layer},
        }
        # surprise factor (stored at write time)
        surprise_val = mem.metadata.get("surprise", 0.5)
        factors["surprise"] = {"value": round(surprise_val, 3), "weight": 0.0,
                               "note": "recorded at write time, used for initial importance adjustment"}

        composite = sum(f["value"] * f["weight"] for f in factors.values())
        return {"memory_id": mem.id, "composite_score": round(min(1.0, max(0.0, composite)), 4), "factors": factors}

    # --- Memory map ---

    def _memory_map(self, args: dict):
        stats = self.store.get_stats()

        # top entities per layer
        layers_detail = {}
        for layer in ["working", "episodic", "semantic", "procedural", "codebase"]:
            top = self.store.conn.execute(
                """SELECT e.canonical_name, COUNT(em.memory_id) as cnt
                   FROM entity_mentions em
                   JOIN memories m ON m.id = em.memory_id
                   JOIN entities e ON e.id = em.entity_id
                   WHERE m.layer = ? AND m.forgotten = 0
                   GROUP BY e.id ORDER BY cnt DESC LIMIT 5""",
                (layer,),
            ).fetchall()
            layers_detail[layer] = {
                "count": stats["memories"].get(layer, 0),
                "top_entities": [{"name": r["canonical_name"], "count": r["cnt"]} for r in top],
            }

        # oldest and newest
        oldest = self.store.conn.execute(
            "SELECT fact_date, content FROM memories WHERE forgotten=0 AND fact_date IS NOT NULL ORDER BY fact_date ASC LIMIT 1"
        ).fetchone()
        newest = self.store.conn.execute(
            "SELECT fact_date, content FROM memories WHERE forgotten=0 AND fact_date IS NOT NULL ORDER BY fact_date DESC LIMIT 1"
        ).fetchone()

        # recent activity
        recent_writes = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type LIKE '%write%' AND created_at > ?",
            (time.time() - 3600,),
        ).fetchone()["cnt"]
        recent_reads = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE event_type LIKE '%read%' OR event_type = 'recall' AND created_at > ?",
            (time.time() - 3600,),
        ).fetchone()["cnt"]

        return {
            **stats,
            "layers": layers_detail,
            "date_range": {
                "oldest": dict(oldest) if oldest else None,
                "newest": dict(newest) if newest else None,
            },
            "last_hour": {"writes": recent_writes, "reads": recent_reads},
        }

    # --- Dedup & maintenance ---

    def _dedup(self, args: dict):
        from engram.dedup import auto_dedup
        return auto_dedup(self.store, threshold=args.get("threshold", 0.92),
                          max_merges=args.get("max_merges", 50))

    def _find_duplicates_tool(self, args: dict):
        from engram.dedup import find_duplicates
        dupes = find_duplicates(self.store, threshold=args.get("threshold", 0.92),
                                limit=args.get("limit", 500))
        return {"duplicates": [
            {"memory_1": {"id": m1.id, "content": m1.content[:150], "layer": m1.layer},
             "memory_2": {"id": m2.id, "content": m2.content[:150], "layer": m2.layer},
             "similarity": round(sim, 4)}
            for m1, m2, sim in dupes[:args.get("limit", 20)]
        ]}

    # --- Conversation tools ---

    def _ingest_sessions(self, args: dict):
        from engram.conversations import ingest_all_sessions
        return ingest_all_sessions(self.store, limit=args.get("limit", 20))

    # --- Session tools ---

    def _recent_session_memories(self, since_ts: float, limit: int = 40):
        rows = self.store.conn.execute(
            """SELECT * FROM memories
               WHERE forgotten = 0 AND created_at >= ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (since_ts, limit),
        ).fetchall()
        return [self.store._row_to_memory(r) for r in rows]

    def _recent_session_events(self, since_ts: float, limit: int = 80):
        rows = self.store.conn.execute(
            "SELECT * FROM events WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (since_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def _build_session_handoff(self, session_id: str | None = None, limit: int = 8):
        sid = session_id or self._session_id
        since_ts = self._session_started_at if sid == self._session_id else 0.0
        diary_rows = self.store.get_diary(limit=50, session_id=sid)
        recent_events = self._recent_session_events(since_ts, limit=80)
        recent_memories = self._recent_session_memories(since_ts, limit=60)

        decisions = []
        errors = []
        project_updates = []
        facts = []
        open_loops = []
        touched_entities = set()

        for mem in recent_memories:
            content = mem.content.strip()
            preview = content.split("\n")[0][:180]
            if content.lower().startswith("decision:"):
                decisions.append({"id": mem.id, "content": preview, "importance": round(mem.importance, 3)})
            elif content.lower().startswith("error:"):
                errors.append({"id": mem.id, "content": preview, "importance": round(mem.importance, 3)})
            elif mem.memory_type == MemoryType.FACT:
                facts.append({"id": mem.id, "content": preview, "importance": round(mem.importance, 3)})
            else:
                project_updates.append({"id": mem.id, "content": preview, "importance": round(mem.importance, 3)})

            entity_rows = self.store.conn.execute(
                """SELECT e.canonical_name
                   FROM entity_mentions em
                   JOIN entities e ON e.id = em.entity_id
                   WHERE em.memory_id = ?
                   LIMIT 5""",
                (mem.id,),
            ).fetchall()
            for row in entity_rows:
                touched_entities.add(row["canonical_name"])

        for row in diary_rows:
            text = row["text"]
            lower = text.lower()
            if any(token in lower for token in ["todo", "next", "block", "blocked", "unresolved", "follow up", "follow-up", "remaining"]):
                open_loops.append(text[:220])

        recall_queries = []
        for event in recent_events:
            if event.get("event_type") == "recall" and event.get("detail"):
                recall_queries.append(event["detail"])
        recall_queries = list(dict.fromkeys(recall_queries))

        summary_lines = [
            f"Session {sid}",
            f"Diary entries: {len(diary_rows)}",
            f"Recent searches: {len(recall_queries)}",
            f"Memories written: {len([e for e in recent_events if 'write' in (e.get('event_type') or '') or e.get('event_type') == 'memory_edit'])}",
        ]
        if decisions:
            summary_lines.append("Decisions: " + " | ".join(item["content"] for item in decisions[:3]))
        if open_loops:
            summary_lines.append("Open loops: " + " | ".join(open_loops[:3]))
        elif project_updates:
            summary_lines.append("Recent work: " + " | ".join(item["content"] for item in project_updates[:3]))

        handoff = {
            "session_id": sid,
            "generated_at": time.time(),
            "summary": "\n".join(summary_lines),
            "current_state": [row["text"] for row in diary_rows[:limit]],
            "decisions": decisions[:limit],
            "errors": errors[:limit],
            "facts": facts[:limit],
            "recent_work": project_updates[:limit],
            "open_loops": open_loops[:limit],
            "recent_queries": recall_queries[:limit],
            "touched_entities": sorted(touched_entities)[:limit],
            "suggested_queries": _suggest_resume_queries(open_loops, touched_entities, recall_queries, limit),
            "stats": {
                "diary_entries": len(diary_rows),
                "recent_queries": len(recall_queries),
                "memories_written": len([e for e in recent_events if "write" in (e.get("event_type") or "") or e.get("event_type") == "memory_edit"]),
                "recent_events": len(recent_events),
            },
        }
        return handoff

    def _refresh_session_handoff(self):
        handoff = self._build_session_handoff(self._session_id)
        self.store.save_session_handoff(self._session_id, handoff["summary"], handoff)
        return handoff

    def _session_summary(self, args: dict):
        """Generate a summary of the current session from diary + recent events."""
        handoff = self._build_session_handoff(self._session_id)
        self.store.save_session_handoff(self._session_id, handoff["summary"], handoff)
        return {
            "summary": handoff["summary"],
            "diary_entries": handoff["stats"]["diary_entries"],
            "searches": handoff["stats"]["recent_queries"],
            "writes": handoff["stats"]["memories_written"],
            "open_loops": handoff["open_loops"],
            "decisions": handoff["decisions"],
        }

    def _session_handoff(self, args: dict):
        session_id = args.get("session_id") or self._session_id
        handoff = self._build_session_handoff(session_id, limit=args.get("limit", 8))
        if args.get("save", True):
            self.store.save_session_handoff(session_id, handoff["summary"], handoff)
        return handoff

    def _session_checkpoint(self, args: dict):
        note = (args.get("note") or "").strip()
        if note:
            entry = f"[checkpoint] {note}"
            _session_diary.append(entry)
            self.store.write_diary(entry, session_id=self._session_id)
        handoff = self._build_session_handoff(self._session_id, limit=args.get("limit", 8))
        handoff["checkpoint_note"] = note or None
        self.store.save_session_handoff(self._session_id, handoff["summary"], handoff)
        return handoff

    def _resume_context(self, args: dict):
        session_id = args.get("session_id")
        limit = args.get("limit", 3)
        if session_id:
            handoff = self.store.get_session_handoff(session_id)
            if not handoff:
                generated = self._build_session_handoff(session_id, limit=8)
                return {"handoffs": [generated], "latest": generated, "generated": True}
            return {"handoffs": [handoff], "latest": handoff}

        handoffs = self.store.list_session_handoffs(limit=limit)
        latest = handoffs[0] if handoffs else None
        if not latest:
            latest = self._build_session_handoff(self._session_id, limit=8)
            handoffs = [latest]
        return {
            "latest": latest,
            "handoffs": handoffs,
            "note": "Use the latest handoff summary, open loops, and decisions to resume work quickly."
        }

    # --- Backlinks ---

    def _backlinks(self, args: dict):
        """Find memories that share entities with the given memory."""
        mem_id = args["memory_id"]
        # get entities linked to this memory
        entity_rows = self.store.conn.execute(
            "SELECT entity_id FROM entity_mentions WHERE memory_id = ?", (mem_id,)
        ).fetchall()
        if not entity_rows:
            return {"backlinks": [], "note": "Memory has no linked entities"}

        entity_ids = [r["entity_id"] for r in entity_rows]
        placeholders = ",".join("?" * len(entity_ids))

        # find other memories sharing those entities
        rows = self.store.conn.execute(
            f"""SELECT DISTINCT m.id, m.content, m.layer, m.importance,
                       e.canonical_name as via_entity
                FROM entity_mentions em
                JOIN memories m ON m.id = em.memory_id
                JOIN entities e ON e.id = em.entity_id
                WHERE em.entity_id IN ({placeholders})
                  AND em.memory_id != ?
                  AND m.forgotten = 0
                ORDER BY m.importance DESC
                LIMIT 20""",
            entity_ids + [mem_id],
        ).fetchall()
        return {"backlinks": [dict(r) for r in rows]}

    # --- Batch operations ---

    def _batch_tag(self, args: dict):
        results = hybrid_search(args["query"], self.store, self.config,
                                top_k=args.get("top_k", 10), rerank=False)
        tagged = 0
        for r in results:
            mem = r.memory
            tags = set(mem.metadata.get("tags", []))
            tags.update(args["tags"])
            mem.metadata["tags"] = sorted(tags)
            self.store.conn.execute("UPDATE memories SET metadata = ? WHERE id = ?",
                                   (json.dumps(mem.metadata), mem.id))
            tagged += 1
        self.store.conn.commit()
        return {"tagged": tagged, "tags_added": args["tags"]}

    def _recompute_importance(self, args: dict):
        rows = self.store.conn.execute(
            "SELECT * FROM memories WHERE forgotten = 0"
        ).fetchall()
        updated = 0
        for row in rows:
            mem = self.store._row_to_memory(row)
            new_imp = compute_importance(mem)
            if abs(new_imp - mem.importance) > 0.01:
                self.store.update_importance(mem.id, new_imp)
                updated += 1
        return {"total": len(rows), "updated": updated}

    # --- Codebase tools ---

    def _scan_codebase(self, args: dict):
        from engram.codebase import scan_codebase
        return scan_codebase(args["path"], self.store, args.get("project_name"))

    def _recall_code(self, args: dict):
        query = args["query"]
        project = args.get("project")
        top_k = args.get("top_k", 10)

        # search within codebase layer only
        results = hybrid_search(query, self.store, self.config, top_k=top_k * 2, rerank=False)
        filtered = []
        for r in results:
            if r.memory.layer != MemoryLayer.CODEBASE:
                continue
            if project and r.memory.metadata.get("project") != project:
                continue
            filtered.append({"id": r.memory.id, "content": r.memory.content,
                             "score": round(r.score, 4),
                             "project": r.memory.metadata.get("project"),
                             "type": r.memory.metadata.get("type"),
                             "file": r.memory.metadata.get("file")})
            if len(filtered) >= top_k:
                break
        return {"results": filtered}

    def _list_projects(self, args: dict):
        rows = self.store.conn.execute(
            """SELECT json_extract(metadata, '$.project') as project,
                      COUNT(*) as memory_count,
                      json_extract(metadata, '$.type') as types
               FROM memories
               WHERE layer = 'codebase' AND forgotten = 0
                 AND json_extract(metadata, '$.project') IS NOT NULL
               GROUP BY project"""
        ).fetchall()
        projects = {}
        for r in rows:
            p = r["project"]
            if p not in projects:
                projects[p] = {"name": p, "memories": 0}
            projects[p]["memories"] += r["memory_count"]

        # get type breakdown per project
        for p in projects:
            type_rows = self.store.conn.execute(
                """SELECT json_extract(metadata, '$.type') as type, COUNT(*) as cnt
                   FROM memories WHERE layer='codebase' AND forgotten=0
                   AND json_extract(metadata, '$.project') = ?
                   GROUP BY type""",
                (p,),
            ).fetchall()
            projects[p]["types"] = {r["type"]: r["cnt"] for r in type_rows}

        return {"projects": list(projects.values())}

    # --- Skill selection ---

    def _get_skills(self, args: dict):
        query = args["query"]
        max_skills = args.get("max_skills", 3)
        should_format = args.get("format", True)

        selection = select_skills(query, self.store, self.config, max_skills=max_skills)

        if should_format:
            formatted = format_skills(selection)
            return {
                "should_inject": selection.should_inject,
                "confidence": round(selection.confidence, 3),
                "task_novelty": round(selection.task_novelty, 3),
                "domain_coverage": round(selection.domain_coverage, 3),
                "reason": selection.reason,
                "skill_count": len(selection.skills),
                "context": formatted if formatted else None,
            }
        else:
            return {
                "should_inject": selection.should_inject,
                "confidence": round(selection.confidence, 3),
                "task_novelty": round(selection.task_novelty, 3),
                "domain_coverage": round(selection.domain_coverage, 3),
                "reason": selection.reason,
                "skills": [
                    {"id": m.id, "content": m.content, "layer": m.layer,
                     "importance": m.importance}
                    for m in selection.skills
                ],
            }

    # --- Deep retrieval ---

    def _train_reranker(self, args: dict):
        result = self._reranker.train(
            self.store,
            lr=args.get("learning_rate", 0.01),
            epochs=args.get("epochs", 50),
        )
        return result

    def _reranker_status(self, args: dict):
        return {
            "trained": self._reranker.is_trained,
            "model_path": str(self._reranker.model_path) if self._reranker.model_path else None,
            "model_exists": self._reranker.model_path.exists() if self._reranker.model_path else False,
        }

    # --- Cognitive scaffolding ---

    def _recall_hints(self, args: dict):
        """Return memory hints — truncated snippets + entities to trigger recognition.

        Inspired by "Your Brain on ChatGPT" (Kosmyna et al.): full context
        replacement weakens cognition. Hints trigger recognition without
        replacing the recall process.
        """
        self._sweep_working()
        results = hybrid_search(args["query"], self.store, self.config,
                                top_k=args.get("top_k", 10),
                                deep_reranker=self._reranker)

        hint_length = args.get("hint_length", 60)
        hints = []

        for r in results:
            mem = r.memory
            content = mem.content.strip()

            # extract first meaningful line as title
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            title = lines[0][:hint_length] if lines else content[:hint_length]

            # truncate with ellipsis if needed
            if len(title) < len(lines[0] if lines else content):
                title += "..."

            # get linked entities
            entity_rows = self.store.conn.execute(
                """SELECT e.canonical_name, e.entity_type
                FROM entity_mentions em
                JOIN entities e ON e.id = em.entity_id
                WHERE em.memory_id = ?
                LIMIT 5""",
                (mem.id,),
            ).fetchall()
            entities = [{"name": r["canonical_name"], "type": r["entity_type"]}
                        for r in entity_rows]

            hints.append({
                "id": mem.id,
                "hint": title,
                "layer": mem.layer,
                "importance": round(mem.importance, 2),
                "date": mem.fact_date,
                "entities": entities,
                "score": round(r.score, 4),
            })

        return {
            "query": args["query"],
            "hints": hints,
            "note": "Use recall with memory IDs to get full content if needed.",
        }

    # --- Drift detection ---

    def _drift_check(self, args: dict):
        report = run_drift_check(
            self.store,
            search_roots=args.get("search_roots"),
            project_root=args.get("project_root"),
            layers=args.get("layers"),
            check_functions=args.get("check_functions", True),
        )
        result = report.to_dict()
        # add summary for quick reading
        error_count = sum(1 for i in report.issues if i.severity == "error")
        warn_count = sum(1 for i in report.issues if i.severity == "warning")
        info_count = sum(1 for i in report.issues if i.severity == "info")
        result["summary"] = (
            f"Drift score: {report.score}/100 | "
            f"{error_count} errors, {warn_count} warnings, {info_count} info | "
            f"{report.claims_valid}/{report.claims_verified} claims valid | "
            f"{report.stale_memories} stale memories"
        )
        return result

    def _drift_fix(self, args: dict):
        report = run_drift_check(
            self.store,
            search_roots=args.get("search_roots"),
            project_root=args.get("project_root"),
        )
        dry_run = args.get("dry_run", True)
        return auto_fix_drift(self.store, report, dry_run=dry_run)

    # --- Pattern extraction ---

    def _extract_patterns(self, args: dict):
        patterns = extract_patterns_from_session(
            self.store,
            self.config,
            hours=args.get("hours", 4.0),
            novelty_threshold=args.get("novelty_threshold", 0.25),
        )

        if args.get("dry_run", False):
            return {
                "patterns": [
                    {"title": p.title, "category": p.category,
                     "novelty": p.novelty, "should_store": p.should_store,
                     "source_events": p.source_events,
                     "content_preview": p.content[:200]}
                    for p in patterns
                ],
                "total": len(patterns),
                "would_store": sum(1 for p in patterns if p.should_store),
                "dry_run": True,
            }

        result = store_patterns(patterns, self.store, self.config)
        return result

    # --- Quality metrics (AgeMem-inspired) ---

    def _quality_metrics(self, args: dict):
        """Compute memory system quality metrics from access patterns."""
        # storage quality: what fraction of stored memories ever get recalled?
        total = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0"
        ).fetchone()["cnt"]
        accessed = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0 AND access_count > 0"
        ).fetchone()["cnt"]
        storage_quality = round(accessed / max(1, total), 3)

        # curation ratio: how many memories have been actively curated?
        curated = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0 AND "
            "(json_extract(metadata, '$.invalidated') = 1 OR "
            " json_extract(metadata, '$.evolution_count') > 0 OR "
            " json_extract(metadata, '$.confirmations') > 0)"
        ).fetchone()["cnt"]
        curation_ratio = round(curated / max(1, total), 3)

        # retrieval relevance: average access count of recently accessed memories
        recent_accessed = self.store.conn.execute(
            "SELECT AVG(access_count) as avg_access FROM memories WHERE forgotten=0 AND access_count > 0"
        ).fetchone()["avg_access"] or 0

        # enrichment coverage: how many memories have enriched metadata?
        enriched = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0 AND "
            "json_extract(metadata, '$.keywords') IS NOT NULL"
        ).fetchone()["cnt"]
        enrichment_ratio = round(enriched / max(1, total), 3)

        # evolved memories count
        evolved = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0 AND "
            "json_extract(metadata, '$.evolution_count') > 0"
        ).fetchone()["cnt"]

        # confirmed memories count
        confirmed = self.store.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE forgotten=0 AND "
            "json_extract(metadata, '$.confirmations') > 0"
        ).fetchone()["cnt"]

        return {
            "storage_quality": storage_quality,
            "curation_ratio": curation_ratio,
            "avg_access_count": round(recent_accessed, 2),
            "enrichment_ratio": enrichment_ratio,
            "total_memories": total,
            "accessed_memories": accessed,
            "curated_memories": curated,
            "enriched_memories": enriched,
            "evolved_memories": evolved,
            "confirmed_memories": confirmed,
        }

    # --- Embedding compression ---

    def _compress_embeddings(self, args: dict):
        from engram.quantize import compress_old_embeddings
        return compress_old_embeddings(self.store, self.config,
                                        dry_run=args.get("dry_run", True))

    # --- Community detection ---

    def _detect_communities(self, args: dict):
        from engram.communities import detect_communities, generate_community_summaries
        result = detect_communities(self.store, min_community_size=args.get("min_size", 3))
        if args.get("generate_summaries") and result.get("communities", 0) > 0:
            summaries = generate_community_summaries(self.store, self.config)
            result["summaries_generated"] = summaries
        return result

    # --- Negative knowledge ---

    def _remember_negative(self, args: dict):
        parts = [f"NEGATIVE KNOWLEDGE: {args['content']}"]
        if args.get("context"):
            parts.append(f"Why this matters: {args['context']}")
        if args.get("scope"):
            parts.append(f"Scope: {args['scope']}")
        content = "\n".join(parts)

        result = self._remember({
            "content": content,
            "source_type": SourceType.HUMAN,
            "layer": MemoryLayer.SEMANTIC,
            "memory_type": MemoryType.FACT,
            "importance": args.get("importance", 0.75),
        })
        result["type"] = "negative_knowledge"
        return result

    # --- Status & type tools (v0.3.0) ---

    def _update_status(self, args: dict):
        self.store.update_status(args["memory_id"], args["new_status"], args.get("reason"))
        return {"memory_id": args["memory_id"], "new_status": args["new_status"], "reason": args.get("reason")}

    def _recall_by_type(self, args: dict):
        mems = self.store.get_memories_by_type(args["memory_type"], args.get("limit", 20))
        return [{"id": m.id, "content": m.content, "memory_type": m.memory_type,
                 "status": m.status, "layer": m.layer, "importance": m.importance,
                 "fact_date": m.fact_date} for m in mems]

    def _status_history(self, args: dict):
        history = self.store.get_status_history(args["memory_id"])
        mem = self.store.get_memory(args["memory_id"])
        result = {"memory_id": args["memory_id"], "transitions": history}
        if mem:
            result["current_status"] = mem.status
            result["content_preview"] = mem.content[:100]
        return result

    def _response(self, req_id, result):
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _error(self, req_id, message):
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": message}}


def run_mcp(config: Config):
    import threading

    server = MCPServer(config)

    # warm up models in background so first recall is fast
    def _warmup():
        try:
            from engram.embeddings import warmup, set_backend, get_backend
            # set backend from config (auto / mlx / sentence_transformers)
            if config.embedding_backend != "auto":
                set_backend(config.embedding_backend)
            warmup(config.embedding_model, config.cross_encoder_model)
            server.store.get_all_embeddings()
            backend = get_backend()
            sys.stderr.write(f"engram: models warmed up (backend={backend})\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"engram: warmup error: {e}\n")
            sys.stderr.flush()

    threading.Thread(target=_warmup, daemon=True).start()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = server.handle_request(request)
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"engram: error: {e}\n")
            sys.stderr.flush()
            error = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(error) + "\n")
            sys.stdout.flush()


def run_mcp_sse(config: Config, port: int = 8421):
    """Run MCP server over HTTP with SSE transport.

    Endpoints:
        POST /mcp  — JSON-RPC request, returns JSON-RPC response
        GET  /sse  — SSE stream for server-initiated messages (notifications)
        GET  /health — health check

    Use this when you can't use stdio (e.g. remote agents, browser-based clients).
    """
    import asyncio
    import threading
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from sse_starlette.sse import EventSourceResponse
    import uvicorn

    server = MCPServer(config)

    # warmup
    def _warmup():
        try:
            from engram.embeddings import warmup as _warmup_models
            if config.embedding_backend != "auto":
                from engram.embeddings import set_backend
                set_backend(config.embedding_backend)
            _warmup_models(config.embedding_model, config.cross_encoder_model)
        except Exception:
            pass

    threading.Thread(target=_warmup, daemon=True).start()

    app = FastAPI(title="Engram MCP (SSE)", version="0.4.0")

    # SSE subscribers
    _sse_queues: list[asyncio.Queue] = []

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        body = await request.json()
        response = server.handle_request(body)
        if response:
            # broadcast to SSE subscribers
            for q in _sse_queues:
                try:
                    q.put_nowait(response)
                except asyncio.QueueFull:
                    pass
            return JSONResponse(content=response)
        return JSONResponse(content={"jsonrpc": "2.0", "result": "ok"})

    @app.get("/sse")
    async def sse_endpoint():
        queue = asyncio.Queue(maxsize=100)
        _sse_queues.append(queue)

        async def event_generator():
            try:
                while True:
                    msg = await queue.get()
                    yield {"event": "message", "data": json.dumps(msg)}
            except asyncio.CancelledError:
                pass
            finally:
                _sse_queues.remove(queue)

        return EventSourceResponse(event_generator())

    @app.get("/health")
    async def health():
        stats = server.store.get_stats()
        ann_ready = server.store.ann_index.ready if server.store.ann_index else False
        return {
            "status": "ok",
            "memories": stats["memories"]["total"],
            "entities": stats["entities"],
            "ann_ready": ann_ready,
        }

    print(f"Starting Engram MCP (SSE) on http://127.0.0.1:{port}")
    print(f"  POST /mcp   — JSON-RPC endpoint")
    print(f"  GET  /sse   — SSE stream")
    print(f"  GET  /health — health check")
    uvicorn.run(app, host="127.0.0.1", port=port)
