"""MCP tool server — JSON-RPC over stdio with 24 tools."""

from __future__ import annotations

import json
import sys
import time
import uuid
from typing import Any

from engram.config import Config
from engram.store import Store, Memory, MemoryLayer, SourceType
from engram.embeddings import embed_documents
from engram.retrieval import search as hybrid_search
from engram.entities import process_entities_for_memory
from engram.extractor import generate_hypothetical_queries
from engram.layers import get_context_layers, format_context
from engram.compress import compress_memories
from engram.lifecycle import compute_importance
from engram.consolidator import consolidate

TOOLS = [
    # Read tools
    {"name": "recall", "description": "Search memories using hybrid retrieval (dense + BM25 + graph + cross-encoder)", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 10}}, "required": ["query"]}},
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
    {"name": "remember", "description": "Store a memory", "inputSchema": {"type": "object", "properties": {"content": {"type": "string"}, "source_type": {"type": "string", "default": "remember:human"}, "layer": {"type": "string", "default": "episodic"}, "importance": {"type": "number", "default": 0.7}}, "required": ["content"]}},
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
]

_session_diary: list[str] = []


class MCPServer:
    def __init__(self, config: Config):
        self.config = config
        self.store = Store(config)
        self.store.init_db()

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
        }
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")
        return handler(args)

    # --- Read tools ---

    def _recall(self, args: dict):
        self._sweep_working()
        results = hybrid_search(args["query"], self.store, self.config, top_k=args.get("top_k", 10))
        return [{"id": r.memory.id, "content": r.memory.content, "score": round(r.score, 4),
                 "layer": r.memory.layer, "fact_date": r.memory.fact_date,
                 "importance": r.memory.importance} for r in results]

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
        mem = Memory(
            id=str(uuid.uuid4()), content=args["content"],
            source_type=args.get("source_type", SourceType.HUMAN),
            layer=args.get("layer", MemoryLayer.EPISODIC),
            importance=args.get("importance", 0.7),
        )
        emb = embed_documents([mem.content], self.config.embedding_model)
        if emb.size > 0:
            mem.embedding = emb[0]
        hqs = []
        try:
            hqs = generate_hypothetical_queries(mem.content, self.config)
        except Exception:
            pass
        self.store.save_memory(mem, hypothetical_queries=hqs)
        process_entities_for_memory(self.store, mem.id, mem.content)
        return {"id": mem.id, "status": "stored"}

    def _remember_interaction(self, args: dict):
        content = f"Q: {args['question']}\nA: {args['answer']}"
        return self._remember({"content": content, "source_type": SourceType.INTERACTION,
                               "importance": args.get("importance", 0.5)})

    def _remember_decision(self, args: dict):
        content = f"Decision: {args['decision']}\nRationale: {args.get('rationale', '')}"
        return self._remember({"content": content, "source_type": SourceType.AI,
                               "layer": MemoryLayer.PROCEDURAL,
                               "importance": args.get("importance", 0.8)})

    def _remember_error(self, args: dict):
        content = f"Error: {args['error']}\nPrevention: {args.get('prevention', '')}"
        return self._remember({"content": content, "source_type": SourceType.AI,
                               "layer": MemoryLayer.PROCEDURAL,
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

    # --- Diary tools ---

    def _diary_write(self, args: dict):
        entry = f"[{time.strftime('%H:%M:%S')}] {args['entry']}"
        _session_diary.append(entry)
        return {"status": "written", "entries": len(_session_diary)}

    def _diary_read(self, args: dict):
        return {"diary": _session_diary}

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
            from engram.embeddings import warmup
            warmup(config.embedding_model, config.cross_encoder_model)
            server.store.get_all_embeddings()
            sys.stderr.write("engram: models warmed up\n")
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
