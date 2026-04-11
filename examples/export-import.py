"""Export, backup, and import engram memories.

Shows how to use the export/import API for:
- backing up your memory database
- migrating between machines
- sharing memories between engram instances
- filtering exports by layer
"""

import json
import tempfile
from pathlib import Path

from engram.config import Config
from engram.store import Store, Memory
from engram.embeddings import embed_documents

import uuid


def main():
    config = Config.load()
    store = Store(config)
    store.init_db()

    # --- seed some memories if empty ---
    if store.count_memories()["total"] == 0:
        print("seeding sample memories...")
        samples = [
            ("the API uses JWT tokens with 1h expiry", "procedural", 0.8),
            ("switched from Redis to SQLite for simplicity", "semantic", 0.9),
            ("the deploy broke because of a missing env var", "episodic", 0.6),
        ]
        for content, layer, imp in samples:
            mem = Memory(id=str(uuid.uuid4()), content=content, layer=layer, importance=imp)
            emb = embed_documents([content], config.embedding_model)
            mem.embedding = emb[0]
            store.save_memory(mem)

    stats = store.get_stats()
    print(f"current state: {stats['memories']['total']} memories")

    # --- export to JSON ---
    export_path = Path(tempfile.mkdtemp()) / "backup.json"
    print(f"\nexporting to {export_path}...")

    rows = store.conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 ORDER BY created_at"
    ).fetchall()

    import base64
    memories = []
    for row in rows:
        mem = {
            "id": row["id"],
            "content": row["content"],
            "layer": row["layer"],
            "importance": row["importance"],
            "created_at": row["created_at"],
            "chunk_hash": row["chunk_hash"],
        }
        if row["embedding"]:
            mem["embedding_b64"] = base64.b64encode(row["embedding"]).decode()
        memories.append(mem)

    export_data = {
        "version": "1.0",
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "memories": memories,
    }
    export_path.write_text(json.dumps(export_data, indent=2))
    print(f"exported {len(memories)} memories ({export_path.stat().st_size / 1024:.1f} KB)")

    # --- import into a fresh db ---
    import_config = Config()
    import_config.db_path = str(Path(tempfile.mkdtemp()) / "imported.db")
    import_store = Store(import_config)
    import_store.init_db()

    print(f"\nimporting into {import_config.db_path}...")
    data = json.loads(export_path.read_text())

    import numpy as np
    imported = 0
    for mem_data in data["memories"]:
        mem = Memory(
            id=mem_data["id"],
            content=mem_data["content"],
            layer=mem_data["layer"],
            importance=mem_data["importance"],
            created_at=mem_data["created_at"],
            chunk_hash=mem_data["chunk_hash"],
        )
        if "embedding_b64" in mem_data:
            emb_bytes = base64.b64decode(mem_data["embedding_b64"])
            mem.embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()

        import_store.save_memory(mem)
        imported += 1

    print(f"imported {imported} memories")
    print(f"new db stats: {import_store.get_stats()}")

    store.close()
    import_store.close()


if __name__ == "__main__":
    main()
