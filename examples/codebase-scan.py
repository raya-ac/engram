"""Codebase scanning — extract compressed code knowledge.

Point engram at a project directory and it extracts file trees,
function/class signatures, import graphs, and config files into
codebase-layer memories. Uses ~10x fewer tokens than raw code.
"""

import sys

from engram.config import Config
from engram.store import Store
from engram.codebase import scan_project
from engram.retrieval import search as hybrid_search


def main():
    if len(sys.argv) < 2:
        print("usage: python codebase-scan.py <project-path> [project-name]")
        print("example: python codebase-scan.py ~/projects/myapp myapp")
        sys.exit(1)

    project_path = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else None

    config = Config.load()
    store = Store(config)
    store.init_db()

    print(f"scanning {project_path}...")
    result = scan_project(project_path, store, config, project_name=project_name)

    print(f"\nscan complete:")
    print(f"  files scanned: {result.get('files_scanned', 0)}")
    print(f"  memories created: {result.get('memories_created', 0)}")
    print(f"  functions found: {result.get('functions', 0)}")
    print(f"  classes found: {result.get('classes', 0)}")

    # now search the codebase layer
    print("\n--- searching codebase memories ---")
    results = hybrid_search("main entry point", store, config, top_k=3, rerank=False)
    for r in results:
        if r.memory.layer == "codebase":
            print(f"  [{r.memory.layer}] {r.memory.content[:100]}...")

    store.close()


if __name__ == "__main__":
    main()
