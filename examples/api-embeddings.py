"""Using API embedding backends (Voyage, OpenAI, Gemini).

Shows how to switch engram's embedding backend from local models
to cloud APIs. Useful when you want higher quality embeddings
or don't have a GPU.

Requirements:
    pip install -e ".[api]"    # or just .[voyage], .[openai], .[gemini]

    export VOYAGE_API_KEY="your-key"     # https://dash.voyageai.com/
    export OPENAI_API_KEY="your-key"
    export GEMINI_API_KEY="your-key"
"""

import os
import sys

from engram.embeddings import (
    embed_query, embed_documents,
    get_backend, get_model_dim, set_default_model,
)


def demo_backend(model_name: str):
    """Show how a specific model works."""
    backend = get_backend(model_name)
    dim = get_model_dim(model_name)
    print(f"\n--- {model_name} ---")
    print(f"  backend: {backend}")
    print(f"  dim: {dim}")

    try:
        vec = embed_query("test query about memory systems", model_name)
        print(f"  embed_query: shape={vec.shape}, norm={float(vec @ vec)**0.5:.4f}")

        vecs = embed_documents(["doc one", "doc two", "doc three"], model_name)
        print(f"  embed_documents: shape={vecs.shape}")

        # similarity check
        q = embed_query("memory retrieval", model_name)
        sims = vecs @ q
        print(f"  similarities: {[f'{s:.3f}' for s in sims]}")

    except (ValueError, ImportError) as e:
        print(f"  skipped: {e}")


def main():
    print("engram multi-backend embedding demo")
    print("=" * 50)

    # local model (always works)
    demo_backend("BAAI/bge-small-en-v1.5")

    # API models (need keys)
    if os.environ.get("VOYAGE_API_KEY"):
        demo_backend("voyage-3.5")
        demo_backend("voyage-3.5-lite")
    else:
        print("\n--- voyage-3.5 ---")
        print("  skipped: set VOYAGE_API_KEY to test")

    if os.environ.get("OPENAI_API_KEY"):
        demo_backend("text-embedding-3-small")
    else:
        print("\n--- text-embedding-3-small ---")
        print("  skipped: set OPENAI_API_KEY to test")

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        demo_backend("gemini-embedding-001")
    else:
        print("\n--- gemini-embedding-001 ---")
        print("  skipped: set GEMINI_API_KEY to test")

    # show how to set default model for the whole session
    print("\n--- setting default model ---")
    print(f"  before: default uses local backend")
    set_default_model("voyage-3.5")
    print(f"  after set_default_model('voyage-3.5'): all embed_query()/embed_documents()")
    print(f"  calls without explicit model_name will use voyage-3.5")

    # reset
    set_default_model("BAAI/bge-small-en-v1.5")


if __name__ == "__main__":
    main()
