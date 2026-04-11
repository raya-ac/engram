"""FastAPI web UI with SSE event stream."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from engram.config import Config
from engram.store import Store


def create_app(config: Config | None = None) -> FastAPI:
    if config is None:
        config = Config.load()

    app = FastAPI(title="Engram", version="0.1.0")

    # set embedding backend + default model from config
    from engram.embeddings import set_backend, set_default_model
    if config.embedding_backend and config.embedding_backend != "auto":
        set_backend(config.embedding_backend)
    set_default_model(config.embedding_model)

    store = Store(config)
    store.init_db()

    app.state.store = store
    app.state.config = config

    template_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    app.state.templates = templates

    from engram.web.routes import router
    app.include_router(router)

    # warm up models + ANN index in background thread so first query is fast
    def _warmup():
        from engram.embeddings import warmup
        warmup(config.embedding_model, config.cross_encoder_model)
        # init ANN index (loads from disk or rebuilds)
        store.init_ann_index(background=False)
        # prime embedding cache as fallback
        if not (store.ann_index and store.ann_index.ready):
            store.get_all_embeddings()

    threading.Thread(target=_warmup, daemon=True).start()

    return app
