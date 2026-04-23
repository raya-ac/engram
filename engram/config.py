"""Configuration with env var > config file > defaults priority."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RetrievalConfig:
    top_k: int = 10
    rrf_k: int = 60
    min_confidence: float = 0.60
    rerank_candidates: int = 20
    dense_multiplier: int = 3
    bm25_multiplier: int = 3
    enable_query_expansion: bool = True
    exact_match_boost: float = 1.22
    search_cache_size: int = 128


@dataclass
class LifecycleConfig:
    forgetting_half_life_days: int = 30
    archive_after_days: int = 90
    archive_min_importance: float = 0.3
    archive_min_accesses: int = 3
    promote_importance: float = 0.7
    promote_accesses: int = 5
    cluster_threshold: float = 0.8
    cluster_min_size: int = 5
    retention_mode: str = "huber"  # l2 | huber | elastic
    huber_delta: float = 0.5      # transition point (in half-lives) for huber mode
    elastic_l1_ratio: float = 0.3  # L1 weight for elastic mode (0=pure L2, 1=pure L1)


@dataclass
class LLMConfig:
    backend: str = "claude_cli"  # claude_cli | mlx | openai | anthropic
    model: str = "claude-sonnet-4-20250514"
    mlx_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    api_key: str = ""  # for openai/anthropic backends; defaults to env var if empty


@dataclass
class ANNConfig:
    enabled: bool = True
    m: int = 32
    ef_construction: int = 200
    ef_search: int = 100
    max_elements: int = 500_000
    index_path: str = "~/.local/share/engram/hnsw.index"

    @property
    def resolved_index_path(self) -> Path:
        return Path(os.path.expanduser(self.index_path))


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8420
    auth_token: str = ""  # set to enable bearer token auth on the web UI


@dataclass
class Config:
    storage_backend: str = "sqlite"  # sqlite | postgres
    db_path: str = "~/.local/share/engram/memory.db"
    postgres_dsn: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_backend: str = "auto"  # auto | mlx | sentence_transformers | voyage | openai | gemini
    embedding_dim: int = 384
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    web: WebConfig = field(default_factory=WebConfig)
    ann: ANNConfig = field(default_factory=ANNConfig)

    @property
    def resolved_db_path(self) -> Path:
        p = Path(os.path.expanduser(self.db_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def normalized_storage_backend(self) -> str:
        backend = (self.storage_backend or "sqlite").strip().lower()
        return backend if backend in {"sqlite", "postgres"} else "sqlite"

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        raw: dict = {}
        candidates = [
            path,
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".config" / "engram" / "config.yaml",
        ]
        for p in candidates:
            if p and Path(p).exists():
                with open(p) as f:
                    raw = yaml.safe_load(f) or {}
                break

        cfg = cls()
        for k in ("storage_backend", "db_path", "postgres_dsn", "embedding_model", "cross_encoder_model", "embedding_backend", "embedding_dim"):
            env = os.environ.get(f"ENGRAM_{k.upper()}")
            if env:
                setattr(cfg, k, type(getattr(cfg, k))(env))
            elif k in raw:
                setattr(cfg, k, raw[k])

        if "retrieval" in raw:
            for k, v in raw["retrieval"].items():
                if hasattr(cfg.retrieval, k):
                    setattr(cfg.retrieval, k, v)
        if "lifecycle" in raw:
            for k, v in raw["lifecycle"].items():
                if hasattr(cfg.lifecycle, k):
                    setattr(cfg.lifecycle, k, v)
        if "llm" in raw:
            for k, v in raw["llm"].items():
                if hasattr(cfg.llm, k):
                    setattr(cfg.llm, k, v)
        if "web" in raw:
            for k, v in raw["web"].items():
                if hasattr(cfg.web, k):
                    setattr(cfg.web, k, v)
        if "ann" in raw:
            for k, v in raw["ann"].items():
                if hasattr(cfg.ann, k):
                    setattr(cfg.ann, k, v)

        # auto-detect embedding dim from model name if dim wasn't explicitly set
        if "embedding_dim" not in raw and not os.environ.get("ENGRAM_EMBEDDING_DIM"):
            from engram.embeddings import get_model_dim
            detected = get_model_dim(cfg.embedding_model)
            if detected:
                cfg.embedding_dim = detected

        return cfg
