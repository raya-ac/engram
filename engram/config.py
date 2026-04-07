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


@dataclass
class LLMConfig:
    backend: str = "claude_cli"
    model: str = "claude-sonnet-4-20250514"
    mlx_model: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8420


@dataclass
class Config:
    db_path: str = "~/.local/share/engram/memory.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @property
    def resolved_db_path(self) -> Path:
        p = Path(os.path.expanduser(self.db_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

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
        for k in ("db_path", "embedding_model", "cross_encoder_model", "embedding_dim"):
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

        return cfg
