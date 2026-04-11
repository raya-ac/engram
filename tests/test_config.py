"""Tests for config.py."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from engram.config import Config, ANNConfig


class TestConfigDefaults:
    def test_defaults(self):
        cfg = Config()
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding_dim == 384
        assert cfg.embedding_backend == "auto"
        assert cfg.retrieval.top_k == 10
        assert cfg.ann.enabled is True
        assert cfg.ann.m == 32

    def test_ann_config(self):
        cfg = Config()
        assert cfg.ann.ef_construction == 200
        assert cfg.ann.ef_search == 100
        assert cfg.ann.max_elements == 500_000

    def test_web_config(self):
        cfg = Config()
        assert cfg.web.host == "127.0.0.1"
        assert cfg.web.port == 8420
        assert cfg.web.auth_token == ""


class TestConfigFile:
    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "embedding_model": "voyage-3.5",
                "embedding_backend": "voyage",
                "retrieval": {"top_k": 20},
                "ann": {"m": 64},
                "web": {"port": 9000, "auth_token": "secret123"},
            }, f)
            f.flush()

            cfg = Config.load(f.name)
            assert cfg.embedding_model == "voyage-3.5"
            assert cfg.embedding_backend == "voyage"
            assert cfg.embedding_dim == 1024  # auto-detected from model
            assert cfg.retrieval.top_k == 20
            assert cfg.ann.m == 64
            assert cfg.web.port == 9000
            assert cfg.web.auth_token == "secret123"

            os.unlink(f.name)


class TestAutoDim:
    def test_auto_dim_voyage(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"embedding_model": "voyage-3.5"}, f)
            f.flush()
            cfg = Config.load(f.name)
            assert cfg.embedding_dim == 1024
            os.unlink(f.name)

    def test_auto_dim_openai(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"embedding_model": "text-embedding-3-small"}, f)
            f.flush()
            cfg = Config.load(f.name)
            assert cfg.embedding_dim == 1536
            os.unlink(f.name)

    def test_explicit_dim_not_overridden(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"embedding_model": "voyage-3.5", "embedding_dim": 512}, f)
            f.flush()
            cfg = Config.load(f.name)
            assert cfg.embedding_dim == 512  # explicit wins
            os.unlink(f.name)


class TestEnvOverride:
    def test_env_overrides_config(self):
        os.environ["ENGRAM_EMBEDDING_DIM"] = "768"
        try:
            cfg = Config.load()
            assert cfg.embedding_dim == 768
        finally:
            del os.environ["ENGRAM_EMBEDDING_DIM"]
