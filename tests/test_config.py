"""Tests for config.py."""

import os
import json
import builtins
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml

from engram.config import Config, ConfigError, DormantRecallConfig


@pytest.fixture(autouse=True)
def isolated_config_environment(monkeypatch, tmp_path):
    """Never read the developer's config or credentials in config unit tests."""
    env_names = {item["env"] for item in Config.schema()["fields"].values()}
    env_names.update({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"})
    for name in env_names:
        # Register absent names as well, since successful loads propagate HF aliases.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(Config, "_default_paths", staticmethod(lambda: [tmp_path / "default.yaml"]))


def write_config(tmp_path, values):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


class TestConfigDefaults:
    def test_defaults(self):
        cfg = Config()
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.embedding_dim == 384
        assert cfg.embedding_backend == "auto"
        assert cfg.storage_backend == "sqlite"
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
                "storage_backend": "postgres",
                "postgres_dsn": "postgresql://localhost:5432/engram",
                "embedding_model": "voyage-3.5",
                "embedding_backend": "voyage",
                "retrieval": {"top_k": 20},
                "ann": {"m": 64},
                "web": {"port": 9000, "auth_token": "secret123"},
            }, f)
            f.flush()

            cfg = Config.load(f.name)
            assert cfg.normalized_storage_backend == "postgres"
            assert cfg.postgres_dsn == "postgresql://localhost:5432/engram"
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

    def test_storage_backend_env_override(self):
        os.environ["ENGRAM_STORAGE_BACKEND"] = "postgres"
        os.environ["ENGRAM_POSTGRES_DSN"] = "postgresql://db/engram"
        try:
            cfg = Config.load()
            assert cfg.normalized_storage_backend == "postgres"
            assert cfg.postgres_dsn == "postgresql://db/engram"
        finally:
            del os.environ["ENGRAM_STORAGE_BACKEND"]
            del os.environ["ENGRAM_POSTGRES_DSN"]


class TestHfToken:
    def test_default_hf_token_empty(self):
        cfg = Config()
        assert cfg.hf_token == ""

    def test_hf_token_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"hf_token": "hf_test_token_123"}, f)
            f.flush()
            # Clean env to test purely yaml
            old_hf = os.environ.pop("HF_TOKEN", None)
            old_hub = os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            try:
                cfg = Config.load(f.name)
                assert cfg.hf_token == "hf_test_token_123"
                assert os.environ.get("HF_TOKEN") == "hf_test_token_123"
            finally:
                if old_hf:
                    os.environ["HF_TOKEN"] = old_hf
                else:
                    os.environ.pop("HF_TOKEN", None)
                if old_hub:
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = old_hub
                else:
                    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
                os.unlink(f.name)

    def test_hf_token_from_env(self):
        os.environ["ENGRAM_HF_TOKEN"] = "hf_env_token_456"
        try:
            cfg = Config.load()
            assert cfg.hf_token == "hf_env_token_456"
        finally:
            del os.environ["ENGRAM_HF_TOKEN"]


class TestValidation:
    def test_explicit_missing_path_does_not_fall_back(self, tmp_path):
        (tmp_path / "default.yaml").write_text("retrieval:\n  top_k: 15\n")
        assert Config.load().retrieval.top_k == 15
        with pytest.raises(ConfigError, match="config_file"):
            Config.load(tmp_path / "missing.yaml")

    def test_unreadable_path_has_safe_error(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, {})
        def denied(*args, **kwargs):
            raise PermissionError("credential-secret-in-os-error")
        monkeypatch.setattr(Path, "read_text", denied)
        with pytest.raises(ConfigError, match="config_file") as exc:
            Config.load(path)
        assert "credential-secret" not in str(exc.value)
        assert exc.value.__suppress_context__

    def test_directory_is_not_a_config(self, tmp_path):
        with pytest.raises(ConfigError, match="config_file"):
            Config.load(tmp_path)

    @pytest.mark.parametrize("document", ["[]", "false", "0", "null", "text", "retrieval: null", "ann: []", "web: 42", "llm: true"])
    def test_malformed_shapes(self, tmp_path, document):
        path = tmp_path / "settings.yaml"
        path.write_text(document)
        with pytest.raises(ConfigError):
            Config.load(path)

    @pytest.mark.parametrize("document", ["hf_token: [private-secret", "hf_token: !!python/object:private-secret {}", "hf_token: a\nhf_token: b", "retrieval:\n  top_k: 2\n  top_k: 3", "[secret]: value", "!!map []"])
    def test_unsafe_or_ambiguous_yaml_is_rejected_without_snippets(self, tmp_path, document):
        path = tmp_path / "settings.yaml"
        path.write_text(document)
        with pytest.raises(ConfigError, match="config_file") as exc:
            Config.load(path)
        assert "private-secret" not in str(exc.value)
        assert "hf_token:" not in str(exc.value)

    @pytest.mark.parametrize("document", ["hf_token: 2026-99-99", "embedding_dim: " + "9" * 5000])
    def test_yaml_constructor_errors_are_wrapped_safely(self, tmp_path, document):
        path = tmp_path / "settings.yaml"
        path.write_text(document)
        with pytest.raises(ConfigError, match="config_file") as exc:
            Config.load(path)
        assert "2026-99-99" not in str(exc.value)
        assert "999999" not in str(exc.value)

    @pytest.mark.parametrize("values, field", [
        ({"private-secret": 4}, "config_file"),
        ({"retrieval": {"private-secret": 4}}, "retrieval"),
        ({"dormant_recall": {"validate": "private-secret"}}, "dormant_recall"),
        ({"ann": {"resolved_index_path": "private-secret"}}, "ann"),
    ])
    def test_unknown_keys_are_not_ignored_or_echoed(self, tmp_path, values, field):
        with pytest.raises(ConfigError, match=field) as exc:
            Config.load(write_config(tmp_path, values))
        assert "private-secret" not in str(exc.value)

    @pytest.mark.parametrize("values, field", [
        ({"storage_backend": "typo-secret"}, "storage_backend"),
        ({"embedding_backend": "typo-secret"}, "embedding_backend"),
        ({"embedding_dim": True}, "embedding_dim"),
        ({"embedding_dim": "384"}, "embedding_dim"),
        ({"hf_token": ["private-secret"]}, "hf_token"),
        ({"postgres_dsn": 123}, "postgres_dsn"),
        ({"embedding_model": " "}, "embedding_model"),
        ({"retrieval": {"top_k": 2.5}}, "retrieval.top_k"),
        ({"retrieval": {"min_confidence": "0.6"}}, "retrieval.min_confidence"),
        ({"retrieval": {"enable_query_expansion": "false"}}, "retrieval.enable_query_expansion"),
        ({"retrieval": {"rerank_passage_fallback": 0}}, "retrieval.rerank_passage_fallback"),
        ({"retrieval": {"min_confidence": float("nan")}}, "retrieval.min_confidence"),
        ({"retrieval": {"exact_match_boost": float("inf")}}, "retrieval.exact_match_boost"),
        ({"retrieval": {"top_k": 0}}, "retrieval.top_k"),
        ({"retrieval": {"rrf_k": -1}}, "retrieval.rrf_k"),
        ({"retrieval": {"rerank_fusion_alpha": 1.01}}, "retrieval.rerank_fusion_alpha"),
        ({"retrieval": {"rerank_passage_floor": -0.001}}, "retrieval.rerank_passage_floor"),
        ({"lifecycle": {"forgetting_half_life_days": 0}}, "lifecycle.forgetting_half_life_days"),
        ({"lifecycle": {"huber_delta": 0}}, "lifecycle.huber_delta"),
        ({"lifecycle": {"retention_mode": "typo-secret"}}, "lifecycle.retention_mode"),
        ({"lifecycle": {"cluster_min_size": 1}}, "lifecycle.cluster_min_size"),
        ({"llm": {"backend": "typo-secret"}}, "llm.backend"),
        ({"llm": {"api_key": {"private-secret": 1}}}, "llm.api_key"),
        ({"web": {"port": 65536}}, "web.port"),
        ({"web": {"auth_token": False}}, "web.auth_token"),
        ({"ann": {"enabled": "false"}}, "ann.enabled"),
        ({"ann": {"m": 0}}, "ann.m"),
        ({"ann": {"max_elements": 0}}, "ann.max_elements"),
        ({"dormant_recall": {"mode": "visible"}}, "dormant_recall.mode"),
        ({"dormant_recall": {"max_bonus": 0.11}}, "dormant_recall.max_bonus"),
        ({"dormant_recall": {"cooldown_days": True}}, "dormant_recall.cooldown_days"),
    ])
    def test_invalid_values_name_only_the_field_and_rule(self, tmp_path, values, field):
        with pytest.raises(ConfigError, match=field) as exc:
            Config.load(write_config(tmp_path, values))
        assert "private-secret" not in str(exc.value)
        assert "typo-secret" not in str(exc.value)

    def test_invalid_file_is_not_hidden_by_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENGRAM_RETRIEVAL_TOP_K", "12")
        with pytest.raises(ConfigError, match="retrieval.top_k"):
            Config.load(write_config(tmp_path, {"retrieval": {"top_k": "private-secret"}}))

    def test_validate_constructed_config_and_backend_property(self):
        assert Config().validate().retrieval.top_k == 10
        with pytest.raises(ConfigError, match="storage_backend"):
            _ = Config(storage_backend="typo-secret").normalized_storage_backend
        with pytest.raises(ConfigError, match="retrieval"):
            Config(retrieval={"top_k": 3}).validate()
        with pytest.raises(ConfigError, match="postgres_dsn"):
            Config(storage_backend="postgres").validate()
        assert Config(storage_backend=" POSTGRES ", postgres_dsn="postgresql://localhost/db").validate().normalized_storage_backend == "postgres"

    def test_dormant_standalone_validation_remains_available(self):
        with pytest.raises(ConfigError, match="dormant_recall.candidate_limit"):
            DormantRecallConfig(candidate_limit=2.5).validate()

    @pytest.mark.parametrize("document", ["", "# comments only\n", "{}", "retrieval: {}"])
    def test_empty_configs_keep_defaults(self, tmp_path, document):
        path = tmp_path / "settings.yaml"
        path.write_text(document)
        assert Config.load(path).retrieval.top_k == 10

    def test_yaml_merge_keeps_standard_explicit_override_semantics(self, tmp_path):
        path = tmp_path / "settings.yaml"
        path.write_text("retrieval:\n  <<: {top_k: 2}\n  top_k: 3\n")
        assert Config.load(path).retrieval.top_k == 3

    def test_numeric_boundaries_and_independent_passage_floor(self, tmp_path):
        cfg = Config.load(write_config(tmp_path, {
            "retrieval": {"min_confidence": 0, "rerank_passage_floor": 1, "rerank_fusion_alpha": 1, "search_cache_size": 0},
            "web": {"port": 65535}, "lifecycle": {"elastic_l1_ratio": 0},
        }))
        assert cfg.retrieval.min_confidence == 0
        assert cfg.retrieval.rerank_passage_floor == 1

    def test_invalid_load_does_not_propagate_hf_secret(self, tmp_path):
        with pytest.raises(ConfigError):
            Config.load(write_config(tmp_path, {"hf_token": "private-secret", "retrieval": {"top_k": 0}}))
        assert "HF_TOKEN" not in os.environ
        assert "HUGGING_FACE_HUB_TOKEN" not in os.environ


class TestCompleteEnvironmentOverrides:
    @pytest.mark.parametrize("name, value, expected", [
        ("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "0.72", 0.72),
        ("ENGRAM_RETRIEVAL_ENABLE_QUERY_EXPANSION", "false", False),
        ("ENGRAM_RETRIEVAL_PRESERVE_PRIOR_CANDIDATE", "TRUE", True),
        ("ENGRAM_RETRIEVAL_RERANK_PASSAGE_FALLBACK", "0", False),
        ("ENGRAM_ANN_ENABLED", "1", True),
        ("ENGRAM_ANN_EF_SEARCH", "250", 250),
        ("ENGRAM_LIFECYCLE_RETENTION_MODE", "elastic", "elastic"),
        ("ENGRAM_LLM_MODEL", "test-model", "test-model"),
        ("ENGRAM_WEB_PORT", "1234", 1234),
        ("ENGRAM_DORMANT_RECALL_MODE", "shadow", "shadow"),
    ])
    def test_nested_env_overrides_and_sources(self, monkeypatch, name, value, expected):
        monkeypatch.setenv(name, value)
        cfg = Config.load()
        path = next(key for key, item in Config.schema()["fields"].items() if item["env"] == name)
        section, field = path.split(".")
        assert getattr(getattr(cfg, section), field) == expected
        assert cfg.describe()["sources"][path] == f"env:{name}"

    @pytest.mark.parametrize("name, value, field", [
        ("ENGRAM_RETRIEVAL_TOP_K", "2.5", "retrieval.top_k"),
        ("ENGRAM_RETRIEVAL_TOP_K", "1_000", "retrieval.top_k"),
        ("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "nan", "retrieval.min_confidence"),
        ("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "inf", "retrieval.min_confidence"),
        ("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "", "retrieval.min_confidence"),
        ("ENGRAM_ANN_ENABLED", "private-secret", "ann.enabled"),
        ("ENGRAM_ANN_ENABLED", "yes", "ann.enabled"),
        ("ENGRAM_LLM_BACKEND", "private-secret", "llm.backend"),
        ("ENGRAM_WEB_PORT", "0", "web.port"),
        ("ENGRAM_DORMANT_RECALL_CANDIDATE_LIMIT", "201", "dormant_recall.candidate_limit"),
    ])
    def test_bad_environment_values_are_safe(self, monkeypatch, name, value, field):
        monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError, match=field) as exc:
            Config.load()
        assert "private-secret" not in str(exc.value)

    def test_env_over_file_over_default_with_secret_clear(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, {"retrieval": {"min_confidence": 0.7}, "web": {"auth_token": "private-secret", "port": 9000}})
        monkeypatch.setenv("ENGRAM_RETRIEVAL_MIN_CONFIDENCE", "0.8")
        monkeypatch.setenv("ENGRAM_WEB_AUTH_TOKEN", "")
        description = Config.load(path).describe()
        assert description["values"]["retrieval"]["min_confidence"] == 0.8
        assert description["values"]["web"]["auth_token"] == ""
        assert description["sources"]["retrieval.min_confidence"] == "env:ENGRAM_RETRIEVAL_MIN_CONFIDENCE"
        assert description["sources"]["web.port"] == "file"
        assert description["sources"]["retrieval.top_k"] == "default"

    def test_hf_alias_precedence_and_redaction(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, {"hf_token": "private-file-secret"})
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "private-hub-secret")
        assert Config.load(path).describe()["sources"]["hf_token"] == "env:HUGGING_FACE_HUB_TOKEN"
        monkeypatch.setenv("HF_TOKEN", "private-hf-secret")
        assert Config.load(path).describe()["sources"]["hf_token"] == "env:HF_TOKEN"
        monkeypatch.setenv("ENGRAM_HF_TOKEN", "private-engram-secret")
        cfg = Config.load(path)
        assert cfg.hf_token == "private-engram-secret"
        assert cfg.describe()["sources"]["hf_token"] == "env:ENGRAM_HF_TOKEN"
        assert "private-" not in json.dumps(cfg.describe())

    def test_empty_legacy_hf_alias_does_not_mask_other_alias(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "private-hub-secret")
        assert Config.load().hf_token == "private-hub-secret"

    def test_effective_hf_token_matches_both_runtime_aliases(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "private-old-hf-secret")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "private-old-hub-secret")
        monkeypatch.setenv("ENGRAM_HF_TOKEN", "private-effective-secret")
        cfg = Config.load()
        assert os.environ["HF_TOKEN"] == os.environ["HUGGING_FACE_HUB_TOKEN"] == cfg.hf_token
        assert cfg.describe()["sources"]["hf_token"] == "env:ENGRAM_HF_TOKEN"
        assert "private-" not in json.dumps(cfg.describe())

    def test_explicit_empty_hf_setting_removes_stale_runtime_aliases(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "private-old-hf-secret")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "private-old-hub-secret")
        monkeypatch.setenv("ENGRAM_HF_TOKEN", "")
        cfg = Config.load()
        assert cfg.hf_token == ""
        assert "HF_TOKEN" not in os.environ
        assert "HUGGING_FACE_HUB_TOKEN" not in os.environ
        assert cfg.describe()["sources"]["hf_token"] == "env:ENGRAM_HF_TOKEN"

    @pytest.mark.parametrize("backend, alias", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
    def test_llm_provider_alias_is_fallback_only(self, tmp_path, monkeypatch, backend, alias):
        monkeypatch.setenv(alias, "private-provider-secret")
        path = write_config(tmp_path, {"llm": {"backend": backend}})
        cfg = Config.load(path)
        assert cfg.llm.api_key == "private-provider-secret"
        assert cfg.describe()["sources"]["llm.api_key"] == f"env:{alias}"
        path = write_config(tmp_path, {"llm": {"backend": backend, "api_key": "private-file-secret"}})
        assert Config.load(path).llm.api_key == "private-file-secret"
        monkeypatch.setenv("ENGRAM_LLM_API_KEY", "private-engram-secret")
        assert Config.load(path).llm.api_key == "private-engram-secret"


class TestConfigReports:
    def test_all_fields_have_schema_and_default_provenance(self):
        cfg = Config()
        description = cfg.describe()
        schema = Config.schema()["fields"]
        expected = set()
        for item in fields(Config):
            value = getattr(cfg, item.name)
            if hasattr(value, "__dataclass_fields__"):
                expected.update(f"{item.name}.{leaf.name}" for leaf in fields(value))
            else:
                expected.add(item.name)
        assert schema.keys() == expected == description["sources"].keys()
        assert set(description["sources"].values()) == {"default"}
        assert description["config_file"] is None
        for entry in schema.values():
            assert entry["help"] and entry["env"].startswith("ENGRAM_")
            assert entry["type"] in {"string", "integer", "number", "boolean"}
        json.dumps(description, allow_nan=False)
        json.dumps(Config.schema(), allow_nan=False)

    def test_describe_redacts_every_secret_but_preserves_live_values(self, tmp_path):
        secret = "private-secret-value"
        cfg = Config.load(write_config(tmp_path, {"hf_token": secret, "postgres_dsn": secret, "llm": {"api_key": secret}, "web": {"auth_token": secret}}))
        description = cfg.describe()
        assert secret not in json.dumps(description)
        assert description["values"]["hf_token"] == "<redacted>"
        assert description["values"]["postgres_dsn"] == "<redacted>"
        assert description["values"]["llm"]["api_key"] == "<redacted>"
        assert description["values"]["web"]["auth_token"] == "<redacted>"
        assert cfg.hf_token == cfg.postgres_dsn == cfg.llm.api_key == cfg.web.auth_token == secret

    def test_provenance_is_not_serialized_as_settings(self, tmp_path):
        cfg = Config.load(write_config(tmp_path, {"retrieval": {"top_k": 11}}))
        assert asdict(cfg).keys() == {item.name for item in fields(Config)}
        assert all(not name.startswith("_") for name in asdict(cfg))
        assert cfg.describe()["config_file"] == str(tmp_path / "settings.yaml")
        assert cfg.describe()["sources"]["retrieval.top_k"] == "file"

    def test_derived_dimension_and_explicit_dimension_provenance(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, {"embedding_model": "voyage-3.5"})
        cfg = Config.load(path)
        assert cfg.embedding_dim == 1024
        assert cfg.describe()["sources"]["embedding_dim"] == "derived:embedding_model"
        path = write_config(tmp_path, {"embedding_model": "voyage-3.5", "embedding_dim": 512})
        assert Config.load(path).describe()["sources"]["embedding_dim"] == "file"
        monkeypatch.setenv("ENGRAM_EMBEDDING_DIM", "256")
        cfg = Config.load(path)
        assert cfg.embedding_dim == 256
        assert cfg.describe()["sources"]["embedding_dim"] == "env:ENGRAM_EMBEDDING_DIM"

    def test_unknown_dimension_warning_is_safe(self, tmp_path):
        cfg = Config.load(write_config(tmp_path, {"embedding_model": "private-custom-model"}))
        assert cfg.embedding_dim == 384
        assert cfg.describe()["warnings"]
        assert "private-custom-model" not in json.dumps(cfg.describe()["warnings"])

    def test_constructed_and_runtime_override_sources(self, tmp_path):
        cfg = Config(embedding_dim=128)
        assert cfg.describe()["sources"]["embedding_dim"] == "derived:constructed"
        cfg = Config.load(write_config(tmp_path, {"retrieval": {"top_k": 11}}))
        cfg.retrieval.top_k = 12
        assert cfg.describe()["sources"]["retrieval.top_k"] == "derived:runtime override"

    def test_schema_cannot_mutate_validation(self):
        Config.schema()["fields"]["storage_backend"]["constraints"]["enum"].append("invalid")
        with pytest.raises(ConfigError, match="storage_backend"):
            Config(storage_backend="invalid").validate()

    def test_load_and_reports_do_not_create_database_or_index_directories(self, tmp_path, monkeypatch):
        path = write_config(tmp_path, {"db_path": str(tmp_path / "db-parent" / "memory.db"), "ann": {"index_path": str(tmp_path / "index-parent" / "hnsw.index")}})
        def forbidden(*args, **kwargs):
            raise AssertionError("configuration inspection attempted a filesystem mutation")
        monkeypatch.setattr(Path, "mkdir", forbidden)
        Config.load(path).describe()
        Config.schema()
        assert not (tmp_path / "db-parent").exists()
        assert not (tmp_path / "index-parent").exists()

    def test_config_reports_do_not_import_service_or_model_runtimes(self, tmp_path, monkeypatch):
        original_import = builtins.__import__
        forbidden = {"engram.store", "engram.store_postgres", "torch", "sentence_transformers", "mlx", "openai", "anthropic", "voyageai", "psycopg"}
        def guarded_import(name, *args, **kwargs):
            if any(name == module or name.startswith(module + ".") for module in forbidden):
                raise AssertionError("configuration inspection imported a service or model runtime")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        cfg = Config.load(write_config(tmp_path, {"embedding_model": "voyage-3.5"}))
        assert cfg.embedding_dim == 1024
        cfg.describe()
        Config.schema()
