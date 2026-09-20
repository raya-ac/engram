"""Configuration with env var > config file > defaults priority."""

from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """An invalid configuration, with a message safe to show to users."""


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
    rerank_fusion_alpha: float = 0.0
    preserve_prior_candidate: bool = True
    rerank_passage_fallback: bool = True
    rerank_passage_floor: float = 0.001


@dataclass
class DormantRecallConfig:
    # Opt in explicitly. Version one never injects suggestions into recall.
    mode: str = "off"  # off | shadow
    candidate_limit: int = 50
    dormancy_days: float = 30.0
    min_relevance: float = 0.75  # raw cosine, not truth/confidence
    max_bonus: float = 0.05
    rerank_candidates: int = 12
    min_rerank_score: float = 0.6  # model score, not a probability
    cooldown_days: float = 7.0
    feedback_cooldown_days: float = 30.0
    log_max_events: int = 1000
    log_retention_days: float = 30.0

    def validate(self):
        _validate_section(self, DormantRecallConfig, "dormant_recall")


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
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    embedding_backend: str = "auto"  # auto | mlx | sentence_transformers | voyage | openai | gemini
    embedding_dim: int = 384
    hf_token: str = ""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dormant_recall: DormantRecallConfig = field(default_factory=DormantRecallConfig)
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
        _validate_value("storage_backend", self.storage_backend, "sqlite")
        return self.storage_backend.strip().lower()

    def validate(self) -> Config:
        """Validate constructed or loaded settings without opening any services."""
        for item in fields(self):
            value = getattr(self, item.name)
            section = _SECTIONS.get(item.name)
            if section:
                _validate_section(value, section, item.name)
            else:
                _validate_value(item.name, value, item.default)
        if self.normalized_storage_backend == "postgres" and not self.postgres_dsn.strip():
            raise ConfigError("postgres_dsn must be nonempty when storage_backend is postgres")
        return self

    def describe(self) -> dict:
        """Return effective settings and provenance, with credentials redacted.

        Provenance is deliberately outside the dataclass fields, so existing
        dataclass serialization still contains only configurable settings.
        """
        self.validate()
        values = asdict(self)
        current = _flatten(values)
        defaults = _flatten(asdict(type(self)()))
        loaded = getattr(self, "_loaded_values", None)
        sources = dict(getattr(self, "_sources", {}))
        for name, value in current.items():
            if loaded is not None and value != loaded[name]:
                sources[name] = "derived:runtime override"
            elif name not in sources:
                sources[name] = "default" if value == defaults[name] else "derived:constructed"
            if name in _SECRETS and value:
                _set_nested(values, name, "<redacted>")
        return {
            "config_file": getattr(self, "_config_file", None),
            "values": values,
            "sources": sources,
            "warnings": list(getattr(self, "_warnings", [])),
        }

    @classmethod
    def schema(cls) -> dict:
        """Describe every supported field without reading files or environment."""
        result = {}
        for name, default in _flatten(asdict(cls())).items():
            aliases = _ENV_ALIASES.get(name, ())
            if name == "llm.api_key":
                aliases = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            constraints = copy.deepcopy(_CONSTRAINTS.get(name, {}))
            if type(default) is float:
                constraints["finite"] = True
            if type(default) is bool:
                constraints["environment_values"] = ["true", "false", "1", "0"]
            if name == "postgres_dsn":
                constraints["nonempty_when"] = {"storage_backend": "postgres"}
            result[name] = {
                "type": _TYPE_NAMES[type(default)],
                "default": default,
                "constraints": constraints,
                "help": _HELP[name],
                "env": _env_name(name),
                "aliases": list(aliases),
                "secret": name in _SECRETS,
            }
        return {"fields": result}

    @staticmethod
    def _default_paths() -> list[Path]:
        return [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".config" / "engram" / "config.yaml",
        ]

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load the first default file, or require the explicitly supplied file.

        Every field supports ENGRAM_<SECTION>_<FIELD>. Environment booleans
        accept true/false or 1/0 only. Invalid file values are rejected even if
        an environment variable would override them.
        """
        selected = None
        try:
            if path is not None:
                selected = Path(path).expanduser()
            else:
                selected = next((p for p in cls._default_paths() if p.exists()), None)
        except (OSError, TypeError, ValueError, RuntimeError):
            raise ConfigError("config_file must be an accessible file path") from None
        raw = _read_config(selected) if selected is not None else {}
        return cls._resolve(raw, selected=selected)

    @classmethod
    def from_mapping(cls, values: dict, *, apply_environment: bool = True) -> Config:
        """Validate proposed settings before setup writes a configuration file.

        Uses the same defaults and environment precedence as load(), without
        discovering or reading a file. Disabling environment application also
        disables provider credential fallback and token propagation.
        """
        if not isinstance(apply_environment, bool):
            raise ConfigError("apply_environment must be a boolean")
        return cls._resolve(values, apply_environment=apply_environment)

    @classmethod
    def _resolve(cls, raw: dict, *, selected: Path | None = None,
                 apply_environment: bool = True) -> Config:
        cfg = cls()
        defaults = _flatten(asdict(cfg))
        cfg._sources = dict.fromkeys(defaults, "default")
        cfg._warnings = []
        cfg._config_file = str(selected.absolute()) if selected is not None else None
        for name, value in _file_values(raw, cfg).items():
            _validate_value(name, value, defaults[name])
            _set_field(cfg, name, value)
            cfg._sources[name] = "file" if selected is not None else "derived:provided"
        for name, default in (defaults.items() if apply_environment else ()):
            primary = _env_name(name)
            env = primary if primary in os.environ else next(
                (alias for alias in _ENV_ALIASES.get(name, ()) if os.environ.get(alias)), None,
            )
            if env is not None:
                value = _parse_env(name, os.environ[env], default)
                _set_field(cfg, name, value)
                cfg._sources[name] = f"env:{env}"

        # Preserve the provider-specific fallback already used by llm.py.
        if apply_environment and not cfg.llm.api_key:
            alias = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(cfg.llm.backend)
            if alias and os.environ.get(alias):
                cfg.llm.api_key = os.environ[alias]
                cfg._sources["llm.api_key"] = f"env:{alias}"

        cfg.validate()
        if cfg._sources["embedding_dim"] == "default":
            # This is a static registry lookup; it does not load a model.
            from engram.embeddings import get_model_dim
            detected = get_model_dim(cfg.embedding_model)
            if detected is not None:
                cfg.embedding_dim = detected
                cfg._sources["embedding_dim"] = "derived:embedding_model"
            else:
                cfg._warnings.append("embedding_dim: model is not in the dimension registry; set its dimension explicitly")
        cfg.validate()
        cfg._loaded_values = _flatten(asdict(cfg))
        if apply_environment and cfg.hf_token:
            # Model clients consume these aliases directly. Keep their runtime
            # value consistent with the precedence reported by this config.
            os.environ["HF_TOKEN"] = cfg.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = cfg.hf_token
        elif apply_environment and "ENGRAM_HF_TOKEN" in os.environ:
            # An explicit empty primary setting must not expose a stale alias
            # to a provider after configuration reports no supplied token.
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return cfg


_SECTIONS = {
    "retrieval": RetrievalConfig, "dormant_recall": DormantRecallConfig,
    "lifecycle": LifecycleConfig, "llm": LLMConfig, "web": WebConfig, "ann": ANNConfig,
}
_TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_SECRETS = {"hf_token", "postgres_dsn", "llm.api_key", "web.auth_token"}
_ENV_ALIASES = {"hf_token": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
_CONSTRAINTS = {
    "storage_backend": {"enum": ["sqlite", "postgres"], "normalize": "strip/lower"},
    "embedding_backend": {"enum": ["auto", "mlx", "sentence_transformers", "voyage", "openai", "gemini"]},
    "llm.backend": {"enum": ["claude_cli", "mlx", "openai", "anthropic"]},
    "lifecycle.retention_mode": {"enum": ["l2", "huber", "elastic"]},
    "dormant_recall.mode": {"enum": ["off", "shadow"]},
    "embedding_dim": {"minimum": 1},
    "retrieval.top_k": {"minimum": 1},
    "retrieval.rrf_k": {"minimum": 1},
    "retrieval.min_confidence": {"minimum": 0, "maximum": 1},
    "retrieval.rerank_candidates": {"minimum": 1},
    "retrieval.dense_multiplier": {"minimum": 1},
    "retrieval.bm25_multiplier": {"minimum": 1},
    "retrieval.exact_match_boost": {"minimum": 0},
    "retrieval.search_cache_size": {"minimum": 0},
    "retrieval.rerank_fusion_alpha": {"minimum": 0, "maximum": 1},
    "retrieval.rerank_passage_floor": {"minimum": 0, "maximum": 1},
    "lifecycle.forgetting_half_life_days": {"minimum": 1},
    "lifecycle.archive_after_days": {"minimum": 0},
    "lifecycle.archive_min_importance": {"minimum": 0, "maximum": 1},
    "lifecycle.archive_min_accesses": {"minimum": 0},
    "lifecycle.promote_importance": {"minimum": 0, "maximum": 1},
    "lifecycle.promote_accesses": {"minimum": 0},
    "lifecycle.cluster_threshold": {"minimum": 0, "maximum": 1},
    "lifecycle.cluster_min_size": {"minimum": 2},
    "lifecycle.huber_delta": {"exclusive_minimum": 0},
    "lifecycle.elastic_l1_ratio": {"minimum": 0, "maximum": 1},
    "ann.m": {"minimum": 2},
    "ann.ef_construction": {"minimum": 1},
    "ann.ef_search": {"minimum": 1},
    "ann.max_elements": {"minimum": 1},
    "web.port": {"minimum": 1, "maximum": 65535},
    "dormant_recall.candidate_limit": {"minimum": 1, "maximum": 200},
    "dormant_recall.dormancy_days": {"minimum": 1, "maximum": 3650},
    "dormant_recall.min_relevance": {"minimum": 0.5, "maximum": 1},
    "dormant_recall.max_bonus": {"minimum": 0, "maximum": 0.1},
    "dormant_recall.rerank_candidates": {"minimum": 1, "maximum": 50},
    "dormant_recall.min_rerank_score": {"minimum": -10, "maximum": 10},
    "dormant_recall.cooldown_days": {"minimum": 1, "maximum": 365},
    "dormant_recall.feedback_cooldown_days": {"minimum": 1, "maximum": 365},
    "dormant_recall.log_max_events": {"minimum": 1, "maximum": 10000},
    "dormant_recall.log_retention_days": {"minimum": 1, "maximum": 365},
}
for _name in ("db_path", "embedding_model", "cross_encoder_model", "llm.model", "llm.mlx_model", "ann.index_path", "web.host"):
    _CONSTRAINTS[_name] = {"nonempty": True}

_HELP = {
    "storage_backend": "Memory storage engine. Postgres requires postgres_dsn.",
    "db_path": "SQLite database path; ~ expands when the path is used.",
    "postgres_dsn": "PostgreSQL connection string, required for postgres storage. Redacted in reports.",
    "embedding_model": "Embedding model identifier; known models supply the default embedding_dim.",
    "cross_encoder_model": "Reranker model identifier. Local models return raw logits.",
    "embedding_backend": "Embedding runtime. Auto selects a local runtime; known hosted models select their provider.",
    "embedding_dim": "Vector dimension. Explicit file or environment values override the known-model registry.",
    "hf_token": "Hugging Face token. ENGRAM_HF_TOKEN takes precedence over HF_TOKEN and HUGGING_FACE_HUB_TOKEN.",
    "retrieval.top_k": "Maximum number of results requested by default.",
    "retrieval.rrf_k": "Reciprocal-rank fusion smoothing constant.",
    "retrieval.min_confidence": "Minimum final score for returned results; independent of the passage retry floor.",
    "retrieval.rerank_candidates": "Maximum candidates considered by the cross-encoder reranker.",
    "retrieval.dense_multiplier": "Dense candidate count as a multiple of top_k.",
    "retrieval.bm25_multiplier": "BM25 candidate count as a multiple of top_k.",
    "retrieval.enable_query_expansion": "Expand retrieval queries with related terms.",
    "retrieval.exact_match_boost": "Multiplier applied to exact text matches.",
    "retrieval.search_cache_size": "Retrieval cache capacity; zero disables result caching.",
    "retrieval.rerank_fusion_alpha": "Weight of the prior ranking in reranker score fusion; zero uses the reranker score alone.",
    "retrieval.preserve_prior_candidate": "Keep the strongest eligible prior candidate when selection would otherwise remove it.",
    "retrieval.rerank_passage_fallback": "Allow a bounded lexical excerpt retry for local rerankers.",
    "retrieval.rerank_passage_floor": "Early local excerpt retry floor; zero disables all excerpt retries. Rejected long memories can also retry before the final confidence gate.",
    "dormant_recall.mode": "Off disables dormant recall; shadow records suggestions without injecting recall results.",
    "dormant_recall.candidate_limit": "Maximum dormant memories considered.",
    "dormant_recall.dormancy_days": "Minimum time since a dormant memory was accessed.",
    "dormant_recall.min_relevance": "Minimum raw cosine similarity, not a truth or confidence probability.",
    "dormant_recall.max_bonus": "Maximum dormant ranking bonus.",
    "dormant_recall.rerank_candidates": "Maximum dormant candidates reranked.",
    "dormant_recall.min_rerank_score": "Minimum dormant reranker model score, not a probability.",
    "dormant_recall.cooldown_days": "Cooldown between repeated dormant suggestions.",
    "dormant_recall.feedback_cooldown_days": "Cooldown after dormant suggestion feedback.",
    "dormant_recall.log_max_events": "Maximum retained dormant events.",
    "dormant_recall.log_retention_days": "Dormant event retention period.",
    "lifecycle.forgetting_half_life_days": "Base retention half-life, before trust weighting.",
    "lifecycle.archive_after_days": "Minimum age before automatic archival is considered.",
    "lifecycle.archive_min_importance": "Memories below this importance may be archived.",
    "lifecycle.archive_min_accesses": "Memories with fewer accesses may be archived.",
    "lifecycle.promote_importance": "Minimum importance for promotion.",
    "lifecycle.promote_accesses": "Minimum access count for promotion.",
    "lifecycle.cluster_threshold": "Similarity threshold for memory clustering.",
    "lifecycle.cluster_min_size": "Minimum memories in a cluster.",
    "lifecycle.retention_mode": "Retention curve: l2, huber, or elastic.",
    "lifecycle.huber_delta": "Huber transition point in half-lives.",
    "lifecycle.elastic_l1_ratio": "Elastic retention L1 weight; zero is pure L2, one pure L1.",
    "llm.backend": "Text generation provider or local runtime.",
    "llm.model": "Text generation model for Claude CLI or hosted providers.",
    "llm.mlx_model": "MLX text generation model identifier.",
    "llm.api_key": "Hosted LLM credential. When empty, the selected provider's ANTHROPIC_API_KEY or OPENAI_API_KEY is used.",
    "web.host": "Web server bind address.",
    "web.port": "Web server TCP port.",
    "web.auth_token": "Bearer token for web authentication; empty disables token authentication.",
    "ann.enabled": "Enable the approximate nearest-neighbor index.",
    "ann.m": "HNSW neighbor connections per node.",
    "ann.ef_construction": "HNSW construction search budget.",
    "ann.ef_search": "HNSW query search budget.",
    "ann.max_elements": "Initial HNSW index capacity.",
    "ann.index_path": "HNSW index file path; ~ expands when the path is used.",
}


def _flatten(values: dict) -> dict:
    result = {}
    for name, value in values.items():
        if isinstance(value, dict):
            result.update((f"{name}.{key}", leaf) for key, leaf in value.items())
        else:
            result[name] = value
    return result


def _set_nested(values: dict, name: str, value) -> None:
    if "." in name:
        section, key = name.split(".")
        values[section][key] = value
    else:
        values[name] = value


def _set_field(config: Config, name: str, value) -> None:
    if "." in name:
        section, key = name.split(".")
        setattr(getattr(config, section), key, value)
    else:
        setattr(config, name, value)


def _env_name(name: str) -> str:
    return "ENGRAM_" + name.replace(".", "_").upper()


def _validate_section(value, cls, name: str) -> None:
    if not isinstance(value, cls):
        raise ConfigError(f"{name} must be a {cls.__name__} section")
    for item in fields(cls):
        _validate_value(f"{name}.{item.name}", getattr(value, item.name), item.default)


def _validate_value(name: str, value, default) -> None:
    kind = type(default)
    if kind is float:
        valid = type(value) in (int, float)
    else:
        valid = type(value) is kind
    if not valid:
        raise ConfigError(f"{name} must be {_TYPE_NAMES[kind]}")
    if kind is float:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ConfigError(f"{name} must be finite")
    rules = _CONSTRAINTS.get(name, {})
    if "enum" in rules:
        comparable = value.strip().lower() if rules.get("normalize") else value
        if comparable not in rules["enum"]:
            raise ConfigError(f"{name} must be one of: {', '.join(rules['enum'])}")
    if rules.get("nonempty") and not value.strip():
        raise ConfigError(f"{name} must be nonempty")
    for key, operator, phrase in (("minimum", lambda a, b: a >= b, "at least"), ("maximum", lambda a, b: a <= b, "at most"), ("exclusive_minimum", lambda a, b: a > b, "greater than")):
        if key in rules and not operator(value, rules[key]):
            raise ConfigError(f"{name} must be {phrase} {rules[key]}")


def _parse_env(name: str, value: str, default):
    kind = type(default)
    try:
        if kind is bool:
            text = value.strip().lower()
            if text not in {"true", "false", "1", "0"}:
                raise ConfigError(f"{name} environment value must be true, false, 1, or 0")
            parsed = text in {"true", "1"}
        elif kind is int:
            if not re.fullmatch(r"[+-]?[0-9]+", value.strip()):
                raise ValueError
            parsed = int(value)
        elif kind is float:
            parsed = float(value)
        else:
            parsed = value
    except ConfigError:
        raise
    except (ValueError, OverflowError):
        raise ConfigError(f"{name} environment value must be {_TYPE_NAMES[kind]}") from None
    _validate_value(name, parsed, default)
    return parsed


def _file_values(raw, config: Config) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError("config_file must contain a mapping of configuration fields")
    known = {item.name for item in fields(config)}
    result = {}
    for name, value in raw.items():
        if name not in known:
            raise ConfigError("config_file contains an unknown configuration field")
        if name in _SECTIONS:
            if not isinstance(value, dict):
                raise ConfigError(f"{name} must be a mapping")
            section_fields = {item.name for item in fields(_SECTIONS[name])}
            for key, leaf in value.items():
                if key not in section_fields:
                    raise ConfigError(f"{name} contains an unknown configuration field")
                result[f"{name}.{key}"] = leaf
        else:
            result[name] = value
    return result


class _ConfigLoader(yaml.SafeLoader):
    """Reject ambiguous duplicate keys rather than silently choosing one."""

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ConfigError("config_file mappings must contain configuration fields")
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("config_file mapping keys must be strings")
            if key in seen:
                raise ConfigError("config_file must not contain duplicate mapping keys")
            seen.add(key)
        self.flatten_mapping(node)
        return super().construct_mapping(node, deep=deep)


def _read_config(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        raise ConfigError("config_file must be a readable UTF-8 file") from None
    loader = None
    try:
        loader = _ConfigLoader(text)
        node = loader.get_single_node()
        return {} if node is None else loader.construct_document(node)
    except ConfigError:
        raise
    except (yaml.YAMLError, ValueError, OverflowError, RecursionError):
        raise ConfigError("config_file must contain valid YAML") from None
    finally:
        if loader is not None:
            loader.dispose()
