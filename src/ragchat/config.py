"""Configuration loading.

All tunable values live in ``config.yaml``. Code reads them through :class:`Config`
so that a chunk-size sweep or a threshold change never requires a source edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


@dataclass(frozen=True)
class PathsConfig:
    articles_dir: Path
    legacy_articles_dir: Path
    index_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    default_strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str
    n_features: int
    ngram_min: int
    ngram_max: int
    lowercase: bool

    @property
    def ngram_range(self) -> tuple[int, int]:
        return (self.ngram_min, self.ngram_max)


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class GenerationConfig:
    backend: str
    model: str
    max_tokens: int
    effort: str
    max_context_chunks: int


@dataclass(frozen=True)
class GroundingConfig:
    min_top_score: float
    min_evidence_coverage: float
    refusal_message: str


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    grounding: GroundingConfig
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not cfg_path.is_file():
            raise ConfigError(f"config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        try:
            paths_raw = raw["paths"]
            paths = PathsConfig(
                articles_dir=_resolve(paths_raw["articles_dir"]),
                legacy_articles_dir=_resolve(paths_raw["legacy_articles_dir"]),
                index_dir=_resolve(paths_raw["index_dir"]),
                results_dir=_resolve(paths_raw["results_dir"]),
            )
            chunking = ChunkingConfig(**raw["chunking"])
            embedding = EmbeddingConfig(**raw["embedding"])
            retrieval = RetrievalConfig(**raw["retrieval"])
            generation = GenerationConfig(**raw["generation"])
            grounding_raw = dict(raw["grounding"])
            grounding_raw["refusal_message"] = grounding_raw["refusal_message"].strip()
            grounding = GroundingConfig(**grounding_raw)
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"malformed config file {cfg_path}: {exc}") from exc

        return cls(
            paths=paths,
            chunking=chunking,
            embedding=embedding,
            retrieval=retrieval,
            generation=generation,
            grounding=grounding,
            raw=raw,
        )

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def with_grounding(self, *, min_evidence_coverage: float) -> "Config":
        """Return a copy with a different evidence threshold (used by the bonus run)."""
        return replace(
            self,
            grounding=replace(self.grounding, min_evidence_coverage=min_evidence_coverage),
        )

    def with_chunking(self, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> "Config":
        """Return a copy with chunk sizing overridden (used by the sweep)."""
        chunking = replace(
            self.chunking,
            chunk_size=chunk_size if chunk_size is not None else self.chunking.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else self.chunking.chunk_overlap,
        )
        return replace(self, chunking=chunking)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def anthropic_credentials_present() -> bool:
    """True when the Anthropic SDK has some credential source available.

    An unset ``ANTHROPIC_API_KEY`` does not by itself mean there are no
    credentials, so an ``ant auth login`` profile on disk counts too.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR")
    candidates = [Path(config_dir)] if config_dir else []
    candidates += [Path.home() / ".config" / "anthropic"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Anthropic")
    return any((c / "credentials").is_dir() for c in candidates)
