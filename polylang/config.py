"""Runtime configuration, read from environment variables (prefix ``POLYLANG_``).

Every setting has a sensible default so the engine runs with zero configuration.
A ``.env`` file in the working directory is loaded automatically if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default, cast=str):
    raw = os.environ.get(f"POLYLANG_{name}")
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass
class Settings:
    # Storage
    data_dir: str = "./storage"
    persist: bool = True

    # Embeddings: "sentence-transformers" (real multilingual model) or "hashing" (offline fallback)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str | None = None
    hashing_dim: int = 1024

    # Chunking
    chunk_size: int = 800          # characters
    chunk_overlap: int = 120       # characters

    # Retrieval
    top_k: int = 4
    fusion: str = "rrf"            # "rrf" (reciprocal rank fusion) or "weighted"
    dense_weight: float = 0.6      # used when fusion == "weighted"
    rrf_k: int = 60
    min_score: float = 0.0         # drop fused hits below this (weighted mode only)

    # Semantic cache
    cache_enabled: bool = True
    cache_threshold: float = 0.88
    cache_max_entries: int = 5000
    cache_ttl_seconds: int = 86400  # 0 disables expiry
    # True: a question in any language can reuse an answer cached for another language
    # (the original project's goal). False: only reuse answers within the same language.
    cache_cross_lingual: bool = True

    # Generation: "extractive" (no LLM), "openai", "anthropic", "ollama"
    llm_provider: str = "extractive"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 600
    llm_timeout: float = 60.0

    # API
    api_key: str = ""              # if set, write endpoints require header X-API-Key
    max_upload_mb: int = 25
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        _load_dotenv()
        s = cls()
        for f in fields(cls):
            if f.name == "cors_origins":
                raw = os.environ.get("POLYLANG_CORS_ORIGINS")
                if raw:
                    s.cors_origins = [o.strip() for o in raw.split(",") if o.strip()]
                continue
            default = getattr(s, f.name)
            if f.name == "embedding_device":
                setattr(s, f.name, _env("EMBEDDING_DEVICE", None))
                continue
            cast = type(default) if default is not None else str
            setattr(s, f.name, _env(f.name.upper(), default, cast))
        # Provider-specific API key fallbacks
        if not s.llm_api_key:
            if s.llm_provider == "openai":
                s.llm_api_key = os.environ.get("OPENAI_API_KEY", "")
            elif s.llm_provider == "anthropic":
                s.llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        for k, v in overrides.items():
            if not hasattr(s, k):
                raise AttributeError(f"Unknown setting: {k}")
            setattr(s, k, v)
        s.validate()
        return s

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.fusion not in {"rrf", "weighted"}:
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        if not 0.0 <= self.dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        if not 0.0 < self.cache_threshold <= 1.0:
            raise ValueError("cache_threshold must be in (0, 1]")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.llm_provider not in {"extractive", "openai", "anthropic", "ollama"}:
            raise ValueError("llm_provider must be extractive, openai, anthropic or ollama")
        if self.embedding_backend not in {"sentence-transformers", "hashing"}:
            raise ValueError("embedding_backend must be 'sentence-transformers' or 'hashing'")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)
