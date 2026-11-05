"""Embedding backends.

* ``SentenceTransformerEmbedder`` – real multilingual model; this is what makes
  cross-lingual retrieval and caching work (an English and a Swahili question
  land close together in vector space).
* ``HashingEmbedder`` – deterministic, dependency-free character n-gram hashing.
  It is *not* cross-lingual; it exists so the engine, tests and CI run offline.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from polylang.text import normalize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or set POLYLANG_EMBEDDING_BACKEND=hashing."
            ) from exc
        log.info("Loading embedding model %s ...", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        self.dim = int(getter()) if getter else int(self.embed(["dim probe"]).shape[1])
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            [normalize(t) for t in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _l2_normalize(vecs)


class HashingEmbedder:
    """Character 3-5 gram feature hashing with signed buckets (offline fallback)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = f" {normalize(text).lower()} "
        for n in (3, 4, 5):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2_normalize(np.stack([self._vector(t) for t in texts]))


def build_embedder(backend: str, model: str, device: str | None = None, hashing_dim: int = 1024) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder(hashing_dim)
    return SentenceTransformerEmbedder(model, device=device)
