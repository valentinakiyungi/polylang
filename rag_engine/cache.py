"""Cross-lingual semantic cache.

A query is embedded once; if a previously answered query lies within
``threshold`` cosine similarity (and shares the same retrieval scope), its answer
is returned without retrieval or generation. With a multilingual embedding model
"How does semantic caching work?" and "Je, semantic caching hufanyaje kazi?" hit
the same entry.

Lookups are a single matrix-vector product, so they stay fast for thousands of
entries. Eviction is LRU, with an optional TTL.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class CacheEntry:
    id: str
    query: str
    language: str
    scope: str
    payload: dict
    created_at: float
    last_hit_at: float
    hits: int = 0


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


@dataclass
class CacheMatch:
    entry: CacheEntry
    similarity: float


class SemanticCache:
    def __init__(self, dim: int, threshold: float = 0.88, max_entries: int = 5000,
                 ttl_seconds: int = 86400, clock=time.time):
        self.dim = dim
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: list[CacheEntry] = []
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ core
    def lookup(self, query_vec: np.ndarray, scope: str = "") -> CacheMatch | None:
        with self._lock:
            self.stats.lookups += 1
            self._expire()
            if not self._entries:
                self.stats.misses += 1
                return None
            sims = self._mat @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
            if scope is not None:
                mask = np.array([e.scope == scope for e in self._entries])
                sims = np.where(mask, sims, -np.inf)
            best = int(np.argmax(sims))
            score = float(sims[best])
            if not np.isfinite(score) or score < self.threshold:
                self.stats.misses += 1
                return None
            entry = self._entries[best]
            entry.hits += 1
            entry.last_hit_at = self._clock()
            self.stats.hits += 1
            return CacheMatch(entry, score)

    def store(self, query: str, query_vec: np.ndarray, payload: dict,
              language: str = "und", scope: str = "") -> CacheEntry:
        with self._lock:
            now = self._clock()
            entry = CacheEntry(uuid.uuid4().hex[:12], query, language, scope, payload, now, now)
            self._entries.append(entry)
            self._mat = np.vstack([self._mat, np.asarray(query_vec, dtype=np.float32).reshape(1, -1)])
            while len(self._entries) > self.max_entries:
                lru = min(range(len(self._entries)), key=lambda i: self._entries[i].last_hit_at)
                self._drop([lru])
                self.stats.evictions += 1
            return entry

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            for i, e in enumerate(self._entries):
                if e.id == entry_id:
                    self._drop([i])
                    return True
            return False

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries = []
            self._mat = np.zeros((0, self.dim), dtype=np.float32)
            if n:
                self.stats.invalidations += n
            return n

    def entries(self) -> list[CacheEntry]:
        with self._lock:
            self._expire()
            return sorted(self._entries, key=lambda e: e.last_hit_at, reverse=True)

    # --------------------------------------------------------------- helpers
    def _drop(self, positions: list[int]) -> None:
        drop = set(positions)
        keep = [i for i in range(len(self._entries)) if i not in drop]
        self._entries = [self._entries[i] for i in keep]
        self._mat = self._mat[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def _expire(self) -> None:
        if not self.ttl_seconds or not self._entries:
            return
        cutoff = self._clock() - self.ttl_seconds
        stale = [i for i, e in enumerate(self._entries) if e.created_at < cutoff]
        if stale:
            self._drop(stale)
            self.stats.expirations += len(stale)

    # ----------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            meta = {"entries": [asdict(e) for e in self._entries], "stats": asdict(self.stats)}
            tmp = directory / "cache.json.tmp"
            tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            tmp.replace(directory / "cache.json")
            np.save(directory / "cache_vectors.npy", self._mat)

    def load(self, directory: Path) -> None:
        meta_path, vec_path = directory / "cache.json", directory / "cache_vectors.npy"
        if not (meta_path.is_file() and vec_path.is_file()):
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mat = np.load(vec_path, allow_pickle=False)
        entries = [CacheEntry(**e) for e in meta.get("entries", [])]
        if len(entries) != len(mat) or (len(mat) and mat.shape[1] != self.dim):
            return  # incompatible (e.g. embedding model changed) -> start empty
        with self._lock:
            self._entries, self._mat = entries, mat.astype(np.float32)
            self.stats = CacheStats(**meta.get("stats", {}))
