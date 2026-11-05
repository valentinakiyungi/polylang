"""Incremental BM25 (Okapi) index. Supports add/remove without a full rebuild."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from rag_engine.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter] = {}      # chunk_id -> term frequencies
        self._len: dict[str, int] = {}
        self._df: Counter = Counter()          # term -> number of chunks containing it
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._tf)

    def add(self, chunk_id: str, text: str) -> None:
        if chunk_id in self._tf:
            self.remove(chunk_id)
        tokens = tokenize(text)
        tf = Counter(tokens)
        self._tf[chunk_id] = tf
        self._len[chunk_id] = len(tokens)
        self._total_len += len(tokens)
        self._df.update(tf.keys())

    def remove(self, chunk_id: str) -> None:
        tf = self._tf.pop(chunk_id, None)
        if tf is None:
            return
        self._total_len -= self._len.pop(chunk_id)
        for term in tf:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def search(self, query: str, top_k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self._tf)
        if n == 0:
            return []
        q_terms = set(tokenize(query))
        if not q_terms:
            return []
        avgdl = self._total_len / n if n else 1.0
        idf = {t: math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5)) for t in q_terms if t in self._df}
        if not idf:
            return []
        results: list[tuple[str, float]] = []
        for cid, tf in self._tf.items():
            if allowed is not None and cid not in allowed:
                continue
            dl = self._len[cid]
            score = 0.0
            for t, w in idf.items():
                f = tf.get(t)
                if f:
                    score += w * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if score > 0:
                results.append((cid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "tf": {k: dict(v) for k, v in self._tf.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> BM25Index:
        idx = cls(data.get("k1", 1.5), data.get("b", 0.75))
        for cid, tf in data.get("tf", {}).items():
            c = Counter(tf)
            idx._tf[cid] = c
            length = int(sum(c.values()))
            idx._len[cid] = length
            idx._total_len += length
            idx._df.update(c.keys())
        return idx


def minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores) if hi > 0 else np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
