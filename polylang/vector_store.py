"""In-memory dense vector index (cosine similarity over L2-normalised vectors) with persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self.dim}")
        self.remove([i for i in ids if i in self._pos])
        start = len(self._ids)
        self._ids.extend(ids)
        for offset, cid in enumerate(ids):
            self._pos[cid] = start + offset
        self._mat = np.vstack([self._mat, vectors])

    def remove(self, ids: list[str]) -> None:
        drop = {self._pos[i] for i in ids if i in self._pos}
        if not drop:
            return
        keep = [p for p in range(len(self._ids)) if p not in drop]
        self._ids = [self._ids[p] for p in keep]
        self._mat = self._mat[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self._pos = {cid: i for i, cid in enumerate(self._ids)}

    def search(self, query_vec: np.ndarray, top_k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        scores = self._mat @ np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if allowed is not None:
            mask = np.array([cid in allowed for cid in self._ids])
            scores = np.where(mask, scores, -np.inf)
        k = min(top_k, len(self._ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self._ids[i], float(scores[i])) for i in idx if np.isfinite(scores[i])]

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(self._ids, dtype=str), mat=self._mat, dim=self.dim)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        data = np.load(path, allow_pickle=False)
        idx = cls(int(data["dim"]))
        ids = [str(i) for i in data["ids"].tolist()]
        if ids:
            idx.add(ids, data["mat"])
        return idx
