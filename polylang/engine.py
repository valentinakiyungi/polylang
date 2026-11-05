"""RAGEngine: document ingestion, hybrid retrieval, semantic caching and generation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from polylang.bm25 import BM25Index, minmax
from polylang.cache import SemanticCache
from polylang.chunking import chunk_text
from polylang.config import Settings
from polylang.embeddings import Embedder, build_embedder
from polylang.llm import ExtractiveGenerator, GenerationError, Generator, Passage, build_generator, no_context_answer
from polylang.loaders import LoadedDocument, load_bytes, load_path
from polylang.text import detect_language
from polylang.vector_store import VectorIndex

log = logging.getLogger(__name__)
STORE_VERSION = 1

# Settings that can be changed while the server runs (from the web UI or PATCH /config).
# They are saved to <data_dir>/settings.json and take precedence over environment variables.
RUNTIME_SETTINGS: dict[str, type] = {
    "top_k": int,
    "fusion": str,
    "dense_weight": float,
    "cache_enabled": bool,
    "cache_threshold": float,
    "cache_cross_lingual": bool,
}
# Changing these changes which passages are retrieved, so cached answers are dropped.
_RETRIEVAL_KEYS = {"top_k", "fusion", "dense_weight", "cache_cross_lingual"}


@dataclass
class Document:
    id: str
    title: str
    language: str
    collection: str
    metadata: dict
    content_hash: str
    char_count: int
    chunk_ids: list[str]
    created_at: float


@dataclass
class Chunk:
    id: str
    doc_id: str
    position: int
    text: str
    language: str


@dataclass
class SourceHit:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    language: str
    text: str
    score: float
    dense_score: float | None
    sparse_score: float | None


@dataclass
class QueryResult:
    query: str
    language: str
    status: str                    # cache_hit | cache_miss | no_context
    answer: str
    sources: list[SourceHit] = field(default_factory=list)
    cache: dict | None = None      # details on hit
    generator: str = ""
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IngestResult:
    document: Document
    created: bool                  # False if an identical document already existed

    def to_dict(self) -> dict:
        d = asdict(self.document)
        d["chunk_count"] = len(d.pop("chunk_ids"))
        d["created"] = self.created
        return d


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RAGEngine:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 generator: Generator | None = None):
        self.settings = settings or Settings.from_env()
        if self.settings.persist:
            self._load_runtime_settings()
        self.embedder = embedder or build_embedder(
            self.settings.embedding_backend, self.settings.embedding_model,
            self.settings.embedding_device, self.settings.hashing_dim)
        self.generator = generator or build_generator(self.settings, self.embedder)
        self._fallback_generator = ExtractiveGenerator(self.embedder)

        self._lock = threading.RLock()
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.vectors = VectorIndex(self.embedder.dim)
        self.bm25 = BM25Index()
        self.cache = SemanticCache(self.embedder.dim, self.settings.cache_threshold,
                                   self.settings.cache_max_entries, self.settings.cache_ttl_seconds)
        self._cache_dirty = 0
        self.started_at = time.time()

        if self.settings.persist:
            self._load()

    # ================================================================ ingest
    def add_document(self, text: str, title: str | None = None, metadata: dict | None = None,
                     doc_id: str | None = None, collection: str = "default",
                     language: str | None = None) -> IngestResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("Document text is empty")
        content_hash = _hash(f"{collection}\n{text}")
        with self._lock:
            if doc_id is None:
                for d in self.documents.values():
                    if d.content_hash == content_hash:
                        return IngestResult(d, created=False)
            elif doc_id in self.documents:
                if self.documents[doc_id].content_hash == content_hash:
                    return IngestResult(self.documents[doc_id], created=False)
                self._remove_document(doc_id)

        doc_id = doc_id or uuid.uuid4().hex[:16]
        lang = language or detect_language(text)
        pieces = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        chunks = [Chunk(f"{doc_id}:{i}", doc_id, i, p, detect_language(p) if len(p) > 40 else lang)
                  for i, p in enumerate(pieces)]
        # Embedding is the slow part: do it outside the lock.
        vecs = self.embedder.embed([c.text for c in chunks])

        doc = Document(
            id=doc_id, title=(title or text[:60].replace("\n", " ")).strip(), language=lang,
            collection=collection or "default", metadata=dict(metadata or {}),
            content_hash=content_hash, char_count=len(text),
            chunk_ids=[c.id for c in chunks], created_at=time.time(),
        )
        with self._lock:
            if doc_id in self.documents:  # concurrent insert of the same id
                self._remove_document(doc_id)
            self.documents[doc_id] = doc
            for c in chunks:
                self.chunks[c.id] = c
                self.bm25.add(c.id, c.text)
            self.vectors.add([c.id for c in chunks], vecs)
            self._invalidate_cache()
            self._save_corpus()
        log.info("Ingested %s (%s, %d chunks)", doc.title, lang, len(chunks))
        return IngestResult(doc, created=True)

    def add_documents(self, docs: list[LoadedDocument], collection: str = "default") -> list[IngestResult]:
        return [self.add_document(d.text, d.title, d.metadata, d.id, collection) for d in docs]

    def ingest_file(self, data: bytes, filename: str, collection: str = "default") -> list[IngestResult]:
        return self.add_documents(load_bytes(data, filename), collection)

    def ingest_path(self, path: str | Path, collection: str = "default") -> list[IngestResult]:
        return self.add_documents(load_path(path), collection)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            self._remove_document(doc_id)
            self._invalidate_cache()
            self._save_corpus()
            return True

    def reset(self) -> None:
        """Remove every document and clear the cache."""
        with self._lock:
            self.documents.clear()
            self.chunks.clear()
            self.vectors = VectorIndex(self.embedder.dim)
            self.bm25 = BM25Index()
            self._invalidate_cache()
            self._save_corpus()

    def _remove_document(self, doc_id: str) -> None:
        doc = self.documents.pop(doc_id)
        self.vectors.remove(doc.chunk_ids)
        for cid in doc.chunk_ids:
            self.chunks.pop(cid, None)
            self.bm25.remove(cid)

    def _invalidate_cache(self) -> None:
        # Cached answers may be stale once the corpus changes.
        if len(self.cache):
            self.cache.clear()
            self._cache_dirty += 1
            self.flush_cache()

    # ============================================================== retrieve
    def retrieve(self, query: str, top_k: int | None = None, collection: str | None = None,
                 query_vec: np.ndarray | None = None) -> list[SourceHit]:
        top_k = top_k or self.settings.top_k
        if query_vec is None:
            query_vec = self.embedder.embed([query])[0]
        pool = max(top_k * 5, 20)
        with self._lock:
            allowed = None
            if collection:
                allowed = {cid for d in self.documents.values() if d.collection == collection
                           for cid in d.chunk_ids}
                if not allowed:
                    return []
            dense = self.vectors.search(query_vec, pool, allowed)
            sparse = self.bm25.search(query, pool, allowed)
            fused = self._fuse(dense, sparse)[:top_k]
            dense_map, sparse_map = dict(dense), dict(sparse)
            hits = []
            for rank, (cid, score) in enumerate(fused, 1):
                c = self.chunks[cid]
                d = self.documents[c.doc_id]
                hits.append(SourceHit(rank, cid, d.id, d.title, c.language, c.text, round(score, 6),
                                      _r(dense_map.get(cid)), _r(sparse_map.get(cid))))
            return hits

    def _fuse(self, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]) -> list[tuple[str, float]]:
        s = self.settings
        if s.fusion == "rrf":
            scores: dict[str, float] = {}
            for ranked, weight in ((dense, s.dense_weight), (sparse, 1 - s.dense_weight)):
                for rank, (cid, _) in enumerate(ranked, 1):
                    scores[cid] = scores.get(cid, 0.0) + 2 * weight / (s.rrf_k + rank)
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)

        ids = list(dict.fromkeys([c for c, _ in dense] + [c for c, _ in sparse]))
        if not ids:
            return []
        dmap, smap = dict(dense), dict(sparse)
        d = np.clip(np.array([dmap.get(i, 0.0) for i in ids]), 0, 1)
        sp = minmax(np.array([smap.get(i, 0.0) for i in ids]))
        combined = s.dense_weight * d + (1 - s.dense_weight) * sp
        out = [(i, float(v)) for i, v in zip(ids, combined, strict=True) if v >= s.min_score]
        return sorted(out, key=lambda x: x[1], reverse=True)

    # ================================================================= query
    def query(self, question: str, top_k: int | None = None, collection: str | None = None,
              use_cache: bool = True, language: str | None = None) -> QueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("Query is empty")
        t0 = time.perf_counter()
        lang = language or detect_language(question)
        top_k = top_k or self.settings.top_k
        qvec = self.embedder.embed([question])[0]
        timings = {"embed": _ms(t0)}
        scope = f"{collection or '*'}|k={top_k}"
        if not self.settings.cache_cross_lingual:
            scope += f"|lang={lang}"
        cache_on = use_cache and self.settings.cache_enabled

        if cache_on:
            t = time.perf_counter()
            match = self.cache.lookup(qvec, scope)
            timings["cache_lookup"] = _ms(t)
            if match:
                timings["total"] = _ms(t0)
                p = match.entry.payload
                return QueryResult(
                    query=question, language=lang, status="cache_hit", answer=p["answer"],
                    sources=[SourceHit(**s) for s in p["sources"]], generator=p.get("generator", ""),
                    cache={"entry_id": match.entry.id, "similarity": round(match.similarity, 4),
                           "matched_query": match.entry.query, "matched_language": match.entry.language,
                           "hits": match.entry.hits},
                    timings_ms=timings,
                )

        t = time.perf_counter()
        sources = self.retrieve(question, top_k, collection, qvec)
        timings["retrieve"] = _ms(t)
        if not sources:
            timings["total"] = _ms(t0)
            return QueryResult(question, lang, "no_context", no_context_answer(lang),
                               generator=self.generator.name, timings_ms=timings)

        passages = [Passage(s.text, s.title, s.score) for s in sources]
        warnings: list[str] = []
        generator_name = self.generator.name
        t = time.perf_counter()
        try:
            answer = self.generator.generate(question, lang, passages)
        except GenerationError as exc:
            log.warning("Generation failed, falling back to extractive: %s", exc)
            warnings.append(f"LLM generation failed, returned extractive answer instead: {exc}")
            answer = self._fallback_generator.generate(question, lang, passages)
            generator_name = self._fallback_generator.name
        timings["generate"] = _ms(t)

        result = QueryResult(question, lang, "cache_miss", answer, sources,
                             generator=generator_name, warnings=warnings)
        if cache_on and not warnings:
            self.cache.store(question, qvec, {"answer": answer, "generator": generator_name,
                                              "sources": [asdict(s) for s in sources]},
                             language=lang, scope=scope)
            self._cache_dirty += 1
            if self._cache_dirty >= 10:
                self.flush_cache()
        timings["total"] = _ms(t0)
        result.timings_ms = timings
        return result

    # ================================================================= stats
    def stats(self) -> dict:
        with self._lock:
            langs = Counter(d.language for d in self.documents.values())
            colls = Counter(d.collection for d in self.documents.values())
            cs = self.cache.stats
            return {
                "documents": len(self.documents),
                "chunks": len(self.chunks),
                "languages": dict(langs),
                "collections": dict(colls),
                "embedder": self.embedder.name,
                "generator": self.generator.name,
                "fusion": self.settings.fusion,
                "cache": {
                    "enabled": self.settings.cache_enabled,
                    "entries": len(self.cache),
                    "threshold": self.cache.threshold,
                    "lookups": cs.lookups, "hits": cs.hits, "misses": cs.misses,
                    "hit_rate": round(cs.hit_rate, 4),
                    "evictions": cs.evictions, "expirations": cs.expirations,
                    "invalidations": cs.invalidations,
                },
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }

    def list_documents(self, collection: str | None = None) -> list[dict]:
        with self._lock:
            out = []
            for d in sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True):
                if collection and d.collection != collection:
                    continue
                row = asdict(d)
                row["chunk_count"] = len(row.pop("chunk_ids"))
                out.append(row)
            return out

    def get_document(self, doc_id: str) -> dict | None:
        with self._lock:
            d = self.documents.get(doc_id)
            if d is None:
                return None
            row = asdict(d)
            row["chunks"] = [asdict(self.chunks[c]) for c in d.chunk_ids]
            row.pop("chunk_ids")
            return row

    # =========================================================== persistence
    @property
    def _dir(self) -> Path:
        return self.settings.data_path

    def _save_corpus(self) -> None:
        if not self.settings.persist:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        manifest = {"version": STORE_VERSION, "embedder": self.embedder.name, "dim": self.embedder.dim}
        corpus = {"documents": [asdict(d) for d in self.documents.values()],
                  "chunks": [asdict(c) for c in self.chunks.values()]}
        tmp = self._dir / "corpus.json.tmp"
        tmp.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._dir / "corpus.json")
        self.vectors.save(self._dir / "vectors.npz")
        (self._dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def flush_cache(self) -> None:
        if self.settings.persist and self._cache_dirty:
            self.cache.save(self._dir)
            self._cache_dirty = 0

    def close(self) -> None:
        with self._lock:
            self.flush_cache()

    # ======================================================== runtime config
    def config(self) -> dict:
        s = self.settings
        return {
            "editable": {k: getattr(s, k) for k in RUNTIME_SETTINGS},
            "server": {
                "embedder": self.embedder.name,
                "generator": self.generator.name,
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "cache_max_entries": s.cache_max_entries,
                "cache_ttl_seconds": s.cache_ttl_seconds,
                "max_upload_mb": s.max_upload_mb,
                "persist": s.persist,
            },
        }

    def update_settings(self, changes: dict) -> dict:
        """Validate and apply runtime setting changes; returns the new config."""
        unknown = set(changes) - set(RUNTIME_SETTINGS)
        if unknown:
            raise ValueError(f"These settings can't be changed at runtime: {', '.join(sorted(unknown))}")
        coerced = {}
        for key, value in changes.items():
            typ = RUNTIME_SETTINGS[key]
            if typ is bool and not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
            if typ is int and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{key} must be a whole number")
            if typ is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{key} must be a number")
            if typ is str and not isinstance(value, str):
                raise ValueError(f"{key} must be text")
            coerced[key] = typ(value)
        with self._lock:
            candidate = dataclasses.replace(self.settings, **coerced)
            candidate.validate()
            if candidate.top_k > 50:
                raise ValueError("top_k must be 50 or less")
            changed = {k for k, v in coerced.items() if getattr(self.settings, k) != v}
            for k in changed:
                setattr(self.settings, k, coerced[k])
            self.cache.threshold = self.settings.cache_threshold
            if changed & _RETRIEVAL_KEYS:
                self._invalidate_cache()
            self._save_runtime_settings()
            return self.config()

    def _save_runtime_settings(self) -> None:
        if not self.settings.persist:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self.settings, k) for k in RUNTIME_SETTINGS}
        tmp = self._dir / "settings.json.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._dir / "settings.json")

    def _load_runtime_settings(self) -> None:
        path = self._dir / "settings.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text("utf-8"))
            values = {k: RUNTIME_SETTINGS[k](v) for k, v in data.items() if k in RUNTIME_SETTINGS}
            candidate = dataclasses.replace(self.settings, **values)
            candidate.validate()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning("Ignoring invalid %s: %s", path, exc)
            return
        for k, v in values.items():
            setattr(self.settings, k, v)

    def _load(self) -> None:
        corpus_path = self._dir / "corpus.json"
        if not corpus_path.is_file():
            return
        manifest_path = self._dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
        corpus = json.loads(corpus_path.read_text("utf-8"))
        self.documents = {d["id"]: Document(**d) for d in corpus.get("documents", [])}
        self.chunks = {c["id"]: Chunk(**c) for c in corpus.get("chunks", [])}
        for c in self.chunks.values():
            self.bm25.add(c.id, c.text)

        same_model = manifest.get("embedder") == self.embedder.name and manifest.get("dim") == self.embedder.dim
        vec_path = self._dir / "vectors.npz"
        if same_model and vec_path.is_file():
            self.vectors = VectorIndex.load(vec_path)
        if set(self.vectors._pos) != set(self.chunks):
            log.warning("Vector index missing or built with a different model; re-embedding %d chunks",
                        len(self.chunks))
            self.vectors = VectorIndex(self.embedder.dim)
            ids = list(self.chunks)
            for i in range(0, len(ids), 256):
                batch = ids[i:i + 256]
                self.vectors.add(batch, self.embedder.embed([self.chunks[c].text for c in batch]))
            self._save_corpus()
        elif same_model:
            self.cache.load(self._dir)
        log.info("Loaded %d documents / %d chunks from %s", len(self.documents), len(self.chunks), self._dir)


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 2)


def _r(v: float | None) -> float | None:
    return None if v is None else round(float(v), 6)
