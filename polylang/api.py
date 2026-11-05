"""FastAPI application exposing the engine over HTTP, plus the web UI."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from polylang import __version__
from polylang.config import Settings
from polylang.engine import RAGEngine
from polylang.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------- schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, examples=["Je, semantic caching hufanyaje kazi?"])
    top_k: int | None = Field(None, ge=1, le=50)
    collection: str | None = None
    use_cache: bool = True
    language: str | None = Field(None, description="Force answer language (ISO-639-1); auto-detected if omitted")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(None, ge=1, le=50)
    collection: str | None = None


class DocumentIn(BaseModel):
    text: str = Field(..., min_length=1)
    title: str | None = None
    id: str | None = Field(None, max_length=128, pattern=r"^[A-Za-z0-9._\-]+$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    language: str | None = None


class SettingsPatch(BaseModel):
    top_k: int | None = Field(None, ge=1, le=50)
    fusion: str | None = Field(None, pattern=r"^(rrf|weighted)$")
    dense_weight: float | None = Field(None, ge=0, le=1)
    cache_enabled: bool | None = None
    cache_threshold: float | None = Field(None, gt=0, le=1)
    cache_cross_lingual: bool | None = None


class DocumentsIn(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=1000)
    collection: str = "default"


# -------------------------------------------------------------------- app
def create_app(engine: RAGEngine | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (engine.settings if engine else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or RAGEngine(settings)
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(
        title="Polylang",
        version=__version__,
        description="Multilingual RAG with dense + BM25 hybrid retrieval and a cross-lingual semantic cache.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])

    def get_engine(request: Request) -> RAGEngine:
        return request.app.state.engine

    def require_key(x_api_key: str | None = Header(None)) -> None:
        if settings.api_key and not (x_api_key and hmac.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # ---- meta
    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", tags=["meta"])
    def health(eng: RAGEngine = Depends(get_engine)):
        return {"status": "ok", "version": __version__, "documents": len(eng.documents),
                "auth_required": bool(settings.api_key)}

    @app.get("/stats", tags=["meta"], dependencies=[Depends(require_key)])
    def stats(eng: RAGEngine = Depends(get_engine)):
        return eng.stats()

    @app.get("/config", tags=["meta"], dependencies=[Depends(require_key)])
    def get_config(eng: RAGEngine = Depends(get_engine)):
        return eng.config()

    @app.patch("/config", tags=["meta"], dependencies=[Depends(require_key)])
    def patch_config(body: SettingsPatch, eng: RAGEngine = Depends(get_engine)):
        return eng.update_settings(body.model_dump(exclude_none=True))

    # ---- querying
    @app.post("/query", tags=["query"], dependencies=[Depends(require_key)])
    def query(req: QueryRequest, eng: RAGEngine = Depends(get_engine)):
        return eng.query(req.query, req.top_k, req.collection, req.use_cache, req.language).to_dict()

    @app.post("/retrieve", tags=["query"], dependencies=[Depends(require_key)])
    def retrieve(req: RetrieveRequest, eng: RAGEngine = Depends(get_engine)):
        from dataclasses import asdict
        return {"query": req.query, "results": [asdict(h) for h in eng.retrieve(req.query, req.top_k, req.collection)]}

    # ---- documents
    @app.post("/documents", tags=["documents"], status_code=201, dependencies=[Depends(require_key)])
    def add_documents(body: DocumentsIn, eng: RAGEngine = Depends(get_engine)):
        results = [eng.add_document(d.text, d.title, d.metadata, d.id, body.collection, d.language)
                   for d in body.documents]
        return {"ingested": [r.to_dict() for r in results]}

    @app.post("/documents/upload", tags=["documents"], status_code=201, dependencies=[Depends(require_key)])
    async def upload(files: list[UploadFile] = File(...), collection: str = Form("default"),
                     eng: RAGEngine = Depends(get_engine)):
        limit = settings.max_upload_mb * 1024 * 1024
        out, errors = [], []
        for f in files:
            data = await f.read(limit + 1)
            if len(data) > limit:
                errors.append({"file": f.filename, "error": f"File exceeds {settings.max_upload_mb} MB"})
                continue
            try:
                # Embedding is CPU-bound: keep it off the event loop.
                from starlette.concurrency import run_in_threadpool
                results = await run_in_threadpool(eng.ingest_file, data, f.filename or "upload.txt", collection)
                out.extend(r.to_dict() for r in results)
            except (UnsupportedFileError, ValueError) as exc:
                errors.append({"file": f.filename, "error": str(exc)})
            except Exception as exc:  # corrupt PDFs etc.
                log.exception("Failed to ingest %s", f.filename)
                errors.append({"file": f.filename, "error": f"Could not parse file: {exc}"})
        if not out and errors:
            raise HTTPException(status_code=400, detail=errors)
        return {"ingested": out, "errors": errors}

    @app.get("/documents", tags=["documents"], dependencies=[Depends(require_key)])
    def list_documents(collection: str | None = Query(None), eng: RAGEngine = Depends(get_engine)):
        docs = eng.list_documents(collection)
        return {"count": len(docs), "documents": docs}

    @app.get("/documents/{doc_id}", tags=["documents"], dependencies=[Depends(require_key)])
    def get_document(doc_id: str, eng: RAGEngine = Depends(get_engine)):
        doc = eng.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc

    @app.delete("/documents/{doc_id}", tags=["documents"], dependencies=[Depends(require_key)])
    def delete_document(doc_id: str, eng: RAGEngine = Depends(get_engine)):
        if not eng.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": doc_id}

    @app.delete("/documents", tags=["documents"], dependencies=[Depends(require_key)])
    def reset(confirm: bool = Query(False), eng: RAGEngine = Depends(get_engine)):
        if not confirm:
            raise HTTPException(status_code=400, detail="Pass ?confirm=true to delete every document")
        n = len(eng.documents)
        eng.reset()
        return {"deleted_documents": n}

    # ---- cache
    @app.get("/cache", tags=["cache"], dependencies=[Depends(require_key)])
    def cache_entries(limit: int = Query(50, ge=1, le=1000), eng: RAGEngine = Depends(get_engine)):
        entries = eng.cache.entries()[:limit]
        return {"count": len(eng.cache), "entries": [
            {"id": e.id, "query": e.query, "language": e.language, "hits": e.hits,
             "created_at": e.created_at, "last_hit_at": e.last_hit_at,
             "answer_preview": e.payload.get("answer", "")[:200]} for e in entries]}

    @app.delete("/cache", tags=["cache"], dependencies=[Depends(require_key)])
    def clear_cache(eng: RAGEngine = Depends(get_engine)):
        n = eng.cache.clear()
        eng._cache_dirty += 1
        eng.flush_cache()
        return {"cleared": n}

    @app.delete("/cache/{entry_id}", tags=["cache"], dependencies=[Depends(require_key)])
    def delete_cache_entry(entry_id: str, eng: RAGEngine = Depends(get_engine)):
        if not eng.cache.delete(entry_id):
            raise HTTPException(status_code=404, detail="Cache entry not found")
        eng._cache_dirty += 1
        return {"deleted": entry_id}

    @app.get("/meta/supported-files", tags=["meta"])
    def supported_files():
        return {"extensions": sorted(SUPPORTED_EXTENSIONS)}

    return app
