import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from rag_engine import RAGEngine  # noqa: E402
from rag_engine.api import create_app  # noqa: E402


@pytest.fixture
def client(settings):
    with TestClient(create_app(RAGEngine(settings))) as c:
        yield c


@pytest.fixture
def secured(settings):
    settings.api_key = "s3cret"
    with TestClient(create_app(RAGEngine(settings))) as c:
        yield c


def test_ui_and_health(client):
    assert "Cross-Lingual RAG" in client.get("/").text
    assert client.get("/health").json()["status"] == "ok"


def test_document_lifecycle_and_query(client):
    r = client.post("/documents", json={"collection": "geo", "documents": [
        {"id": "kili", "title": "Kilimanjaro", "text": "Kilimanjaro is the highest mountain in Africa."},
        {"text": "Lake Victoria is the largest lake in Africa."},
    ]})
    assert r.status_code == 201 and len(r.json()["ingested"]) == 2

    assert client.get("/documents").json()["count"] == 2
    assert client.get("/documents/kili").json()["chunks"][0]["text"].startswith("Kilimanjaro")
    assert client.get("/documents/nope").status_code == 404

    q = client.post("/query", json={"query": "highest mountain in Africa", "collection": "geo"}).json()
    assert q["status"] == "cache_miss" and q["sources"][0]["doc_id"] == "kili"
    q2 = client.post("/query", json={"query": "highest mountain in Africa", "collection": "geo"}).json()
    assert q2["status"] == "cache_hit"

    assert client.get("/cache").json()["count"] == 1
    assert client.get("/stats").json()["cache"]["hits"] == 1
    assert client.post("/retrieve", json={"query": "largest lake"}).json()["results"][0]["text"].startswith("Lake")

    assert client.delete("/documents/kili").json() == {"deleted": "kili"}
    assert client.delete("/documents").status_code == 400
    assert client.delete("/documents?confirm=true").json()["deleted_documents"] == 1


def test_upload(client):
    files = [("files", ("notes.md", b"# Notes\nArusha is near Mount Meru.", "text/markdown")),
             ("files", ("bad.exe", b"MZ", "application/octet-stream"))]
    r = client.post("/documents/upload", files=files, data={"collection": "notes"})
    assert r.status_code == 201
    body = r.json()
    assert len(body["ingested"]) == 1 and body["errors"][0]["file"] == "bad.exe"
    r = client.post("/documents/upload", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400


def test_validation(client):
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/documents", json={"documents": [{"text": "x", "id": "bad id!"}]}).status_code == 422


def test_cache_endpoints(client):
    client.post("/documents", json={"documents": [{"text": "Serengeti hosts the great migration."}]})
    client.post("/query", json={"query": "great migration"})
    entry = client.get("/cache").json()["entries"][0]
    assert client.delete(f"/cache/{entry['id']}").json() == {"deleted": entry["id"]}
    assert client.delete("/cache/missing").status_code == 404
    client.post("/query", json={"query": "great migration"})
    assert client.delete("/cache").json()["cleared"] == 1


def test_api_key(secured):
    assert secured.get("/health").status_code == 200
    assert secured.get("/").status_code == 200
    assert secured.get("/stats").status_code == 401
    assert secured.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get("/stats", headers={"X-API-Key": "s3cret"}).status_code == 200
