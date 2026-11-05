import json

import numpy as np
import pytest

from polylang.bm25 import BM25Index, minmax
from polylang.cache import SemanticCache
from polylang.chunking import chunk_text
from polylang.config import Settings
from polylang.embeddings import HashingEmbedder
from polylang.llm import ExtractiveGenerator, Passage
from polylang.loaders import UnsupportedFileError, load_bytes
from polylang.text import detect_language, tokenize
from polylang.vector_store import VectorIndex


@pytest.mark.parametrize("text,lang", [
    ("How does semantic caching work?", "en"),
    ("Je, mfumo huu hutumia nini?", "sw"),
    ("Watoto wanacheza mpira", "sw"),
    ("Как работает семантический кэш?", "ru"),
    ("Як працює кеш?", "uk"),
    ("Comment fonctionne le cache sémantique ?", "fr"),
    ("¿Cómo funciona la caché?", "es"),
    ("Wie funktioniert das und was ist der Vorteil?", "de"),
    ("语义缓存如何工作", "zh"),
    ("machine learning human", "en"),
    ("12345", "und"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_tokenize_unicode():
    assert tokenize("Hello, Мир! Habari-yako") == ["hello", "мир", "habari", "yako"]
    assert tokenize("语义缓存") == ["语义", "义缓", "缓存"]


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share some text
    assert any(c1[-20:].split()[-1] in c2 for c1, c2 in zip(chunks, chunks[1:], strict=False))
    assert chunk_text("   ") == []
    assert chunk_text("short") == ["short"]


def test_chunking_hard_wraps_long_sentence():
    chunks = chunk_text("word " * 500, chunk_size=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)


def test_hashing_embedder_normalised_and_similar():
    e = HashingEmbedder(512)
    v = e.embed(["semantic caching", "semantic caches", "volcano eruption"])
    assert v.shape == (3, 512)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert e.embed([]).shape == (0, 512)


def test_bm25_add_remove_and_persist():
    idx = BM25Index()
    idx.add("a", "rust pytorch inference")
    idx.add("b", "semantic cache latency")
    assert idx.search("rust", 5)[0][0] == "a"
    assert idx.search("cache", 5, allowed={"a"}) == []
    restored = BM25Index.from_dict(json.loads(json.dumps(idx.to_dict())))
    assert restored.search("latency", 5)[0][0] == "b"
    idx.remove("a")
    assert idx.search("rust", 5) == []
    assert len(idx) == 1


def test_minmax():
    assert minmax(np.array([])).size == 0
    assert list(minmax(np.array([2.0, 4.0]))) == [0.0, 1.0]
    assert list(minmax(np.array([3.0, 3.0]))) == [1.0, 1.0]


def test_vector_index(tmp_path):
    e = HashingEmbedder(128)
    idx = VectorIndex(128)
    idx.add(["x", "y"], e.embed(["alpha beta", "gamma delta"]))
    assert idx.search(e.embed(["alpha beta"])[0], 1)[0][0] == "x"
    idx.add(["x"], e.embed(["gamma delta"]))  # replace
    assert len(idx) == 2
    idx.save(tmp_path / "v.npz")
    loaded = VectorIndex.load(tmp_path / "v.npz")
    assert len(loaded) == 2
    loaded.remove(["x", "y"])
    assert loaded.search(e.embed(["a"])[0], 3) == []


def test_cache_hit_miss_scope_eviction_ttl(tmp_path):
    now = [1000.0]
    c = SemanticCache(4, threshold=0.9, max_entries=2, ttl_seconds=100, clock=lambda: now[0])
    v1, v2, v3 = np.eye(4, dtype=np.float32)[:3]
    c.store("q1", v1, {"answer": "a1"}, scope="s")
    assert c.lookup(v1, "s").entry.payload["answer"] == "a1"
    assert c.lookup(v1, "other") is None           # scoped
    assert c.lookup(v2, "s") is None               # dissimilar
    now[0] += 1
    c.store("q2", v2, {"answer": "a2"}, scope="s")
    now[0] += 1
    c.lookup(v1, "s")                               # q1 recently used
    now[0] += 1
    c.store("q3", v3, {"answer": "a3"}, scope="s")  # evicts LRU = q2
    assert {e.query for e in c.entries()} == {"q1", "q3"}
    assert c.stats.evictions == 1

    c.save(tmp_path)
    c2 = SemanticCache(4, clock=lambda: now[0])
    c2.load(tmp_path)
    assert len(c2) == 2

    now[0] += 1000                                  # TTL expiry
    assert c.lookup(v1, "s") is None
    assert len(c) == 0 and c.stats.expirations == 2


def test_cache_ignores_incompatible_persisted_dim(tmp_path):
    c = SemanticCache(4)
    c.store("q", np.ones(4, dtype=np.float32) / 2, {"answer": "x"})
    c.save(tmp_path)
    other = SemanticCache(8)
    other.load(tmp_path)
    assert len(other) == 0


def test_extractive_generator_cites_sources():
    gen = ExtractiveGenerator(HashingEmbedder(256))
    out = gen.generate("what reduces latency", "en", [
        Passage("Semantic caching reduces latency a lot. Volcanoes erupt rarely.", "t1", 1.0),
        Passage("Unrelated text about gardening and tomatoes in summer.", "t2", 0.5),
    ])
    assert "Semantic caching reduces latency" in out and "[1]" in out


def test_loaders_text_json_html_csv():
    assert load_bytes("Привет мир".encode("cp1251"), "a.txt")[0].text == "Привет мир"
    docs = load_bytes(b'[{"id": "1", "text": "hello"}, "plain string"]', "k.json")
    assert [d.id for d in docs] == ["1", None] and docs[1].text == "plain string"
    docs = load_bytes(b'{"text": "a"}\n\n{"text": "b", "title": "B"}\n', "k.jsonl")
    assert [d.title for d in docs] == ["k #1", "B"]
    html = load_bytes(b"<html><script>x()</script><p>Hello</p><p>World</p></html>", "p.html")[0].text
    assert "Hello" in html and "x()" not in html
    assert "a, b" in load_bytes(b"a,b\nc,d", "t.csv")[0].text
    with pytest.raises(UnsupportedFileError):
        load_bytes(b"x", "file.exe")
    with pytest.raises(UnsupportedFileError):
        load_bytes(b'[{"title": "no text"}]', "bad.json")


def test_loaders_docx():
    docx = pytest.importorskip("docx")
    import io
    d = docx.Document()
    d.add_paragraph("Habari ya dunia")
    buf = io.BytesIO()
    d.save(buf)
    assert load_bytes(buf.getvalue(), "x.docx")[0].text == "Habari ya dunia"


def test_settings_validation(monkeypatch):
    monkeypatch.setenv("POLYLANG_TOP_K", "7")
    monkeypatch.setenv("POLYLANG_CACHE_ENABLED", "false")
    monkeypatch.setenv("POLYLANG_CORS_ORIGINS", "http://a.com, http://b.com")
    s = Settings.from_env(embedding_backend="hashing")
    assert s.top_k == 7 and s.cache_enabled is False and s.cors_origins == ["http://a.com", "http://b.com"]
    with pytest.raises(ValueError):
        Settings.from_env(chunk_overlap=10_000)
    with pytest.raises(ValueError):
        Settings.from_env(fusion="magic")
    with pytest.raises(AttributeError):
        Settings.from_env(nope=1)
