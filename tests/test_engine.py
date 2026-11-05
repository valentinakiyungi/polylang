import json
import threading

import pytest

from polylang import RAGEngine
from polylang.cli import main as cli_main
from polylang.llm import GenerationError


def test_query_cache_miss_then_hit(loaded):
    r1 = loaded.query("How does semantic caching reduce latency?")
    assert r1.status == "cache_miss"
    assert r1.sources[0].doc_id == "cache"
    assert "[1]" in r1.answer
    r2 = loaded.query("How does semantic caching reduce the latency?")
    assert r2.status == "cache_hit"
    assert r2.answer == r1.answer
    assert r2.cache["matched_query"] == r1.query
    assert loaded.stats()["cache"]["hits"] == 1


def test_bypass_cache(loaded):
    loaded.query("What is retrieval augmented generation?")
    assert loaded.query("What is retrieval augmented generation?", use_cache=False).status == "cache_miss"


def test_cache_scoped_by_collection_and_topk(loaded):
    loaded.query("Rust PyTorch inference")
    assert loaded.query("Rust PyTorch inference", top_k=1).status == "cache_miss"
    assert loaded.query("Rust PyTorch inference", collection="default").status == "cache_miss"


def test_ingest_invalidates_cache(loaded):
    loaded.query("What is retrieval augmented generation?")
    assert len(loaded.cache) == 1
    loaded.add_document("A brand new fact about Arusha.", doc_id="new")
    assert len(loaded.cache) == 0
    assert loaded.query("What is retrieval augmented generation?").status == "cache_miss"


def test_dedup_and_replace(engine):
    a = engine.add_document("Same text here.")
    b = engine.add_document("Same text here.")
    assert a.created and not b.created and a.document.id == b.document.id
    engine.add_document("Version one of the doc.", doc_id="d")
    engine.add_document("Version two of the doc, now with more words.", doc_id="d")
    assert len(engine.documents) == 2
    assert "Version two" in engine.get_document("d")["chunks"][0]["text"]
    assert len(engine.chunks) == 2


def test_collections_filter(engine):
    engine.add_document("Lions live in the Serengeti.", collection="wildlife")
    engine.add_document("Lions is also the name of a rugby team.", collection="sport")
    hits = engine.retrieve("lions", collection="wildlife")
    assert len(hits) == 1 and "Serengeti" in hits[0].text
    assert engine.retrieve("lions", collection="missing") == []
    assert len(engine.list_documents("sport")) == 1


def test_weighted_fusion(settings):
    settings.fusion = "weighted"
    eng = RAGEngine(settings)
    eng.add_document("Kilimanjaro is the highest mountain in Africa.")
    eng.add_document("Lake Victoria is the largest lake in Africa.")
    hits = eng.retrieve("highest mountain")
    assert "Kilimanjaro" in hits[0].text and 0 <= hits[0].score <= 1


def test_empty_inputs(engine):
    with pytest.raises(ValueError):
        engine.add_document("   ")
    with pytest.raises(ValueError):
        engine.query("")
    r = engine.query("anything at all")
    assert r.status == "no_context" and r.sources == []
    assert engine.query("Je, kuna nini?").answer.startswith("Sikupata")


def test_delete_and_reset(loaded):
    assert loaded.delete_document("rust")
    assert not loaded.delete_document("rust")
    assert all(h.doc_id != "rust" for h in loaded.retrieve("Rust PyTorch"))
    loaded.reset()
    assert loaded.stats()["documents"] == 0 and len(loaded.vectors) == 0 and len(loaded.bm25) == 0


def test_persistence_roundtrip(settings, loaded):
    loaded.query("What is retrieval augmented generation?")
    loaded.close()
    again = RAGEngine(settings)
    assert set(again.documents) == {"rust", "rag", "cache", "sw"}
    assert len(again.vectors) == len(again.chunks)
    assert again.query("What is retrieval augmented generation?").status == "cache_hit"


def test_model_change_triggers_reembed(settings, loaded):
    loaded.close()
    settings.hashing_dim = 256
    again = RAGEngine(settings)
    assert again.vectors.dim == 256 and len(again.vectors) == len(again.chunks)
    assert len(again.cache) == 0
    assert again.retrieve("Rust PyTorch")[0].doc_id in {"rust", "sw"}


def test_llm_failure_falls_back(settings):
    class Broken:
        name = "broken"

        def generate(self, *a):
            raise GenerationError("boom")

    eng = RAGEngine(settings, generator=Broken())
    eng.add_document("Dar es Salaam is the largest city in Tanzania.")
    r = eng.query("largest city in Tanzania")
    assert r.status == "cache_miss" and r.warnings and "Dar es Salaam" in r.answer
    assert r.generator == "extractive"
    assert len(eng.cache) == 0  # failures are not cached


def test_llm_receives_language_and_passages(settings):
    seen = {}

    class Spy:
        name = "spy"

        def generate(self, question, language, passages):
            seen.update(q=question, lang=language, n=len(passages))
            return "jibu [1]"

    eng = RAGEngine(settings, generator=Spy())
    eng.add_document("Semantic caching hupunguza muda wa kusubiri.")
    r = eng.query("Je, semantic caching hufanya nini?")
    assert r.answer == "jibu [1]" and seen["lang"] == "sw" and seen["n"] == 1


def test_large_document_chunks(engine):
    text = " ".join(f"Paragraph {i} describes the Serengeti migration in detail." for i in range(300))
    res = engine.add_document(text, doc_id="big")
    assert len(res.document.chunk_ids) > 5
    assert len(engine.vectors) == len(res.document.chunk_ids)


def test_concurrent_ingest_and_query(engine):
    errors = []

    def work(i):
        try:
            engine.add_document(f"Document number {i} about topic {i}.", doc_id=f"d{i}")
            engine.query(f"topic {i}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(engine.documents) == 16 and len(engine.vectors) == len(engine.chunks)


def test_cli(tmp_path, capsys):
    kb = tmp_path / "kb.jsonl"
    kb.write_text(json.dumps({"id": "k1", "title": "Kili", "text": "Kilimanjaro is 5895 metres tall."}) + "\n")
    base = ["--offline", "--data-dir", str(tmp_path / "s")]
    assert cli_main(base + ["ingest", str(kb)]) == 0
    assert cli_main(base + ["query", "How", "tall", "is", "Kilimanjaro?"]) == 0
    assert "5895" in capsys.readouterr().out
    assert cli_main(base + ["query", "How tall is Kilimanjaro?", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "cache_hit"
    assert cli_main(base + ["list"]) == 0
    assert cli_main(base + ["stats"]) == 0
    assert cli_main(base + ["clear-cache"]) == 0
    assert cli_main(base + ["delete", "k1"]) == 0
    assert cli_main(base + ["delete", "k1"]) == 1
    assert cli_main(base + ["ingest", str(tmp_path / "missing.txt")]) == 2


def test_runtime_settings_update_and_persist(settings, loaded):
    loaded.query("What is retrieval augmented generation?")
    cfg = loaded.update_settings({"cache_threshold": 0.95})
    assert cfg["editable"]["cache_threshold"] == 0.95 and loaded.cache.threshold == 0.95
    assert len(loaded.cache) == 1  # threshold change keeps cached answers
    loaded.update_settings({"top_k": 2, "fusion": "weighted"})
    assert len(loaded.cache) == 0  # retrieval change drops them
    assert len(loaded.retrieve("Rust")) <= 2
    loaded.close()
    again = RAGEngine(settings.__class__.from_env(embedding_backend="hashing", data_dir=settings.data_dir))
    assert again.settings.top_k == 2 and again.settings.fusion == "weighted"
    assert again.cache.threshold == 0.95


@pytest.mark.parametrize("bad", [
    {"chunk_size": 10}, {"top_k": "5"}, {"top_k": 0}, {"top_k": 99},
    {"fusion": "magic"}, {"cache_threshold": 1.5}, {"cache_enabled": "yes"},
])
def test_runtime_settings_rejects_bad_values(loaded, bad):
    before = loaded.config()["editable"]
    with pytest.raises(ValueError):
        loaded.update_settings(bad)
    assert loaded.config()["editable"] == before
