import pytest

from rag_engine import RAGEngine, Settings


@pytest.fixture
def settings(tmp_path):
    return Settings.from_env(embedding_backend="hashing", data_dir=str(tmp_path / "store"),
                             llm_provider="extractive", cache_threshold=0.9)


@pytest.fixture
def engine(settings):
    eng = RAGEngine(settings)
    yield eng
    eng.close()


SAMPLE = [
    ("rust", "High-throughput AI systems use Rust and PyTorch for low-latency inference. "
             "Rust handles request routing and memory-safe concurrency."),
    ("rag", "Retrieval-Augmented Generation combines search with a large language model. "
            "Relevant passages are retrieved and given to the model as context."),
    ("cache", "Semantic caching reduces latency by reusing answers to questions that mean the same thing. "
              "A cosine similarity threshold decides whether a cached answer is reused."),
    ("sw", "Mifumo ya akili bandia hutumia Rust na PyTorch kwa uchakataji wa haraka."),
]


@pytest.fixture
def loaded(engine):
    for doc_id, text in SAMPLE:
        engine.add_document(text, title=doc_id, doc_id=doc_id)
    return engine
