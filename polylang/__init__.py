"""Polylang: multilingual RAG with hybrid retrieval and a cross-lingual semantic cache."""

from polylang.config import Settings
from polylang.engine import QueryResult, RAGEngine

__all__ = ["RAGEngine", "QueryResult", "Settings"]
__version__ = "1.0.0"
