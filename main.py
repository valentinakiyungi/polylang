"""Backwards-compatible entry point so ``uvicorn main:app`` keeps working.

Prefer ``rag-engine serve`` or ``uvicorn rag_engine.server:app``.
"""

from rag_engine.server import app  # noqa: F401

if __name__ == "__main__":
    import sys

    from rag_engine.cli import main

    sys.exit(main(["serve", *sys.argv[1:]]))
