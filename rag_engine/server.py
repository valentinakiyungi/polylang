"""ASGI entry point: ``uvicorn rag_engine.server:app``."""

import logging
import os

from rag_engine.api import create_app

logging.basicConfig(level=os.environ.get("RAG_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()
