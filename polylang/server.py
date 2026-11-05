"""ASGI entry point: ``uvicorn polylang.server:app``."""

import logging
import os

from polylang.api import create_app

logging.basicConfig(level=os.environ.get("POLYLANG_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()
