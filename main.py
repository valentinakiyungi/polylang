"""Backwards-compatible entry point so ``uvicorn main:app`` keeps working.

Prefer ``polylang serve`` or ``uvicorn polylang.server:app``.
"""

from polylang.server import app  # noqa: F401

if __name__ == "__main__":
    import sys

    from polylang.cli import main

    sys.exit(main(["serve", *sys.argv[1:]]))
