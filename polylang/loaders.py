"""Turn files into plain text. PDF and DOCX support use optional dependencies."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".csv",
                        ".json", ".jsonl", ".pdf", ".docx"}


class UnsupportedFileError(ValueError):
    pass


@dataclass
class LoadedDocument:
    text: str
    title: str
    metadata: dict = field(default_factory=dict)
    id: str | None = None


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1251")  # common for legacy Russian text files
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _strip_html(html: str) -> str:
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style"}:
                self._skip += 1
            elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in {"script", "style"} and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                self.out.append(data)

    p = _P()
    p.feed(html)
    return "".join(p.out)


def load_bytes(data: bytes, filename: str) -> list[LoadedDocument]:
    """Parse an uploaded file. JSON/JSONL may contain several documents."""
    ext = Path(filename).suffix.lower()
    title = Path(filename).stem
    meta = {"source": filename}

    if ext in {".txt", ".md", ".markdown", ".rst"}:
        return [LoadedDocument(_decode(data), title, meta)]
    if ext in {".html", ".htm"}:
        return [LoadedDocument(_strip_html(_decode(data)), title, meta)]
    if ext == ".csv":
        rows = list(csv.reader(io.StringIO(_decode(data))))
        text = "\n".join(", ".join(cell for cell in row if cell) for row in rows)
        return [LoadedDocument(text, title, meta)]
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise UnsupportedFileError("PDF support needs `pip install pypdf`") from exc
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return [LoadedDocument(text, title, {**meta, "pages": len(reader.pages)})]
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError as exc:
            raise UnsupportedFileError("DOCX support needs `pip install python-docx`") from exc
        d = docx.Document(io.BytesIO(data))
        text = "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
        return [LoadedDocument(text, title, meta)]
    if ext in {".json", ".jsonl"}:
        return _load_json(_decode(data), filename, ext)
    raise UnsupportedFileError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def _load_json(raw: str, filename: str, ext: str) -> list[LoadedDocument]:
    """Accepts a list of objects / JSONL lines with a ``text`` field (plus optional
    ``id``, ``title``, ``metadata``), or a single such object."""
    if ext == ".jsonl":
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("documents", [parsed])
        else:
            items = []
    docs: list[LoadedDocument] = []
    for n, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            raise UnsupportedFileError(f"{filename}: item {n} has no 'text' field")
        meta = dict(item.get("metadata") or {})
        meta.setdefault("source", filename)
        docs.append(LoadedDocument(
            text=str(item["text"]),
            title=str(item.get("title") or f"{Path(filename).stem} #{n + 1}"),
            metadata=meta,
            id=str(item["id"]) if item.get("id") is not None else None,
        ))
    return docs


def load_path(path: str | Path) -> list[LoadedDocument]:
    """Load a single file, or every supported file under a directory (recursively)."""
    path = Path(path)
    if path.is_dir():
        docs: list[LoadedDocument] = []
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                for d in load_bytes(p.read_bytes(), p.name):
                    d.metadata["source"] = str(p)
                    docs.append(d)
        return docs
    if not path.is_file():
        raise FileNotFoundError(path)
    docs = load_bytes(path.read_bytes(), path.name)
    for d in docs:
        d.metadata["source"] = str(path)
    return docs
