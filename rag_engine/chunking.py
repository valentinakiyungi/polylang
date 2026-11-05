"""Split documents into overlapping, sentence-aligned chunks."""

from __future__ import annotations

import re

from rag_engine.text import split_sentences


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Greedy sentence packing up to ``chunk_size`` characters, carrying ``overlap``
    characters of trailing context into the next chunk. Over-long sentences are
    hard-wrapped on whitespace."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    for sent in split_sentences(text):
        while len(sent) > chunk_size:
            cut = sent.rfind(" ", 0, chunk_size)
            cut = cut if cut > chunk_size // 2 else chunk_size
            pieces.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            pieces.append(sent)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]
            current = f"{tail} {piece}".strip() if tail else piece
            if len(current) > chunk_size:
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks
