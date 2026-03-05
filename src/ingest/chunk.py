from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextChunk:
    text: str
    source: str        # filename or paper title
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text
        source: Identifier for the source document (e.g. filename)
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (period + space)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1  # include the period

        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append(TextChunk(
                text=chunk_text_str,
                source=source,
                chunk_index=index,
                start_char=start,
                end_char=end,
            ))
            index += 1

        start = end - overlap  # overlap with previous chunk

    return chunks


def chunks_from_pdf_text(text: str, filename: str) -> list[TextChunk]:
    """Convenience wrapper: chunk already-extracted PDF text."""
    return chunk_text(text, source=filename)