from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from src.ingest.chunk import TextChunk

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "models/text-embedding-004"


def embed_text(text: str) -> list[float]:
    """Embed a single string using Gemini embedding model."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_chunks(chunks: list[TextChunk]) -> list[dict]:
    """
    Embed a list of TextChunks.

    Returns a list of dicts with keys:
        - text
        - source
        - chunk_index
        - embedding
    """
    embedded = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i + 1}/{len(chunks)} from {chunk.source}...")
        embedding = embed_text(chunk.text)
        embedded.append({
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "embedding": embedding,
        })
    return embedded