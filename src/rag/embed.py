from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.ingest.chunk import TextChunk

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "text-embedding-004"


def embed_text(text: str) -> list[float]:
    """Embed a single string using Gemini embedding model."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


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