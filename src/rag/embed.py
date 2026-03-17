from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai

from src.ingest.chunk import TextChunk

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

EMBEDDING_MODEL = "models/gemini-embedding-001"
REQUESTS_PER_MINUTE = 80  # stay safely under the 100/min free tier limit
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # ~0.75 seconds


def embed_text(text: str) -> list[float]:
    """Embed a single string using Gemini embedding model."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def embed_chunks(chunks: list[TextChunk]) -> list[dict]:
    """
    Embed a list of TextChunks with rate limiting.

    Returns a list of dicts with keys:
        - text
        - source
        - chunk_index
        - embedding
    """
    embedded = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i + 1}/{total} from {chunk.source}...")
        embedding = embed_text(chunk.text)
        embedded.append({
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "embedding": embedding,
        })
        # Rate limiting: pause between requests to stay under free tier quota
        if i < total - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return embedded