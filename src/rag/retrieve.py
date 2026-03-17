from __future__ import annotations

from src.rag.embed import embed_text
from src.rag.vector_store import VectorStore


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Args:
        query: Natural language research question
        store: Populated VectorStore instance
        top_k: Number of chunks to return

    Returns:
        List of dicts with keys: text, source, chunk_index, score
    """
    query_embedding = embed_text(query)
    results = store.search(query_embedding, top_k=top_k)
    return results


def format_retrieved_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for LLM input.
    Includes source citation for each chunk.
    """
    parts = []
    for i, result in enumerate(results):
        parts.append(
            f"[Source {i + 1}: {result['source']}]\n{result['text']}"
        )
    return "\n\n---\n\n".join(parts)


def get_unique_sources(results: list[dict]) -> list[str]:
    """Return deduplicated list of source filenames from results."""
    seen = set()
    sources = []
    for r in results:
        if r["source"] not in seen:
            seen.add(r["source"])
            sources.append(r["source"])
    return sources