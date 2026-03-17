from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    Simple FAISS-backed vector store for text chunks.

    Stores:
        - FAISS index (for similarity search)
        - Metadata list (text, source, chunk_index per entry)
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata: list[dict] = []

    def add(self, embedded_chunks: list[dict]) -> None:
        """Add embedded chunks to the store."""
        vectors = np.array(
            [chunk["embedding"] for chunk in embedded_chunks],
            dtype=np.float32,
        )
        self.index.add(vectors)
        for chunk in embedded_chunks:
            self.metadata.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            })

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Search for the top-k most similar chunks.

        Returns list of dicts with keys: text, source, chunk_index, score
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(dist)
            results.append(result)

        return results

    def save(self, path: str | Path) -> None:
        """Save index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Vector store saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load index and metadata from disk."""
        path = Path(path)
        store = cls()
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        store.dimension = store.index.d
        print(f"Vector store loaded from {path} ({len(store.metadata)} chunks)")
        return store

    def __len__(self) -> int:
        return self.index.ntotal