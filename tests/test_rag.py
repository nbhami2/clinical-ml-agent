import numpy as np
from src.rag.vector_store import VectorStore
from src.rag.retrieve import format_retrieved_context, get_unique_sources


def _make_store() -> VectorStore:
    """Helper: create a small vector store with fake embeddings."""
    store = VectorStore(dimension=4)
    fake_chunks = [
        {"text": "Sepsis prediction using XGBoost.", "source": "paper1.pdf", "chunk_index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
        {"text": "LSTM model for readmission.", "source": "paper2.pdf", "chunk_index": 0, "embedding": [0.9, 0.8, 0.7, 0.6]},
        {"text": "Logistic regression baseline.", "source": "paper1.pdf", "chunk_index": 1, "embedding": [0.2, 0.2, 0.3, 0.3]},
    ]
    store.add(fake_chunks)
    return store


def test_vector_store_add():
    """Store should contain correct number of chunks after adding."""
    store = _make_store()
    assert len(store) == 3


def test_vector_store_search():
    """Search should return top-k results."""
    store = _make_store()
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    results = store.search(query[0].tolist(), top_k=2)
    assert len(results) == 2
    assert "text" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]


def test_format_retrieved_context():
    """Formatted context should include source labels."""
    results = [
        {"text": "Some finding.", "source": "paper1.pdf", "chunk_index": 0, "score": 0.1},
    ]
    context = format_retrieved_context(results)
    assert "Source 1" in context
    assert "paper1.pdf" in context
    assert "Some finding." in context


def test_get_unique_sources():
    """Should deduplicate sources correctly."""
    results = [
        {"text": "A", "source": "paper1.pdf", "chunk_index": 0, "score": 0.1},
        {"text": "B", "source": "paper1.pdf", "chunk_index": 1, "score": 0.2},
        {"text": "C", "source": "paper2.pdf", "chunk_index": 0, "score": 0.3},
    ]
    sources = get_unique_sources(results)
    assert sources == ["paper1.pdf", "paper2.pdf"]