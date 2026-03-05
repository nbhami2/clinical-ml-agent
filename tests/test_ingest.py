from src.ingest.chunk import chunk_text, TextChunk


def test_chunk_basic():
    """Should produce multiple chunks from long text."""
    text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(text, source="test_paper.pdf")
    assert len(chunks) > 1
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_overlap():
    """Each chunk should be within the expected size range."""
    text = "Word " * 1000
    chunks = chunk_text(text, source="test.pdf", chunk_size=500, overlap=100)
    for chunk in chunks:
        assert len(chunk.text) <= 600  # allow small overage for sentence boundaries


def test_chunk_source():
    """Source label should be preserved in all chunks."""
    text = "Some text content. " * 100
    chunks = chunk_text(text, source="my_paper.pdf")
    assert all(c.source == "my_paper.pdf" for c in chunks)


def test_chunk_empty_text():
    """Empty text should return no chunks."""
    chunks = chunk_text("", source="empty.pdf")
    assert chunks == []