from __future__ import annotations

import re
from pathlib import Path

import fitz  # pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    for page in doc:
        pages.append(page.get_text())

    doc.close()
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Basic cleaning: remove excessive whitespace and non-printable characters."""
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(pdf_path: str | Path) -> str:
    """Full pipeline: extract + clean text from a PDF."""
    raw = extract_text_from_pdf(pdf_path)
    return clean_text(raw)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.parse_pdf <path_to_pdf>")
        sys.exit(1)

    text = load_pdf(sys.argv[1])
    print(text[:2000])  # Preview first 2000 chars