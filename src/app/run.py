from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from src.ingest.parse_pdf import load_pdf
from src.ingest.chunk import chunks_from_pdf_text
from src.rag.embed import embed_chunks, embed_text
from src.rag.vector_store import VectorStore
from src.rag.retrieve import retrieve, format_retrieved_context, get_unique_sources
from src.extract.extract_paper import extract_paper_safe
from src.extract.normalize import normalize_extraction
from src.synth.evidence_table import evidence_table_to_csv, summarize_evidence_table
from src.synth.report_writer import write_report, save_report


DATA_DIR = Path("data/raw")
VECTOR_STORE_DIR = Path("data/processed/vector_store")


def load_or_build_vector_store(pdf_paths: list[Path]) -> VectorStore:
    """
    Load vector store from disk if it exists, otherwise build it
    from all PDFs in data/raw/.
    """
    if VECTOR_STORE_DIR.exists():
        print("Loading existing vector store...")
        return VectorStore.load(VECTOR_STORE_DIR)

    print(f"Building vector store from {len(pdf_paths)} PDFs...")
    store = VectorStore()

    for pdf_path in pdf_paths:
        print(f"  Processing: {pdf_path.name}")
        text = load_pdf(pdf_path)
        chunks = chunks_from_pdf_text(text, filename=pdf_path.name)
        embedded = embed_chunks(chunks)
        store.add(embedded)

    store.save(VECTOR_STORE_DIR)
    return store


def run_pipeline(question: str) -> None:
    """Full pipeline: question -> retrieved papers -> extractions -> report."""

    # --- Setup run directory ---
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "run_id": run_id,
        "question": question,
        "steps": [],
    }

    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    # --- Step 1: Find PDFs ---
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"ERROR: No PDFs found in {DATA_DIR}/")
        print("Please add clinical ML paper PDFs to data/raw/ and try again.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDFs in data/raw/")
    trace["steps"].append({"step": "find_pdfs", "count": len(pdf_paths)})

    # --- Step 2: Build or load vector store ---
    store = load_or_build_vector_store(pdf_paths)
    trace["steps"].append({"step": "vector_store", "chunks": len(store)})

    # --- Step 3: Retrieve relevant chunks ---
    print(f"\nRetrieving relevant chunks for question...")
    results = retrieve(question, store, top_k=10)
    sources = get_unique_sources(results)

    print(f"  Retrieved {len(results)} chunks from {len(sources)} papers:")
    for s in sources:
        print(f"    - {s}")

    retrieved_path = out_dir / "retrieved.json"
    retrieved_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    trace["steps"].append({
        "step": "retrieve",
        "chunks_retrieved": len(results),
        "sources": sources,
    })

    # --- Step 4: Extract schema from each source paper ---
    print(f"\nExtracting structured schema from {len(sources)} papers...")
    extractions = []
    extraction_records = []

    for pdf_name in sources:
        pdf_path = DATA_DIR / pdf_name
        if not pdf_path.exists():
            print(f"  WARNING: {pdf_name} not found, skipping.")
            continue

        print(f"  Extracting: {pdf_name}")
        text = load_pdf(pdf_path)
        extraction, error = extract_paper_safe(text)

        if error:
            print(f"    ERROR: {error}")
            extraction_records.append({"source": pdf_name, "error": error})
            continue

        extraction = normalize_extraction(extraction)
        extractions.append(extraction)
        extraction_records.append({
            "source": pdf_name,
            "title": extraction.citation.title,
            "year": extraction.citation.year,
            "model": extraction.modeling.model_family,
            "auroc": extraction.metrics.auroc,
            "external_validation": extraction.validation.external_validation,
            "calibration_reported": extraction.metrics.calibration_reported,
        })
        print(f"    OK: {extraction.citation.title} ({extraction.citation.year})")

    extractions_path = out_dir / "extractions.jsonl"
    with open(extractions_path, "w", encoding="utf-8") as f:
        for record in extraction_records:
            f.write(json.dumps(record) + "\n")

    trace["steps"].append({
        "step": "extract",
        "papers_extracted": len(extractions),
        "papers_failed": len(sources) - len(extractions),
    })

    if not extractions:
        print("\nERROR: No papers were successfully extracted. Cannot write report.")
        sys.exit(1)

    # --- Step 5: Build evidence table ---
    print(f"\nBuilding evidence table...")
    csv_str = evidence_table_to_csv(extractions)
    (out_dir / "evidence_table.csv").write_text(csv_str, encoding="utf-8")
    stats = summarize_evidence_table(extractions)
    print(f"  Stats: {stats}")
    trace["steps"].append({"step": "evidence_table", "stats": stats})

    # --- Step 6: Write report ---
    print(f"\nGenerating report...")
    report_text = write_report(question, extractions)
    report_path = save_report(report_text, str(out_dir))
    trace["steps"].append({"step": "write_report", "path": report_path})

    # --- Save trace ---
    (out_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, default=str), encoding="utf-8"
    )

    # --- Done ---
    print(f"\n{'='*60}")
    print(f"DONE. Outputs written to: {out_dir}/")
    print(f"  report.md")
    print(f"  evidence_table.csv")
    print(f"  extractions.jsonl")
    print(f"  retrieved.json")
    print(f"  trace.json")
    print(f"{'='*60}\n")


def main() -> None:
    print("Clinical ML Research Assistant")
    print("--------------------------------")
    question = input("Enter your research question: ").strip()
    if not question:
        raise SystemExit("No question provided.")
    run_pipeline(question)


if __name__ == "__main__":
    main()