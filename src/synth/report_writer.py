from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.extract.schema import PaperExtraction
from src.synth.evidence_table import (
    evidence_table_to_csv,
    summarize_evidence_table,
)

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = "models/gemini-2.0-flash"

SYNTHESIS_PROMPT = """
You are an expert clinical ML researcher writing a structured evidence synthesis report.

You have been given:
1. A research question
2. A summary of extracted data from {n_papers} clinical ML papers
3. An evidence table (CSV) showing key fields per paper

Write a structured Markdown report with the following sections:

## Background
Brief context for the research question (2-3 sentences).

## Evidence Summary
Synthesize findings across the papers. Mention specific papers by title when making
claims. Cover: models used, validation strategies, performance metrics, and any
consensus or disagreements across studies.

## Methodological Critique
Identify recurring weaknesses across the studies. Specifically address:
- Data leakage risks
- Calibration reporting gaps
- External validation coverage
- Subgroup/fairness analysis gaps

## Proposed Validation Study
Based on the evidence gaps identified, propose a concrete external validation study
design. Include: population, data sources, validation strategy, metrics to report,
and fairness considerations.

## Limitations of This Synthesis
What are the limitations of this automated synthesis?

---

Research Question:
{question}

Evidence Summary Statistics:
{stats}

Evidence Table (CSV):
{evidence_csv}

Paper Key Findings:
{findings}

Write the full report now. Be specific, cite paper titles, and be critical.
"""


def build_findings_text(extractions: list[PaperExtraction]) -> str:
    """Format key findings from each paper for the synthesis prompt."""
    lines = []
    for p in extractions:
        title = p.citation.title
        finding = p.key_findings or "No key findings extracted."
        auroc = f"AUROC={p.metrics.auroc}" if p.metrics.auroc else "AUROC=NR"
        ext_val = "external validation: yes" if p.validation.external_validation else "external validation: no"
        calib = "calibration reported: yes" if p.metrics.calibration_reported else "calibration reported: no"
        lines.append(f"- {title} ({p.citation.year}): {finding} [{auroc}, {ext_val}, {calib}]")
    return "\n".join(lines)


def write_report(
    question: str,
    extractions: list[PaperExtraction],
) -> str:
    """
    Generate a full Markdown report from a research question + extractions.

    Returns the report as a string.
    """
    stats = summarize_evidence_table(extractions)
    evidence_csv = evidence_table_to_csv(extractions)
    findings = build_findings_text(extractions)

    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        n_papers=len(extractions),
        stats=str(stats),
        evidence_csv=evidence_csv[:3000],
        findings=findings,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text.strip()


def save_report(report_text: str, out_dir: str) -> str:
    """Save report markdown to the output directory."""
    path = Path(out_dir) / "report.md"
    path.write_text(report_text, encoding="utf-8")
    return str(path)