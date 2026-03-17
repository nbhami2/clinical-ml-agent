from __future__ import annotations

import csv
import io

from src.extract.schema import PaperExtraction


def build_evidence_table(extractions: list[PaperExtraction]) -> list[dict]:
    """
    Convert a list of PaperExtractions into a flat list of dicts
    suitable for display or CSV export.
    """
    rows = []
    for paper in extractions:
        rows.append({
            "title": paper.citation.title,
            "year": paper.citation.year,
            "venue": paper.citation.venue or "",
            "task": paper.task or "",
            "setting": paper.population.setting or "",
            "sample_size": paper.population.sample_size or "",
            "outcome_prevalence": paper.population.outcome_prevalence or "",
            "modality": paper.data.modality or "",
            "model_family": paper.modeling.model_family or "",
            "split_type": paper.validation.split_type or "",
            "external_validation": paper.validation.external_validation,
            "auroc": paper.metrics.auroc or "",
            "auprc": paper.metrics.auprc or "",
            "calibration_reported": paper.metrics.calibration_reported,
            "calibration_method": paper.metrics.calibration_method or "",
            "subgroup_analysis": paper.bias_fairness.subgroup_analysis,
            "leakage_risks": paper.data.leakage_risks or "",
            "limitations": paper.limitations or "",
            "key_findings": paper.key_findings or "",
        })
    return rows


def evidence_table_to_csv(extractions: list[PaperExtraction]) -> str:
    """Export evidence table as a CSV string."""
    rows = build_evidence_table(extractions)
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def summarize_evidence_table(extractions: list[PaperExtraction]) -> dict:
    """
    Compute aggregate statistics across all extracted papers.
    Useful for the report's 'Evidence Overview' section.
    """
    total = len(extractions)
    if total == 0:
        return {}

    auroc_values = [
        p.metrics.auroc for p in extractions if p.metrics.auroc is not None
    ]
    external_val_count = sum(
        1 for p in extractions if p.validation.external_validation
    )
    calibration_count = sum(
        1 for p in extractions if p.metrics.calibration_reported
    )
    subgroup_count = sum(
        1 for p in extractions if p.bias_fairness.subgroup_analysis
    )
    model_families = [
        p.modeling.model_family for p in extractions
        if p.modeling.model_family is not None
    ]
    split_types = [
        p.validation.split_type for p in extractions
        if p.validation.split_type is not None
    ]

    return {
        "total_papers": total,
        "auroc_mean": round(sum(auroc_values) / len(auroc_values), 3) if auroc_values else None,
        "auroc_min": min(auroc_values) if auroc_values else None,
        "auroc_max": max(auroc_values) if auroc_values else None,
        "external_validation_rate": round(external_val_count / total, 2),
        "calibration_reporting_rate": round(calibration_count / total, 2),
        "subgroup_analysis_rate": round(subgroup_count / total, 2),
        "model_families": list(set(model_families)),
        "split_types": list(set(split_types)),
    }