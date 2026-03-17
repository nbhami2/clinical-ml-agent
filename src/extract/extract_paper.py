from __future__ import annotations

import json
import os

from dotenv import load_dotenv
import google.generativeai as genai

from src.extract.schema import (
    PaperExtraction,
    Citation,
    Population,
    DataInfo,
    Modeling,
    Validation,
    Metrics,
    FairnessInfo,
)

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = "gemini-2.0-flash"

EXTRACTION_PROMPT = """
You are an expert clinical ML researcher. Extract structured information from the
following clinical machine learning paper text.

Return ONLY a valid JSON object with exactly this structure. Use null for any fields
you cannot find. Do not include any explanation or markdown - just the raw JSON.

{{
    "citation": {{
        "title": "string",
        "authors": ["string"],
        "year": integer,
        "venue": "string or null",
        "doi": "string or null"
    }},
    "task": "string or null",
    "population": {{
        "setting": "string or null",
        "inclusion_criteria": "string or null",
        "exclusion_criteria": "string or null",
        "sample_size": integer or null,
        "outcome_prevalence": float or null
    }},
    "data": {{
        "modality": "string or null",
        "feature_types": ["string"] or null,
        "missingness_handling": "string or null",
        "leakage_risks": "string or null"
    }},
    "modeling": {{
        "model_family": "string or null",
        "feature_selection": "string or null",
        "hyperparameter_tuning": "string or null",
        "outcome": "string or null"
    }},
    "validation": {{
        "split_type": "string or null",
        "cross_validation": "string or null",
        "external_validation": true or false,
        "external_validation_details": "string or null"
    }},
    "metrics": {{
        "auroc": float or null,
        "auprc": float or null,
        "sensitivity": float or null,
        "specificity": float or null,
        "calibration_reported": true or false,
        "calibration_method": "string or null",
        "other_metrics": ["string"] or null
    }},
    "bias_fairness": {{
        "subgroup_analysis": true or false,
        "subgroups_evaluated": ["string"] or null,
        "fairness_metrics": "string or null"
    }},
    "baseline_comparators": "string or null",
    "key_findings": "string or null",
    "limitations": "string or null",
    "reproducibility": "string or null"
}}

Paper text:
{paper_text}
"""


def extract_paper(paper_text: str, max_chars: int = 12000) -> PaperExtraction:
    """
    Extract structured schema from raw paper text using Gemini.

    Args:
        paper_text: Raw text from a clinical ML paper
        max_chars: Truncate text to this length to stay within token limits

    Returns:
        PaperExtraction pydantic model
    """
    # Truncate to avoid hitting token limits - focus on beginning/methods/results
    truncated = paper_text[:max_chars]

    prompt = EXTRACTION_PROMPT.format(paper_text=truncated)

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)

    raw = response.text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    data = json.loads(raw)
    return PaperExtraction(**data)


def extract_paper_safe(paper_text: str) -> tuple[PaperExtraction | None, str | None]:
    """
    Safe wrapper around extract_paper.
    Returns (extraction, None) on success or (None, error_message) on failure.
    """
    try:
        extraction = extract_paper(paper_text)
        return extraction, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
    except Exception as e:
        return None, f"Extraction failed: {e}"