from __future__ import annotations

from src.extract.schema import PaperExtraction

# Map common variants to canonical metric names
AUROC_ALIASES = {
    "auc", "auroc", "c-statistic", "c statistic", "area under roc",
    "area under the roc", "area under receiver operating characteristic",
}

AUPRC_ALIASES = {
    "auprc", "average precision", "area under precision recall",
    "area under the precision recall curve",
}

CALIBRATION_ALIASES = {
    "brier score", "calibration plot", "calibration curve",
    "hosmer-lemeshow", "hosmer lemeshow", "reliability diagram",
    "calibration slope", "calibration intercept",
}

MODEL_FAMILY_MAP = {
    "lr": "logistic regression",
    "logistic": "logistic regression",
    "xgb": "XGBoost",
    "xgboost": "XGBoost",
    "rf": "random forest",
    "random forest": "random forest",
    "svm": "SVM",
    "support vector machine": "SVM",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "bert": "BERT",
    "gbt": "gradient boosting",
    "gradient boosted": "gradient boosting",
    "lightgbm": "LightGBM",
    "gbm": "gradient boosting",
}


def normalize_model_family(raw: str | None) -> str | None:
    """Normalize model family name to a canonical form."""
    if raw is None:
        return None
    key = raw.lower().strip()
    return MODEL_FAMILY_MAP.get(key, raw)


def normalize_extraction(extraction: PaperExtraction) -> PaperExtraction:
    """
    Apply normalization passes to a PaperExtraction.
    Returns a new PaperExtraction with cleaned fields.
    """
    # Normalize model family
    if extraction.modeling.model_family:
        extraction.modeling.model_family = normalize_model_family(
            extraction.modeling.model_family
        )

    # Infer calibration_reported from other_metrics if not already set
    if not extraction.metrics.calibration_reported and extraction.metrics.other_metrics:
        for metric in extraction.metrics.other_metrics:
            if metric.lower() in CALIBRATION_ALIASES:
                extraction.metrics.calibration_reported = True
                extraction.metrics.calibration_method = metric
                break

    return extraction