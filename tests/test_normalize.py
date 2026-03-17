from src.extract.normalize import normalize_model_family, normalize_extraction
from src.extract.schema import (
    PaperExtraction, Citation, Population, DataInfo,
    Modeling, Validation, Metrics, FairnessInfo,
)


def _base_extraction(**kwargs) -> PaperExtraction:
    return PaperExtraction(
        citation=Citation(title="Test", authors=["A"], year=2023),
        population=Population(),
        data=DataInfo(),
        modeling=Modeling(**kwargs.get("modeling", {})),
        validation=Validation(),
        metrics=Metrics(**kwargs.get("metrics", {})),
        bias_fairness=FairnessInfo(),
    )


def test_normalize_model_family_known():
    assert normalize_model_family("xgb") == "XGBoost"
    assert normalize_model_family("lr") == "logistic regression"
    assert normalize_model_family("rf") == "random forest"


def test_normalize_model_family_unknown():
    """Unknown model names should pass through unchanged."""
    assert normalize_model_family("MyCustomNet") == "MyCustomNet"


def test_normalize_model_family_none():
    assert normalize_model_family(None) is None


def test_normalize_extraction_model():
    extraction = _base_extraction(modeling={"model_family": "xgb"})
    result = normalize_extraction(extraction)
    assert result.modeling.model_family == "XGBoost"


def test_normalize_calibration_inferred():
    """calibration_reported should be inferred from other_metrics."""
    extraction = _base_extraction(
        metrics={"calibration_reported": False, "other_metrics": ["Brier score"]}
    )
    result = normalize_extraction(extraction)
    assert result.metrics.calibration_reported is True
    assert result.metrics.calibration_method == "Brier score"