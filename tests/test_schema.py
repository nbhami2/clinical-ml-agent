from src.extract.schema import PaperExtraction, Citation, Population, DataInfo, Modeling, Validation, Metrics, FairnessInfo


def test_paper_extraction_minimal():
    """Schema should construct with only required fields."""
    paper = PaperExtraction(
        citation=Citation(title="Test Paper", authors=["Smith J"], year=2023),
        population=Population(),
        data=DataInfo(),
        modeling=Modeling(),
        validation=Validation(),
        metrics=Metrics(),
        bias_fairness=FairnessInfo(),
    )
    assert paper.citation.title == "Test Paper"
    assert paper.validation.external_validation is False
    assert paper.metrics.calibration_reported is False


def test_paper_extraction_full():
    """Schema should accept all fields correctly."""
    paper = PaperExtraction(
        citation=Citation(title="Sepsis Prediction", authors=["Jones A", "Lee B"], year=2022, venue="Nature Medicine"),
        task="Sepsis early warning in ICU",
        population=Population(setting="ICU", sample_size=5000, outcome_prevalence=0.15),
        data=DataInfo(modality="EHR", feature_types=["labs", "vitals"], leakage_risks="None identified"),
        modeling=Modeling(model_family="XGBoost", outcome="Sepsis onset within 6 hours"),
        validation=Validation(split_type="temporal", external_validation=True),
        metrics=Metrics(auroc=0.85, calibration_reported=True, calibration_method="Brier score"),
        bias_fairness=FairnessInfo(subgroup_analysis=True, subgroups_evaluated=["age", "sex"]),
        key_findings="XGBoost achieved AUROC 0.85 with strong temporal validation.",
    )
    assert paper.metrics.auroc == 0.85
    assert paper.validation.external_validation is True
    assert "age" in paper.bias_fairness.subgroups_evaluated