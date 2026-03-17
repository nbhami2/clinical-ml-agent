from src.extract.schema import (
    PaperExtraction, Citation, Population, DataInfo,
    Modeling, Validation, Metrics, FairnessInfo,
)
from src.synth.evidence_table import (
    build_evidence_table,
    evidence_table_to_csv,
    summarize_evidence_table,
)
from src.synth.report_writer import build_findings_text


def _make_extraction(title: str, year: int, auroc: float,
                     external_val: bool, calibration: bool) -> PaperExtraction:
    return PaperExtraction(
        citation=Citation(title=title, authors=["Author A"], year=year),
        task="Sepsis prediction",
        population=Population(setting="ICU", sample_size=1000),
        data=DataInfo(modality="EHR", leakage_risks="None identified"),
        modeling=Modeling(model_family="XGBoost"),
        validation=Validation(split_type="temporal", external_validation=external_val),
        metrics=Metrics(auroc=auroc, calibration_reported=calibration),
        bias_fairness=FairnessInfo(subgroup_analysis=False),
        key_findings="Model performed well on held-out test set.",
    )


def test_build_evidence_table():
    extractions = [
        _make_extraction("Paper A", 2022, 0.85, True, True),
        _make_extraction("Paper B", 2023, 0.78, False, False),
    ]
    table = build_evidence_table(extractions)
    assert len(table) == 2
    assert table[0]["title"] == "Paper A"
    assert table[1]["auroc"] == 0.78


def test_evidence_table_to_csv():
    extractions = [_make_extraction("Paper A", 2022, 0.85, True, True)]
    csv_str = evidence_table_to_csv(extractions)
    assert "Paper A" in csv_str
    assert "auroc" in csv_str


def test_summarize_evidence_table():
    extractions = [
        _make_extraction("Paper A", 2022, 0.85, True, True),
        _make_extraction("Paper B", 2023, 0.78, False, False),
        _make_extraction("Paper C", 2021, 0.91, True, True),
    ]
    stats = summarize_evidence_table(extractions)
    assert stats["total_papers"] == 3
    assert stats["external_validation_rate"] == round(2 / 3, 2)
    assert stats["calibration_reporting_rate"] == round(2 / 3, 2)
    assert stats["auroc_mean"] == round((0.85 + 0.78 + 0.91) / 3, 3)


def test_summarize_empty():
    stats = summarize_evidence_table([])
    assert stats == {}


def test_build_findings_text():
    extractions = [_make_extraction("Paper A", 2022, 0.85, True, True)]
    text = build_findings_text(extractions)
    assert "Paper A" in text
    assert "AUROC=0.85" in text
    assert "external validation: yes" in text