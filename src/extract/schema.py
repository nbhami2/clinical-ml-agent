from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str = Field(description="Full title of the paper")
    authors: list[str] = Field(description="List of author names")
    year: int = Field(description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal or conference name")
    doi: Optional[str] = Field(default=None, description="DOI or URL if available")


class Population(BaseModel):
    setting: Optional[str] = Field(default=None, description="Clinical setting e.g. ICU, ED, inpatient")
    inclusion_criteria: Optional[str] = Field(default=None, description="Who was included")
    exclusion_criteria: Optional[str] = Field(default=None, description="Who was excluded")
    sample_size: Optional[int] = Field(default=None, description="Total number of patients or samples")
    outcome_prevalence: Optional[float] = Field(default=None, description="Prevalence of the predicted outcome (0-1)")


class DataInfo(BaseModel):
    modality: Optional[str] = Field(default=None, description="Data type e.g. EHR, imaging, wearables, claims")
    feature_types: Optional[list[str]] = Field(default=None, description="Types of features used e.g. labs, vitals, demographics")
    missingness_handling: Optional[str] = Field(default=None, description="How missing data was handled")
    leakage_risks: Optional[str] = Field(default=None, description="Any potential data leakage identified")


class Modeling(BaseModel):
    model_family: Optional[str] = Field(default=None, description="e.g. logistic regression, XGBoost, LSTM, Transformer")
    feature_selection: Optional[str] = Field(default=None, description="Feature selection strategy if any")
    hyperparameter_tuning: Optional[str] = Field(default=None, description="Tuning strategy e.g. grid search, Bayesian")
    outcome: Optional[str] = Field(default=None, description="Predicted outcome and time horizon")


class Validation(BaseModel):
    split_type: Optional[str] = Field(default=None, description="e.g. random split, temporal split, site split")
    cross_validation: Optional[str] = Field(default=None, description="CV strategy if used")
    external_validation: bool = Field(default=False, description="Whether external validation was performed")
    external_validation_details: Optional[str] = Field(default=None, description="Details of external validation dataset")


class Metrics(BaseModel):
    auroc: Optional[float] = Field(default=None, description="Area under ROC curve")
    auprc: Optional[float] = Field(default=None, description="Area under precision-recall curve")
    sensitivity: Optional[float] = Field(default=None, description="Sensitivity / recall")
    specificity: Optional[float] = Field(default=None, description="Specificity")
    calibration_reported: bool = Field(default=False, description="Whether calibration was reported")
    calibration_method: Optional[str] = Field(default=None, description="e.g. Brier score, calibration plot, Hosmer-Lemeshow")
    other_metrics: Optional[list[str]] = Field(default=None, description="Any other reported metrics")


class FairnessInfo(BaseModel):
    subgroup_analysis: bool = Field(default=False, description="Whether subgroup analysis was performed")
    subgroups_evaluated: Optional[list[str]] = Field(default=None, description="e.g. age, sex, race, insurance")
    fairness_metrics: Optional[str] = Field(default=None, description="Any fairness metrics reported")


class PaperExtraction(BaseModel):
    citation: Citation
    task: Optional[str] = Field(default=None, description="High-level prediction task e.g. sepsis early warning")
    population: Population
    data: DataInfo
    modeling: Modeling
    validation: Validation
    metrics: Metrics
    bias_fairness: FairnessInfo
    baseline_comparators: Optional[str] = Field(default=None, description="Comparator models or clinical scores used")
    key_findings: Optional[str] = Field(default=None, description="Main results summarized in 1-2 sentences")
    limitations: Optional[str] = Field(default=None, description="Stated or inferred limitations")
    reproducibility: Optional[str] = Field(default=None, description="Code/data availability statement")