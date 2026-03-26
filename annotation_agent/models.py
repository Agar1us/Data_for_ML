from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: list[float]
    mask_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationSpec:
    task_name: str
    task_description: str
    object_prompts: list[str]
    classes: dict[str, str]
    examples: dict[str, list[str]]
    edge_cases: list[dict[str, Any]]
    edge_case_counts: dict[str, int]
    guidelines: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetrics:
    kappa: float | None
    percent_agreement: float | None
    label_distribution: dict[str, int]
    object_detection_rate: float
    mask_rate: float
    object_confidence_mean: float
    object_confidence_std: float
    low_confidence_count: int
    low_confidence_ratio: float
    no_detection_count: int
    no_detection_ratio: float
    confusion_matrix: dict[str, dict[str, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    run_dir: str
    labels_csv: str = ""
    quality_metrics_json: str = ""
    annotation_spec_md: str = ""
    labelstudio_import_json: str = ""
    labelstudio_review_json: str = ""
    object_prompts: list[str] | None = None
    label_assignment_mode: str = ""
    task_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
