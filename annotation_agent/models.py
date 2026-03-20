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
    classes: dict[str, str]
    examples: dict[str, list[str]]
    edge_cases: list[dict[str, Any]]
    guidelines: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetrics:
    kappa: float | None
    percent_agreement: float | None
    label_distribution: dict[str, int]
    folder_match_rate: float
    confidence_mean: float
    confidence_std: float
    low_confidence_count: int
    low_confidence_ratio: float
    confusion_matrix: dict[str, dict[str, int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    run_dir: str
    labeled_csv: str
    manual_review_manifest: str
    quality_metrics_json: str = ""
    annotation_spec_md: str = ""
    labelstudio_import_json: str = ""
    labelstudio_review_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
