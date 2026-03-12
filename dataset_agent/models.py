from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetQuerySpec(BaseModel):
    query: str = Field(description="Original user query")
    modalities: list[Literal["image", "text"]] = Field(default_factory=lambda: ["image", "text"])
    requested_classes: list[str] = Field(default_factory=list)


class DatasetCandidate(BaseModel):
    source: Literal["huggingface", "kaggle", "web"]
    dataset_id: str
    title: str
    url: str
    modality: Literal["image", "text", "multimodal", "unknown", "other"] = "unknown"
    description: str | None = None
    classes: list[str] = Field(default_factory=list)
    downloads: int | None = None
    examples: int | None = None
    size_bytes: int | None = None
    license: str | None = None
    local_dir: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilterDecision(BaseModel):
    source: str
    dataset_id: str
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    missing_classes: list[str] = Field(default_factory=list)


class WebCollectedAsset(BaseModel):
    class_name: str
    query: str
    page_url: str
    image_url: str
    local_path: str


class CollectionManifest(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query_spec: DatasetQuerySpec
    runtime_config: dict[str, Any]
    selected_datasets: list[DatasetCandidate] = Field(default_factory=list)
    rejected_datasets: list[FilterDecision] = Field(default_factory=list)
    web_collected_assets: list[WebCollectedAsset] = Field(default_factory=list)
    covered_classes: list[str] = Field(default_factory=list)
    missing_classes: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
