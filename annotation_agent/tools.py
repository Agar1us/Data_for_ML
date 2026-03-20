from __future__ import annotations

import json
import logging
import math
import re
import os
import shutil
import threading
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    def tool(func):  # type: ignore[misc]
        return func

from annotation_agent.config import (
    COLUMN_ALL_DETECTIONS_JSON,
    COLUMN_BBOX_XYXY,
    COLUMN_CONFIDENCE,
    COLUMN_FILE_PATH,
    COLUMN_FILENAME,
    COLUMN_FOLDER_LABEL,
    COLUMN_FOLDER_MATCH,
    COLUMN_HAS_MASK,
    COLUMN_IMAGE_HEIGHT,
    COLUMN_IMAGE_WIDTH,
    COLUMN_MASK_PATH,
    COLUMN_PREDICTED_LABEL,
    DEFAULT_BOUNDARY_CONFIDENCE_RANGE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_REVIEW_LINK_MODE,
    NO_DETECTION_LABEL,
    SUPPORTED_IMG_EXTENSIONS,
    ToolResult,
)
from annotation_agent.models import AnnotationSpec, Detection, QualityMetrics
from annotation_agent.reporting import ensure_run_layout, make_run_id


logger = logging.getLogger(__name__)
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_safe_json(item) for item in value.tolist()]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _safe_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    return value


def _json_success(payload: ToolResult) -> str:
    return json.dumps({"success": True, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _json_error(message: str, **payload: Any) -> str:
    return json.dumps({"success": False, "error": message, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _parse_json_arg(payload: str, argument_name: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Argument '{argument_name}' must be valid JSON.") from exc


def _write_json_artifact(payload: ToolResult, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(payload), handle, ensure_ascii=False, indent=2)
    return str(path)


def _read_json_artifact(path: str) -> ToolResult:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {artifact_path}")
    return payload


def _write_markdown(markdown: str, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def _read_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Table does not exist: {table_path}")
    return pd.read_csv(table_path)


def _write_table(df: pd.DataFrame, output_path: str) -> str:
    table_path = Path(output_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)
    return str(table_path)


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return text.strip("_") or "dataset"


def _list_class_dirs(dataset_dir: str) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir())


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMG_EXTENSIONS


def _scan_dataset_records(dataset_dir: str) -> tuple[list[ToolResult], dict[str, int], list[str]]:
    records: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    empty_classes: list[str] = []
    for class_dir in _list_class_dirs(dataset_dir):
        image_paths = sorted(path for path in class_dir.rglob("*") if path.is_file() and _is_supported_image(path))
        class_counts[class_dir.name] = len(image_paths)
        if not image_paths:
            empty_classes.append(class_dir.name)
        for image_path in image_paths:
            records.append(
                {
                    COLUMN_FILE_PATH: str(image_path.resolve()),
                    COLUMN_FILENAME: image_path.name,
                    COLUMN_FOLDER_LABEL: class_dir.name,
                }
            )
    return records, class_counts, empty_classes


def _inspect_dataset_records(records: list[ToolResult], dataset_dir: str) -> ToolResult:
    broken: list[dict[str, str]] = []
    for record in records:
        error = _validate_image(record[COLUMN_FILE_PATH])
        if error:
            broken.append({COLUMN_FILE_PATH: record[COLUMN_FILE_PATH], "error": error})
    class_counts: dict[str, int] = {}
    for record in records:
        class_counts[str(record[COLUMN_FOLDER_LABEL])] = class_counts.get(str(record[COLUMN_FOLDER_LABEL]), 0) + 1
    all_classes = {path.name for path in _list_class_dirs(dataset_dir)}
    empty_classes = sorted(all_classes - set(class_counts))
    return {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "rows": len(records),
        "valid_files": len(records) - len(broken),
        "broken_files": broken,
        "broken_count": len(broken),
        "class_counts": {**{name: 0 for name in empty_classes}, **class_counts},
        "empty_classes": empty_classes,
    }


def _open_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _validate_image(path: str) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as exc:
        return str(exc)


def _load_yoloe(model_path: str):
    with _MODEL_CACHE_LOCK:
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]
        try:
            from ultralytics import YOLOE  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics with YOLOE support is required for image auto-labeling. "
                "Install the annotation agent requirements first."
            ) from exc
        _MODEL_CACHE[model_path] = YOLOE(model_path)
        return _MODEL_CACHE[model_path]


def _model_version_from_path(model_path: str) -> str:
    return Path(model_path).name or DEFAULT_MODEL_PATH


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Expected tensor-like value, got None.")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _extract_detections(result: Any, classes: list[str], mask_key_prefix: str = "") -> tuple[list[Detection], dict[str, np.ndarray]]:
    detections: list[Detection] = []
    mask_payloads: dict[str, np.ndarray] = {}
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections, mask_payloads

    class_ids = _tensor_to_numpy(getattr(boxes, "cls", []))
    confidences = _tensor_to_numpy(getattr(boxes, "conf", []))
    xyxy = _tensor_to_numpy(getattr(boxes, "xyxy", []))
    masks = getattr(result, "masks", None)
    mask_data = getattr(masks, "data", None) if masks is not None else None

    for index in range(len(class_ids)):
        class_index = int(class_ids[index])
        label = classes[class_index] if 0 <= class_index < len(classes) else str(class_index)
        mask_key = None
        if mask_data is not None and len(mask_data) > index:
            mask_key = f"{mask_key_prefix}{index}"
            mask_payloads[mask_key] = _tensor_to_numpy(mask_data[index])
        detections.append(
            Detection(
                label=label,
                confidence=float(confidences[index]),
                bbox=[float(value) for value in xyxy[index].tolist()],
                mask_key=mask_key,
            )
        )
    return detections, mask_payloads


def _normalize_bbox_json(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [float(item) for item in json.loads(value)]
    return []


def _parse_detection_list(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return []


def _boundary_range(payload: str = "") -> tuple[float, float]:
    if not payload:
        return DEFAULT_BOUNDARY_CONFIDENCE_RANGE
    parsed = _parse_json_arg(payload, "boundary_range_json")
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError("boundary_range_json must be a JSON list with two floats.")
    low, high = float(parsed[0]), float(parsed[1])
    if low > high:
        raise ValueError("boundary_range_json must be ordered [low, high].")
    return low, high


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _guess_label_column(df: pd.DataFrame) -> str:
    for candidate in ("human_label", "label", "annotated_label", "predicted_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Human labels file must contain one of: human_label, label, annotated_label, predicted_label.")


def _class_definitions(classes: list[str]) -> dict[str, str]:
    return {
        label: (
            f"Assign '{label}' when the primary visible subject is best described as "
            f"'{label.replace('_', ' ').replace('-', ' ')}' and this category is more specific than the alternatives."
        )
        for label in classes
    }


def _warn_deprecated_argument(tool_name: str, argument_name: str, replacement: str = "") -> None:
    message = f"{tool_name}: argument '{argument_name}' is deprecated and ignored."
    if replacement:
        message += f" Use {replacement} instead."
    warnings.warn(message, DeprecationWarning, stacklevel=2)
    logger.warning(message)


def _merge_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _save_mask_array(mask: np.ndarray, masks_dir: str, filename: str, row_index: int, image_format: str = "npy") -> str:
    mask_dir_path = Path(masks_dir)
    mask_dir_path.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    binary_mask = (np.asarray(mask) > 0).astype(np.uint8)
    if image_format.lower() == "png":
        mask_path = mask_dir_path / f"{stem}_{row_index}.png"
        Image.fromarray(binary_mask * 255).save(mask_path)
    else:
        mask_path = mask_dir_path / f"{stem}_{row_index}.npy"
        np.save(mask_path, binary_mask)
    return str(mask_path)


def _persist_review_pointer(source: Path, target: Path, link_mode: str) -> None:
    if link_mode == "manifest_only":
        return
    if link_mode == "symlink":
        try:
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(source, target)
            return
        except OSError:
            logger.warning("Failed to create symlink %s -> %s, falling back to copy2.", target, source, exc_info=True)
    shutil.copy2(source, target)


def _render_spec_markdown(spec: AnnotationSpec) -> str:
    lines = [
        f"# Annotation Specification: {spec.task_name}",
        "",
        "## Task Description",
        spec.task_description,
        "",
        "## Classes",
    ]
    for label, definition in spec.classes.items():
        lines.extend([f"### {label}", f"Definition: {definition}", "", "Examples:"])
        examples = spec.examples.get(label, [])
        if not examples:
            lines.append("1. No high-confidence examples available.")
        else:
            for index, example in enumerate(examples, start=1):
                lines.append(f"{index}. `{example}`")
        lines.append("")
    lines.extend(["## Edge Cases", "| File | Reason | Prediction | Folder |", "| --- | --- | --- | --- |"])
    if spec.edge_cases:
        for item in spec.edge_cases:
            lines.append(
                f"| `{item.get('file_path', '')}` | {item.get('reason', '')} | "
                f"{item.get('predicted_label', '')} | {item.get('folder_label', '')} |"
            )
    else:
        lines.append("| None | No edge cases detected | | |")
    lines.extend(["", "## Instructions", spec.guidelines.strip(), ""])
    return "\n".join(lines)


def _quality_payload(df: pd.DataFrame, confidence_threshold: float) -> QualityMetrics:
    predicted = (
        df[COLUMN_PREDICTED_LABEL].fillna("").astype(str).replace("", NO_DETECTION_LABEL).value_counts().to_dict()
        if COLUMN_PREDICTED_LABEL in df.columns
        else {}
    )
    confidence_series = pd.to_numeric(df.get(COLUMN_CONFIDENCE, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    folder_match_series = (
        df.get(COLUMN_FOLDER_MATCH, pd.Series(dtype=bool)).map(_bool_from_value)
        if COLUMN_FOLDER_MATCH in df.columns
        else pd.Series(dtype=bool)
    )
    low_conf_mask = confidence_series < float(confidence_threshold)
    metrics = QualityMetrics(
        kappa=None,
        percent_agreement=None,
        label_distribution={str(key): int(value) for key, value in predicted.items()},
        folder_match_rate=float(folder_match_series.mean()) if not folder_match_series.empty else 0.0,
        confidence_mean=float(confidence_series.mean()) if not confidence_series.empty else 0.0,
        confidence_std=float(confidence_series.std(ddof=0)) if len(confidence_series) > 1 else 0.0,
        low_confidence_count=int(low_conf_mask.sum()),
        low_confidence_ratio=float(low_conf_mask.mean()) if len(confidence_series) else 0.0,
        confusion_matrix=None,
    )
    return metrics


def _compute_agreement(df_auto: pd.DataFrame, df_human: pd.DataFrame, confidence_threshold: float) -> dict[str, Any]:
    label_column = _guess_label_column(df_human)
    if COLUMN_FILENAME not in df_human.columns:
        raise ValueError(f"Human labels file must include a '{COLUMN_FILENAME}' column.")

    auto = df_auto.copy()
    human = df_human.copy()
    auto["filename_key"] = auto[COLUMN_FILENAME].map(_merge_key)
    human["filename_key"] = human[COLUMN_FILENAME].map(_merge_key)

    if COLUMN_FILE_PATH in auto.columns and COLUMN_FILE_PATH in human.columns:
        auto["file_path_key"] = auto[COLUMN_FILE_PATH].map(_merge_key)
        human["file_path_key"] = human[COLUMN_FILE_PATH].map(_merge_key)
        merge_columns = ["file_path_key"]
    elif COLUMN_FOLDER_LABEL in auto.columns and COLUMN_FOLDER_LABEL in human.columns:
        auto["folder_label_key"] = auto[COLUMN_FOLDER_LABEL].map(_merge_key)
        human["folder_label_key"] = human[COLUMN_FOLDER_LABEL].map(_merge_key)
        merge_columns = ["folder_label_key", "filename_key"]
    else:
        merge_columns = ["filename_key"]

    human_columns = merge_columns + [label_column]
    merged = auto.merge(
        human[human_columns].rename(columns={label_column: "human_label"}),
        on=merge_columns,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping filenames between auto labels and human labels.")

    auto_labels = merged[COLUMN_PREDICTED_LABEL].fillna("").astype(str)
    human_labels = merged["human_label"].fillna("").astype(str)
    non_empty = (auto_labels != "") & (human_labels != "")
    merged = merged.loc[non_empty].reset_index(drop=True)
    if merged.empty:
        raise ValueError("Overlapping rows exist, but at least one side has only empty labels.")

    labels = sorted(set(merged[COLUMN_PREDICTED_LABEL]).union(set(merged["human_label"])))
    if len(labels) < 2:
        raise ValueError("Need at least two distinct labels across auto and human annotations to compute Cohen's kappa.")

    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
    except ImportError as exc:
        raise ImportError("scikit-learn is required to compute agreement metrics with human labels.") from exc

    matrix = confusion_matrix(merged["human_label"], merged[COLUMN_PREDICTED_LABEL], labels=labels)
    confusion_payload = {
        human_label: {pred_label: int(matrix[row_index, col_index]) for col_index, pred_label in enumerate(labels)}
        for row_index, human_label in enumerate(labels)
    }
    base_metrics = _quality_payload(df_auto, confidence_threshold).to_dict()
    base_metrics.update(
        {
            "kappa": float(cohen_kappa_score(merged["human_label"], merged[COLUMN_PREDICTED_LABEL], labels=labels)),
            "percent_agreement": float((merged["human_label"] == merged[COLUMN_PREDICTED_LABEL]).mean()),
            "confusion_matrix": confusion_payload,
            "agreement_rows": int(len(merged)),
            "human_label_column": label_column,
        }
    )
    return base_metrics


def _clamp_percentage(value: float) -> float:
    return max(0.0, min(100.0, round(value, 4)))


def _bbox_to_labelstudio(bbox: list[float], image_width: int, image_height: int) -> dict[str, float]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 values, got: {bbox}")
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "x": _clamp_percentage((x1 / image_width) * 100.0),
        "y": _clamp_percentage((y1 / image_height) * 100.0),
        "width": _clamp_percentage((width / image_width) * 100.0),
        "height": _clamp_percentage((height / image_height) * 100.0),
        "rotation": 0,
    }


def _labelstudio_record(row: pd.Series, image_reference: str, model_version: str) -> ToolResult:
    predicted_label = str(row.get(COLUMN_PREDICTED_LABEL) or "").strip()
    bbox = _normalize_bbox_json(row.get(COLUMN_BBOX_XYXY))
    predictions: list[dict[str, Any]] = []
    if predicted_label and predicted_label != NO_DETECTION_LABEL and len(bbox) == 4:
        width = int(row.get(COLUMN_IMAGE_WIDTH) or 0)
        height = int(row.get(COLUMN_IMAGE_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            width, height = _open_image_size(str(row[COLUMN_FILE_PATH]))
        predictions = [
            {
                "model_version": model_version,
                "result": [
                    {
                        "id": f"pred-{row.name}",
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image",
                        "value": {
                            **_bbox_to_labelstudio(bbox, width, height),
                            "rectanglelabels": [predicted_label],
                        },
                        "score": float(row.get(COLUMN_CONFIDENCE, 0.0) or 0.0),
                    }
                ],
            }
        ]
    return {
        "data": {"image": image_reference},
        "predictions": predictions,
    }


def prepare_run_dir_impl(artifacts_dir: str, dataset_name: str = "", run_id: str = "") -> ToolResult:
    prefix = _slugify(dataset_name) if dataset_name else "annotation"
    final_run_id = run_id or make_run_id(prefix)
    run_dir = ensure_run_layout(Path(artifacts_dir) / final_run_id)
    return {
        "run_id": final_run_id,
        "run_dir": str(run_dir),
        "reports_dir": str(run_dir / "reports"),
        "summary_dir": str(run_dir / "summary"),
        "labeled_dir": str(run_dir / "cleaned_or_labeled"),
        "manual_review_dir": str(run_dir / "manual_review"),
        "masks_dir": str(run_dir / "masks"),
    }


def infer_classes_from_folders_impl(dataset_dir: str) -> ToolResult:
    _, class_counts, empty_classes = _scan_dataset_records(dataset_dir)
    classes = sorted(class_counts.keys())
    return {
        "dataset_dir": dataset_dir,
        "classes": classes,
        "class_counts": class_counts,
        "empty_classes": empty_classes,
    }


def scan_image_dataset_impl(dataset_dir: str, output_path: str = "") -> ToolResult:
    records, class_counts, empty_classes = _scan_dataset_records(dataset_dir)
    df = pd.DataFrame(records, columns=[COLUMN_FILE_PATH, COLUMN_FILENAME, COLUMN_FOLDER_LABEL])
    payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "rows": int(len(df)),
        "class_counts": class_counts,
        "empty_classes": empty_classes,
        "columns": list(df.columns),
    }
    if output_path:
        payload["output_path"] = _write_table(df, output_path)
    return payload


def validate_image_dataset_impl(dataset_dir: str, output_path: str = "") -> ToolResult:
    records, _, _ = _scan_dataset_records(dataset_dir)
    payload = _inspect_dataset_records(records, dataset_dir)
    if output_path:
        payload["output_path"] = _write_json_artifact(payload, output_path)
    return payload


def inspect_image_dataset_impl(dataset_dir: str, dataset_output_path: str = "", validation_output_path: str = "") -> ToolResult:
    records, class_counts, empty_classes = _scan_dataset_records(dataset_dir)
    df = pd.DataFrame(records, columns=[COLUMN_FILE_PATH, COLUMN_FILENAME, COLUMN_FOLDER_LABEL])
    scan_payload: ToolResult = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "rows": int(len(df)),
        "class_counts": class_counts,
        "empty_classes": empty_classes,
        "columns": list(df.columns),
    }
    if dataset_output_path:
        scan_payload["output_path"] = _write_table(df, dataset_output_path)
    validation_payload = _inspect_dataset_records(records, dataset_dir)
    if validation_output_path:
        validation_payload["output_path"] = _write_json_artifact(validation_payload, validation_output_path)
    return {"scan": scan_payload, "validation": validation_payload}


def run_yoloe_labeling_impl(
    dataset_csv_path: str,
    classes: list[str],
    output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    masks_dir: str = "",
    batch_size: int = 16,
    report_output_path: str = "",
) -> dict[str, Any]:
    if not classes:
        raise ValueError("classes must be a non-empty list of class names.")
    dataset_df = _read_table(dataset_csv_path)
    if dataset_df.empty:
        raise ValueError("Dataset manifest is empty; nothing to label.")

    model = _load_yoloe(model_path)
    if not hasattr(model, "set_classes"):
        raise AttributeError("Loaded YOLOE model does not expose set_classes().")
    model.set_classes(classes)

    results_rows: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    file_paths = dataset_df[COLUMN_FILE_PATH].astype(str).tolist()
    masks_dir = str(masks_dir or "")

    for batch_start in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[batch_start : batch_start + batch_size]
        predictions = model.predict(batch_paths, conf=0.05, verbose=False)
        for local_index, prediction in enumerate(predictions):
            row_index = batch_start + local_index
            source_row = dataset_df.iloc[row_index]
            detections, batch_masks = _extract_detections(prediction, classes, mask_key_prefix=f"row_{row_index}_")
            best = max(detections, key=lambda item: item.confidence, default=None)
            predicted_label = best.label if best else NO_DETECTION_LABEL
            mask_path = ""
            image_width, image_height = _open_image_size(str(source_row[COLUMN_FILE_PATH]))
            if best and best.mask_key and masks_dir and best.mask_key in batch_masks:
                mask_path = _save_mask_array(batch_masks[best.mask_key], masks_dir, str(source_row[COLUMN_FILENAME]), row_index)
            results_rows.append(
                {
                    COLUMN_FILE_PATH: source_row[COLUMN_FILE_PATH],
                    COLUMN_FILENAME: source_row[COLUMN_FILENAME],
                    COLUMN_FOLDER_LABEL: source_row[COLUMN_FOLDER_LABEL],
                    COLUMN_PREDICTED_LABEL: predicted_label,
                    COLUMN_CONFIDENCE: float(best.confidence if best else 0.0),
                    COLUMN_BBOX_XYXY: json.dumps(best.bbox if best else []),
                    COLUMN_HAS_MASK: bool(mask_path),
                    COLUMN_MASK_PATH: mask_path,
                    COLUMN_ALL_DETECTIONS_JSON: json.dumps([item.to_dict() for item in detections], ensure_ascii=False),
                    COLUMN_FOLDER_MATCH: bool(best and best.label == source_row[COLUMN_FOLDER_LABEL]),
                    COLUMN_IMAGE_WIDTH: image_width,
                    COLUMN_IMAGE_HEIGHT: image_height,
                }
            )

    labeled_df = pd.DataFrame(results_rows)
    saved_csv = _write_table(labeled_df, output_path)
    payload = {
        "dataset_csv_path": dataset_csv_path,
        "output_path": saved_csv,
        "rows": int(len(labeled_df)),
        "classes": classes,
        "model_version": _model_version_from_path(model_path),
        "detections_found": int((labeled_df[COLUMN_PREDICTED_LABEL] != NO_DETECTION_LABEL).sum()),
        "no_detection_count": int((labeled_df[COLUMN_PREDICTED_LABEL] == NO_DETECTION_LABEL).sum()),
        "mask_count": int(labeled_df[COLUMN_HAS_MASK].sum()),
    }
    if report_output_path:
        payload["report_output_path"] = _write_json_artifact(payload, report_output_path)
    return payload


def save_segmentation_masks_impl(input_path: str, output_path: str, masks_dir: str) -> ToolResult:
    df = _read_table(input_path)
    if COLUMN_MASK_PATH not in df.columns:
        df[COLUMN_MASK_PATH] = ""
    if COLUMN_HAS_MASK not in df.columns:
        df[COLUMN_HAS_MASK] = df[COLUMN_MASK_PATH].astype(str).str.strip().replace("", pd.NA).notna()
    mask_dir_path = Path(masks_dir)
    mask_dir_path.mkdir(parents=True, exist_ok=True)
    saved_csv = _write_table(df, output_path)
    return {
        "output_path": saved_csv,
        "saved_mask_count": int(pd.Series(df[COLUMN_MASK_PATH]).astype(str).str.strip().replace("", np.nan).notna().sum()),
        "masks_dir": str(mask_dir_path),
    }


def split_low_confidence_examples_impl(
    input_path: str,
    review_dir: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    output_confident_path: str = "",
    output_uncertain_path: str = "",
    manifest_output_path: str = "",
    link_mode: str = DEFAULT_REVIEW_LINK_MODE,
) -> ToolResult:
    df = _read_table(input_path)
    df[COLUMN_CONFIDENCE] = pd.to_numeric(df.get(COLUMN_CONFIDENCE), errors="coerce").fillna(0.0)
    confident = df[df[COLUMN_CONFIDENCE] >= float(threshold)].reset_index(drop=True)
    uncertain = df[df[COLUMN_CONFIDENCE] < float(threshold)].reset_index(drop=True)
    review_root = Path(review_dir)
    review_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for _, row in uncertain.iterrows():
        class_name = str(row.get(COLUMN_FOLDER_LABEL) or row.get(COLUMN_PREDICTED_LABEL) or "unassigned")
        target_dir = review_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        source = Path(str(row[COLUMN_FILE_PATH]))
        target = target_dir / source.name
        _persist_review_pointer(source, target, link_mode)
        manifest_rows.append(
            {
                COLUMN_FILE_PATH: str(source),
                "review_path": str(target),
                COLUMN_FILENAME: row[COLUMN_FILENAME],
                COLUMN_FOLDER_LABEL: row.get(COLUMN_FOLDER_LABEL, ""),
                COLUMN_PREDICTED_LABEL: row.get(COLUMN_PREDICTED_LABEL, ""),
                COLUMN_CONFIDENCE: float(row.get(COLUMN_CONFIDENCE, 0.0) or 0.0),
            }
        )

    payload = {
        "threshold": float(threshold),
        "review_dir": str(review_root),
        "link_mode": link_mode,
        "confident_count": int(len(confident)),
        "uncertain_count": int(len(uncertain)),
    }
    if output_confident_path:
        payload["confident_output_path"] = _write_table(confident, output_confident_path)
    if output_uncertain_path:
        payload["uncertain_output_path"] = _write_table(uncertain, output_uncertain_path)
    if manifest_output_path:
        manifest_df = pd.DataFrame(
            manifest_rows,
            columns=[
                "file_path",
                "review_path",
                "filename",
                "folder_label",
                "predicted_label",
                "confidence",
            ],
        )
        payload["manifest_output_path"] = _write_table(manifest_df, manifest_output_path)
    return payload


def summarize_annotation_examples_impl(
    input_path: str,
    task: str,
    boundary_range_json: str = "",
    output_path: str = "",
    class_definitions: dict[str, str] | None = None,
) -> ToolResult:
    df = _read_table(input_path)
    if df.empty:
        raise ValueError("Labeled dataset is empty.")
    low, high = _boundary_range(boundary_range_json)

    class_candidates = [
        item
        for item in sorted(set(df[COLUMN_FOLDER_LABEL].fillna("").astype(str)).union(set(df[COLUMN_PREDICTED_LABEL].fillna("").astype(str))))
        if item and item != NO_DETECTION_LABEL
    ]
    examples: dict[str, list[str]] = {}
    for label in class_candidates:
        rows = df[df[COLUMN_FOLDER_LABEL].fillna(df[COLUMN_PREDICTED_LABEL]) == label].copy()
        if rows.empty:
            rows = df[df[COLUMN_PREDICTED_LABEL].fillna("") == label].copy()
        rows[COLUMN_CONFIDENCE] = pd.to_numeric(rows.get(COLUMN_CONFIDENCE), errors="coerce").fillna(0.0)
        selected = rows.sort_values(COLUMN_CONFIDENCE, ascending=False).head(3)
        examples[label] = selected[COLUMN_FILE_PATH].astype(str).tolist()

    edge_cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        confidence = float(row.get(COLUMN_CONFIDENCE, 0.0) or 0.0)
        predicted_label = str(row.get(COLUMN_PREDICTED_LABEL) or "")
        reasons: list[str] = []
        if low <= confidence <= high:
            reasons.append("boundary_confidence")
        if not _bool_from_value(row.get(COLUMN_FOLDER_MATCH, False)):
            reasons.append("folder_mismatch")
        detections = _parse_detection_list(row.get(COLUMN_ALL_DETECTIONS_JSON))
        detection_labels = {str(item.get("label", "")).strip() for item in detections if str(item.get("label", "")).strip()}
        if len(detection_labels) > 1:
            reasons.append("multi_class_detection")
        if predicted_label == NO_DETECTION_LABEL:
            reasons.append("no_detection")
        if reasons:
            edge_cases.append(
                {
                    COLUMN_FILE_PATH: str(row[COLUMN_FILE_PATH]),
                    "reason": ", ".join(reasons),
                    COLUMN_PREDICTED_LABEL: predicted_label,
                    COLUMN_FOLDER_LABEL: str(row.get(COLUMN_FOLDER_LABEL) or ""),
                    COLUMN_CONFIDENCE: confidence,
                }
            )

    definitions = _class_definitions(class_candidates)
    if class_definitions:
        definitions.update({str(key): str(value) for key, value in class_definitions.items() if str(key) in definitions})

    summary = AnnotationSpec(
        task_name=task,
        task_description=f"Image annotation task for '{task}'. Use the examples and class definitions below to label images consistently.",
        classes=definitions,
        examples=examples,
        edge_cases=edge_cases,
        guidelines=(
            "Assign the class that best matches the dominant object or scene in each image. "
            "When the image is ambiguous, prefer the folder taxonomy, note uncertainty, and send low-confidence cases for manual review. "
            f"When no class fits, use {NO_DETECTION_LABEL} and leave the bounding box empty."
        ),
    ).to_dict()
    payload = {
        "input_path": input_path,
        "task": task,
        "class_count": len(class_candidates),
        "edge_case_count": len(edge_cases),
        "summary": summary,
    }
    if output_path:
        payload["output_path"] = _write_json_artifact(payload, output_path)
    return payload


def generate_annotation_spec_impl(summary_path: str, task: str, output_path: str, spec_markdown: str = "") -> ToolResult:
    summary_payload = _read_json_artifact(summary_path)
    summary = summary_payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Summary artifact is missing the 'summary' object.")
    spec = AnnotationSpec(
        task_name=str(summary.get("task_name") or task),
        task_description=str(summary.get("task_description") or f"Image annotation task for '{task}'."),
        classes={str(key): str(value) for key, value in dict(summary.get("classes", {})).items()},
        examples={str(key): [str(item) for item in value] for key, value in dict(summary.get("examples", {})).items()},
        edge_cases=list(summary.get("edge_cases", [])),
        guidelines=str(summary.get("guidelines") or ""),
    )
    markdown = spec_markdown.strip() or _render_spec_markdown(spec)
    saved_path = _write_markdown(markdown, output_path)
    return {"summary_path": summary_path, "task": task, "output_path": saved_path}


def compute_annotation_quality_impl(
    input_path: str,
    output_path: str = "",
    human_labels_path: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ToolResult:
    df = _read_table(input_path)
    if human_labels_path:
        metrics = _compute_agreement(df, _read_table(human_labels_path), float(confidence_threshold))
    else:
        metrics = _quality_payload(df, float(confidence_threshold)).to_dict()
    payload = {
        "input_path": input_path,
        "human_labels_path": human_labels_path,
        "confidence_threshold": float(confidence_threshold),
        **metrics,
    }
    if output_path:
        payload["output_path"] = _write_json_artifact(payload, output_path)
    return payload


def export_labelstudio_predictions_impl(
    input_path: str,
    output_path: str,
    review_output_path: str = "",
    review_manifest_path: str = "",
    base_image_url: str = "/data/local-files/?d=",
    model_version: str | None = None,
) -> ToolResult:
    df = _read_table(input_path)
    effective_model_version = model_version or DEFAULT_MODEL_PATH
    all_records = []
    for _, row in df.iterrows():
        image_reference = f"{base_image_url}{row[COLUMN_FILE_PATH]}" if base_image_url else str(row[COLUMN_FILE_PATH])
        all_records.append(_labelstudio_record(row, image_reference, model_version=effective_model_version))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(_safe_json(all_records), ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "input_path": input_path,
        "output_path": str(output_file),
        "record_count": len(all_records),
        "model_version": effective_model_version,
    }
    if review_output_path:
        review_df = df.iloc[0:0].copy()
        if review_manifest_path and Path(review_manifest_path).exists():
            manifest = _read_table(review_manifest_path)
            review_df = df[df[COLUMN_FILENAME].isin(manifest[COLUMN_FILENAME].astype(str))].reset_index(drop=True)
        review_records = []
        for _, row in review_df.iterrows():
            image_reference = f"{base_image_url}{row[COLUMN_FILE_PATH]}" if base_image_url else str(row[COLUMN_FILE_PATH])
            review_records.append(_labelstudio_record(row, image_reference, model_version=effective_model_version))
        review_file = Path(review_output_path)
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps(_safe_json(review_records), ensure_ascii=False, indent=2), encoding="utf-8")
        payload["review_output_path"] = str(review_file)
        payload["review_record_count"] = len(review_records)
    return payload


@tool
def prepare_run_dir(artifacts_dir: str, dataset_name: str = "", run_id: str = "") -> str:
    """
    Create a run directory with the standard annotation artifact layout.

    Args:
        artifacts_dir: Base directory where runs are stored.
        dataset_name: Optional human-readable dataset name for the run prefix.
        run_id: Optional explicit run identifier. Empty = generate one.

    Returns:
        JSON with run_dir and standard subdirectories.
    """
    logger.info("prepare_run_dir called: artifacts_dir=%s dataset_name=%s", artifacts_dir, dataset_name)
    try:
        return _json_success(prepare_run_dir_impl(artifacts_dir, dataset_name=dataset_name, run_id=run_id))
    except Exception as exc:
        logger.exception("prepare_run_dir failed")
        return _json_error(str(exc))


@tool
def infer_classes_from_folders(dataset_dir: str) -> str:
    """
    Infer class names from first-level subdirectories of the dataset folder.

    Args:
        dataset_dir: Folder containing one subdirectory per class.

    Returns:
        JSON with inferred classes and empty-class information.
    """
    logger.info("infer_classes_from_folders called: dataset_dir=%s", dataset_dir)
    try:
        return _json_success(infer_classes_from_folders_impl(dataset_dir))
    except Exception as exc:
        logger.exception("infer_classes_from_folders failed")
        return _json_error(str(exc))


@tool
def scan_image_dataset(dataset_dir: str, output_path: str = "") -> str:
    """
    Scan a folder-organized image dataset into a flat manifest.

    Args:
        dataset_dir: Folder containing one subdirectory per class.
        output_path: Optional CSV path for the manifest.

    Returns:
        JSON with dataset summary and optional output_path.
    """
    logger.info("scan_image_dataset called: dataset_dir=%s", dataset_dir)
    try:
        return _json_success(scan_image_dataset_impl(dataset_dir, output_path=output_path))
    except Exception as exc:
        logger.exception("scan_image_dataset failed")
        return _json_error(str(exc))


@tool
def validate_image_dataset(dataset_dir: str, output_path: str = "") -> str:
    """
    Validate that image files exist and can be opened.

    Args:
        dataset_dir: Folder containing one subdirectory per class.
        output_path: Optional JSON artifact path.

    Returns:
        JSON with valid/broken counts, empty classes, and broken file details.
    """
    logger.info("validate_image_dataset called: dataset_dir=%s", dataset_dir)
    try:
        return _json_success(validate_image_dataset_impl(dataset_dir, output_path=output_path))
    except Exception as exc:
        logger.exception("validate_image_dataset failed")
        return _json_error(str(exc))


@tool
def run_yoloe_labeling(
    dataset_csv_path: str,
    classes_json: str,
    output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    mask_payload_path: str = "",
    masks_dir: str = "",
    batch_size: int = 16,
    report_output_path: str = "",
) -> str:
    """
    Run YOLOE inference over the dataset manifest and save the labeled manifest.

    Args:
        dataset_csv_path: CSV from scan_image_dataset.
        classes_json: JSON list of class names to set on YOLOE.
        output_path: CSV path for labeled output.
        model_path: YOLOE model path.
        mask_payload_path: Deprecated NPZ path placeholder kept for backwards compatibility.
        masks_dir: Optional directory where masks are flushed immediately.
        batch_size: Batch size for YOLOE predict.
        report_output_path: Optional JSON report path.

    Returns:
        JSON with labeled output path, basic detection stats, and optional report path.
    """
    logger.info("run_yoloe_labeling called: dataset_csv_path=%s model_path=%s", dataset_csv_path, model_path)
    try:
        classes = _parse_json_arg(classes_json, "classes_json")
        if not isinstance(classes, list) or not classes:
            raise ValueError("classes_json must be a non-empty JSON list of class names.")
        if mask_payload_path:
            _warn_deprecated_argument("run_yoloe_labeling", "mask_payload_path", replacement="masks_dir")
        return _json_success(
            run_yoloe_labeling_impl(
                dataset_csv_path=dataset_csv_path,
                classes=[str(item) for item in classes],
                output_path=output_path,
                model_path=model_path,
                masks_dir=masks_dir,
                batch_size=batch_size,
                report_output_path=report_output_path,
            )
        )
    except Exception as exc:
        logger.exception("run_yoloe_labeling failed")
        return _json_error(str(exc))


@tool
def save_segmentation_masks(
    input_path: str,
    output_path: str,
    masks_dir: str,
    mask_payload_path: str = "",
    image_format: str = "npy",
) -> str:
    """
    Persist raw segmentation masks from the YOLOE run into files and update mask_path.

    Args:
        input_path: Labeled CSV that may contain mask_key references.
        output_path: Final labeled CSV path with updated mask_path values.
        masks_dir: Directory where masks are saved.
        mask_payload_path: Optional NPZ path with raw mask arrays.
        image_format: "npy" or "png".

    Returns:
        JSON with saved CSV path and number of persisted masks.
    """
    logger.info("save_segmentation_masks called: input_path=%s masks_dir=%s", input_path, masks_dir)
    try:
        if mask_payload_path:
            _warn_deprecated_argument("save_segmentation_masks", "mask_payload_path")
        if image_format and image_format.lower() != "npy":
            _warn_deprecated_argument("save_segmentation_masks", "image_format", replacement="run_yoloe_labeling(..., masks_dir=...)")
        return _json_success(save_segmentation_masks_impl(input_path=input_path, output_path=output_path, masks_dir=masks_dir))
    except Exception as exc:
        logger.exception("save_segmentation_masks failed")
        return _json_error(str(exc))


@tool
def split_low_confidence_examples(
    input_path: str,
    review_dir: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    output_confident_path: str = "",
    output_uncertain_path: str = "",
    manifest_output_path: str = "",
    link_mode: str = DEFAULT_REVIEW_LINK_MODE,
) -> str:
    """
    Split labeled examples by confidence and copy low-confidence images for review.

    Args:
        input_path: Labeled CSV path.
        review_dir: Directory where low-confidence images are copied.
        threshold: Confidence threshold below which examples are routed for review.
        output_confident_path: Optional CSV path for confident examples.
        output_uncertain_path: Optional CSV path for uncertain examples.
        manifest_output_path: Optional CSV manifest path for review items.
        link_mode: "symlink", "copy", or "manifest_only" for review assets.

    Returns:
        JSON with counts and saved artifact paths.
    """
    logger.info("split_low_confidence_examples called: input_path=%s threshold=%s", input_path, threshold)
    try:
        return _json_success(
            split_low_confidence_examples_impl(
                input_path=input_path,
                review_dir=review_dir,
                threshold=threshold,
                output_confident_path=output_confident_path,
                output_uncertain_path=output_uncertain_path,
                manifest_output_path=manifest_output_path,
                link_mode=link_mode,
            )
        )
    except Exception as exc:
        logger.exception("split_low_confidence_examples failed")
        return _json_error(str(exc))


@tool
def summarize_annotation_examples(
    input_path: str,
    task: str,
    boundary_range_json: str = "",
    output_path: str = "",
    class_definitions_json: str = "",
) -> str:
    """
    Summarize high-confidence examples and edge cases for spec generation.

    Args:
        input_path: Labeled CSV path.
        task: Task name or description used in the spec.
        boundary_range_json: Optional JSON list [low, high] for boundary-confidence edge cases.
        output_path: Optional JSON path to save the summary.
        class_definitions_json: Optional JSON object with explicit class definitions.

    Returns:
        JSON with class definitions, examples, edge cases, and guidance context.
    """
    logger.info("summarize_annotation_examples called: input_path=%s task=%s", input_path, task)
    try:
        class_definitions = _parse_json_arg(class_definitions_json, "class_definitions_json") if class_definitions_json else None
        return _json_success(
            summarize_annotation_examples_impl(
                input_path=input_path,
                task=task,
                boundary_range_json=boundary_range_json,
                output_path=output_path,
                class_definitions=class_definitions,
            )
        )
    except Exception as exc:
        logger.exception("summarize_annotation_examples failed")
        return _json_error(str(exc))


@tool
def generate_annotation_spec(summary_path: str, task: str, output_path: str, spec_markdown: str = "") -> str:
    """
    Save an annotation spec Markdown document from a structured summary.

    Args:
        summary_path: JSON artifact path from summarize_annotation_examples.
        task: Task name or description.
        output_path: Markdown file path.
        spec_markdown: Optional LLM-authored Markdown. Empty = use deterministic fallback template.

    Returns:
        JSON with saved Markdown path.
    """
    logger.info("generate_annotation_spec called: summary_path=%s output_path=%s", summary_path, output_path)
    try:
        return _json_success(
            generate_annotation_spec_impl(summary_path=summary_path, task=task, output_path=output_path, spec_markdown=spec_markdown)
        )
    except Exception as exc:
        logger.exception("generate_annotation_spec failed")
        return _json_error(str(exc))


@tool
def compute_annotation_quality(
    input_path: str,
    output_path: str = "",
    human_labels_path: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Compute annotation quality metrics for auto labels, optionally against human labels.

    Args:
        input_path: Labeled CSV path.
        output_path: Optional JSON artifact path.
        human_labels_path: Optional CSV path with filename + human labels.
        confidence_threshold: Threshold used for low-confidence stats.

    Returns:
        JSON with label distribution, confidence stats, folder_match_rate, and optional agreement metrics.
    """
    logger.info("compute_annotation_quality called: input_path=%s human_labels_path=%s", input_path, human_labels_path)
    try:
        return _json_success(
            compute_annotation_quality_impl(
                input_path=input_path,
                output_path=output_path,
                human_labels_path=human_labels_path,
                confidence_threshold=confidence_threshold,
            )
        )
    except Exception as exc:
        logger.exception("compute_annotation_quality failed")
        return _json_error(str(exc))


@tool
def export_labelstudio_predictions(
    input_path: str,
    output_path: str,
    review_output_path: str = "",
    review_manifest_path: str = "",
    base_image_url: str = "/data/local-files/?d=",
    model_version: str = "",
) -> str:
    """
    Export auto-label predictions to Label Studio import JSON files.

    Args:
        input_path: Labeled CSV path.
        output_path: JSON output path for all images.
        review_output_path: Optional JSON output path for low-confidence review images.
        review_manifest_path: Optional manifest CSV from split_low_confidence_examples.
        base_image_url: Prefix used to build Label Studio local-files image references.
        model_version: Optional model version string stored in Label Studio predictions.

    Returns:
        JSON with saved output paths and record counts.
    """
    logger.info("export_labelstudio_predictions called: input_path=%s", input_path)
    try:
        return _json_success(
            export_labelstudio_predictions_impl(
                input_path=input_path,
                output_path=output_path,
                review_output_path=review_output_path,
                review_manifest_path=review_manifest_path,
                base_image_url=base_image_url,
                model_version=model_version or None,
            )
        )
    except Exception as exc:
        logger.exception("export_labelstudio_predictions failed")
        return _json_error(str(exc))
