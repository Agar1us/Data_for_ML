from __future__ import annotations

import json
import logging
import math
import re
import threading
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
from PIL import Image

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    def tool(func):  # type: ignore[misc]
        return func

from annotation_agent.config import (
    AL_HUMAN_LABELS_FILE_NAME,
    AL_LABELS_COLUMNS,
    AL_LABELS_FILE_NAME,
    COLUMN_ALL_DETECTIONS_JSON,
    COLUMN_BBOX_XYXY,
    COLUMN_FILE_PATH,
    COLUMN_FILENAME,
    COLUMN_FOLDER_LABEL,
    COLUMN_HAS_MASK,
    COLUMN_IMAGE_HEIGHT,
    COLUMN_IMAGE_WIDTH,
    COLUMN_LABEL_SOURCE,
    COLUMN_MASK_PATH,
    COLUMN_OBJECT_CONFIDENCE,
    COLUMN_OBJECT_DETECTED,
    COLUMN_OBJECT_LABEL,
    DEFAULT_BOUNDARY_CONFIDENCE_RANGE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LABEL_ASSIGNMENT_MODE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK_MODE,
    NO_DETECTION_LABEL,
    SUPPORTED_IMG_EXTENSIONS,
    ToolResult,
)
from annotation_agent.models import AnnotationSpec, Detection, QualityMetrics
from annotation_agent.reporting import ensure_run_layout


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


def _resolve_labelstudio_image_path(image_reference: str, local_files_document_root: str = "") -> str:
    reference = str(image_reference or "").strip()
    if not reference:
        raise ValueError("Label Studio export row is missing the 'image' field.")
    marker = "/data/local-files/?d="
    if reference.startswith(marker):
        relative_path = unquote(reference[len(marker):])
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return str(candidate.resolve())
        root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
        return str((root / candidate).resolve())
    candidate = Path(unquote(reference))
    if candidate.is_absolute():
        return str(candidate.resolve())
    root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
    return str((root / candidate).resolve())


def _parse_labelstudio_rectangles(payload: Any) -> list[dict[str, Any]]:
    if payload is None or payload == "":
        return []
    parsed = payload
    if isinstance(payload, str):
        parsed = json.loads(payload)
    if not isinstance(parsed, list):
        return []
    rectangles: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            rectangles.append(item)
    return rectangles


def _label_from_rectangles(rectangles: list[dict[str, Any]]) -> str:
    for item in rectangles:
        labels = item.get("rectanglelabels")
        if isinstance(labels, list) and labels:
            label = str(labels[0]).strip()
            if label:
                return label
    return ""


def _rectangle_to_xyxy(item: dict[str, Any]) -> tuple[list[float], int, int]:
    original_width = int(round(float(item.get("original_width", 0) or 0)))
    original_height = int(round(float(item.get("original_height", 0) or 0)))
    if original_width <= 0 or original_height <= 0:
        raise ValueError("Label Studio rectangle is missing original image dimensions.")
    x = float(item.get("x", 0.0) or 0.0)
    y = float(item.get("y", 0.0) or 0.0)
    width = float(item.get("width", 0.0) or 0.0)
    height = float(item.get("height", 0.0) or 0.0)
    x1 = (x / 100.0) * original_width
    y1 = (y / 100.0) * original_height
    x2 = ((x + width) / 100.0) * original_width
    y2 = ((y + height) / 100.0) * original_height
    return [float(x1), float(y1), float(x2), float(y2)], original_width, original_height


def _select_primary_bbox(detections: list[dict[str, Any]]) -> list[float] | None:
    if not detections:
        return None

    def area(item: dict[str, Any]) -> float:
        bbox = item.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            return -1.0
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    best = max(detections, key=area)
    bbox = best.get("bbox", [])
    return [float(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None


def _needs_manual_review(row: pd.Series, confidence_threshold: float) -> bool:
    confidence = float(row.get(COLUMN_OBJECT_CONFIDENCE, 0.0) or 0.0)
    object_detected = _bool_from_value(row.get(COLUMN_OBJECT_DETECTED, False))
    object_label = str(row.get(COLUMN_OBJECT_LABEL) or "").strip()
    return (not object_detected) or object_label == NO_DETECTION_LABEL or confidence < float(confidence_threshold)


def _build_object_level_labels_dataframe(
    df: pd.DataFrame,
    *,
    is_human_verified: bool,
    default_split: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        file_path = str(row.get(COLUMN_FILE_PATH) or "").strip()
        if not file_path:
            continue
        filename = str(row.get(COLUMN_FILENAME) or Path(file_path).name)
        image_width = int(float(row.get(COLUMN_IMAGE_WIDTH) or 0) or 0)
        image_height = int(float(row.get(COLUMN_IMAGE_HEIGHT) or 0) or 0)
        if image_width <= 0 or image_height <= 0:
            image_width, image_height = _open_image_size(file_path)
        class_label = str(row.get(COLUMN_FOLDER_LABEL) or "").strip()
        detections = _parse_detection_list(row.get(COLUMN_ALL_DETECTIONS_JSON))
        if not detections:
            fallback_bbox = _normalize_bbox_json(row.get(COLUMN_BBOX_XYXY))
            if len(fallback_bbox) == 4 and _bool_from_value(row.get(COLUMN_OBJECT_DETECTED, False)):
                detections = [{"bbox": fallback_bbox}]
        split = default_split
        if not is_human_verified:
            split = "review" if _needs_manual_review(row, confidence_threshold) else default_split
        for detection in detections:
            bbox = detection.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            rows.append(
                {
                    "file_path": str(Path(file_path).resolve()),
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "class_label": class_label,
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                    "is_human_verified": bool(is_human_verified),
                    "split": split,
                }
            )
    return pd.DataFrame(rows, columns=AL_LABELS_COLUMNS)


def build_object_labels_impl(
    input_path: str,
    output_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ToolResult:
    df = _read_table(input_path)
    labels_df = _build_object_level_labels_dataframe(
        df,
        is_human_verified=False,
        default_split="labeled",
        confidence_threshold=float(confidence_threshold),
    )
    saved_path = _write_table(labels_df, output_path)
    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": saved_path,
        "rows": int(len(labels_df)),
        "confidence_threshold": float(confidence_threshold),
    }


def convert_labelstudio_export_to_object_labels_impl(
    export_path: str,
    output_path: str,
    local_files_document_root: str = "",
) -> ToolResult:
    export_df = _read_table(export_path)
    required_columns = {"image", "label"}
    missing = sorted(required_columns - set(export_df.columns))
    if missing:
        raise ValueError(f"Label Studio CSV export is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in export_df.iterrows():
        file_path = _resolve_labelstudio_image_path(str(row.get("image") or ""), local_files_document_root=local_files_document_root)
        rectangles = _parse_labelstudio_rectangles(row.get("label"))
        for rectangle in rectangles:
            bbox_xyxy, width, height = _rectangle_to_xyxy(rectangle)
            class_label = _label_from_rectangles([rectangle])
            if not class_label:
                continue
            rows.append(
                {
                    "file_path": str(Path(file_path).resolve()),
                    "image_width": int(width),
                    "image_height": int(height),
                    "class_label": class_label,
                    "x1": float(bbox_xyxy[0]),
                    "y1": float(bbox_xyxy[1]),
                    "x2": float(bbox_xyxy[2]),
                    "y2": float(bbox_xyxy[3]),
                    "is_human_verified": True,
                    "split": "human_review",
                }
            )
    labels_df = pd.DataFrame(rows, columns=AL_LABELS_COLUMNS)
    saved_path = _write_table(labels_df, output_path)
    return {
        "export_path": str(Path(export_path).resolve()),
        "output_path": saved_path,
        "rows": int(len(labels_df)),
        "local_files_document_root": str(Path(local_files_document_root).resolve()) if local_files_document_root else str(Path.cwd().resolve()),
    }


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
    for candidate in ("human_label", "label", "annotated_label", "folder_label", "predicted_label"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Human labels file must contain one of: human_label, label, annotated_label, folder_label, predicted_label."
    )


def _class_definitions(classes: list[str]) -> dict[str, str]:
    return {
        label: (
            f"Assign '{label}' when the full image should be categorized under "
            f"'{label.replace('_', ' ').replace('-', ' ')}' according to the dataset taxonomy."
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


def _render_spec_markdown(spec: AnnotationSpec) -> str:
    lines = [
        f"# Annotation Specification: {spec.task_name}",
        "",
        "## Task Description",
        spec.task_description,
        "",
        "Object prompts used for localization/segmentation:",
        ", ".join(f"`{prompt}`" for prompt in spec.object_prompts) if spec.object_prompts else "No explicit object prompts provided.",
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
    lines.extend(["## Edge Cases"])
    if spec.edge_case_counts:
        lines.extend(["Edge case counts by reason:"])
        for reason, count in sorted(spec.edge_case_counts.items()):
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.extend(["| File | Reason | Object Label | Class |", "| --- | --- | --- | --- |"])
    if spec.edge_cases:
        for item in spec.edge_cases:
            lines.append(
                f"| `{item.get('file_path', '')}` | {item.get('reason', '')} | "
                f"{item.get('object_label', '')} | {item.get('folder_label', '')} |"
            )
    else:
        lines.append("| None | No edge cases detected | | |")
    lines.extend(["", "## Instructions", spec.guidelines.strip(), ""])
    return "\n".join(lines)


def _edge_case_reason_tokens(reason_value: Any) -> list[str]:
    text = str(reason_value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _compact_edge_cases(edge_cases: list[dict[str, Any]], max_total: int = 12, max_per_reason: int = 4, max_per_class_reason: int = 2) -> tuple[list[dict[str, Any]], dict[str, int]]:
    reason_counts: dict[str, int] = {}
    for item in edge_cases:
        for reason in _edge_case_reason_tokens(item.get("reason")):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if len(edge_cases) <= max_total:
        return edge_cases, reason_counts

    sampled: list[dict[str, Any]] = []
    per_reason_used: dict[str, int] = {}
    per_class_reason_used: dict[tuple[str, str], int] = {}
    seen_files: set[str] = set()
    priority_reasons = ("no_detection", "multiple_detections", "boundary_confidence")

    def _try_add(item: dict[str, Any], reason: str) -> bool:
        file_path = str(item.get(COLUMN_FILE_PATH) or "")
        class_label = str(item.get(COLUMN_FOLDER_LABEL) or "")
        key = (reason, class_label)
        if file_path in seen_files:
            return False
        if per_reason_used.get(reason, 0) >= max_per_reason:
            return False
        if per_class_reason_used.get(key, 0) >= max_per_class_reason:
            return False
        sampled.append(item)
        seen_files.add(file_path)
        per_reason_used[reason] = per_reason_used.get(reason, 0) + 1
        per_class_reason_used[key] = per_class_reason_used.get(key, 0) + 1
        return True

    for reason in priority_reasons:
        for item in edge_cases:
            if len(sampled) >= max_total:
                break
            if reason in _edge_case_reason_tokens(item.get("reason")):
                _try_add(item, reason)

    for item in edge_cases:
        if len(sampled) >= max_total:
            break
        reasons = _edge_case_reason_tokens(item.get("reason"))
        if not reasons:
            continue
        for reason in reasons:
            if _try_add(item, reason):
                break

    return sampled, reason_counts


def _quality_payload(df: pd.DataFrame, confidence_threshold: float) -> QualityMetrics:
    assigned = (
        df[COLUMN_FOLDER_LABEL].fillna("").astype(str).replace("", NO_DETECTION_LABEL).value_counts().to_dict()
        if COLUMN_FOLDER_LABEL in df.columns
        else {}
    )
    confidence_series = pd.to_numeric(df.get(COLUMN_OBJECT_CONFIDENCE, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    object_detected_series = (
        df.get(COLUMN_OBJECT_DETECTED, pd.Series(dtype=bool)).map(_bool_from_value)
        if COLUMN_OBJECT_DETECTED in df.columns
        else pd.Series(dtype=bool)
    )
    has_mask_series = (
        df.get(COLUMN_HAS_MASK, pd.Series(dtype=bool)).map(_bool_from_value)
        if COLUMN_HAS_MASK in df.columns
        else pd.Series(dtype=bool)
    )
    low_conf_mask = confidence_series < float(confidence_threshold)
    no_detection_count = int((~object_detected_series).sum()) if not object_detected_series.empty else 0
    metrics = QualityMetrics(
        kappa=None,
        percent_agreement=None,
        label_distribution={str(key): int(value) for key, value in assigned.items()},
        object_detection_rate=float(object_detected_series.mean()) if not object_detected_series.empty else 0.0,
        mask_rate=float(has_mask_series.mean()) if not has_mask_series.empty else 0.0,
        object_confidence_mean=float(confidence_series.mean()) if not confidence_series.empty else 0.0,
        object_confidence_std=float(confidence_series.std(ddof=0)) if len(confidence_series) > 1 else 0.0,
        low_confidence_count=int(low_conf_mask.sum()),
        low_confidence_ratio=float(low_conf_mask.mean()) if len(confidence_series) else 0.0,
        no_detection_count=no_detection_count,
        no_detection_ratio=float(no_detection_count / len(df)) if len(df) else 0.0,
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

    auto_label_column = COLUMN_FOLDER_LABEL
    auto_labels = merged[auto_label_column].fillna("").astype(str)
    human_labels = merged["human_label"].fillna("").astype(str)
    non_empty = (auto_labels != "") & (human_labels != "")
    merged = merged.loc[non_empty].reset_index(drop=True)
    if merged.empty:
        raise ValueError("Overlapping rows exist, but at least one side has only empty labels.")

    labels = sorted(set(merged[auto_label_column]).union(set(merged["human_label"])))
    if len(labels) < 2:
        raise ValueError("Need at least two distinct labels across auto and human annotations to compute Cohen's kappa.")

    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
    except ImportError as exc:
        raise ImportError("scikit-learn is required to compute agreement metrics with human labels.") from exc

    matrix = confusion_matrix(merged["human_label"], merged[auto_label_column], labels=labels)
    confusion_payload = {
        human_label: {pred_label: int(matrix[row_index, col_index]) for col_index, pred_label in enumerate(labels)}
        for row_index, human_label in enumerate(labels)
    }
    base_metrics = _quality_payload(df_auto, confidence_threshold).to_dict()
    base_metrics.update(
        {
            "kappa": float(cohen_kappa_score(merged["human_label"], merged[auto_label_column], labels=labels)),
            "percent_agreement": float((merged["human_label"] == merged[auto_label_column]).mean()),
            "confusion_matrix": confusion_payload,
            "agreement_rows": int(len(merged)),
            "human_label_column": label_column,
            "auto_label_column": auto_label_column,
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


def _labelstudio_image_reference(file_path: str, base_image_url: str, local_files_document_root: str = "") -> str:
    if not base_image_url:
        return str(file_path)
    if base_image_url.startswith("/data/local-files/?d="):
        document_root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
        try:
            relative_path = Path(file_path).resolve().relative_to(document_root)
        except ValueError as exc:
            raise ValueError(
                f"File path '{file_path}' is not inside Label Studio local files document root '{document_root}'."
            ) from exc
        return f"{base_image_url}{relative_path.as_posix()}"
    return f"{base_image_url}{file_path}"


def _labelstudio_record(row: pd.Series, image_reference: str, model_version: str) -> ToolResult:
    assigned_label = str(row.get(COLUMN_FOLDER_LABEL) or "").strip()
    object_detected = _bool_from_value(row.get(COLUMN_OBJECT_DETECTED, False))
    bbox = _normalize_bbox_json(row.get(COLUMN_BBOX_XYXY))
    predictions: list[dict[str, Any]] = []
    if assigned_label and object_detected and len(bbox) == 4:
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
                            "rectanglelabels": [assigned_label],
                        },
                        "score": float(row.get(COLUMN_OBJECT_CONFIDENCE, 0.0) or 0.0),
                    }
                ],
            }
        ]
    return {
        "data": {"image": image_reference},
        "predictions": predictions,
    }


def prepare_run_dir_impl(artifacts_dir: str) -> ToolResult:
    run_dir = ensure_run_layout(Path(artifacts_dir))
    return {
        "run_dir": str(run_dir),
        "reports_dir": str(run_dir / "reports"),
        "summary_dir": str(run_dir / "summary"),
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
    object_prompts: list[str],
    output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    masks_dir: str = "",
    batch_size: int = 16,
    report_output_path: str = "",
    task_mode: str = DEFAULT_TASK_MODE,
    label_assignment_mode: str = DEFAULT_LABEL_ASSIGNMENT_MODE,
) -> dict[str, Any]:
    if not object_prompts:
        raise ValueError("object_prompts must be a non-empty list of prompts.")
    if task_mode != DEFAULT_TASK_MODE:
        raise ValueError(f"Unsupported task_mode: {task_mode}")
    if label_assignment_mode != DEFAULT_LABEL_ASSIGNMENT_MODE:
        raise ValueError(f"Unsupported label_assignment_mode: {label_assignment_mode}")
    dataset_df = _read_table(dataset_csv_path)
    if dataset_df.empty:
        raise ValueError("Dataset manifest is empty; nothing to label.")

    model = _load_yoloe(model_path)
    if not hasattr(model, "set_classes"):
        raise AttributeError("Loaded YOLOE model does not expose set_classes().")
    model.set_classes(object_prompts)

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
            detections, batch_masks = _extract_detections(prediction, object_prompts, mask_key_prefix=f"row_{row_index}_")
            best = max(detections, key=lambda item: item.confidence, default=None)
            object_label = best.label if best else NO_DETECTION_LABEL
            mask_path = ""
            image_width, image_height = _open_image_size(str(source_row[COLUMN_FILE_PATH]))
            if best and best.mask_key and masks_dir and best.mask_key in batch_masks:
                mask_path = _save_mask_array(batch_masks[best.mask_key], masks_dir, str(source_row[COLUMN_FILENAME]), row_index)
            results_rows.append(
                {
                    COLUMN_FILE_PATH: source_row[COLUMN_FILE_PATH],
                    COLUMN_FILENAME: source_row[COLUMN_FILENAME],
                    COLUMN_FOLDER_LABEL: source_row[COLUMN_FOLDER_LABEL],
                    COLUMN_OBJECT_LABEL: object_label,
                    COLUMN_OBJECT_CONFIDENCE: float(best.confidence if best else 0.0),
                    COLUMN_OBJECT_DETECTED: bool(best),
                    COLUMN_BBOX_XYXY: json.dumps(best.bbox if best else []),
                    COLUMN_HAS_MASK: bool(mask_path),
                    COLUMN_MASK_PATH: mask_path,
                    COLUMN_ALL_DETECTIONS_JSON: json.dumps([item.to_dict() for item in detections], ensure_ascii=False),
                    COLUMN_LABEL_SOURCE: label_assignment_mode,
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
        "task_mode": task_mode,
        "label_assignment_mode": label_assignment_mode,
        "object_prompts": object_prompts,
        "model_version": _model_version_from_path(model_path),
        "detections_found": int(labeled_df[COLUMN_OBJECT_DETECTED].sum()),
        "no_detection_count": int((~labeled_df[COLUMN_OBJECT_DETECTED]).sum()),
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


def summarize_annotation_examples_impl(
    input_path: str,
    task: str,
    boundary_range_json: str = "",
    output_path: str = "",
    class_definitions: dict[str, str] | None = None,
    object_prompts: list[str] | None = None,
    label_assignment_mode: str = DEFAULT_LABEL_ASSIGNMENT_MODE,
) -> ToolResult:
    df = _read_table(input_path)
    if df.empty:
        raise ValueError("Labeled dataset is empty.")
    low, high = _boundary_range(boundary_range_json)

    class_candidates = [
        item
        for item in sorted(
            set(df[COLUMN_FOLDER_LABEL].fillna("").astype(str))
        )
        if item and item != NO_DETECTION_LABEL
    ]
    examples: dict[str, list[str]] = {}
    for label in class_candidates:
        rows = df[df[COLUMN_FOLDER_LABEL].fillna("") == label].copy()
        rows[COLUMN_OBJECT_CONFIDENCE] = pd.to_numeric(rows.get(COLUMN_OBJECT_CONFIDENCE), errors="coerce").fillna(0.0)
        selected = rows.sort_values(COLUMN_OBJECT_CONFIDENCE, ascending=False).head(3)
        examples[label] = selected[COLUMN_FILE_PATH].astype(str).tolist()

    raw_edge_cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        confidence = float(row.get(COLUMN_OBJECT_CONFIDENCE, 0.0) or 0.0)
        object_label = str(row.get(COLUMN_OBJECT_LABEL) or "")
        reasons: list[str] = []
        if low <= confidence <= high:
            reasons.append("boundary_confidence")
        detections = _parse_detection_list(row.get(COLUMN_ALL_DETECTIONS_JSON))
        if len(detections) > 1:
            reasons.append("multiple_detections")
        if not _bool_from_value(row.get(COLUMN_OBJECT_DETECTED, False)) or object_label == NO_DETECTION_LABEL:
            reasons.append("no_detection")
        if reasons:
            raw_edge_cases.append(
                {
                    COLUMN_FILE_PATH: str(row[COLUMN_FILE_PATH]),
                    "reason": ", ".join(reasons),
                    COLUMN_OBJECT_LABEL: object_label,
                    COLUMN_FOLDER_LABEL: str(row.get(COLUMN_FOLDER_LABEL) or ""),
                    COLUMN_OBJECT_CONFIDENCE: confidence,
                }
            )
    edge_cases, edge_case_counts = _compact_edge_cases(raw_edge_cases)

    definitions = _class_definitions(class_candidates)
    if class_definitions:
        definitions.update({str(key): str(value) for key, value in class_definitions.items() if str(key) in definitions})

    effective_object_prompts = [str(item).strip() for item in (object_prompts or []) if str(item).strip()]
    summary = AnnotationSpec(
        task_name=task,
        task_description=(
            f"Image annotation task for '{task}'. Use the semantic class taxonomy from the dataset folders. "
            f"Use the object prompts {effective_object_prompts or ['<missing>']} only for object localization and segmentation."
        ),
        object_prompts=effective_object_prompts,
        classes=definitions,
        examples=examples,
        edge_cases=edge_cases,
        edge_case_counts=edge_case_counts,
        guidelines=(
            "Assign the semantic class from the dataset taxonomy, not from the detector output. "
            "Use detector predictions only to localize or segment the target object. "
            f"If the target object is missing or the geometry is low confidence, keep the semantic class from the folder taxonomy, mark the case for manual review, and leave geometry empty when needed. "
            f"Label assignment mode: {label_assignment_mode}."
        ),
    ).to_dict()
    payload = {
        "input_path": input_path,
        "task": task,
        "class_count": len(class_candidates),
        "edge_case_count": len(raw_edge_cases),
        "edge_case_sample_count": len(edge_cases),
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
        object_prompts=[str(item) for item in list(summary.get("object_prompts", []))],
        classes={str(key): str(value) for key, value in dict(summary.get("classes", {})).items()},
        examples={str(key): [str(item) for item in value] for key, value in dict(summary.get("examples", {})).items()},
        edge_cases=list(summary.get("edge_cases", [])),
        edge_case_counts={str(key): int(value) for key, value in dict(summary.get("edge_case_counts", {})).items()},
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
    base_image_url: str = "/data/local-files/?d=",
    model_version: str | None = None,
    local_files_document_root: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ToolResult:
    df = _read_table(input_path)
    effective_model_version = model_version or DEFAULT_MODEL_PATH
    all_records = []
    for _, row in df.iterrows():
        image_reference = _labelstudio_image_reference(
            str(row[COLUMN_FILE_PATH]),
            base_image_url=base_image_url,
            local_files_document_root=local_files_document_root,
        )
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
        review_df = df[df.apply(lambda row: _needs_manual_review(row, float(confidence_threshold)), axis=1)].reset_index(drop=True)
        review_records = []
        for _, row in review_df.iterrows():
            image_reference = _labelstudio_image_reference(
                str(row[COLUMN_FILE_PATH]),
                base_image_url=base_image_url,
                local_files_document_root=local_files_document_root,
            )
            review_records.append(_labelstudio_record(row, image_reference, model_version=effective_model_version))
        review_file = Path(review_output_path)
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps(_safe_json(review_records), ensure_ascii=False, indent=2), encoding="utf-8")
        payload["review_output_path"] = str(review_file)
        payload["review_record_count"] = len(review_records)
        payload["review_confidence_threshold"] = float(confidence_threshold)
    payload["local_files_document_root"] = str(Path(local_files_document_root).resolve()) if local_files_document_root else str(Path.cwd().resolve())
    return payload


@tool
def prepare_run_dir(artifacts_dir: str) -> str:
    """
    Create the standard flat annotation artifact layout.

    Args:
        artifacts_dir: Base annotation directory.

    Returns:
        JSON with the annotation root and standard subdirectories.
    """
    logger.info("prepare_run_dir called: artifacts_dir=%s", artifacts_dir)
    try:
        return _json_success(prepare_run_dir_impl(artifacts_dir))
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
    output_path: str,
    object_prompts_json: str = "",
    model_path: str = DEFAULT_MODEL_PATH,
    mask_payload_path: str = "",
    masks_dir: str = "",
    batch_size: int = 16,
    report_output_path: str = "",
    task_mode: str = DEFAULT_TASK_MODE,
    label_assignment_mode: str = DEFAULT_LABEL_ASSIGNMENT_MODE,
    classes_json: str = "",
) -> str:
    """
    Run YOLOE inference over the dataset manifest and save the labeled manifest.

    Args:
        dataset_csv_path: CSV from scan_image_dataset.
        object_prompts_json: JSON list of object prompts to set on YOLOE.
        output_path: CSV path for labeled output.
        model_path: YOLOE model path.
        mask_payload_path: Deprecated NPZ path placeholder kept for backwards compatibility.
        masks_dir: Optional directory where masks are flushed immediately.
        batch_size: Batch size for YOLOE predict.
        report_output_path: Optional JSON report path.
        task_mode: Annotation task mode. v1 supports image_classification.
        label_assignment_mode: Semantic label assignment source. v1 supports folder_label.
        classes_json: Deprecated alias for object_prompts_json.

    Returns:
        JSON with labeled output path, basic detection stats, and optional report path.
    """
    logger.info("run_yoloe_labeling called: dataset_csv_path=%s model_path=%s", dataset_csv_path, model_path)
    try:
        prompts_payload = object_prompts_json or classes_json
        if classes_json and not object_prompts_json:
            _warn_deprecated_argument("run_yoloe_labeling", "classes_json", replacement="object_prompts_json")
        object_prompts = _parse_json_arg(prompts_payload, "object_prompts_json")
        if not isinstance(object_prompts, list) or not object_prompts:
            raise ValueError("object_prompts_json must be a non-empty JSON list of prompts.")
        if mask_payload_path:
            _warn_deprecated_argument("run_yoloe_labeling", "mask_payload_path", replacement="masks_dir")
        return _json_success(
            run_yoloe_labeling_impl(
                dataset_csv_path=dataset_csv_path,
                object_prompts=[str(item) for item in object_prompts],
                output_path=output_path,
                model_path=model_path,
                masks_dir=masks_dir,
                batch_size=batch_size,
                report_output_path=report_output_path,
                task_mode=task_mode,
                label_assignment_mode=label_assignment_mode,
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
def summarize_annotation_examples(
    input_path: str,
    task: str,
    boundary_range_json: str = "",
    output_path: str = "",
    class_definitions_json: str = "",
    object_prompts_json: str = "",
    label_assignment_mode: str = DEFAULT_LABEL_ASSIGNMENT_MODE,
) -> str:
    """
    Summarize high-confidence examples and edge cases for spec generation.

    Args:
        input_path: Labeled CSV path.
        task: Task name or description used in the spec.
        boundary_range_json: Optional JSON list [low, high] for boundary-confidence edge cases.
        output_path: Optional JSON path to save the summary.
        class_definitions_json: Optional JSON object with explicit class definitions.
        object_prompts_json: Optional JSON list of object prompts used for geometry extraction.
        label_assignment_mode: Source of semantic labels. v1 supports folder_label.

    Returns:
        JSON with class definitions, examples, edge cases, and guidance context.
    """
    logger.info("summarize_annotation_examples called: input_path=%s task=%s", input_path, task)
    try:
        class_definitions = _parse_json_arg(class_definitions_json, "class_definitions_json") if class_definitions_json else None
        object_prompts = _parse_json_arg(object_prompts_json, "object_prompts_json") if object_prompts_json else None
        return _json_success(
            summarize_annotation_examples_impl(
                input_path=input_path,
                task=task,
                boundary_range_json=boundary_range_json,
                output_path=output_path,
                class_definitions=class_definitions,
                object_prompts=object_prompts,
                label_assignment_mode=label_assignment_mode,
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
        JSON with semantic label distribution, object detection stats, confidence stats, and optional agreement metrics.
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
def build_object_labels(
    input_path: str,
    output_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Build a canonical object-level labels.csv for AL/detection from auto-labeled image rows.

    Args:
        input_path: Image-level labeled.csv path.
        output_path: Output CSV path with one row per bbox.
        confidence_threshold: Threshold below which auto-labeled detections are marked split=review.

    Returns:
        JSON with saved output path and row count.
    """
    logger.info("build_object_labels called: input_path=%s", input_path)
    try:
        return _json_success(
            build_object_labels_impl(
                input_path=input_path,
                output_path=output_path,
                confidence_threshold=confidence_threshold,
            )
        )
    except Exception as exc:
        logger.exception("build_object_labels failed")
        return _json_error(str(exc))


@tool
def convert_labelstudio_export_to_object_labels(
    export_path: str,
    output_path: str,
    local_files_document_root: str = "",
) -> str:
    """
    Convert Label Studio CSV export into canonical object-level human_labels.csv for AL/detection.

    Args:
        export_path: CSV export from Label Studio.
        output_path: Output CSV path with one row per human bbox.
        local_files_document_root: Host-side root used for /data/local-files/?d=... references.

    Returns:
        JSON with saved output path and row count.
    """
    logger.info("convert_labelstudio_export_to_object_labels called: export_path=%s", export_path)
    try:
        return _json_success(
            convert_labelstudio_export_to_object_labels_impl(
                export_path=export_path,
                output_path=output_path,
                local_files_document_root=local_files_document_root,
            )
        )
    except Exception as exc:
        logger.exception("convert_labelstudio_export_to_object_labels failed")
        return _json_error(str(exc))


@tool
def export_labelstudio_predictions(
    input_path: str,
    output_path: str,
    review_output_path: str = "",
    base_image_url: str = "/data/local-files/?d=",
    model_version: str = "",
    local_files_document_root: str = "",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """
    Export auto-label predictions to Label Studio import JSON files.

    Args:
        input_path: Labeled CSV path.
        output_path: JSON output path for all images.
        review_output_path: Optional JSON output path for low-confidence review images.
        base_image_url: Prefix used to build Label Studio local-files image references.
        model_version: Optional model version string stored in Label Studio predictions.
        local_files_document_root: Host-side root directory that Label Studio serves for /data/local-files/.
        confidence_threshold: Threshold below which auto-labeled examples are routed to review JSON.

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
                base_image_url=base_image_url,
                model_version=model_version or None,
                local_files_document_root=local_files_document_root,
                confidence_threshold=confidence_threshold,
            )
        )
    except Exception as exc:
        logger.exception("export_labelstudio_predictions failed")
        return _json_error(str(exc))
