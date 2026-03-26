from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import pandas as pd

from al_agent.common import (
    AL_LABEL_COLUMNS,
    CLASS_LABEL_COLUMN,
    DEFAULT_BASE_IMAGE_URL,
    FILE_PATH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IMAGE_ID_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    REVIEWED_IMAGES_COLUMNS,
    SPLIT_COLUMN,
    UNCERTAINTY_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
    _bool_from_value,
    _safe_json,
)


def _bbox_to_labelstudio(bbox: Iterable[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "x": (x1 / width) * 100.0,
        "y": (y1 / height) * 100.0,
        "width": ((x2 - x1) / width) * 100.0,
        "height": ((y2 - y1) / height) * 100.0,
    }


def build_labelstudio_config(classes: list[str]) -> str:
    labels = "\n".join(f'    <Label value="{escape(class_name, quote=True)}"/>' for class_name in classes if class_name)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


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


def predictions_from_labels(labels_df: pd.DataFrame, image_paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    selected = labels_df[labels_df[FILE_PATH_COLUMN].isin(set(image_paths))].reset_index(drop=True)
    if selected.empty:
        return {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    for file_path, group in selected.groupby(FILE_PATH_COLUMN):
        rows: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            rows.append(
                {
                    "class_label": str(row[CLASS_LABEL_COLUMN]),
                    "confidence": 1.0,
                    "bbox": [float(row[X1_COLUMN]), float(row[Y1_COLUMN]), float(row[X2_COLUMN]), float(row[Y2_COLUMN])],
                }
            )
        predictions[str(file_path)] = rows
    return predictions


def export_labelstudio_detection_batch(
    selected_images_df: pd.DataFrame,
    *,
    predictions: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    iteration: int | str,
    strategy: str,
    model_version: str,
    classes: list[str],
    base_image_url: str = DEFAULT_BASE_IMAGE_URL,
    local_files_document_root: str = "",
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "uncertain_manifest.csv"
    labelstudio_path = output_root / "labelstudio_import.json"
    labelstudio_config_path = output_root / "labelstudio_config.xml"

    manifest_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for _, row in selected_images_df.iterrows():
        file_path = str(row[FILE_PATH_COLUMN])
        width = int(row[IMAGE_WIDTH_COLUMN])
        height = int(row[IMAGE_HEIGHT_COLUMN])
        image_predictions = predictions.get(file_path, [])
        max_confidence = max((float(item["confidence"]) for item in image_predictions), default=0.0)
        uncertainty = 1.0 - max_confidence if image_predictions else 1.0
        manifest_rows.append(
            {
                FILE_PATH_COLUMN: file_path,
                IMAGE_WIDTH_COLUMN: width,
                IMAGE_HEIGHT_COLUMN: height,
                IMAGE_ID_COLUMN: str(row[IMAGE_ID_COLUMN]),
                "prediction_count": len(image_predictions),
                "max_confidence": max_confidence,
                UNCERTAINTY_COLUMN: uncertainty,
                "iteration": iteration,
                "strategy": strategy,
            }
        )
        predictions_payload = []
        if image_predictions:
            predictions_payload = [
                {
                    "model_version": model_version,
                    "result": [
                        {
                            "id": f"pred-{row[IMAGE_ID_COLUMN]}-{prediction_index}",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "value": {
                                **_bbox_to_labelstudio(prediction["bbox"], width, height),
                                "rectanglelabels": [str(prediction["class_label"])],
                            },
                            "score": float(prediction["confidence"]),
                        }
                        for prediction_index, prediction in enumerate(image_predictions)
                    ],
                }
            ]
        tasks.append(
            {
                "data": {
                    "image": _labelstudio_image_reference(
                        file_path,
                        base_image_url,
                        local_files_document_root=local_files_document_root,
                    ),
                    FILE_PATH_COLUMN: file_path,
                    IMAGE_ID_COLUMN: str(row[IMAGE_ID_COLUMN]),
                },
                "predictions": predictions_payload,
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    labelstudio_path.write_text(json.dumps(_safe_json(tasks), ensure_ascii=False, indent=2), encoding="utf-8")
    labelstudio_config_path.write_text(build_labelstudio_config(classes), encoding="utf-8")
    return {
        "iteration": iteration,
        "strategy": strategy,
        "manifest_path": str(manifest_path),
        "labelstudio_import_path": str(labelstudio_path),
        "labelstudio_config_path": str(labelstudio_config_path),
        "selected_count": int(len(selected_images_df)),
    }


def _resolve_labelstudio_image_path(image_reference: str, local_files_document_root: str = "") -> str:
    value = str(image_reference or "").strip()
    if not value:
        return ""
    prefix = "/data/local-files/?d="
    if value.startswith(prefix):
        relative = unquote(value[len(prefix) :]).lstrip("/")
        root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
        return str((root / relative).resolve())
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate.resolve())
    root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
    return str((root / value).resolve())


def _parse_labelstudio_rectangles(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return []
        parsed = json.loads(stripped)
    else:
        parsed = payload
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    rectangles = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "rectanglelabels":
            rectangles.append(item)
            continue
        if "value" in item and isinstance(item["value"], dict) and item["value"].get("rectanglelabels"):
            rectangles.append(item)
    return rectangles


def _labelstudio_rectangle_to_row(file_path: str, rectangle: dict[str, Any]) -> dict[str, Any]:
    value = rectangle.get("value") or {}
    labels = value.get("rectanglelabels") or []
    if not labels:
        raise ValueError(f"Rectangle is missing rectanglelabels: {rectangle}")
    image_width = int(rectangle.get("original_width") or 0)
    image_height = int(rectangle.get("original_height") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Rectangle is missing original dimensions: {rectangle}")
    x = float(value.get("x") or 0.0)
    y = float(value.get("y") or 0.0)
    width_pct = float(value.get("width") or 0.0)
    height_pct = float(value.get("height") or 0.0)
    x1 = (x / 100.0) * image_width
    y1 = (y / 100.0) * image_height
    x2 = ((x + width_pct) / 100.0) * image_width
    y2 = ((y + height_pct) / 100.0) * image_height
    return {
        FILE_PATH_COLUMN: str(Path(file_path).resolve()),
        IMAGE_WIDTH_COLUMN: image_width,
        IMAGE_HEIGHT_COLUMN: image_height,
        CLASS_LABEL_COLUMN: str(labels[0]),
        X1_COLUMN: float(x1),
        Y1_COLUMN: float(y1),
        X2_COLUMN: float(x2),
        Y2_COLUMN: float(y2),
        IS_HUMAN_VERIFIED_COLUMN: True,
        SPLIT_COLUMN: "labeled",
    }


def _extract_reviewed_file_paths(export_path: str, local_files_document_root: str = "") -> set[str]:
    path = Path(export_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        image_column = "image" if "image" in df.columns else FILE_PATH_COLUMN
        return {
            _resolve_labelstudio_image_path(str(value), local_files_document_root=local_files_document_root)
            for value in df[image_column].dropna().tolist()
            if str(value).strip()
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    reviewed_paths: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        data = item.get("data") or {}
        image_reference = str(data.get("image") or "").strip()
        file_path = str(data.get(FILE_PATH_COLUMN) or "").strip()
        resolved = _resolve_labelstudio_image_path(image_reference, local_files_document_root=local_files_document_root) if image_reference else ""
        if not resolved and file_path:
            resolved = str(Path(file_path).resolve())
        if resolved:
            reviewed_paths.add(resolved)
    return reviewed_paths


def import_labelstudio_detection_export(
    export_path: str,
    *,
    local_files_document_root: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(export_path)
    if not path.exists():
        raise FileNotFoundError(f"Label Studio export does not exist: {path}")

    rows: list[dict[str, Any]] = []
    reviewed_rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        label_column = "label" if "label" in df.columns else ""
        for _, row in df.iterrows():
            file_path = _resolve_labelstudio_image_path(str(row.get("image") or ""), local_files_document_root=local_files_document_root)
            rectangles = _parse_labelstudio_rectangles(row.get(label_column)) if label_column else []
            for rectangle in rectangles:
                rows.append(_labelstudio_rectangle_to_row(file_path, rectangle))
        reviewed_paths = _extract_reviewed_file_paths(export_path, local_files_document_root=local_files_document_root)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        reviewed_paths = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            data = item.get("data") or {}
            image_reference = str(data.get("image") or "").strip()
            file_path = str(data.get(FILE_PATH_COLUMN) or "").strip()
            resolved = _resolve_labelstudio_image_path(image_reference, local_files_document_root=local_files_document_root) if image_reference else ""
            if not resolved and file_path:
                resolved = str(Path(file_path).resolve())
            if not resolved:
                continue
            reviewed_paths.add(resolved)
            annotations = item.get("annotations") or []
            if not annotations:
                continue
            for annotation in annotations:
                rectangles = _parse_labelstudio_rectangles(annotation.get("result", []))
                for rectangle in rectangles:
                    rows.append(_labelstudio_rectangle_to_row(resolved, rectangle))

    human_labels_df = pd.DataFrame(rows, columns=AL_LABEL_COLUMNS)
    if not human_labels_df.empty:
        human_labels_df[IS_HUMAN_VERIFIED_COLUMN] = True
        human_labels_df[SPLIT_COLUMN] = "labeled"
    known_sizes = (
        human_labels_df[[FILE_PATH_COLUMN, IMAGE_WIDTH_COLUMN, IMAGE_HEIGHT_COLUMN]]
        .drop_duplicates(subset=[FILE_PATH_COLUMN])
        .set_index(FILE_PATH_COLUMN)
        .to_dict("index")
        if not human_labels_df.empty
        else {}
    )
    reviewed_rows = []
    human_labeled_paths = set(human_labels_df[FILE_PATH_COLUMN].tolist())
    for file_path in sorted(reviewed_paths):
        size_payload = known_sizes.get(file_path)
        if size_payload:
            width = int(size_payload[IMAGE_WIDTH_COLUMN])
            height = int(size_payload[IMAGE_HEIGHT_COLUMN])
        elif Path(file_path).exists():
            from al_agent.data import _open_image_size

            width, height = _open_image_size(file_path)
        else:
            width, height = 0, 0
        reviewed_rows.append(
            {
                FILE_PATH_COLUMN: file_path,
                IMAGE_WIDTH_COLUMN: width,
                IMAGE_HEIGHT_COLUMN: height,
                IS_HUMAN_VERIFIED_COLUMN: True,
                "has_boxes": file_path in human_labeled_paths,
                SPLIT_COLUMN: "labeled",
            }
        )
    reviewed_df = pd.DataFrame(reviewed_rows, columns=REVIEWED_IMAGES_COLUMNS)
    return human_labels_df, reviewed_df
