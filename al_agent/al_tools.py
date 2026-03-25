from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("scikit-learn is required for Active Learning.") from exc

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    def tool(func):  # type: ignore[misc]
        return func


logger = logging.getLogger(__name__)

FILE_PATH_COLUMN = "file_path"
IMAGE_WIDTH_COLUMN = "image_width"
IMAGE_HEIGHT_COLUMN = "image_height"
CLASS_LABEL_COLUMN = "class_label"
X1_COLUMN = "x1"
Y1_COLUMN = "y1"
X2_COLUMN = "x2"
Y2_COLUMN = "y2"
IS_HUMAN_VERIFIED_COLUMN = "is_human_verified"
SPLIT_COLUMN = "split"
IMAGE_ID_COLUMN = "image_id"
UNCERTAINTY_COLUMN = "uncertainty_score"

AL_LABEL_COLUMNS = [
    FILE_PATH_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    CLASS_LABEL_COLUMN,
    X1_COLUMN,
    Y1_COLUMN,
    X2_COLUMN,
    Y2_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    SPLIT_COLUMN,
]
REVIEWED_IMAGES_COLUMNS = [
    FILE_PATH_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    "has_boxes",
    SPLIT_COLUMN,
]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_AL_ARTIFACTS_DIR = "data/current_run/al"
DEFAULT_IMAGE_MODEL_PATH = "yolo26x.pt"
DEFAULT_IMAGE_STRATEGIES = ("confidence",)
DEFAULT_N_ITERATIONS = 1
DEFAULT_BATCH_SIZE = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_BASE_IMAGE_URL = "/data/local-files/?d="
DEFAULT_LINK_MODE = "symlink"
DEFAULT_EPOCHS = 5
DEFAULT_IMGSZ = 640
DEFAULT_TRAIN_BATCH = 8
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_WAIT_FOR_HUMAN_FEEDBACK = True
DEFAULT_HUMAN_WAIT_TIMEOUT_SEC = 86400
DEFAULT_HUMAN_POLL_INTERVAL_SEC = 5.0


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
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _json_success(payload: dict[str, Any]) -> str:
    return json.dumps({"success": True, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _json_error(message: str, **payload: Any) -> str:
    return json.dumps({"success": False, "error": message, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _slugify(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "active_learning"


def make_run_id(prefix: str = "al") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def ensure_al_run_layout(base_dir: str | Path) -> Path:
    run_dir = Path(base_dir)
    for subdir in ("reports", "labelstudio", "models", "datasets", "splits", "curves"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def _read_jsonish(config_json: str | dict[str, Any] | None) -> dict[str, Any]:
    if config_json is None or config_json == "":
        return {}
    if isinstance(config_json, dict):
        return dict(config_json)
    candidate = Path(config_json)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(config_json)


def _write_json(path: str | Path, payload: Any) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _write_table(df: pd.DataFrame, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _read_table(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def _stable_image_id(file_path: str) -> str:
    return hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:16]


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _resolve_labels_csv_path(labeled_data_path: str) -> Path:
    input_path = Path(labeled_data_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Labels path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {input_path}")
        return input_path
    candidates = [
        input_path / "reports" / "labels.csv",
        input_path / "labels.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve labels.csv from {input_path}. Expected one of: {candidates}")


def load_detection_labels(labeled_data_path: str) -> pd.DataFrame:
    labels_csv = _resolve_labels_csv_path(labeled_data_path)
    df = pd.read_csv(labels_csv)
    missing = [column for column in AL_LABEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {labels_csv}: {missing}")
    normalized = df.copy()
    normalized[FILE_PATH_COLUMN] = normalized[FILE_PATH_COLUMN].astype(str).map(lambda value: str(Path(value).resolve()))
    for column in (IMAGE_WIDTH_COLUMN, IMAGE_HEIGHT_COLUMN):
        normalized[column] = normalized[column].astype(int)
    for column in (X1_COLUMN, Y1_COLUMN, X2_COLUMN, Y2_COLUMN):
        normalized[column] = normalized[column].astype(float)
    normalized[CLASS_LABEL_COLUMN] = normalized[CLASS_LABEL_COLUMN].astype(str)
    normalized[IS_HUMAN_VERIFIED_COLUMN] = normalized[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)
    normalized[SPLIT_COLUMN] = normalized[SPLIT_COLUMN].astype(str)
    return normalized


def _resolve_dataset_dir(labels_df: pd.DataFrame, config: dict[str, Any]) -> Path:
    configured = str(config.get("dataset_dir") or "").strip()
    if configured:
        dataset_dir = Path(configured).resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Configured dataset_dir does not exist: {dataset_dir}")
        return dataset_dir
    common_root = Path(os.path.commonpath(labels_df[FILE_PATH_COLUMN].tolist())).resolve()
    if common_root.is_file():
        common_root = common_root.parent
    if any(path.is_dir() for path in common_root.iterdir()):
        return common_root
    return common_root.parent


def _open_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _scan_dataset_images(dataset_dir: Path, labels_df: pd.DataFrame) -> pd.DataFrame:
    known_sizes = (
        labels_df[[FILE_PATH_COLUMN, IMAGE_WIDTH_COLUMN, IMAGE_HEIGHT_COLUMN]]
        .drop_duplicates(subset=[FILE_PATH_COLUMN])
        .set_index(FILE_PATH_COLUMN)
        .to_dict("index")
        if not labels_df.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for image_path in sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS):
        resolved = str(image_path.resolve())
        if resolved in known_sizes:
            width = int(known_sizes[resolved][IMAGE_WIDTH_COLUMN])
            height = int(known_sizes[resolved][IMAGE_HEIGHT_COLUMN])
        else:
            width, height = _open_image_size(resolved)
        rows.append(
            {
                FILE_PATH_COLUMN: resolved,
                IMAGE_WIDTH_COLUMN: width,
                IMAGE_HEIGHT_COLUMN: height,
                IMAGE_ID_COLUMN: _stable_image_id(resolved),
            }
        )
    if not rows:
        raise RuntimeError(f"No supported images found under dataset_dir: {dataset_dir}")
    return pd.DataFrame(rows)


def _load_reviewed_images(reviewed_images_path: str = "") -> pd.DataFrame:
    if not reviewed_images_path:
        return pd.DataFrame(columns=REVIEWED_IMAGES_COLUMNS)
    path = Path(reviewed_images_path)
    if not path.exists():
        return pd.DataFrame(columns=REVIEWED_IMAGES_COLUMNS)
    df = pd.read_csv(path)
    for column in REVIEWED_IMAGES_COLUMNS:
        if column not in df.columns:
            if column == "has_boxes":
                df[column] = False
            elif column == IS_HUMAN_VERIFIED_COLUMN:
                df[column] = False
            else:
                df[column] = ""
    df[FILE_PATH_COLUMN] = df[FILE_PATH_COLUMN].astype(str).map(lambda value: str(Path(value).resolve()))
    df[IS_HUMAN_VERIFIED_COLUMN] = df[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)
    df["has_boxes"] = df["has_boxes"].map(_bool_from_value)
    return df[REVIEWED_IMAGES_COLUMNS].drop_duplicates(subset=[FILE_PATH_COLUMN], keep="last")


def _primary_class_per_image(labels_df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if labels_df.empty:
        return mapping
    grouped = labels_df.groupby(FILE_PATH_COLUMN)[CLASS_LABEL_COLUMN].agg(list)
    for file_path, labels in grouped.items():
        if not labels:
            mapping[str(file_path)] = "__negative__"
            continue
        counts = pd.Series(labels).value_counts()
        mapping[str(file_path)] = str(counts.index[0])
    return mapping


def build_image_inventory(
    labels_df: pd.DataFrame,
    *,
    dataset_dir: str | Path,
    reviewed_images_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reviewed = reviewed_images_df if reviewed_images_df is not None else pd.DataFrame(columns=REVIEWED_IMAGES_COLUMNS)
    images_df = _scan_dataset_images(Path(dataset_dir), labels_df)
    label_groups = {
        file_path: group.reset_index(drop=True)
        for file_path, group in labels_df.groupby(FILE_PATH_COLUMN)
    }
    reviewed_map = (
        reviewed.drop_duplicates(subset=[FILE_PATH_COLUMN], keep="last").set_index(FILE_PATH_COLUMN).to_dict("index")
        if not reviewed.empty
        else {}
    )
    primary_labels = _primary_class_per_image(labels_df)
    rows: list[dict[str, Any]] = []
    for _, image_row in images_df.iterrows():
        file_path = str(image_row[FILE_PATH_COLUMN])
        image_width = int(image_row[IMAGE_WIDTH_COLUMN])
        image_height = int(image_row[IMAGE_HEIGHT_COLUMN])
        label_rows = label_groups.get(file_path)
        review_row = reviewed_map.get(file_path, {})
        has_boxes = bool(label_rows is not None and not label_rows.empty)
        if review_row:
            split = str(review_row.get(SPLIT_COLUMN) or "labeled")
            is_human_verified = _bool_from_value(review_row.get(IS_HUMAN_VERIFIED_COLUMN, False))
            has_boxes = _bool_from_value(review_row.get("has_boxes", has_boxes))
        elif has_boxes:
            split = "review" if (label_rows[SPLIT_COLUMN].astype(str) == "review").any() else "labeled"
            is_human_verified = bool(label_rows[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value).any())
        else:
            split = "pool"
            is_human_verified = False
        rows.append(
            {
                FILE_PATH_COLUMN: file_path,
                IMAGE_ID_COLUMN: _stable_image_id(file_path),
                IMAGE_WIDTH_COLUMN: image_width,
                IMAGE_HEIGHT_COLUMN: image_height,
                "primary_class_label": primary_labels.get(file_path, "__negative__"),
                IS_HUMAN_VERIFIED_COLUMN: bool(is_human_verified),
                "has_boxes": bool(has_boxes),
                SPLIT_COLUMN: split,
            }
        )
    return pd.DataFrame(rows)


def _normalize_test_count(n_rows: int, test_size: float | int) -> int:
    if n_rows <= 1:
        return 0
    requested = int(round(n_rows * test_size)) if isinstance(test_size, float) and test_size < 1 else int(test_size)
    requested = max(1, requested)
    return min(requested, n_rows - 1)


def _prepare_detection_splits(
    inventory_df: pd.DataFrame,
    *,
    test_size: float | int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled_images = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    pool_images = inventory_df[inventory_df[SPLIT_COLUMN] != "labeled"].reset_index(drop=True)
    if len(labeled_images) < 2:
        raise ValueError("At least 2 labeled images are required for detection active learning.")
    test_count = _normalize_test_count(len(labeled_images), test_size)
    stratify = labeled_images["primary_class_label"] if labeled_images["primary_class_label"].value_counts().min() >= 2 else None
    if test_count == 0:
        return labeled_images, labeled_images.iloc[0:0].copy(), pool_images
    train_df, test_df = train_test_split(
        labeled_images,
        test_size=test_count,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), pool_images


def _link_or_copy_image(source: str | Path, target: str | Path, link_mode: str = DEFAULT_LINK_MODE) -> None:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if link_mode == "symlink":
        try:
            target_path.symlink_to(source_path.resolve())
            return
        except OSError:
            logger.warning("Falling back to copy for %s -> %s", source_path, target_path)
    shutil.copy2(source_path, target_path)


def _bbox_xyxy_to_yolo(bbox: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    cx = x1 + box_width / 2.0
    cy = y1 + box_height / 2.0
    return cx / width, cy / height, box_width / width, box_height / height


def _materialize_detection_subset(
    image_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    subset_root: Path,
    *,
    class_to_idx: dict[str, int],
    link_mode: str,
) -> None:
    images_dir = subset_root / "images"
    labels_dir = subset_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_groups = {
        file_path: group.reset_index(drop=True)
        for file_path, group in labels_df.groupby(FILE_PATH_COLUMN)
    }
    for _, row in image_df.iterrows():
        file_path = str(row[FILE_PATH_COLUMN])
        source = Path(file_path)
        target_image = images_dir / source.name
        _link_or_copy_image(source, target_image, link_mode=link_mode)
        label_path = labels_dir / f"{source.stem}.txt"
        group = label_groups.get(file_path)
        lines: list[str] = []
        if group is not None and not group.empty:
            width = int(row[IMAGE_WIDTH_COLUMN])
            height = int(row[IMAGE_HEIGHT_COLUMN])
            for _, label_row in group.iterrows():
                class_index = class_to_idx[str(label_row[CLASS_LABEL_COLUMN])]
                bbox = (
                    float(label_row[X1_COLUMN]),
                    float(label_row[Y1_COLUMN]),
                    float(label_row[X2_COLUMN]),
                    float(label_row[Y2_COLUMN]),
                )
                cx, cy, box_width, box_height = _bbox_xyxy_to_yolo(bbox, width, height)
                lines.append(f"{class_index} {cx:.6f} {cy:.6f} {box_width:.6f} {box_height:.6f}")
        label_path.write_text("\n".join(lines), encoding="utf-8")


def build_yolo_detection_dataset(
    train_images_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    val_images_df: pd.DataFrame,
    val_labels_df: pd.DataFrame,
    *,
    classes: list[str],
    output_dir: str | Path,
    link_mode: str = DEFAULT_LINK_MODE,
) -> Path:
    dataset_root = Path(output_dir)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    class_to_idx = {label: index for index, label in enumerate(classes)}
    _materialize_detection_subset(
        train_images_df,
        train_labels_df,
        dataset_root / "train",
        class_to_idx=class_to_idx,
        link_mode=link_mode,
    )
    _materialize_detection_subset(
        val_images_df,
        val_labels_df,
        dataset_root / "val",
        class_to_idx=class_to_idx,
        link_mode=link_mode,
    )
    data_yaml = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "val/images",
        "names": {index: label for index, label in enumerate(classes)},
    }
    _write_json(dataset_root / "data.yaml", data_yaml)
    return dataset_root


class YOLODetectionBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_IMAGE_MODEL_PATH,
        *,
        epochs: int = DEFAULT_EPOCHS,
        imgsz: int = DEFAULT_IMGSZ,
        train_batch: int = DEFAULT_TRAIN_BATCH,
        predict_batch: int = DEFAULT_TRAIN_BATCH,
        device: str | int | None = None,
        link_mode: str = DEFAULT_LINK_MODE,
    ) -> None:
        self.model_path = model_path
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.train_batch = int(train_batch)
        self.predict_batch = int(predict_batch)
        self.device = device
        self.link_mode = link_mode

    def _load_model(self, weights_path: str | None = None) -> Any:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ultralytics is required for detection active learning.") from exc
        load_path = weights_path or self.model_path
        return YOLO(load_path, task="detect")

    def train(
        self,
        *,
        train_images_df: pd.DataFrame,
        train_labels_df: pd.DataFrame,
        val_images_df: pd.DataFrame,
        val_labels_df: pd.DataFrame,
        classes: list[str],
        work_dir: str | Path,
        iteration_name: str,
    ) -> dict[str, Any]:
        dataset_root = build_yolo_detection_dataset(
            train_images_df,
            train_labels_df,
            val_images_df,
            val_labels_df,
            classes=classes,
            output_dir=Path(work_dir) / "datasets" / iteration_name,
            link_mode=self.link_mode,
        )
        model = self._load_model()
        model.train(
            data=str(dataset_root / "data.yaml"),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.train_batch,
            project=str(Path(work_dir) / "models"),
            name=iteration_name,
            exist_ok=True,
            verbose=False,
            device=self.device,
        )
        save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(work_dir) / "models" / iteration_name))
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        weights_path = best_weights if best_weights.exists() else last_weights if last_weights.exists() else Path(self.model_path)
        trained_model = self._load_model(str(weights_path))
        return {
            "model": trained_model,
            "model_dir": str(save_dir),
            "weights_path": str(weights_path),
            "dataset_root": str(dataset_root),
            "classes": classes,
        }

    def predict(
        self,
        model: Any,
        image_paths: Iterable[str],
        classes: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        image_path_list = [str(path) for path in image_paths]
        if not image_path_list:
            return {}
        results = model.predict(
            source=image_path_list,
            imgsz=self.imgsz,
            batch=self.predict_batch,
            verbose=False,
            device=self.device,
        )
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            model_labels = {int(index): str(label) for index, label in names.items()}
        elif isinstance(names, list):
            model_labels = {index: str(label) for index, label in enumerate(names)}
        else:
            model_labels = {index: label for index, label in enumerate(classes)}

        output: dict[str, list[dict[str, Any]]] = {}
        for image_path, result in zip(image_path_list, results):
            boxes = getattr(result, "boxes", None)
            detections: list[dict[str, Any]] = []
            if boxes is not None and len(boxes) > 0:
                class_ids = np.asarray(getattr(getattr(boxes, "cls", []), "cpu", lambda: getattr(boxes, "cls", []))())
                confidences = np.asarray(getattr(getattr(boxes, "conf", []), "cpu", lambda: getattr(boxes, "conf", []))())
                xyxy = np.asarray(getattr(getattr(boxes, "xyxy", []), "cpu", lambda: getattr(boxes, "xyxy", []))())
                for index in range(len(class_ids)):
                    label = model_labels.get(int(class_ids[index]), str(int(class_ids[index])))
                    detections.append(
                        {
                            "class_label": label,
                            "confidence": float(confidences[index]),
                            "bbox": [float(value) for value in xyxy[index].tolist()],
                        }
                    )
            output[str(Path(image_path).resolve())] = detections
        return output


def _iou_xyxy(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _evaluate_detection_metrics(
    *,
    eval_images_df: pd.DataFrame,
    eval_labels_df: pd.DataFrame,
    predictions: dict[str, list[dict[str, Any]]],
    classes: list[str],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    gt_by_image_class: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for _, row in eval_labels_df.iterrows():
        gt_by_image_class[str(row[FILE_PATH_COLUMN])][str(row[CLASS_LABEL_COLUMN])].append(
            [float(row[X1_COLUMN]), float(row[Y1_COLUMN]), float(row[X2_COLUMN]), float(row[Y2_COLUMN])]
        )

    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in classes}
    for _, image_row in eval_images_df.iterrows():
        file_path = str(image_row[FILE_PATH_COLUMN])
        image_predictions = predictions.get(file_path, [])
        pred_by_class: dict[str, list[list[float]]] = defaultdict(list)
        for item in image_predictions:
            pred_by_class[str(item["class_label"])].append([float(value) for value in item["bbox"]])
        for label in classes:
            gt_boxes = list(gt_by_image_class[file_path].get(label, []))
            pred_boxes = list(pred_by_class.get(label, []))
            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            for pred_index, pred_box in enumerate(pred_boxes):
                best_iou = 0.0
                best_gt_index = None
                for gt_index, gt_box in enumerate(gt_boxes):
                    if gt_index in matched_gt:
                        continue
                    iou = _iou_xyxy(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index
                if best_gt_index is not None and best_iou >= iou_threshold:
                    matched_gt.add(best_gt_index)
                    matched_pred.add(pred_index)
            stats[label]["tp"] += len(matched_gt)
            stats[label]["fp"] += len(pred_boxes) - len(matched_pred)
            stats[label]["fn"] += len(gt_boxes) - len(matched_gt)

    class_metrics: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    for label in classes:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "precision_macro": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_macro": float(np.mean(recall_values)) if recall_values else 0.0,
        "f1_macro": float(np.mean(f1_values)) if f1_values else 0.0,
        "class_metrics": class_metrics,
    }


def _confidence_scores(predictions: dict[str, list[dict[str, Any]]], image_paths: list[str]) -> np.ndarray:
    scores = []
    for file_path in image_paths:
        detections = predictions.get(file_path, [])
        if not detections:
            scores.append(1.0)
        else:
            scores.append(1.0 - max(float(item["confidence"]) for item in detections))
    return np.asarray(scores, dtype=float)


def select_uncertain_images(
    image_paths: list[str],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    strategy: str,
    batch_size: int,
    random_state: int,
) -> list[str]:
    if batch_size <= 0 or not image_paths:
        return []
    normalized = strategy.strip().lower()
    if normalized == "random":
        rng = np.random.default_rng(random_state)
        order = np.arange(len(image_paths))
        rng.shuffle(order)
        return [image_paths[int(index)] for index in order[: min(batch_size, len(order))]]
    if normalized != "confidence":
        raise ValueError(f"Unsupported image detection AL strategy: {strategy}")
    scores = _confidence_scores(predictions, image_paths)
    order = np.argsort(-scores)
    return [image_paths[int(index)] for index in order[: min(batch_size, len(order))]]


def _bbox_to_labelstudio(bbox: Iterable[float], width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "x": (x1 / width) * 100.0,
        "y": (y1 / height) * 100.0,
        "width": ((x2 - x1) / width) * 100.0,
        "height": ((y2 - y1) / height) * 100.0,
    }


def export_labelstudio_detection_batch(
    selected_images_df: pd.DataFrame,
    *,
    predictions: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    iteration: int,
    strategy: str,
    model_version: str,
    base_image_url: str = DEFAULT_BASE_IMAGE_URL,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "uncertain_manifest.csv"
    labelstudio_path = output_root / "labelstudio_import.json"

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
                "iteration": int(iteration),
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
                    "image": f"{base_image_url}{file_path}" if base_image_url else file_path,
                    FILE_PATH_COLUMN: file_path,
                    IMAGE_ID_COLUMN: str(row[IMAGE_ID_COLUMN]),
                },
                "predictions": predictions_payload,
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    labelstudio_path.write_text(json.dumps(_safe_json(tasks), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "iteration": iteration,
        "strategy": strategy,
        "manifest_path": str(manifest_path),
        "labelstudio_import_path": str(labelstudio_path),
        "selected_count": int(len(selected_images_df)),
    }


def _resolve_labelstudio_image_path(image_reference: str, local_files_document_root: str = "") -> str:
    reference = str(image_reference or "").strip()
    if not reference:
        return ""
    marker = "/data/local-files/?d="
    if reference.startswith(marker):
        relative_path = reference[len(marker):]
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return str(candidate.resolve())
        root = Path(local_files_document_root).resolve() if local_files_document_root else Path.cwd().resolve()
        return str((root / candidate).resolve())
    candidate = Path(reference)
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
        if not isinstance(item, dict):
            continue
        value = item.get("value") if isinstance(item.get("value"), dict) else item
        if not isinstance(value, dict):
            continue
        rectanglelabels = value.get("rectanglelabels")
        if not isinstance(rectanglelabels, list) or not rectanglelabels:
            continue
        rectangles.append(
            {
                "x": value.get("x", 0.0),
                "y": value.get("y", 0.0),
                "width": value.get("width", 0.0),
                "height": value.get("height", 0.0),
                "rectanglelabels": rectanglelabels,
                "original_width": item.get("original_width", value.get("original_width", 0)),
                "original_height": item.get("original_height", value.get("original_height", 0)),
            }
        )
    return rectangles


def _labelstudio_rectangle_to_row(file_path: str, rectangle: dict[str, Any]) -> dict[str, Any]:
    width = int(round(float(rectangle.get("original_width", 0) or 0)))
    height = int(round(float(rectangle.get("original_height", 0) or 0)))
    if width <= 0 or height <= 0:
        width, height = _open_image_size(file_path)
    x = float(rectangle.get("x", 0.0) or 0.0)
    y = float(rectangle.get("y", 0.0) or 0.0)
    box_width = float(rectangle.get("width", 0.0) or 0.0)
    box_height = float(rectangle.get("height", 0.0) or 0.0)
    x1 = (x / 100.0) * width
    y1 = (y / 100.0) * height
    x2 = ((x + box_width) / 100.0) * width
    y2 = ((y + box_height) / 100.0) * height
    label = str(rectangle.get("rectanglelabels", [""])[0]).strip()
    return {
        FILE_PATH_COLUMN: str(Path(file_path).resolve()),
        IMAGE_WIDTH_COLUMN: int(width),
        IMAGE_HEIGHT_COLUMN: int(height),
        CLASS_LABEL_COLUMN: label,
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
        if "image" not in df.columns:
            return set()
        return {
            _resolve_labelstudio_image_path(str(value), local_files_document_root=local_files_document_root)
            for value in df["image"].tolist()
            if str(value).strip()
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviewed: set[str] = set()
    for task in payload:
        if not isinstance(task, dict):
            continue
        data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
        file_path = str(data.get(FILE_PATH_COLUMN) or "").strip()
        if file_path:
            reviewed.add(str(Path(file_path).resolve()))
            continue
        image_reference = str(data.get("image") or "").strip()
        resolved = _resolve_labelstudio_image_path(image_reference, local_files_document_root=local_files_document_root)
        if resolved:
            reviewed.add(resolved)
    return reviewed


def import_labelstudio_detection_export(
    export_path: str,
    *,
    local_files_document_root: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    export_file = Path(export_path)
    rows: list[dict[str, Any]] = []
    if export_file.suffix.lower() == ".csv":
        export_df = pd.read_csv(export_file)
        if "image" not in export_df.columns:
            raise ValueError(f"Label Studio CSV export is missing 'image': {export_file}")
        label_column = "label" if "label" in export_df.columns else ""
        for _, row in export_df.iterrows():
            file_path = _resolve_labelstudio_image_path(str(row.get("image") or ""), local_files_document_root=local_files_document_root)
            if not file_path:
                continue
            rectangles = _parse_labelstudio_rectangles(row.get(label_column)) if label_column else []
            for rectangle in rectangles:
                label = str(rectangle.get("rectanglelabels", [""])[0]).strip()
                if not label:
                    continue
                rows.append(_labelstudio_rectangle_to_row(file_path, rectangle))
    else:
        payload = json.loads(export_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Label Studio JSON export must be a list of tasks.")
        for task in payload:
            if not isinstance(task, dict):
                continue
            data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
            file_path = str(data.get(FILE_PATH_COLUMN) or "").strip()
            if not file_path:
                file_path = _resolve_labelstudio_image_path(
                    str(data.get("image") or ""),
                    local_files_document_root=local_files_document_root,
                )
            if not file_path:
                continue
            annotations = task.get("annotations", [])
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                rectangles = _parse_labelstudio_rectangles(annotation.get("result", []))
                for rectangle in rectangles:
                    label = str(rectangle.get("rectanglelabels", [""])[0]).strip()
                    if not label:
                        continue
                    rows.append(_labelstudio_rectangle_to_row(file_path, rectangle))
    human_labels_df = pd.DataFrame(rows, columns=AL_LABEL_COLUMNS)

    reviewed_paths = _extract_reviewed_file_paths(export_path, local_files_document_root=local_files_document_root)
    reviewed_rows: list[dict[str, Any]] = []
    known_sizes = (
        human_labels_df[[FILE_PATH_COLUMN, IMAGE_WIDTH_COLUMN, IMAGE_HEIGHT_COLUMN]]
        .drop_duplicates(subset=[FILE_PATH_COLUMN])
        .set_index(FILE_PATH_COLUMN)
        .to_dict("index")
        if not human_labels_df.empty
        else {}
    )
    human_labeled_paths = set(human_labels_df[FILE_PATH_COLUMN].tolist())
    for file_path in sorted(reviewed_paths):
        size_payload = known_sizes.get(file_path)
        if size_payload:
            width = int(size_payload[IMAGE_WIDTH_COLUMN])
            height = int(size_payload[IMAGE_HEIGHT_COLUMN])
        elif Path(file_path).exists():
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


def merge_human_feedback(
    labels_df: pd.DataFrame,
    reviewed_images_df: pd.DataFrame,
    *,
    export_path: str,
    local_files_document_root: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    human_labels_df, reviewed_df = import_labelstudio_detection_export(
        export_path,
        local_files_document_root=local_files_document_root,
    )
    reviewed_paths = set(reviewed_df[FILE_PATH_COLUMN].tolist())
    merged_labels = labels_df[~labels_df[FILE_PATH_COLUMN].isin(reviewed_paths)].copy()
    if not human_labels_df.empty:
        merged_labels = pd.concat([merged_labels, human_labels_df], ignore_index=True)
    merged_labels = merged_labels.sort_values([FILE_PATH_COLUMN, CLASS_LABEL_COLUMN, X1_COLUMN, Y1_COLUMN]).reset_index(drop=True)

    existing_reviewed = reviewed_images_df[~reviewed_images_df[FILE_PATH_COLUMN].isin(reviewed_paths)].copy()
    merged_reviewed = pd.concat([existing_reviewed, reviewed_df], ignore_index=True)
    merged_reviewed = merged_reviewed.drop_duplicates(subset=[FILE_PATH_COLUMN], keep="last").reset_index(drop=True)

    return merged_labels, merged_reviewed, {
        "reviewed_images": len(reviewed_paths),
        "human_box_rows": int(len(human_labels_df)),
        "negative_reviewed_images": int((~reviewed_df["has_boxes"].map(_bool_from_value)).sum()) if not reviewed_df.empty else 0,
    }


def _save_history(history: list[dict[str, Any]], output_dir: str | Path, strategy: str) -> tuple[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    history_json = output_root / f"{strategy}_history.json"
    history_csv = output_root / f"{strategy}_history.csv"
    _write_json(history_json, history)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    return str(history_json), str(history_csv)


def _expected_human_export_path(
    run_dir: Path,
    *,
    strategy: str,
    iteration: int,
    human_feedback_dir: str = "",
) -> Path:
    if human_feedback_dir:
        configured = Path(human_feedback_dir)
        base_dir = configured if configured.is_absolute() else run_dir / configured
    else:
        base_dir = run_dir / "human_feedback"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{strategy}_iteration_{iteration}.json"


def _write_human_feedback_instruction(
    run_dir: Path,
    *,
    strategy: str,
    iteration: int,
    labelstudio_import_path: str,
    expected_human_export_path: Path,
) -> str:
    instruction_path = run_dir / "reports" / f"{strategy}_iteration_{iteration}_human_feedback.txt"
    instruction_path.write_text(
        "\n".join(
            [
                "AL agent is waiting for human feedback.",
                f"Import into Label Studio: {labelstudio_import_path}",
                f"After annotation, export the completed tasks to: {expected_human_export_path}",
                "When the export file appears at that path, the current run will continue automatically.",
            ]
        ),
        encoding="utf-8",
    )
    return str(instruction_path)


def _wait_for_human_export(expected_path: Path, *, timeout_sec: float, poll_interval_sec: float) -> None:
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if interactive:
        print(
            "\n".join(
                [
                    "",
                    "Active learning paused for human feedback.",
                    f"Place the Label Studio export at: {expected_path}",
                    "Type 'done' after you saved the file there.",
                    "Type 'abort' to stop the current AL run.",
                    "",
                ]
            ),
            flush=True,
        )
    start = time.monotonic()
    while True:
        if expected_path.exists() and expected_path.stat().st_size > 0:
            return
        if interactive:
            response = input("> ").strip().lower()
            if response in {"abort", "quit", "exit"}:
                raise RuntimeError(f"AL run aborted while waiting for human feedback: {expected_path}")
            if expected_path.exists() and expected_path.stat().st_size > 0:
                return
            print(f"Export file not found yet: {expected_path}", flush=True)
            if timeout_sec > 0 and (time.monotonic() - start) >= timeout_sec:
                raise TimeoutError(f"Timed out waiting for human feedback export: {expected_path}")
            continue
        if timeout_sec > 0 and (time.monotonic() - start) >= timeout_sec:
            raise TimeoutError(f"Timed out waiting for human feedback export: {expected_path}")
        time.sleep(max(0.5, poll_interval_sec))


def _refresh_inventory_after_feedback(
    *,
    current_labels: pd.DataFrame,
    current_reviewed: pd.DataFrame,
    dataset_dir: Path,
    test_images_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_df = build_image_inventory(current_labels, dataset_dir=dataset_dir, reviewed_images_df=current_reviewed)
    test_paths = set(test_images_df[FILE_PATH_COLUMN].astype(str).tolist())
    labeled_inventory = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    train_images_df = labeled_inventory[~labeled_inventory[FILE_PATH_COLUMN].isin(test_paths)].reset_index(drop=True)
    used_paths = set(train_images_df[FILE_PATH_COLUMN].astype(str).tolist()) | test_paths
    pool_images_df = inventory_df[~inventory_df[FILE_PATH_COLUMN].isin(used_paths)].reset_index(drop=True)
    return inventory_df, train_images_df, pool_images_df


def plot_learning_curves(
    histories_by_strategy: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for strategy, history in histories_by_strategy.items():
        if not history:
            continue
        history_df = pd.DataFrame(history)
        plt.plot(history_df["n_labeled_images"], history_df["f1_macro"], marker="o", label=strategy)
    plt.title("Active Learning Detection Curve")
    plt.xlabel("Number of labeled images")
    plt.ylabel("F1 macro")
    plt.grid(True, alpha=0.3)
    if histories_by_strategy:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return str(output_file)


def _lookup_manual_labels_path(manual_labels_by_iteration: dict[str, Any], strategy: str, iteration: int) -> str:
    if not manual_labels_by_iteration:
        return ""
    strategy_specific = manual_labels_by_iteration.get(strategy, manual_labels_by_iteration)
    if not isinstance(strategy_specific, dict):
        return ""
    return str(strategy_specific.get(str(iteration)) or strategy_specific.get(iteration) or "")


def _run_detection_strategy_cycle(
    *,
    strategy: str,
    task_description: str,
    labels_df: pd.DataFrame,
    reviewed_images_df: pd.DataFrame,
    dataset_dir: Path,
    classes: list[str],
    backend: Any,
    run_dir: Path,
    n_iterations: int,
    batch_size: int,
    test_size: float | int,
    random_state: int,
    base_image_url: str,
    manual_labels_by_iteration: dict[str, Any],
    pre_run_human_export_path: str,
    local_files_document_root: str,
    wait_for_human_feedback: bool,
    human_feedback_dir: str,
    human_wait_timeout_sec: float,
    human_poll_interval_sec: float,
) -> dict[str, Any]:
    strategy_dir = run_dir / "strategies" / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    current_labels = labels_df.copy().reset_index(drop=True)
    current_reviewed = reviewed_images_df.copy().reset_index(drop=True)
    merge_events: list[dict[str, Any]] = []
    if pre_run_human_export_path:
        current_labels, current_reviewed, merge_stats = merge_human_feedback(
            current_labels,
            current_reviewed,
            export_path=pre_run_human_export_path,
            local_files_document_root=local_files_document_root,
        )
        merge_events.append({"iteration": 0, "export_path": pre_run_human_export_path, **merge_stats})

    inventory_df = build_image_inventory(current_labels, dataset_dir=dataset_dir, reviewed_images_df=current_reviewed)
    train_images_df, test_images_df, pool_images_df = _prepare_detection_splits(
        inventory_df,
        test_size=test_size,
        random_state=random_state,
    )

    history: list[dict[str, Any]] = []
    iteration_artifacts: list[dict[str, Any]] = []
    last_model_dir = ""
    model_version = getattr(backend, "model_path", DEFAULT_IMAGE_MODEL_PATH)
    awaiting_human_feedback = False
    human_feedback_instruction_path = ""
    expected_human_export_path = ""

    for iteration in range(1, n_iterations + 1):
        train_labels_df = current_labels[current_labels[FILE_PATH_COLUMN].isin(train_images_df[FILE_PATH_COLUMN])].reset_index(drop=True)
        test_labels_df = current_labels[current_labels[FILE_PATH_COLUMN].isin(test_images_df[FILE_PATH_COLUMN])].reset_index(drop=True)
        train_result = backend.train(
            train_images_df=train_images_df,
            train_labels_df=train_labels_df,
            val_images_df=test_images_df,
            val_labels_df=test_labels_df,
            classes=classes,
            work_dir=strategy_dir,
            iteration_name=f"iteration_{iteration}",
        )
        last_model_dir = str(train_result["model_dir"])
        model_version = str(train_result.get("weights_path") or model_version)
        eval_predictions = backend.predict(train_result["model"], test_images_df[FILE_PATH_COLUMN].tolist(), classes)
        metrics = _evaluate_detection_metrics(
            eval_images_df=test_images_df,
            eval_labels_df=test_labels_df,
            predictions=eval_predictions,
            classes=classes,
        )
        selected_count = 0
        if not pool_images_df.empty:
            pool_predictions = backend.predict(train_result["model"], pool_images_df[FILE_PATH_COLUMN].tolist(), classes)
            selected_paths = select_uncertain_images(
                pool_images_df[FILE_PATH_COLUMN].tolist(),
                pool_predictions,
                strategy=strategy,
                batch_size=min(batch_size, len(pool_images_df)),
                random_state=random_state + iteration,
            )
            if selected_paths:
                selected_count = len(selected_paths)
                selected_images_df = pool_images_df[pool_images_df[FILE_PATH_COLUMN].isin(selected_paths)].reset_index(drop=True)
                iteration_dir = strategy_dir / f"iteration_{iteration}"
                export_payload = export_labelstudio_detection_batch(
                    selected_images_df,
                    predictions=pool_predictions,
                    output_dir=iteration_dir,
                    iteration=iteration,
                    strategy=strategy,
                    model_version=Path(model_version).name,
                    base_image_url=base_image_url,
                )
                manual_labels_path = _lookup_manual_labels_path(manual_labels_by_iteration, strategy, iteration)
                iteration_human_instruction_path = ""
                iteration_expected_human_export_path = ""
                if not manual_labels_path and wait_for_human_feedback:
                    expected_path = _expected_human_export_path(
                        run_dir,
                        strategy=strategy,
                        iteration=iteration,
                        human_feedback_dir=human_feedback_dir,
                    )
                    iteration_expected_human_export_path = str(expected_path)
                    iteration_human_instruction_path = _write_human_feedback_instruction(
                        run_dir,
                        strategy=strategy,
                        iteration=iteration,
                        labelstudio_import_path=export_payload["labelstudio_import_path"],
                        expected_human_export_path=expected_path,
                    )
                    human_feedback_instruction_path = iteration_human_instruction_path
                    expected_human_export_path = iteration_expected_human_export_path
                    _wait_for_human_export(
                        expected_path,
                        timeout_sec=human_wait_timeout_sec,
                        poll_interval_sec=human_poll_interval_sec,
                    )
                    manual_labels_path = str(expected_path)
                iteration_artifacts.append(
                    {
                        **export_payload,
                        "manual_labels_path": manual_labels_path,
                        "human_feedback_instruction_path": iteration_human_instruction_path,
                        "expected_human_export_path": iteration_expected_human_export_path,
                    }
                )
                if manual_labels_path:
                    current_labels, current_reviewed, merge_stats = merge_human_feedback(
                        current_labels,
                        current_reviewed,
                        export_path=manual_labels_path,
                        local_files_document_root=local_files_document_root,
                    )
                    merge_events.append({"iteration": iteration, "export_path": manual_labels_path, **merge_stats})
                    reviewed_paths = set(pd.read_csv(export_payload["manifest_path"])[FILE_PATH_COLUMN].astype(str).tolist())
                    pool_images_df = pool_images_df[~pool_images_df[FILE_PATH_COLUMN].isin(reviewed_paths)].reset_index(drop=True)
                    inventory_df, train_images_df, pool_images_df = _refresh_inventory_after_feedback(
                        current_labels=current_labels,
                        current_reviewed=current_reviewed,
                        dataset_dir=dataset_dir,
                        test_images_df=test_images_df,
                    )
                else:
                    awaiting_human_feedback = True
        history.append(
            {
                "iteration": int(iteration),
                "strategy": strategy,
                "task_description": task_description,
                "n_labeled_images": int(len(train_images_df)),
                "n_labeled_boxes": int(len(current_labels[current_labels[FILE_PATH_COLUMN].isin(train_images_df[FILE_PATH_COLUMN])])),
                "pool_images": int(len(pool_images_df)),
                "selected_count": int(selected_count),
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
        )
        if awaiting_human_feedback or pool_images_df.empty:
            break

    history_json, history_csv = _save_history(history, run_dir / "reports", strategy)
    final_labels_path = run_dir / "reports" / "labels.csv"
    reviewed_images_path = run_dir / "reports" / "reviewed_images.csv"
    current_labels.to_csv(final_labels_path, index=False)
    current_reviewed.to_csv(reviewed_images_path, index=False)
    return {
        "strategy": strategy,
        "history": history,
        "history_json": history_json,
        "history_csv": history_csv,
        "final_model_dir": last_model_dir,
        "labels_csv": str(final_labels_path),
        "reviewed_images_csv": str(reviewed_images_path),
        "iteration_artifacts": iteration_artifacts,
        "labelstudio_import_path": iteration_artifacts[-1]["labelstudio_import_path"] if iteration_artifacts else "",
        "uncertain_manifest_path": iteration_artifacts[-1]["manifest_path"] if iteration_artifacts else "",
        "merge_events": merge_events,
        "awaiting_human_feedback": awaiting_human_feedback,
        "human_feedback_instruction_path": human_feedback_instruction_path,
        "expected_human_export_path": expected_human_export_path,
    }


def image_detection_active_learning_impl(
    *,
    task_description: str,
    labeled_data_path: str,
    config: dict[str, Any] | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    resolved_config = {
        "artifacts_dir": DEFAULT_AL_ARTIFACTS_DIR,
        "n_iterations": DEFAULT_N_ITERATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "model_path": DEFAULT_IMAGE_MODEL_PATH,
        "base_image_url": DEFAULT_BASE_IMAGE_URL,
        "strategies": list(DEFAULT_IMAGE_STRATEGIES),
        "epochs": DEFAULT_EPOCHS,
        "imgsz": DEFAULT_IMGSZ,
        "train_batch": DEFAULT_TRAIN_BATCH,
        "predict_batch": DEFAULT_TRAIN_BATCH,
        "link_mode": DEFAULT_LINK_MODE,
        "manual_labels_by_iteration": {},
        "human_export_path": "",
        "reviewed_images_path": "",
        "dataset_dir": "",
        "local_files_document_root": "",
        "wait_for_human_feedback": DEFAULT_WAIT_FOR_HUMAN_FEEDBACK,
        "human_feedback_dir": "",
        "human_wait_timeout_sec": DEFAULT_HUMAN_WAIT_TIMEOUT_SEC,
        "human_poll_interval_sec": DEFAULT_HUMAN_POLL_INTERVAL_SEC,
    }
    if config:
        resolved_config.update(config)

    labels_df = load_detection_labels(labeled_data_path)
    labels_csv = _resolve_labels_csv_path(labeled_data_path)
    classes = sorted(labels_df[CLASS_LABEL_COLUMN].astype(str).unique().tolist())
    dataset_dir = _resolve_dataset_dir(labels_df, resolved_config)
    run_id = resolved_config.get("run_id") or make_run_id(_slugify(Path(labels_csv).stem))
    run_dir = ensure_al_run_layout(Path(str(resolved_config["artifacts_dir"])) / run_id)
    reviewed_images_df = _load_reviewed_images(str(resolved_config.get("reviewed_images_path") or ""))

    effective_backend = backend or YOLODetectionBackend(
        model_path=str(resolved_config["model_path"]),
        epochs=int(resolved_config["epochs"]),
        imgsz=int(resolved_config["imgsz"]),
        train_batch=int(resolved_config["train_batch"]),
        predict_batch=int(resolved_config["predict_batch"]),
        device=resolved_config.get("device"),
        link_mode=str(resolved_config["link_mode"]),
    )

    strategies = [str(strategy).strip().lower() for strategy in resolved_config["strategies"]]
    strategy_results: dict[str, Any] = {}
    history_paths: dict[str, str] = {}
    for strategy in strategies:
        strategy_result = _run_detection_strategy_cycle(
            strategy=strategy,
            task_description=task_description,
            labels_df=labels_df.copy(),
            reviewed_images_df=reviewed_images_df.copy(),
            dataset_dir=dataset_dir,
            classes=classes,
            backend=effective_backend,
            run_dir=run_dir,
            n_iterations=int(resolved_config["n_iterations"]),
            batch_size=int(resolved_config["batch_size"]),
            test_size=resolved_config["test_size"],
            random_state=int(resolved_config["random_state"]),
            base_image_url=str(resolved_config["base_image_url"]),
            manual_labels_by_iteration=resolved_config.get("manual_labels_by_iteration") or {},
            pre_run_human_export_path=str(resolved_config.get("human_export_path") or ""),
            local_files_document_root=str(resolved_config.get("local_files_document_root") or ""),
            wait_for_human_feedback=_bool_from_value(resolved_config.get("wait_for_human_feedback", DEFAULT_WAIT_FOR_HUMAN_FEEDBACK)),
            human_feedback_dir=str(resolved_config.get("human_feedback_dir") or ""),
            human_wait_timeout_sec=float(resolved_config.get("human_wait_timeout_sec", DEFAULT_HUMAN_WAIT_TIMEOUT_SEC)),
            human_poll_interval_sec=float(resolved_config.get("human_poll_interval_sec", DEFAULT_HUMAN_POLL_INTERVAL_SEC)),
        )
        strategy_results[strategy] = strategy_result
        history_paths[strategy] = strategy_result["history_json"]

    learning_curve_path = plot_learning_curves(
        {strategy: result["history"] for strategy, result in strategy_results.items()},
        run_dir / "curves" / "learning_curve.png",
    )
    primary_strategy = strategies[0]
    primary_result = strategy_results[primary_strategy]
    summary_payload = {
        "task_description": task_description,
        "labels_csv_input": str(labels_csv),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "classes": classes,
        "strategies": strategies,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
        "awaiting_human_feedback": bool(primary_result.get("awaiting_human_feedback", False)),
        "human_feedback_instruction_path": str(primary_result.get("human_feedback_instruction_path") or ""),
        "expected_human_export_path": str(primary_result.get("expected_human_export_path") or ""),
    }
    summary_path = _write_json(run_dir / "reports" / "summary.json", summary_payload)
    return {
        "modality": "image",
        "task_mode": "object_detection",
        "implemented": True,
        "task_description": task_description,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "labels_csv": primary_result["labels_csv"],
        "reviewed_images_csv": primary_result["reviewed_images_csv"],
        "classes": classes,
        "strategy_results": strategy_results,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
        "labelstudio_import_path": primary_result["labelstudio_import_path"],
        "uncertain_manifest_path": primary_result["uncertain_manifest_path"],
        "final_model_dir": primary_result["final_model_dir"],
        "summary_path": summary_path,
        "awaiting_human_feedback": bool(primary_result.get("awaiting_human_feedback", False)),
        "human_feedback_instruction_path": str(primary_result.get("human_feedback_instruction_path") or ""),
        "expected_human_export_path": str(primary_result.get("expected_human_export_path") or ""),
        "config": resolved_config,
    }


def table_classification_active_learning_impl(
    *,
    task_description: str,
    labeled_data_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "modality": "table",
        "implemented": False,
        "task_description": task_description,
        "labeled_data_path": labeled_data_path,
        "config": config or {},
        "message": "Table active learning backend is planned but not implemented in v1.",
    }


@tool
def image_detection_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Run bbox-level active learning for image object detection.

    Args:
        task_description: Natural-language description of the detection task.
        labeled_data_path: Path to annotation reports directory or directly to labels.csv.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON with AL artifacts, histories, learning-curve plot path, and Label Studio export paths.
    """
    logger.info("image_detection_active_learning called: labeled_data_path=%s", labeled_data_path)
    try:
        return _json_success(
            image_detection_active_learning_impl(
                task_description=task_description,
                labeled_data_path=labeled_data_path,
                config=_read_jsonish(config_json),
            )
        )
    except Exception as exc:
        logger.exception("image_detection_active_learning failed")
        return _json_error(str(exc))


@tool
def table_classification_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Placeholder tool for future tabular active-learning support.

    Args:
        task_description: Natural-language description of the classification task.
        labeled_data_path: Path to labeled tabular data.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON payload with implemented=false for the current v1 stub.
    """
    logger.info("table_classification_active_learning called: labeled_data_path=%s", labeled_data_path)
    try:
        return _json_success(
            table_classification_active_learning_impl(
                task_description=task_description,
                labeled_data_path=labeled_data_path,
                config=_read_jsonish(config_json),
            )
        )
    except Exception as exc:
        logger.exception("table_classification_active_learning failed")
        return _json_error(str(exc))
