from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from al_agent.common import (
    AL_LABEL_COLUMNS,
    FILE_PATH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IMAGE_ID_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    MIN_HUMAN_VERIFIED_TEST_IMAGES,
    REVIEWED_IMAGES_COLUMNS,
    SPLIT_COLUMN,
    SUPPORTED_IMAGE_EXTENSIONS,
    _bool_from_value,
    _stable_image_id,
    CLASS_LABEL_COLUMN,
)


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
    for column in ("x1", "y1", "x2", "y2"):
        normalized[column] = normalized[column].astype(float)
    normalized[CLASS_LABEL_COLUMN] = normalized[CLASS_LABEL_COLUMN].astype(str)
    normalized[IS_HUMAN_VERIFIED_COLUMN] = normalized[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)
    normalized[SPLIT_COLUMN] = normalized[SPLIT_COLUMN].astype(str)
    return normalized


def resolve_dataset_dir(labels_df: pd.DataFrame, config: dict[str, Any]) -> Path:
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


def load_reviewed_images(reviewed_images_path: str = "") -> pd.DataFrame:
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


def _select_holdout_rows(
    *,
    labeled_images: pd.DataFrame,
    candidate_df: pd.DataFrame,
    required_count: int,
    random_state: int,
    purpose: str,
) -> pd.DataFrame:
    if required_count <= 0 or candidate_df.empty:
        return candidate_df.iloc[0:0].copy()

    candidate_df = candidate_df.copy()
    candidate_df["primary_class_label"] = candidate_df["primary_class_label"].astype(str)
    class_counts = (
        labeled_images["primary_class_label"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    shuffled = candidate_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in shuffled.to_dict("records"):
        groups.setdefault(str(row["primary_class_label"]), []).append(row)

    selected: list[dict[str, Any]] = []
    group_labels = sorted(groups)
    while len(selected) < required_count:
        progress = False
        for class_label in group_labels:
            class_candidates = groups.get(class_label, [])
            while class_candidates:
                candidate = class_candidates.pop(0)
                if class_counts.get(class_label, 0) <= 1:
                    continue
                selected.append(candidate)
                class_counts[class_label] = class_counts.get(class_label, 0) - 1
                progress = True
                break
            if len(selected) >= required_count:
                break
        if not progress:
            break

    if len(selected) < required_count:
        raise ValueError(
            f"Detection AL cannot reserve {required_count} images for {purpose} while keeping at least "
            "one labeled training image per class. Add more labeled images before continuing."
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def required_human_test_count(inventory_df: pd.DataFrame, test_size: float | int) -> int:
    labeled_images = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    if labeled_images.empty:
        return 0
    test_count = _normalize_test_count(len(labeled_images), test_size)
    return max(MIN_HUMAN_VERIFIED_TEST_IMAGES, test_count) if test_count > 0 else 0


def select_human_test_candidates(
    inventory_df: pd.DataFrame,
    *,
    required_count: int,
    random_state: int,
) -> pd.DataFrame:
    labeled_images = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    non_human = labeled_images[~labeled_images[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)].reset_index(drop=True)
    if required_count <= 0 or non_human.empty:
        return non_human.iloc[0:0].copy()
    candidate_count = min(required_count, len(non_human))
    return _select_holdout_rows(
        labeled_images=labeled_images,
        candidate_df=non_human,
        required_count=candidate_count,
        random_state=random_state,
        purpose="human-verified test bootstrapping",
    )


def prepare_detection_splits(
    inventory_df: pd.DataFrame,
    *,
    test_size: float | int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labeled_images = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    pool_images = inventory_df[inventory_df[SPLIT_COLUMN] != "labeled"].reset_index(drop=True)
    if len(labeled_images) < 2:
        raise ValueError("At least 2 labeled images are required for detection active learning.")
    required_count = required_human_test_count(inventory_df, test_size)
    if required_count == 0:
        return labeled_images, labeled_images.iloc[0:0].copy(), pool_images, {
            "test_split_source": "human_verified_only",
            "n_test_images": 0,
            "n_human_verified_test_candidates": 0,
            "metrics_reliable": False,
        }

    human_verified = labeled_images[labeled_images[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)].reset_index(drop=True)
    if len(human_verified) < required_count:
        raise ValueError(
            "Detection AL requires a human-verified-only test set. "
            f"Need at least {required_count} human-verified labeled images, found {len(human_verified)}."
        )

    effective_test_count = min(required_count, len(human_verified) - 1)
    if effective_test_count <= 0:
        raise ValueError(
            "Need at least 2 human-verified labeled images to build a dedicated detection test set."
        )
    test_df = _select_holdout_rows(
        labeled_images=labeled_images,
        candidate_df=human_verified,
        required_count=effective_test_count,
        random_state=random_state,
        purpose="the human-verified-only test set",
    )
    train_df = labeled_images[~labeled_images[FILE_PATH_COLUMN].isin(test_df[FILE_PATH_COLUMN])].reset_index(drop=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), pool_images, {
        "test_split_source": "human_verified_only",
        "n_test_images": int(len(test_df)),
        "n_human_verified_test_candidates": int(len(human_verified)),
        "metrics_reliable": True,
    }


def refresh_inventory_after_feedback(
    *,
    current_labels: pd.DataFrame,
    current_reviewed: pd.DataFrame,
    dataset_dir: Path,
    test_paths: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory_df = build_image_inventory(current_labels, dataset_dir=dataset_dir, reviewed_images_df=current_reviewed)
    test_images_df = inventory_df[inventory_df[FILE_PATH_COLUMN].isin(test_paths)].reset_index(drop=True)
    labeled_inventory = inventory_df[inventory_df[SPLIT_COLUMN] == "labeled"].reset_index(drop=True)
    train_images_df = labeled_inventory[~labeled_inventory[FILE_PATH_COLUMN].isin(test_paths)].reset_index(drop=True)
    used_paths = set(train_images_df[FILE_PATH_COLUMN].astype(str).tolist()) | test_paths
    pool_images_df = inventory_df[~inventory_df[FILE_PATH_COLUMN].isin(used_paths)].reset_index(drop=True)
    return inventory_df, train_images_df, pool_images_df
