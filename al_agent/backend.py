from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from al_agent.common import (
    CLASS_LABEL_COLUMN,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_IMAGE_MODEL_PATH,
    DEFAULT_LINK_MODE,
    DEFAULT_TRAIN_BATCH,
    FILE_PATH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IMAGE_WIDTH_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
    _stable_image_id,
)

logger = logging.getLogger(__name__)


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
        target_name = f"{_stable_image_id(file_path)}_{source.name}"
        target_image = images_dir / target_name
        _link_or_copy_image(source, target_image, link_mode=link_mode)
        label_path = labels_dir / f"{Path(target_name).stem}.txt"
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
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required to write YOLO data.yaml files.") from exc
    (dataset_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
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
        except ImportError as exc:  # pragma: no cover
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
