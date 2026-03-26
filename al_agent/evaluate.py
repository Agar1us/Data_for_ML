from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from al_agent.common import (
    CLASS_LABEL_COLUMN,
    DEFAULT_IOU_THRESHOLD,
    FILE_PATH_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
)


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


def evaluate_detection_metrics(
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
