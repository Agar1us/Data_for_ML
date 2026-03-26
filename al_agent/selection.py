from __future__ import annotations

from typing import Any

import numpy as np


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
