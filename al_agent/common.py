from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

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
MIN_HUMAN_VERIFIED_TEST_IMAGES = 2
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
