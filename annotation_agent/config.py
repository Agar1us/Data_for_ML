from __future__ import annotations

from typing import Any


ToolResult = dict[str, Any]

SUPPORTED_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BOUNDARY_CONFIDENCE_RANGE = (0.35, 0.65)
DEFAULT_MODEL_PATH = "yoloe-26x-seg.pt"
DEFAULT_ARTIFACTS_DIR = "data/current_run/annotation"
DEFAULT_LLM_MODEL_ID = "gpt-5-mini"
DEFAULT_TASK_MODE = "image_classification"
DEFAULT_LABEL_ASSIGNMENT_MODE = "folder_label"
NO_DETECTION_LABEL = "__NO_DETECTION__"

COLUMN_FILE_PATH = "file_path"
COLUMN_FILENAME = "filename"
COLUMN_FOLDER_LABEL = "folder_label"
COLUMN_OBJECT_LABEL = "object_label"
COLUMN_OBJECT_CONFIDENCE = "object_confidence"
COLUMN_OBJECT_DETECTED = "object_detected"
COLUMN_BBOX_XYXY = "bbox_xyxy"
COLUMN_HAS_MASK = "has_mask"
COLUMN_MASK_PATH = "mask_path"
COLUMN_ALL_DETECTIONS_JSON = "all_detections_json"
COLUMN_LABEL_SOURCE = "label_source"
COLUMN_IMAGE_WIDTH = "image_width"
COLUMN_IMAGE_HEIGHT = "image_height"

AL_LABELS_FILE_NAME = "labels.csv"
AL_HUMAN_LABELS_FILE_NAME = "human_labels.csv"
AL_LABELS_COLUMNS = [
    "file_path",
    "image_width",
    "image_height",
    "class_label",
    "x1",
    "y1",
    "x2",
    "y2",
    "is_human_verified",
    "split",
]
