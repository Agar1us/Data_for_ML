from __future__ import annotations

from typing import Any


ToolResult = dict[str, Any]

SUPPORTED_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BOUNDARY_CONFIDENCE_RANGE = (0.35, 0.65)
DEFAULT_MODEL_PATH = "yoloe-26x-seg.pt"
DEFAULT_ARTIFACTS_DIR = "data/current_run/annotation"
DEFAULT_LLM_MODEL_ID = "gpt-5-mini"
NO_DETECTION_LABEL = "__NO_DETECTION__"
DEFAULT_REVIEW_LINK_MODE = "symlink"

COLUMN_FILE_PATH = "file_path"
COLUMN_FILENAME = "filename"
COLUMN_FOLDER_LABEL = "folder_label"
COLUMN_PREDICTED_LABEL = "predicted_label"
COLUMN_CONFIDENCE = "confidence"
COLUMN_BBOX_XYXY = "bbox_xyxy"
COLUMN_HAS_MASK = "has_mask"
COLUMN_MASK_PATH = "mask_path"
COLUMN_ALL_DETECTIONS_JSON = "all_detections_json"
COLUMN_FOLDER_MATCH = "folder_match"
COLUMN_IMAGE_WIDTH = "image_width"
COLUMN_IMAGE_HEIGHT = "image_height"
