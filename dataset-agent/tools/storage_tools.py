from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from smolagents import tool


def _data_root() -> str:
    return os.getenv("DATASET_AGENT_DATA_DIR", "data")


def _artifacts_root() -> str:
    return os.getenv("DATASET_AGENT_ARTIFACTS_DIR", "collection_artifacts")


@tool
def save_dataset(data: str, dataset_name: str, filename: str) -> str:
    """
    Save collected data to data/<dataset_name>/filename.

    Args:
        data: Dataset content to write to disk.
        dataset_name: Name of the dataset subdirectory.
        filename: Output filename inside the dataset directory.

    Returns:
        A status string with the target path and file size.
    """
    dir_path = os.path.join(_data_root(), dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(data)
    size = os.path.getsize(file_path)
    return f"Saved to {file_path} ({size} bytes)"


@tool
def save_metadata(dataset_name: str, metadata_json: str) -> str:
    """
    Save metadata JSON to data/<dataset_name>/metadata.json.

    Args:
        dataset_name: Name of the dataset subdirectory.
        metadata_json: Metadata payload encoded as a JSON string.

    Returns:
        A status string with the saved metadata path.
    """
    dir_path = os.path.join(_data_root(), dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "metadata.json")

    metadata = json.loads(metadata_json)
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()

    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return f"Metadata saved to {file_path}"


@tool
def write_text_artifact(relative_path: str, content: str) -> str:
    """
    Save a text artifact inside the collection_artifacts directory.

    Args:
        relative_path: Relative path under collection_artifacts where the file will be saved.
        content: Text content to write to the artifact file.

    Returns:
        A status string with the saved artifact path and file size.
    """
    if os.path.isabs(relative_path):
        return "Error: relative_path must be relative to the collection_artifacts directory."

    normalized = os.path.normpath(relative_path)
    if normalized.startswith(".."):
        return "Error: relative_path cannot escape the collection_artifacts directory."

    root = _artifacts_root()
    file_path = os.path.join(root, normalized)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    size = os.path.getsize(file_path)
    return f"Artifact saved to {file_path} ({size} bytes)"
