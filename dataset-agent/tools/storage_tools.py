from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from smolagents import tool
from tools.path_utils import resolve_data_output_dir
from tools.runtime import artifacts_root, data_root


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
    try:
        dir_path = Path(resolve_data_output_dir(dataset_name))
    except ValueError as exc:
        return f"Error: {exc}"

    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / filename
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(data)
    size = file_path.stat().st_size
    return f"Saved to {file_path} ({size} bytes)"


@tool
def save_metadata(dataset_name: str, metadata_json: str) -> str:
    """
    Save metadata JSON to data/metadata.json.

    Args:
        dataset_name: Dataset name for provenance only.
        metadata_json: Metadata payload encoded as a JSON string.

    Returns:
        A status string with the saved metadata path.
    """
    dir_path = data_root()
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / "metadata.json"

    metadata = json.loads(metadata_json)
    metadata["dataset_name"] = dataset_name
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()

    with file_path.open("w", encoding="utf-8") as handle:
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

    root = artifacts_root()
    file_path = root / normalized
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(content)
    size = file_path.stat().st_size
    return f"Artifact saved to {file_path} ({size} bytes)"
