from __future__ import annotations

import json
import os

from smolagents import tool


def _data_root() -> str:
    return os.getenv("DATASET_AGENT_DATA_DIR", "data")


def _resolve_output_dir(path: str) -> str:
    if os.path.isabs(path):
        return path

    normalized = os.path.normpath(path)
    data_root = _data_root()
    data_dir_name = os.path.basename(os.path.normpath(data_root))

    if normalized == data_dir_name:
        normalized = ""
    elif normalized.startswith(f"{data_dir_name}{os.sep}"):
        normalized = normalized[len(data_dir_name) + 1 :]

    return os.path.join(data_root, normalized) if normalized else data_root


def _get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


@tool
def search_kaggle(query: str, max_results: int = 10) -> str:
    """
    Search Kaggle datasets by query.

    Args:
        query: Search query for Kaggle datasets.
        max_results: Maximum number of datasets to return.

    Returns:
        A JSON-formatted string with dataset search results.
    """
    try:
        api = _get_api()
        datasets = api.dataset_list(search=query, sort_by="hottest")
    except Exception as exc:
        return f"Error searching Kaggle: {exc}"

    results = []
    for dataset in datasets[:max_results]:
        results.append(
            {
                "ref": str(dataset),
                "title": getattr(dataset, "title", ""),
                "size": getattr(dataset, "totalBytes", None),
                "description": (getattr(dataset, "subtitle", "") or "")[:200],
            }
        )

    if not results:
        return f"No Kaggle datasets found for query: '{query}'"
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def download_kaggle_dataset(dataset_ref: str, save_dir: str) -> str:
    """
    Download and unzip a Kaggle dataset.

    Args:
        dataset_ref: Kaggle dataset reference in owner/name form.
        save_dir: Directory where the dataset will be unpacked.

    Returns:
        A status string with the output directory and downloaded files.
    """
    target_dir = _resolve_output_dir(save_dir)
    os.makedirs(target_dir, exist_ok=True)

    try:
        api = _get_api()
        api.dataset_download_files(dataset_ref, path=target_dir, unzip=True)
    except Exception as exc:
        return f"Error downloading Kaggle dataset '{dataset_ref}': {exc}"

    files = sorted(os.listdir(target_dir))
    return f"Downloaded {len(files)} files to {target_dir}: {files}"
