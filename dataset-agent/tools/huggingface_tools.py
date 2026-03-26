from __future__ import annotations

import json
import os

from smolagents import tool
from tools.path_utils import resolve_data_output_dir


@tool
def search_huggingface(query: str, max_results: int = 10) -> str:
    """
    Search for datasets on the Hugging Face Hub.

    Args:
        query: Search query for datasets.
        max_results: Maximum number of datasets to return.

    Returns:
        A JSON-formatted string with dataset search results.
    """
    try:
        from huggingface_hub import list_datasets
    except ImportError as exc:
        return f"Unable to import huggingface_hub: {exc}"

    results = []
    try:
        for index, dataset in enumerate(
            list_datasets(search=query, sort="downloads", direction=-1)
        ):
            if index >= max_results:
                break
            results.append(
                {
                    "id": dataset.id,
                    "downloads": getattr(dataset, "downloads", None),
                    "tags": (getattr(dataset, "tags", None) or [])[:5],
                    "description": (getattr(dataset, "description", "") or "")[:200],
                }
            )
    except Exception as exc:
        return f"Error searching Hugging Face: {exc}"

    if not results:
        return f"No datasets found for query: '{query}'"
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def download_hf_dataset(
    dataset_id: str,
    save_dir: str,
    subset: str = "",
    split: str = "",
) -> str:
    """
    Download a dataset from Hugging Face and save it locally.

    Args:
        dataset_id: Hugging Face dataset identifier.
        save_dir: Directory where the downloaded files will be saved.
        subset: Optional dataset configuration name.
        split: Optional dataset split name.

    Returns:
        A status string describing what was downloaded and where it was saved.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return f"Unable to import datasets: {exc}"

    try:
        target_dir = resolve_data_output_dir(save_dir)
    except ValueError as exc:
        return f"Error: {exc}"
    os.makedirs(target_dir, exist_ok=True)

    kwargs = {"path": dataset_id}
    if subset:
        kwargs["name"] = subset
    if split:
        kwargs["split"] = split

    try:
        dataset = load_dataset(**kwargs)
    except Exception as exc:
        return f"Error downloading Hugging Face dataset '{dataset_id}': {exc}"

    if split:
        try:
            file_path = os.path.join(target_dir, f"{split}.csv")
            dataset.to_csv(file_path)
            return f"Downloaded {len(dataset)} records to {file_path}"
        except Exception:
            disk_path = os.path.join(target_dir, split)
            dataset.save_to_disk(disk_path)
            return f"Downloaded split '{split}' to {disk_path}"

    total = 0
    split_names = []
    for split_name, split_dataset in dataset.items():
        try:
            file_path = os.path.join(target_dir, f"{split_name}.csv")
            split_dataset.to_csv(file_path)
        except Exception:
            file_path = os.path.join(target_dir, split_name)
            split_dataset.save_to_disk(file_path)
        total += len(split_dataset)
        split_names.append(split_name)
    return f"Downloaded {total} records ({split_names} splits) to {target_dir}"
