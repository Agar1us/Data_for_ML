from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from al_agent.common import (
    CLASS_LABEL_COLUMN,
    FILE_PATH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    SPLIT_COLUMN,
    X1_COLUMN,
    Y1_COLUMN,
    _bool_from_value,
)
from al_agent.labelstudio import import_labelstudio_detection_export

logger = logging.getLogger(__name__)


def merge_human_feedback(
    labels_df: pd.DataFrame,
    reviewed_images_df: pd.DataFrame,
    *,
    export_path: str,
    local_files_document_root: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    human_labels_df, reviewed_df = import_labelstudio_detection_export(
        export_path,
        local_files_document_root=local_files_document_root,
    )
    reviewed_paths = set(reviewed_df[FILE_PATH_COLUMN].tolist())
    auto_box_count = int(labels_df[labels_df[FILE_PATH_COLUMN].isin(reviewed_paths)].shape[0])
    human_box_count = int(len(human_labels_df))
    possible_incomplete_annotation = auto_box_count > 0 and human_box_count < (auto_box_count * 0.5)
    if possible_incomplete_annotation:
        logger.warning(
            "Human feedback contains significantly fewer boxes than replaced auto-labels: human=%s auto=%s export=%s",
            human_box_count,
            auto_box_count,
            export_path,
        )
    merged_labels = labels_df[~labels_df[FILE_PATH_COLUMN].isin(reviewed_paths)].copy()
    if not human_labels_df.empty:
        merged_labels = pd.concat([merged_labels, human_labels_df], ignore_index=True)
    merged_labels = merged_labels.sort_values([FILE_PATH_COLUMN, CLASS_LABEL_COLUMN, X1_COLUMN, Y1_COLUMN]).reset_index(drop=True)

    existing_reviewed = reviewed_images_df[~reviewed_images_df[FILE_PATH_COLUMN].isin(reviewed_paths)].copy()
    merged_reviewed = pd.concat([existing_reviewed, reviewed_df], ignore_index=True)
    merged_reviewed = merged_reviewed.drop_duplicates(subset=[FILE_PATH_COLUMN], keep="last").reset_index(drop=True)

    return merged_labels, merged_reviewed, {
        "reviewed_images": len(reviewed_paths),
        "human_box_rows": human_box_count,
        "auto_box_rows_replaced": auto_box_count,
        "possible_incomplete_annotation": bool(possible_incomplete_annotation),
        "negative_reviewed_images": int((~reviewed_df["has_boxes"].map(_bool_from_value)).sum()) if not reviewed_df.empty else 0,
    }


def expected_human_export_path(
    run_dir: Path,
    *,
    strategy: str,
    iteration: int | str,
    human_feedback_dir: str = "",
) -> Path:
    if human_feedback_dir:
        configured = Path(human_feedback_dir)
        base_dir = configured if configured.is_absolute() else run_dir / configured
    else:
        base_dir = run_dir / "human_feedback"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{strategy}_iteration_{iteration}.json"


def _find_available_human_export(expected_path: Path, *, used_paths: set[str] | None = None) -> Path | None:
    normalized_used_paths = {str(Path(path).resolve()) for path in (used_paths or set())}
    if (
        expected_path.exists()
        and expected_path.stat().st_size > 0
        and str(expected_path.resolve()) not in normalized_used_paths
    ):
        return expected_path
    parent = expected_path.parent
    if not parent.exists():
        return None
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".json", ".csv"}
        and path.stat().st_size > 0
        and str(path.resolve()) not in normalized_used_paths
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_human_feedback_instruction(
    run_dir: Path,
    *,
    strategy: str,
    iteration: int | str,
    labelstudio_import_path: str,
    labelstudio_config_path: str,
    expected_human_export_path: Path,
) -> str:
    instruction_path = run_dir / "reports" / f"{strategy}_iteration_{iteration}_human_feedback.txt"
    instruction_path.write_text(
        "\n".join(
            [
                "AL agent is waiting for human feedback.",
                f"Use Label Studio config: {labelstudio_config_path}",
                f"Import into Label Studio: {labelstudio_import_path}",
                f"After annotation, export the completed tasks into directory: {expected_human_export_path.parent}",
                f"Preferred filename: {expected_human_export_path.name}",
                "Any non-empty .json or .csv export file in that directory will be accepted automatically.",
            ]
        ),
        encoding="utf-8",
    )
    return str(instruction_path)


def wait_for_human_export(
    expected_path: Path,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
    used_paths: set[str] | None = None,
) -> Path:
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    if interactive:
        print(
            "\n".join(
                [
                    "",
                    "Active learning paused for human feedback.",
                    f"Place the Label Studio export into directory: {expected_path.parent}",
                    f"Preferred filename: {expected_path.name}",
                    "Any new non-empty .json or .csv file in that directory will be accepted.",
                    "Type 'done' after you saved the export.",
                    "Type 'abort' to stop the current AL run.",
                    "",
                ]
            ),
            flush=True,
        )
    start = time.monotonic()
    while True:
        candidate = _find_available_human_export(expected_path, used_paths=used_paths)
        if candidate is not None:
            return candidate
        if interactive:
            response = input("> ").strip().lower()
            if response in {"abort", "quit", "exit"}:
                raise RuntimeError(f"AL run aborted while waiting for human feedback in: {expected_path.parent}")
            candidate = _find_available_human_export(expected_path, used_paths=used_paths)
            if candidate is not None:
                return candidate
            print(f"Export file not found yet in directory: {expected_path.parent}", flush=True)
            if timeout_sec > 0 and (time.monotonic() - start) >= timeout_sec:
                raise TimeoutError(f"Timed out waiting for human feedback export in: {expected_path.parent}")
            continue
        if timeout_sec > 0 and (time.monotonic() - start) >= timeout_sec:
            raise TimeoutError(f"Timed out waiting for human feedback export in: {expected_path.parent}")
        time.sleep(max(0.5, poll_interval_sec))
