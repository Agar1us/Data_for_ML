from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    def tool(func):  # type: ignore[misc]
        return func

from al_agent.backend import YOLODetectionBackend, build_yolo_detection_dataset
from al_agent.common import (
    AL_LABEL_COLUMNS,
    CLASS_LABEL_COLUMN,
    DEFAULT_AL_ARTIFACTS_DIR,
    DEFAULT_BASE_IMAGE_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HUMAN_POLL_INTERVAL_SEC,
    DEFAULT_HUMAN_WAIT_TIMEOUT_SEC,
    DEFAULT_IMAGE_MODEL_PATH,
    DEFAULT_IMAGE_STRATEGIES,
    DEFAULT_IMGSZ,
    DEFAULT_LINK_MODE,
    DEFAULT_N_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_BATCH,
    DEFAULT_WAIT_FOR_HUMAN_FEEDBACK,
    FILE_PATH_COLUMN,
    IMAGE_HEIGHT_COLUMN,
    IMAGE_ID_COLUMN,
    IMAGE_WIDTH_COLUMN,
    IS_HUMAN_VERIFIED_COLUMN,
    SPLIT_COLUMN,
    UNCERTAINTY_COLUMN,
    X1_COLUMN,
    X2_COLUMN,
    Y1_COLUMN,
    Y2_COLUMN,
    _bool_from_value,
    _json_error,
    _json_success,
    _read_jsonish,
    _safe_json,
    _slugify,
    _write_json,
    make_run_id,
    ensure_al_run_layout,
)
from al_agent.data import (
    build_image_inventory,
    load_detection_labels,
    load_reviewed_images,
    prepare_detection_splits,
    refresh_inventory_after_feedback,
    required_human_test_count,
    resolve_dataset_dir,
    select_human_test_candidates,
)
from al_agent.evaluate import evaluate_detection_metrics
from al_agent.feedback import (
    expected_human_export_path as make_expected_human_export_path,
    merge_human_feedback,
    wait_for_human_export as _wait_for_human_export,
    write_human_feedback_instruction,
)
from al_agent.labelstudio import (
    build_labelstudio_config,
    export_labelstudio_detection_batch,
    import_labelstudio_detection_export,
    predictions_from_labels,
)
from al_agent.selection import select_uncertain_images

logger = logging.getLogger(__name__)

_prepare_detection_splits = prepare_detection_splits


def _save_history(history: list[dict[str, Any]], output_dir: str | Path, strategy: str) -> tuple[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    history_json = output_root / f"{strategy}_history.json"
    history_csv = output_root / f"{strategy}_history.csv"
    _write_json(history_json, history)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    return str(history_json), str(history_csv)


def plot_learning_curves(
    histories_by_strategy: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> str:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for strategy, history in histories_by_strategy.items():
        if not history:
            continue
        history_df = pd.DataFrame(history)
        plt.plot(history_df["n_labeled_images"], history_df["f1_macro"], marker="o", label=strategy)
    plt.title("Active Learning Detection Curve")
    plt.xlabel("Number of labeled images")
    plt.ylabel("F1 macro")
    plt.grid(True, alpha=0.3)
    if histories_by_strategy:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return str(output_file)


def _lookup_manual_labels_path(manual_labels_by_iteration: dict[str, Any], strategy: str, iteration: int) -> str:
    if not manual_labels_by_iteration:
        return ""
    strategy_specific = manual_labels_by_iteration.get(strategy, manual_labels_by_iteration)
    if not isinstance(strategy_specific, dict):
        return ""
    return str(strategy_specific.get(str(iteration)) or strategy_specific.get(iteration) or "")


def _print_training_banner(strategy: str, iteration: int | str, classes: list[str]) -> None:
    rendered_classes = ", ".join(str(class_name) for class_name in classes if str(class_name).strip()) or "<none>"
    print(
        f"[al-agent] training detection model | strategy={strategy} | iteration={iteration} | classes=[{rendered_classes}]",
        flush=True,
    )


def _ensure_human_verified_test_protocol(
    *,
    current_labels: pd.DataFrame,
    current_reviewed: pd.DataFrame,
    dataset_dir: Path,
    classes: list[str],
    run_dir: Path,
    strategy: str,
    test_size: float | int,
    random_state: int,
    base_image_url: str,
    local_files_document_root: str,
    wait_for_human_feedback: bool,
    human_feedback_dir: str,
    human_wait_timeout_sec: float,
    human_poll_interval_sec: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    set[str],
    dict[str, Any],
    list[dict[str, Any]],
    str,
    str,
    set[str],
]:
    merge_events: list[dict[str, Any]] = []
    human_feedback_instruction_path = ""
    expected_human_export_path_str = ""
    used_human_export_paths: set[str] = set()

    inventory_df = build_image_inventory(current_labels, dataset_dir=dataset_dir, reviewed_images_df=current_reviewed)
    required_count = required_human_test_count(inventory_df, test_size)
    human_verified_count = int(
        inventory_df[
            (inventory_df[SPLIT_COLUMN] == "labeled") & inventory_df[IS_HUMAN_VERIFIED_COLUMN].map(_bool_from_value)
        ].shape[0]
    )

    if human_verified_count < required_count:
        if not wait_for_human_feedback:
            raise RuntimeError(
                "Detection AL requires a dedicated human-verified-only test set. "
                f"Need at least {required_count} human-verified labeled images, found {human_verified_count}. "
                "Re-run with human feedback enabled so the test bootstrap batch can be annotated."
            )
        needed = required_count - human_verified_count
        selected_images_df = select_human_test_candidates(
            inventory_df,
            required_count=needed,
            random_state=random_state,
        )
        if selected_images_df.empty:
            raise RuntimeError(
                "Detection AL could not build the initial human-verified test bootstrap batch. "
                "There are not enough labeled images available."
            )
        setup_dir = run_dir / "strategies" / strategy / "human_test_setup"
        export_payload = export_labelstudio_detection_batch(
            selected_images_df,
            predictions=predictions_from_labels(current_labels, selected_images_df[FILE_PATH_COLUMN].astype(str).tolist()),
            output_dir=setup_dir,
            iteration="test_setup",
            strategy=strategy,
            model_version="human_test_setup",
            classes=classes,
            base_image_url=base_image_url,
            local_files_document_root=local_files_document_root,
        )
        expected_path = make_expected_human_export_path(
            run_dir,
            strategy=f"{strategy}_test_setup",
            iteration=0,
            human_feedback_dir=human_feedback_dir,
        )
        human_feedback_instruction_path = write_human_feedback_instruction(
            run_dir,
            strategy=f"{strategy}_test_setup",
            iteration=0,
            labelstudio_import_path=export_payload["labelstudio_import_path"],
            labelstudio_config_path=export_payload["labelstudio_config_path"],
            expected_human_export_path=expected_path,
        )
        expected_human_export_path_str = str(expected_path)
        actual_export_path = _wait_for_human_export(
            expected_path,
            timeout_sec=human_wait_timeout_sec,
            poll_interval_sec=human_poll_interval_sec,
            used_paths=used_human_export_paths,
        )
        used_human_export_paths.add(str(Path(actual_export_path).resolve()))
        current_labels, current_reviewed, merge_stats = merge_human_feedback(
            current_labels,
            current_reviewed,
            export_path=str(actual_export_path),
            local_files_document_root=local_files_document_root,
        )
        updated_classes = sorted(current_labels[CLASS_LABEL_COLUMN].astype(str).unique().tolist())
        if updated_classes != classes:
            logger.warning("AL class list updated after human test bootstrap: %s -> %s", classes, updated_classes)
            classes[:] = updated_classes
        merge_events.append(
            {
                "iteration": 0,
                "phase": "human_test_bootstrap",
                "export_path": str(actual_export_path),
                **merge_stats,
            }
        )
        inventory_df = build_image_inventory(current_labels, dataset_dir=dataset_dir, reviewed_images_df=current_reviewed)

    train_images_df, test_images_df, pool_images_df, split_meta = prepare_detection_splits(
        inventory_df,
        test_size=test_size,
        random_state=random_state,
    )
    test_paths = set(test_images_df[FILE_PATH_COLUMN].astype(str).tolist())
    return (
        current_labels,
        current_reviewed,
        train_images_df,
        test_images_df,
        pool_images_df,
        test_paths,
        split_meta,
        merge_events,
        human_feedback_instruction_path,
        expected_human_export_path_str,
        used_human_export_paths,
    )


def _run_detection_strategy_cycle(
    *,
    strategy: str,
    task_description: str,
    labels_df: pd.DataFrame,
    reviewed_images_df: pd.DataFrame,
    dataset_dir: Path,
    classes: list[str],
    backend: Any,
    run_dir: Path,
    n_iterations: int,
    batch_size: int,
    test_size: float | int,
    random_state: int,
    base_image_url: str,
    manual_labels_by_iteration: dict[str, Any],
    pre_run_human_export_path: str,
    local_files_document_root: str,
    wait_for_human_feedback: bool,
    human_feedback_dir: str,
    human_wait_timeout_sec: float,
    human_poll_interval_sec: float,
) -> dict[str, Any]:
    strategy_dir = run_dir / "strategies" / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    current_labels = labels_df.copy().reset_index(drop=True)
    current_reviewed = reviewed_images_df.copy().reset_index(drop=True)
    local_classes = list(classes)
    merge_events: list[dict[str, Any]] = []
    used_human_export_paths: set[str] = set()
    if pre_run_human_export_path:
        used_human_export_paths.add(str(Path(pre_run_human_export_path).resolve()))
        current_labels, current_reviewed, merge_stats = merge_human_feedback(
            current_labels,
            current_reviewed,
            export_path=pre_run_human_export_path,
            local_files_document_root=local_files_document_root,
        )
        local_classes = sorted(current_labels[CLASS_LABEL_COLUMN].astype(str).unique().tolist())
        merge_events.append({"iteration": 0, "phase": "pre_run_merge", "export_path": pre_run_human_export_path, **merge_stats})

    (
        current_labels,
        current_reviewed,
        train_images_df,
        test_images_df,
        pool_images_df,
        test_paths,
        split_meta,
        bootstrap_merge_events,
        bootstrap_instruction_path,
        bootstrap_expected_path,
        bootstrap_used_human_export_paths,
    ) = _ensure_human_verified_test_protocol(
        current_labels=current_labels,
        current_reviewed=current_reviewed,
        dataset_dir=dataset_dir,
        classes=local_classes,
        run_dir=run_dir,
        strategy=strategy,
        test_size=test_size,
        random_state=random_state,
        base_image_url=base_image_url,
        local_files_document_root=local_files_document_root,
        wait_for_human_feedback=wait_for_human_feedback,
        human_feedback_dir=human_feedback_dir,
        human_wait_timeout_sec=human_wait_timeout_sec,
        human_poll_interval_sec=human_poll_interval_sec,
    )
    merge_events.extend(bootstrap_merge_events)
    used_human_export_paths.update(bootstrap_used_human_export_paths)

    history: list[dict[str, Any]] = []
    iteration_artifacts: list[dict[str, Any]] = []
    last_model_dir = ""
    model_version = getattr(backend, "model_path", DEFAULT_IMAGE_MODEL_PATH)
    awaiting_human_feedback = False
    human_feedback_instruction_path = bootstrap_instruction_path
    expected_human_export_path_str = bootstrap_expected_path

    for iteration in range(1, n_iterations + 1):
        train_labels_df = current_labels[current_labels[FILE_PATH_COLUMN].isin(train_images_df[FILE_PATH_COLUMN])].reset_index(drop=True)
        test_labels_df = current_labels[current_labels[FILE_PATH_COLUMN].isin(test_images_df[FILE_PATH_COLUMN])].reset_index(drop=True)
        _print_training_banner(strategy, iteration, local_classes)
        train_result = backend.train(
            train_images_df=train_images_df,
            train_labels_df=train_labels_df,
            val_images_df=test_images_df,
            val_labels_df=test_labels_df,
            classes=local_classes,
            work_dir=strategy_dir,
            iteration_name=f"iteration_{iteration}",
        )
        last_model_dir = str(train_result["model_dir"])
        model_version = str(train_result.get("weights_path") or model_version)
        eval_predictions = backend.predict(train_result["model"], test_images_df[FILE_PATH_COLUMN].tolist(), local_classes)
        metrics = evaluate_detection_metrics(
            eval_images_df=test_images_df,
            eval_labels_df=test_labels_df,
            predictions=eval_predictions,
            classes=local_classes,
        )
        history.append(
            {
                "iteration": int(iteration),
                "strategy": strategy,
                "phase": "selection_round",
                "task_description": task_description,
                "n_labeled_images": int(len(train_images_df)),
                "n_labeled_boxes": int(len(train_labels_df)),
                "n_test_images": int(split_meta["n_test_images"]),
                "test_split_source": str(split_meta["test_split_source"]),
                "metrics_reliable": bool(split_meta["metrics_reliable"]),
                "pool_images": int(len(pool_images_df)),
                "selected_count": 0,
                "precision_macro": float(metrics["precision_macro"]),
                "recall_macro": float(metrics["recall_macro"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
        )
        selected_count = 0
        if not pool_images_df.empty:
            pool_predictions = backend.predict(train_result["model"], pool_images_df[FILE_PATH_COLUMN].tolist(), local_classes)
            selected_paths = select_uncertain_images(
                pool_images_df[FILE_PATH_COLUMN].tolist(),
                pool_predictions,
                strategy=strategy,
                batch_size=min(batch_size, len(pool_images_df)),
                random_state=random_state + iteration,
            )
            if selected_paths:
                selected_count = len(selected_paths)
                history[-1]["selected_count"] = int(selected_count)
                selected_images_df = pool_images_df[pool_images_df[FILE_PATH_COLUMN].isin(selected_paths)].reset_index(drop=True)
                iteration_dir = strategy_dir / f"iteration_{iteration}"
                export_payload = export_labelstudio_detection_batch(
                    selected_images_df,
                    predictions=pool_predictions,
                    output_dir=iteration_dir,
                    iteration=iteration,
                    strategy=strategy,
                    model_version=Path(model_version).name,
                    classes=local_classes,
                    base_image_url=base_image_url,
                    local_files_document_root=local_files_document_root,
                )
                manual_labels_path = _lookup_manual_labels_path(manual_labels_by_iteration, strategy, iteration)
                iteration_human_instruction_path = ""
                iteration_expected_human_export_path = ""
                if not manual_labels_path and wait_for_human_feedback:
                    expected_path = make_expected_human_export_path(
                        run_dir,
                        strategy=strategy,
                        iteration=iteration,
                        human_feedback_dir=human_feedback_dir,
                    )
                    iteration_expected_human_export_path = str(expected_path)
                    iteration_human_instruction_path = write_human_feedback_instruction(
                        run_dir,
                        strategy=strategy,
                        iteration=iteration,
                        labelstudio_import_path=export_payload["labelstudio_import_path"],
                        labelstudio_config_path=export_payload["labelstudio_config_path"],
                        expected_human_export_path=expected_path,
                    )
                    human_feedback_instruction_path = iteration_human_instruction_path
                    expected_human_export_path_str = iteration_expected_human_export_path
                    actual_export_path = _wait_for_human_export(
                        expected_path,
                        timeout_sec=human_wait_timeout_sec,
                        poll_interval_sec=human_poll_interval_sec,
                        used_paths=used_human_export_paths,
                    )
                    manual_labels_path = str(actual_export_path)
                iteration_artifacts.append(
                    {
                        **export_payload,
                        "manual_labels_path": manual_labels_path,
                        "human_feedback_instruction_path": iteration_human_instruction_path,
                        "expected_human_export_path": iteration_expected_human_export_path,
                    }
                )
                if manual_labels_path:
                    used_human_export_paths.add(str(Path(manual_labels_path).resolve()))
                    current_labels, current_reviewed, merge_stats = merge_human_feedback(
                        current_labels,
                        current_reviewed,
                        export_path=manual_labels_path,
                        local_files_document_root=local_files_document_root,
                    )
                    updated_classes = sorted(current_labels[CLASS_LABEL_COLUMN].astype(str).unique().tolist())
                    if updated_classes != local_classes:
                        logger.warning("AL class list updated after human feedback: %s -> %s", local_classes, updated_classes)
                        local_classes = updated_classes
                    merge_events.append({"iteration": iteration, "phase": "review_merge", "export_path": manual_labels_path, **merge_stats})
                    reviewed_paths = set(pd.read_csv(export_payload["manifest_path"])[FILE_PATH_COLUMN].astype(str).tolist())
                    pool_images_df = pool_images_df[~pool_images_df[FILE_PATH_COLUMN].isin(reviewed_paths)].reset_index(drop=True)
                    _, train_images_df, pool_images_df = refresh_inventory_after_feedback(
                        current_labels=current_labels,
                        current_reviewed=current_reviewed,
                        dataset_dir=dataset_dir,
                        test_paths=test_paths,
                    )
                    if iteration == n_iterations:
                        post_train_labels_df = current_labels[
                            current_labels[FILE_PATH_COLUMN].isin(train_images_df[FILE_PATH_COLUMN])
                        ].reset_index(drop=True)
                        post_test_labels_df = current_labels[
                            current_labels[FILE_PATH_COLUMN].isin(test_images_df[FILE_PATH_COLUMN])
                        ].reset_index(drop=True)
                        _print_training_banner(strategy, f"{iteration}_post_feedback", local_classes)
                        post_train_result = backend.train(
                            train_images_df=train_images_df,
                            train_labels_df=post_train_labels_df,
                            val_images_df=test_images_df,
                            val_labels_df=post_test_labels_df,
                            classes=local_classes,
                            work_dir=strategy_dir,
                            iteration_name=f"iteration_{iteration}_post_feedback",
                        )
                        last_model_dir = str(post_train_result["model_dir"])
                        model_version = str(post_train_result.get("weights_path") or model_version)
                        post_eval_predictions = backend.predict(
                            post_train_result["model"],
                            test_images_df[FILE_PATH_COLUMN].tolist(),
                            local_classes,
                        )
                        post_metrics = evaluate_detection_metrics(
                            eval_images_df=test_images_df,
                            eval_labels_df=post_test_labels_df,
                            predictions=post_eval_predictions,
                            classes=local_classes,
                        )
                        history.append(
                            {
                                "iteration": int(iteration),
                                "strategy": strategy,
                                "phase": "post_feedback_retrain",
                                "task_description": task_description,
                                "n_labeled_images": int(len(train_images_df)),
                                "n_labeled_boxes": int(len(post_train_labels_df)),
                                "n_test_images": int(split_meta["n_test_images"]),
                                "test_split_source": str(split_meta["test_split_source"]),
                                "metrics_reliable": bool(split_meta["metrics_reliable"]),
                                "pool_images": int(len(pool_images_df)),
                                "selected_count": 0,
                                "precision_macro": float(post_metrics["precision_macro"]),
                                "recall_macro": float(post_metrics["recall_macro"]),
                                "f1_macro": float(post_metrics["f1_macro"]),
                            }
                        )
                else:
                    awaiting_human_feedback = True
        if awaiting_human_feedback or pool_images_df.empty:
            break

    history_json, history_csv = _save_history(history, run_dir / "reports", strategy)
    final_labels_path = run_dir / "reports" / "labels.csv"
    reviewed_images_path = run_dir / "reports" / "reviewed_images.csv"
    current_labels.to_csv(final_labels_path, index=False)
    current_reviewed.to_csv(reviewed_images_path, index=False)
    return {
        "strategy": strategy,
        "history": history,
        "history_json": history_json,
        "history_csv": history_csv,
        "final_model_dir": last_model_dir,
        "labels_csv": str(final_labels_path),
        "reviewed_images_csv": str(reviewed_images_path),
        "iteration_artifacts": iteration_artifacts,
        "labelstudio_import_path": iteration_artifacts[-1]["labelstudio_import_path"] if iteration_artifacts else "",
        "labelstudio_config_path": iteration_artifacts[-1]["labelstudio_config_path"] if iteration_artifacts else "",
        "uncertain_manifest_path": iteration_artifacts[-1]["manifest_path"] if iteration_artifacts else "",
        "merge_events": merge_events,
        "awaiting_human_feedback": awaiting_human_feedback,
        "human_feedback_instruction_path": human_feedback_instruction_path,
        "expected_human_export_path": expected_human_export_path_str,
    }


def image_detection_active_learning_impl(
    *,
    task_description: str,
    labeled_data_path: str,
    config: dict[str, Any] | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    resolved_config = {
        "artifacts_dir": DEFAULT_AL_ARTIFACTS_DIR,
        "n_iterations": DEFAULT_N_ITERATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "model_path": DEFAULT_IMAGE_MODEL_PATH,
        "base_image_url": DEFAULT_BASE_IMAGE_URL,
        "strategies": list(DEFAULT_IMAGE_STRATEGIES),
        "epochs": 5,
        "imgsz": DEFAULT_IMGSZ,
        "train_batch": DEFAULT_TRAIN_BATCH,
        "predict_batch": DEFAULT_TRAIN_BATCH,
        "link_mode": DEFAULT_LINK_MODE,
        "manual_labels_by_iteration": {},
        "human_export_path": "",
        "reviewed_images_path": "",
        "dataset_dir": "",
        "local_files_document_root": "",
        "wait_for_human_feedback": DEFAULT_WAIT_FOR_HUMAN_FEEDBACK,
        "human_feedback_dir": "",
        "human_wait_timeout_sec": DEFAULT_HUMAN_WAIT_TIMEOUT_SEC,
        "human_poll_interval_sec": DEFAULT_HUMAN_POLL_INTERVAL_SEC,
    }
    if config:
        resolved_config.update(config)

    labels_df = load_detection_labels(labeled_data_path)
    labels_csv = Path(labeled_data_path) if Path(labeled_data_path).is_file() else Path(labeled_data_path) / "reports" / "labels.csv"
    classes = sorted(labels_df[CLASS_LABEL_COLUMN].astype(str).unique().tolist())
    dataset_dir = resolve_dataset_dir(labels_df, resolved_config)
    run_id = resolved_config.get("run_id") or make_run_id(_slugify(Path(labels_csv).stem))
    run_dir = ensure_al_run_layout(Path(str(resolved_config["artifacts_dir"])) / run_id)
    reviewed_images_df = load_reviewed_images(str(resolved_config.get("reviewed_images_path") or ""))

    effective_backend = backend or YOLODetectionBackend(
        model_path=str(resolved_config["model_path"]),
        epochs=int(resolved_config["epochs"]),
        imgsz=int(resolved_config["imgsz"]),
        train_batch=int(resolved_config["train_batch"]),
        predict_batch=int(resolved_config["predict_batch"]),
        device=resolved_config.get("device"),
        link_mode=str(resolved_config["link_mode"]),
    )

    strategies = [str(strategy).strip().lower() for strategy in resolved_config["strategies"]]
    strategy_results: dict[str, Any] = {}
    history_paths: dict[str, str] = {}
    for strategy in strategies:
        strategy_result = _run_detection_strategy_cycle(
            strategy=strategy,
            task_description=task_description,
            labels_df=labels_df.copy(),
            reviewed_images_df=reviewed_images_df.copy(),
            dataset_dir=dataset_dir,
            classes=classes,
            backend=effective_backend,
            run_dir=run_dir,
            n_iterations=int(resolved_config["n_iterations"]),
            batch_size=int(resolved_config["batch_size"]),
            test_size=resolved_config["test_size"],
            random_state=int(resolved_config["random_state"]),
            base_image_url=str(resolved_config["base_image_url"]),
            manual_labels_by_iteration=resolved_config.get("manual_labels_by_iteration") or {},
            pre_run_human_export_path=str(resolved_config.get("human_export_path") or ""),
            local_files_document_root=str(resolved_config.get("local_files_document_root") or ""),
            wait_for_human_feedback=_bool_from_value(resolved_config.get("wait_for_human_feedback", DEFAULT_WAIT_FOR_HUMAN_FEEDBACK)),
            human_feedback_dir=str(resolved_config.get("human_feedback_dir") or ""),
            human_wait_timeout_sec=float(resolved_config.get("human_wait_timeout_sec", DEFAULT_HUMAN_WAIT_TIMEOUT_SEC)),
            human_poll_interval_sec=float(resolved_config.get("human_poll_interval_sec", DEFAULT_HUMAN_POLL_INTERVAL_SEC)),
        )
        strategy_results[strategy] = strategy_result
        history_paths[strategy] = strategy_result["history_json"]

    learning_curve_path = plot_learning_curves(
        {strategy: result["history"] for strategy, result in strategy_results.items()},
        run_dir / "curves" / "learning_curve.png",
    )
    primary_strategy = strategies[0]
    primary_result = strategy_results[primary_strategy]
    summary_payload = {
        "task_description": task_description,
        "labels_csv_input": str(labels_csv),
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "classes": classes,
        "strategies": strategies,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
        "awaiting_human_feedback": bool(primary_result.get("awaiting_human_feedback", False)),
        "human_feedback_instruction_path": str(primary_result.get("human_feedback_instruction_path") or ""),
        "expected_human_export_path": str(primary_result.get("expected_human_export_path") or ""),
    }
    summary_path = _write_json(run_dir / "reports" / "summary.json", summary_payload)
    return {
        "modality": "image",
        "task_mode": "object_detection",
        "implemented": True,
        "task_description": task_description,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "labels_csv": primary_result["labels_csv"],
        "reviewed_images_csv": primary_result["reviewed_images_csv"],
        "classes": classes,
        "strategy_results": strategy_results,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
        "labelstudio_import_path": primary_result["labelstudio_import_path"],
        "labelstudio_config_path": primary_result["labelstudio_config_path"],
        "uncertain_manifest_path": primary_result["uncertain_manifest_path"],
        "final_model_dir": primary_result["final_model_dir"],
        "summary_path": summary_path,
        "awaiting_human_feedback": bool(primary_result.get("awaiting_human_feedback", False)),
        "human_feedback_instruction_path": str(primary_result.get("human_feedback_instruction_path") or ""),
        "expected_human_export_path": str(primary_result.get("expected_human_export_path") or ""),
        "config": resolved_config,
    }


def table_classification_active_learning_impl(
    *,
    task_description: str,
    labeled_data_path: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "modality": "table",
        "task_mode": "classification",
        "implemented": False,
        "task_description": task_description,
        "labeled_data_path": labeled_data_path,
        "config": config or {},
        "message": "Table classification active learning is not implemented yet.",
    }


@tool
def image_detection_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Run bbox-level active learning for image object detection.

    Args:
        task_description: Natural-language description of the detection task.
        labeled_data_path: Path to annotation reports directory or directly to labels.csv.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON with AL artifacts, histories, learning-curve plot path, and Label Studio export paths.
    """
    logger.info("image_detection_active_learning called: labeled_data_path=%s", labeled_data_path)
    try:
        return _json_success(
            image_detection_active_learning_impl(
                task_description=task_description,
                labeled_data_path=labeled_data_path,
                config=_read_jsonish(config_json),
            )
        )
    except Exception as exc:
        logger.exception("image_detection_active_learning failed")
        return _json_error(str(exc))


@tool
def table_classification_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Placeholder wrapper for future table active learning support.

    Args:
        task_description: Natural-language description of the tabular task.
        labeled_data_path: Path to the labeled tabular dataset artifact.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON stub describing the current unimplemented state.
    """
    logger.info("table_classification_active_learning called: labeled_data_path=%s", labeled_data_path)
    try:
        return _json_success(
            table_classification_active_learning_impl(
                task_description=task_description,
                labeled_data_path=labeled_data_path,
                config=_read_jsonish(config_json),
            )
        )
    except Exception as exc:
        logger.exception("table_classification_active_learning failed")
        return _json_error(str(exc))
