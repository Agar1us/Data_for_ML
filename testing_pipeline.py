from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
PIPELINE_STAGES = ("dataset", "quality", "annotation", "al")
def load_root_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _ensure_current_run_dirs(current_run_root: Path) -> dict[str, Path]:
    directories = {
        "current_run_root": current_run_root,
        "collection_dir": current_run_root / "collection",
        "quality_dir": current_run_root / "quality",
        "annotation_dir": current_run_root / "annotation",
        "al_dir": current_run_root / "al",
        "logs_dir": current_run_root / "logs",
        "collection_artifacts_dir": current_run_root / "collection_artifacts",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _make_log_dir_name(query: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(char if char.isalnum() else "_" for char in query[:40]).strip("_")
    safe_query = safe_query or "request"
    return f"{timestamp}_{safe_query}"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _memory_records(agent: Any, stage_name: str) -> list[dict[str, Any]]:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return []

    records: list[dict[str, Any]] = []
    system_prompt = getattr(getattr(memory, "system_prompt", None), "system_prompt", "")
    if system_prompt:
        records.append(
            {
                "record_type": "system_prompt",
                "stage": stage_name,
                "agent_name": getattr(agent, "name", None) or getattr(agent, "agent_name", type(agent).__name__),
                "data": system_prompt,
            }
        )

    if hasattr(memory, "get_full_steps"):
        steps = memory.get_full_steps()
    else:
        raw_steps = getattr(memory, "steps", [])
        steps = [getattr(step, "dict", lambda: {"data": str(step)})() for step in raw_steps]

    for index, step in enumerate(steps):
        records.append(
            {
                "record_type": "memory_step",
                "stage": stage_name,
                "step": index,
                "agent_name": getattr(agent, "name", None) or getattr(agent, "agent_name", type(agent).__name__),
                "data": step,
            }
        )
    return records


def _save_stage_logs(
    *,
    log_dir: Path,
    file_name: str,
    stage_name: str,
    inputs: dict[str, Any],
    result: dict[str, Any],
    agent: Any | None = None,
) -> str:
    records: list[dict[str, Any]] = [
        {"record_type": "stage_inputs", "stage": stage_name, "data": inputs},
        {"record_type": "stage_result", "stage": stage_name, "data": result},
    ]
    if agent is not None:
        records.extend(_memory_records(agent, stage_name))
        last_result = getattr(agent, "last_result", None)
        if last_result is not None:
            records.append({"record_type": "agent_last_result", "stage": stage_name, "data": last_result})
    output_path = log_dir / file_name
    _write_jsonl(output_path, records)
    return str(output_path)


def _contains_images(directory: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS for path in directory.rglob("*"))


def _class_subdirs(directory: Path) -> list[Path]:
    return [child for child in sorted(directory.iterdir()) if child.is_dir() and _contains_images(child)]


def _latest_tree_mtime(directory: Path) -> float:
    latest = directory.stat().st_mtime
    for path in directory.rglob("*"):
        try:
            latest = max(latest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def _class_id(label: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", label.strip(), flags=re.UNICODE)
    compact = re.sub(r"_+", "_", normalized).strip("_")
    return compact or "class"


def _write_class_mapping(dataset_dir: Path, query: str) -> tuple[Path, dict[str, int], list[str]]:
    class_dirs = _class_subdirs(dataset_dir)
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories with images found under dataset root: {dataset_dir}")

    class_mapping_path = dataset_dir / "class_mapping.json"
    raw_counts: dict[str, int] = {}
    class_mapping: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_class_ids: dict[str, str] = {}
    for class_dir in class_dirs:
        label = class_dir.name
        class_id = _class_id(label)
        if class_id in seen_class_ids and seen_class_ids[class_id] != label:
            warnings.append(
                f"class_id collision detected: '{label}' and '{seen_class_ids[class_id]}' both map to '{class_id}'"
            )
        else:
            seen_class_ids[class_id] = label
        image_paths = sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        raw_counts[label] = len(image_paths)
        class_mapping.append(
            {
                "class_id": class_id,
                "folder_label": label,
                "display_label": label,
                "description": "",
                "sample_query": image_paths[0].relative_to(class_dir).parts[0] if image_paths and len(image_paths[0].relative_to(class_dir).parts) > 1 else query,
            }
        )

    class_mapping_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_dir.resolve()),
                "classes": class_mapping,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return class_mapping_path, raw_counts, warnings


def _write_annotation_config(dataset_dir: Path, object_prompts: list[str] | None) -> Path | None:
    prompts = [str(item).strip() for item in (object_prompts or []) if str(item).strip()]
    if not prompts:
        return None
    config_path = dataset_dir / "annotation_config.json"
    config_path.write_text(
        json.dumps({"object_prompts": prompts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path


def _read_annotation_config_prompts(dataset_dir: Path) -> list[str]:
    candidates = [dataset_dir / "annotation_config.json"]
    if dataset_dir.parent != dataset_dir:
        candidates.append(dataset_dir.parent / "annotation_config.json")
    for config_path in candidates:
        if not config_path.exists():
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        prompts = [str(item).strip() for item in list(payload.get("object_prompts", [])) if str(item).strip()]
        if prompts:
            return prompts
    return []

def _normalize_prompt_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _extract_prompt_list_from_text(result_text: str, field_name: str) -> list[str]:
    pattern = rf"{field_name}\s*:\s*(.+)"
    match = re.search(pattern, result_text, flags=re.IGNORECASE)
    if not match:
        return []

    raw_value = match.group(1).strip()
    if "\n" in raw_value:
        raw_value = raw_value.splitlines()[0].strip()
    raw_value = raw_value.lstrip("-").strip()
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value.strip("\"'")

    return _normalize_prompt_list(parsed)


def _extract_object_prompts_from_dataset_result(result: Any) -> list[str]:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            prompts = _extract_prompt_list_from_text(result, "object_prompts")
            if prompts:
                return prompts
            return _extract_prompt_list_from_text(result, "object_prompt")
    if not isinstance(payload, dict):
        return []
    prompts = _normalize_prompt_list(payload.get("object_prompts"))
    if prompts:
        return prompts
    return _normalize_prompt_list(payload.get("object_prompt"))


def _invalid_object_prompts_shape_message(result: Any) -> str:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return ""
    if not isinstance(payload, dict):
        return ""

    raw_prompts = payload.get("object_prompts")
    if isinstance(raw_prompts, dict):
        return (
            "dataset-agent returned object_prompts as a dict of per-class values. "
            "annotation stage requires one generic English object prompt, for example ['swan']."
        )

    raw_prompt = payload.get("object_prompt")
    if isinstance(raw_prompt, dict):
        return (
            "dataset-agent returned object_prompt as a dict. "
            "annotation stage requires one generic English object prompt, for example 'swan'."
        )
    return ""


def _prepare_dataset_stage_artifacts(dataset_dir: Path, query: str, object_prompts: list[str] | None) -> dict[str, Any]:
    class_mapping_path, raw_counts, warnings = _write_class_mapping(dataset_dir, query)
    annotation_config_path = _write_annotation_config(dataset_dir, object_prompts)
    class_labels = sorted(raw_counts.keys())
    return {
        "dataset_root": str(dataset_dir.resolve()),
        "class_mapping_json": str(class_mapping_path.resolve()),
        "annotation_config_json": str(annotation_config_path.resolve()) if annotation_config_path else "",
        "class_labels": class_labels,
        "class_dirs": [str((dataset_dir / label).resolve()) for label in class_labels],
        "raw_counts": raw_counts,
        "warnings": warnings,
    }


def _resolve_annotation_object_prompts(
    dataset_stage: dict[str, Any],
    *,
    dataset_dir: Path,
    explicit_prompts: list[str] | None = None,
) -> list[str]:
    prompts = [str(item).strip() for item in (explicit_prompts or []) if str(item).strip()]
    if prompts:
        return prompts
    prompts = _read_annotation_config_prompts(dataset_dir)
    if prompts:
        return prompts
    prompts = _extract_object_prompts_from_dataset_result(dataset_stage.get("result"))
    if prompts:
        return prompts
    invalid_shape_message = _invalid_object_prompts_shape_message(dataset_stage.get("result"))
    if invalid_shape_message:
        raise RuntimeError(invalid_shape_message)
    raise RuntimeError(
        "annotation object prompts are required for the annotation stage. "
        "Pass --annotation-object-prompt ..., provide collection/annotation_config.json, "
        "or make dataset-agent return object_prompts in its structured result."
    )


def _validate_dataset_stage(dataset_stage: dict[str, Any]) -> None:
    required_paths = ["dataset_root", "class_mapping_json"]
    for key in required_paths:
        path = Path(str(dataset_stage.get(key, "")))
        if not path.exists():
            raise RuntimeError(f"Dataset stage is missing required artifact '{key}': {path}")
    if not dataset_stage.get("class_labels"):
        raise RuntimeError("Dataset stage produced no class labels.")


def _validate_folder_classification_dataset(dataset_dir: Path, stage_name: str) -> None:
    if not dataset_dir.exists():
        raise RuntimeError(f"{stage_name} dataset directory does not exist: {dataset_dir}")
    class_dirs = _class_subdirs(dataset_dir)
    if not class_dirs:
        raise RuntimeError(
            f"{stage_name} dataset is not a valid folder-based image classification dataset: {dataset_dir}"
        )


def snapshot_image_datasets(collection_dir: Path) -> dict[str, float]:
    snapshots: dict[str, float] = {}
    if not collection_dir.exists():
        return snapshots

    candidates = [collection_dir, *[path for path in collection_dir.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if _class_subdirs(candidate):
            snapshots[str(candidate.resolve())] = _latest_tree_mtime(candidate)
    return snapshots


def discover_latest_image_dataset(collection_dir: Path, before_snapshot: dict[str, float] | None = None) -> Path:
    if not collection_dir.exists():
        raise RuntimeError(f"Collection directory does not exist: {collection_dir}")

    before_snapshot = before_snapshot or {}
    candidates: list[tuple[tuple[float, float, int, int], Path]] = []

    directories = [collection_dir, *[path for path in collection_dir.rglob("*") if path.is_dir()]]
    for candidate in directories:
        class_dirs = _class_subdirs(candidate)
        if not class_dirs:
            continue

        resolved = candidate.resolve()
        latest_mtime = _latest_tree_mtime(candidate)
        previous_mtime = before_snapshot.get(str(resolved), 0.0)
        is_new = 1 if str(resolved) not in before_snapshot else 0
        changed_score = latest_mtime - previous_mtime
        depth_score = -len(candidate.relative_to(collection_dir).parts)
        rank = (float(is_new), changed_score, len(class_dirs), depth_score)
        candidates.append((rank, resolved))

    if not candidates:
        raise RuntimeError(
            "No folder-based image dataset was found under the collection directory. "
            "Expected a dataset root with class subdirectories that contain image files."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _try_discover_latest_image_dataset(root_dir: Path) -> Path | None:
    try:
        return discover_latest_image_dataset(root_dir)
    except RuntimeError:
        return None


def _normalize_stage_selection(stages: list[str] | None) -> list[str]:
    selected = [stage.strip().lower() for stage in (stages or list(PIPELINE_STAGES)) if stage.strip()]
    invalid = [stage for stage in selected if stage not in PIPELINE_STAGES]
    if invalid:
        raise ValueError(f"Unsupported stages: {invalid}. Allowed values: {list(PIPELINE_STAGES)}")
    return selected or list(PIPELINE_STAGES)


def _ensure_dataset_agent_import_path() -> Path:
    dataset_agent_root = PROJECT_ROOT / "dataset-agent"
    if str(dataset_agent_root) not in sys.path:
        sys.path.insert(0, str(dataset_agent_root))
    return dataset_agent_root


def run_dataset_stage(
    *,
    query: str,
    collection_dir: Path,
    logs_dir: Path,
    collection_artifacts_dir: Path,
    max_results: int = 10,
) -> dict[str, Any]:
    dataset_agent_root = _ensure_dataset_agent_import_path()

    from dotenv import load_dotenv

    from config import AgentConfig, PROJECT_ROOT as DATASET_PROJECT_ROOT
    from main import ensure_dependencies, save_agent_logs, setup_logging
    from tools.runtime import set_runtime_context

    load_dotenv(DATASET_PROJECT_ROOT / ".env")

    ensure_dependencies()
    from agents.orchestrator import create_orchestrator

    config = AgentConfig(
        max_search_results=max_results,
        data_dir=str(collection_dir),
        logs_dir=str(logs_dir),
        artifacts_dir=str(collection_artifacts_dir),
    )

    config.data_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)

    set_runtime_context(
        data_root=config.data_root,
        logs_root=config.logs_root,
        artifacts_root=config.artifacts_root,
        yandex_headless=config.yandex_headless,
        yandex_manual_captcha_timeout=config.yandex_manual_captcha_timeout,
        yandex_profile_dir=config.yandex_profile_dir,
    )

    log_dir = Path(setup_logging(config, query))
    set_runtime_context(
        data_root=config.data_root,
        logs_root=config.logs_root,
        artifacts_root=config.artifacts_root,
        run_log_dir=log_dir,
        yandex_headless=config.yandex_headless,
        yandex_manual_captcha_timeout=config.yandex_manual_captcha_timeout,
        yandex_profile_dir=config.yandex_profile_dir,
    )
    orchestrator = create_orchestrator(config)

    try:
        result = orchestrator.run(query)
    finally:
        save_agent_logs(orchestrator, str(log_dir))

    return {
        "stage": "dataset",
        "query": query,
        "result": result,
        "data_dir": str(config.data_root),
        "logs_dir": str(config.logs_root),
        "artifacts_dir": str(config.artifacts_root),
        "log_dir": str(log_dir),
        "dataset_agent_root": str(dataset_agent_root),
    }


def run_quality_stage(
    *,
    dataset_dir: Path,
    quality_dir: Path,
    log_dir: Path,
    task_description: str,
    model_id: str = "gpt-5-mini",
    hash_func_name: str = "phash",
    hash_size: int = 16,
    threshold: int = 8,
) -> dict[str, Any]:
    from data_quality_tools_agent import ToolBasedDataQualityAgent

    agent = ToolBasedDataQualityAgent(
        model_id=model_id,
        artifacts_dir=str(quality_dir),
        task_description=task_description,
    )
    result = agent.deduplicate_images(
        input_dir=str(dataset_dir),
        output_dir=str(quality_dir),
        hash_func_name=hash_func_name,
        hash_size=hash_size,
        threshold=threshold,
        dry_run=False,
    )
    stage_result = {"stage": "quality", **result}
    log_path = _save_stage_logs(
        log_dir=log_dir,
        file_name="quality_agent_log.jsonl",
        stage_name="quality",
        inputs={
            "dataset_dir": str(dataset_dir),
            "quality_dir": str(quality_dir),
            "task_description": task_description,
            "model_id": model_id,
            "hash_func_name": hash_func_name,
            "hash_size": hash_size,
            "threshold": threshold,
        },
        result=stage_result,
        agent=agent,
    )
    stage_result["log_path"] = log_path
    return stage_result


def run_annotation_stage(
    *,
    dataset_dir: Path,
    annotation_dir: Path,
    log_dir: Path,
    task: str,
    object_prompts: list[str] | None = None,
    model_id: str = "gpt-5-mini",
    model_path: str = "yoloe-26x-seg.pt",
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    from annotation_agent import AnnotationAgent

    agent = AnnotationAgent(
        modality="image",
        object_prompts=object_prompts,
        task_mode="image_classification",
        label_assignment_mode="folder_label",
        confidence_threshold=confidence_threshold,
        model_path=model_path,
        model_id=model_id,
        artifacts_dir=str(annotation_dir),
    )
    result = agent.run_pipeline(dataset_dir=str(dataset_dir), task=task)
    stage_result = {"stage": "annotation", **result}
    log_path = _save_stage_logs(
        log_dir=log_dir,
        file_name="annotation_agent_log.jsonl",
        stage_name="annotation",
        inputs={
            "dataset_dir": str(dataset_dir),
            "annotation_dir": str(annotation_dir),
            "task": task,
            "object_prompts": object_prompts or [],
            "model_id": model_id,
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
        },
        result=stage_result,
        agent=agent,
    )
    stage_result["log_path"] = log_path
    return stage_result


def _resolve_al_labels_source(annotation_dir: Path, al_dir: Path, annotation_stage: dict[str, Any], annotation_stage_selected: bool) -> Path:
    if annotation_stage_selected:
        stage_path = str(annotation_stage.get("labels_csv") or "").strip()
        if stage_path:
            candidate = Path(stage_path).resolve()
            if candidate.exists():
                return candidate
        candidate = annotation_dir / "reports" / "labels.csv"
        if candidate.exists():
            return candidate
        raise RuntimeError(f"Annotation stage did not produce labels.csv: {candidate}")
    for candidate in (al_dir / "reports" / "labels.csv", annotation_dir / "reports" / "labels.csv"):
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not resolve bbox-level labels.csv for al stage.")


def _resolve_al_reviewed_images_source(al_dir: Path) -> str:
    candidate = al_dir / "reports" / "reviewed_images.csv"
    return str(candidate.resolve()) if candidate.exists() else ""


def run_al_stage(
    *,
    dataset_dir: Path,
    labels_path: Path,
    al_dir: Path,
    log_dir: Path,
    task_description: str,
    human_export_path: str = "",
    model_path: str = "yolo26x.pt",
    batch_size: int = 20,
    n_iterations: int = 1,
    test_size: float = 0.2,
    strategies: list[str] | None = None,
    wait_for_human_feedback: bool = True,
    human_feedback_dir: str = "",
    human_wait_timeout_sec: float = 86400.0,
    human_poll_interval_sec: float = 5.0,
) -> dict[str, Any]:
    from al_agent.agent import ALAgent

    config = {
        "artifacts_dir": str(al_dir),
        "dataset_dir": str(dataset_dir),
        "model_path": model_path,
        "batch_size": int(batch_size),
        "n_iterations": int(n_iterations),
        "test_size": test_size,
        "strategies": list(strategies or ["confidence"]),
        "human_export_path": human_export_path,
        "reviewed_images_path": _resolve_al_reviewed_images_source(al_dir),
        "wait_for_human_feedback": bool(wait_for_human_feedback),
        "human_feedback_dir": human_feedback_dir,
        "human_wait_timeout_sec": float(human_wait_timeout_sec),
        "human_poll_interval_sec": float(human_poll_interval_sec),
    }
    al_agent = ALAgent()
    result = al_agent.run(
        task_description=task_description,
        labeled_data_path=str(labels_path),
        modality="image",
        config_json=json.dumps(config, ensure_ascii=False),
    )
    stage_result = {"stage": "al", **result}
    log_path = _save_stage_logs(
        log_dir=log_dir,
        file_name="al_agent_log.jsonl",
        stage_name="al",
        inputs={
            "dataset_dir": str(dataset_dir),
            "labels_path": str(labels_path),
            "al_dir": str(al_dir),
            "task_description": task_description,
            "human_export_path": human_export_path,
            "model_path": model_path,
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "test_size": test_size,
            "strategies": list(strategies or ["confidence"]),
            "wait_for_human_feedback": wait_for_human_feedback,
            "human_feedback_dir": human_feedback_dir,
            "human_wait_timeout_sec": human_wait_timeout_sec,
            "human_poll_interval_sec": human_poll_interval_sec,
        },
        result=stage_result,
        agent=al_agent,
    )
    stage_result["log_path"] = log_path
    return stage_result


def run_four_agent_pipeline(
    *,
    query: str,
    task: str = "image_classification",
    current_run_root: Path | None = None,
    stages: list[str] | None = None,
    annotation_object_prompts: list[str] | None = None,
    dataset_max_results: int = 10,
    quality_model_id: str = "gpt-5-mini",
    annotation_model_id: str = "gpt-5-mini",
    annotation_model_path: str = "yoloe-26x-seg.pt",
    annotation_confidence_threshold: float = 0.5,
    al_model_path: str = "yolo26x.pt",
    al_batch_size: int = 20,
    al_n_iterations: int = 1,
    al_test_size: float = 0.2,
    al_human_export_path: str = "",
    al_strategies: list[str] | None = None,
    al_wait_for_human_feedback: bool = True,
    al_human_feedback_dir: str = "",
    al_human_wait_timeout_sec: float = 86400.0,
    al_human_poll_interval_sec: float = 5.0,
    dedup_hash_func_name: str = "phash",
    dedup_hash_size: int = 16,
    dedup_threshold: int = 8,
) -> dict[str, Any]:
    load_root_dotenv()
    selected_stages = _normalize_stage_selection(stages)
    if {"dataset", "quality"} & set(selected_stages) and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run dataset-agent and quality-agent in the chained pipeline.")

    root = current_run_root or (PROJECT_ROOT / "data" / "current_run")
    directories = _ensure_current_run_dirs(root)
    before_snapshot = snapshot_image_datasets(directories["collection_dir"]) if "dataset" in selected_stages else {}
    pipeline_log_dir = directories["logs_dir"] / _make_log_dir_name(query)
    if "dataset" not in selected_stages:
        pipeline_log_dir.mkdir(parents=True, exist_ok=True)

    dataset_stage: dict[str, Any]
    if "dataset" in selected_stages:
        dataset_stage = run_dataset_stage(
            query=query,
            collection_dir=directories["collection_dir"],
            logs_dir=directories["logs_dir"],
            collection_artifacts_dir=directories["collection_artifacts_dir"],
            max_results=dataset_max_results,
        )
        dataset_log_dir = Path(str(dataset_stage.get("log_dir") or pipeline_log_dir))
        dataset_log_dir.mkdir(parents=True, exist_ok=True)
        pipeline_log_dir = dataset_log_dir
        dataset_dir = discover_latest_image_dataset(directories["collection_dir"], before_snapshot=before_snapshot)
    else:
        dataset_dir = discover_latest_image_dataset(directories["collection_dir"])
        dataset_stage = {
            "stage": "dataset",
            "skipped": True,
            "reason": "dataset stage was not selected; using latest existing collection dataset",
        }

    dataset_stage_artifacts = _prepare_dataset_stage_artifacts(dataset_dir, query, None)
    dataset_stage.update(dataset_stage_artifacts)
    _validate_dataset_stage(dataset_stage)
    _validate_folder_classification_dataset(dataset_dir, "dataset-stage")
    resolved_object_prompts = _resolve_annotation_object_prompts(
        dataset_stage,
        dataset_dir=dataset_dir,
        explicit_prompts=annotation_object_prompts,
    )
    annotation_config_path = _write_annotation_config(dataset_dir, resolved_object_prompts)
    if annotation_config_path is not None:
        dataset_stage["annotation_config_json"] = str(annotation_config_path.resolve())

    quality_stage: dict[str, Any]
    if "quality" in selected_stages:
        quality_stage = run_quality_stage(
            dataset_dir=dataset_dir,
            quality_dir=directories["quality_dir"],
            log_dir=pipeline_log_dir,
            task_description=query,
            model_id=quality_model_id,
            hash_func_name=dedup_hash_func_name,
            hash_size=dedup_hash_size,
            threshold=dedup_threshold,
        )
        cleaned_dataset_dir = Path(quality_stage["output_dir"])
    else:
        existing_quality_dataset = _try_discover_latest_image_dataset(directories["quality_dir"])
        cleaned_dataset_dir = existing_quality_dataset or dataset_dir
        quality_stage = {
            "stage": "quality",
            "skipped": True,
            "reason": "quality stage was not selected; using latest existing quality dataset when available",
            "output_dir": str(cleaned_dataset_dir),
        }

    _validate_folder_classification_dataset(cleaned_dataset_dir, "quality-stage")

    if "annotation" in selected_stages:
        annotation_stage = run_annotation_stage(
            dataset_dir=cleaned_dataset_dir,
            annotation_dir=directories["annotation_dir"],
            log_dir=pipeline_log_dir,
            task=task,
            object_prompts=resolved_object_prompts,
            model_id=annotation_model_id,
            model_path=annotation_model_path,
            confidence_threshold=annotation_confidence_threshold,
        )
    else:
        annotation_stage = {
            "stage": "annotation",
            "skipped": True,
            "reason": "annotation stage was not selected",
            "object_prompts": resolved_object_prompts,
        }

    if "al" in selected_stages:
        al_labels_path = _resolve_al_labels_source(
            directories["annotation_dir"],
            directories["al_dir"],
            annotation_stage,
            annotation_stage_selected="annotation" in selected_stages,
        )
        al_stage = run_al_stage(
            dataset_dir=cleaned_dataset_dir,
            labels_path=al_labels_path,
            al_dir=directories["al_dir"],
            log_dir=pipeline_log_dir,
            task_description=task,
            human_export_path=al_human_export_path,
            model_path=al_model_path,
            batch_size=al_batch_size,
            n_iterations=al_n_iterations,
            test_size=al_test_size,
            strategies=al_strategies,
            wait_for_human_feedback=al_wait_for_human_feedback,
            human_feedback_dir=al_human_feedback_dir,
            human_wait_timeout_sec=al_human_wait_timeout_sec,
            human_poll_interval_sec=al_human_poll_interval_sec,
        )
    else:
        al_stage = {
            "stage": "al",
            "skipped": True,
            "reason": "al stage was not selected",
        }

    summary = {
        "query": query,
        "task": task,
        "selected_stages": selected_stages,
        "current_run_root": str(root),
        "log_dir": str(pipeline_log_dir),
        "dataset_dir": str(dataset_dir),
        "quality_dataset_dir": str(cleaned_dataset_dir),
        "annotation_object_prompts": resolved_object_prompts,
        "dataset_warnings": dataset_stage.get("warnings", []),
        "quality_warnings": quality_stage.get("warnings", []),
        "annotation_warnings": annotation_stage.get("warnings", []),
        "al_warnings": al_stage.get("warnings", []),
        "stages": {
            "dataset": dataset_stage,
            "quality": quality_stage,
            "annotation": annotation_stage,
            "al": al_stage,
        },
    }
    summary_path = root / "pipeline_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> int:
    load_root_dotenv()
    parser = argparse.ArgumentParser(
        description="Run dataset-agent, quality-agent, annotation-agent, and al-agent sequentially from the repository root."
    )
    parser.add_argument("query", type=str, help="User request for dataset collection and downstream processing.")
    parser.add_argument(
        "--task",
        type=str,
        default="image_classification",
        help="Annotation task description passed into annotation-agent.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(PIPELINE_STAGES),
        default=list(PIPELINE_STAGES),
        help="Subset of pipeline stages to run, e.g. --stages dataset quality or --stages annotation.",
    )
    parser.add_argument(
        "--current-run-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "current_run",
        help="Root directory for shared pipeline artifacts.",
    )
    parser.add_argument(
        "--dataset-max-results",
        type=int,
        default=10,
        help="Maximum search results per source for dataset-agent.",
    )
    parser.add_argument(
        "--quality-model-id",
        type=str,
        default="gpt-5-mini",
        help="Model id for quality-agent.",
    )
    parser.add_argument(
        "--annotation-object-prompt",
        action="append",
        dest="annotation_object_prompts",
        default=[],
        help="Repeatable generic object prompt used by annotation-agent geometry extraction, e.g. --annotation-object-prompt swan.",
    )
    parser.add_argument(
        "--annotation-model-id",
        type=str,
        default="gpt-5-mini",
        help="Model id for annotation-agent spec generation.",
    )
    parser.add_argument(
        "--annotation-model-path",
        type=str,
        default="yoloe-26x-seg.pt",
        help="YOLOE model checkpoint path for annotation-agent auto-labeling.",
    )
    parser.add_argument(
        "--annotation-confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for annotation-agent manual-review split.",
    )
    parser.add_argument(
        "--al-model-path",
        type=str,
        default="yolo26x.pt",
        help="YOLO detection checkpoint path for al-agent.",
    )
    parser.add_argument(
        "--al-batch-size",
        type=int,
        default=20,
        help="Number of uncertain images to export per AL iteration.",
    )
    parser.add_argument(
        "--al-n-iterations",
        type=int,
        default=1,
        help="Maximum number of AL iterations to execute in one run.",
    )
    parser.add_argument(
        "--al-test-size",
        type=float,
        default=0.2,
        help="Holdout fraction or count for AL detector evaluation.",
    )
    parser.add_argument(
        "--al-human-export-path",
        type=str,
        default="",
        help="Optional Label Studio export path to merge before the current AL iteration.",
    )
    parser.add_argument(
        "--al-wait-for-human-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pause the AL run, ask the user to annotate the exported batch, and continue in the same process.",
    )
    parser.add_argument(
        "--al-human-feedback-dir",
        type=str,
        default="",
        help="Optional directory where AL waits for human Label Studio exports per iteration.",
    )
    parser.add_argument(
        "--al-human-wait-timeout-sec",
        type=float,
        default=86400.0,
        help="Timeout in seconds while waiting for a human export during AL.",
    )
    parser.add_argument(
        "--al-human-poll-interval-sec",
        type=float,
        default=5.0,
        help="Polling interval in seconds for human export files in non-interactive AL runs.",
    )
    parser.add_argument(
        "--al-strategy",
        action="append",
        dest="al_strategies",
        default=[],
        help="Repeatable AL query strategy, e.g. --al-strategy confidence.",
    )
    parser.add_argument(
        "--dedup-hash",
        type=str,
        default="phash",
        choices=["phash", "dhash", "ahash", "whash"],
        help="Hash function for image deduplication.",
    )
    parser.add_argument(
        "--dedup-hash-size",
        type=int,
        default=16,
        help="Hash size for image deduplication.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=int,
        default=8,
        help="Hamming-distance threshold for near-duplicate image removal.",
    )
    args = parser.parse_args()

    summary = run_four_agent_pipeline(
        query=args.query,
        task=args.task,
        current_run_root=args.current_run_root,
        stages=args.stages,
        annotation_object_prompts=args.annotation_object_prompts or None,
        dataset_max_results=args.dataset_max_results,
        quality_model_id=args.quality_model_id,
        annotation_model_id=args.annotation_model_id,
        annotation_model_path=args.annotation_model_path,
        annotation_confidence_threshold=args.annotation_confidence_threshold,
        al_model_path=args.al_model_path,
        al_batch_size=args.al_batch_size,
        al_n_iterations=args.al_n_iterations,
        al_test_size=args.al_test_size,
        al_human_export_path=args.al_human_export_path,
        al_strategies=args.al_strategies or None,
        al_wait_for_human_feedback=args.al_wait_for_human_feedback,
        al_human_feedback_dir=args.al_human_feedback_dir,
        al_human_wait_timeout_sec=args.al_human_wait_timeout_sec,
        al_human_poll_interval_sec=args.al_human_poll_interval_sec,
        dedup_hash_func_name=args.dedup_hash,
        dedup_hash_size=args.dedup_hash_size,
        dedup_threshold=args.dedup_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
