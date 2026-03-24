from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("scikit-learn is required for Active Learning.") from exc

try:
    from smolagents import tool
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    def tool(func):  # type: ignore[misc]
        return func


logger = logging.getLogger(__name__)

FILE_PATH_COLUMN = "file_path"
FILENAME_COLUMN = "filename"
LABEL_COLUMN = "folder_label"
IMAGE_ID_COLUMN = "image_id"
SUGGESTED_LABEL_COLUMN = "suggested_label"
CONFIDENCE_COLUMN = "confidence"
HUMAN_LABEL_COLUMN = "human_label"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_AL_ARTIFACTS_DIR = "al_artifacts"
DEFAULT_IMAGE_MODEL_PATH = "yolo26x-cls.pt"
DEFAULT_IMAGE_STRATEGIES = ("confidence", "random")
DEFAULT_INITIAL_SIZE = 50
DEFAULT_N_ITERATIONS = 5
DEFAULT_BATCH_SIZE = 20
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_BASE_IMAGE_URL = "/data/local-files/?d="
DEFAULT_LINK_MODE = "symlink"
DEFAULT_EPOCHS = 5
DEFAULT_IMGSZ = 224
DEFAULT_TRAIN_BATCH = 16


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
    for subdir in ("reports", "curves", "splits", "strategies", "labelstudio", "models"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: str | Path, payload: Any) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _read_jsonish(config_json: str | dict[str, Any] | None) -> dict[str, Any]:
    if config_json is None or config_json == "":
        return {}
    if isinstance(config_json, dict):
        return dict(config_json)
    candidate = Path(config_json)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(config_json)


def _stable_image_id(file_path: str) -> str:
    return hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:16]


def _resolve_labeled_csv_path(labeled_data_path: str) -> Path:
    input_path = Path(labeled_data_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Labeled data path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {input_path}")
        return input_path
    candidates = [
        input_path / "cleaned_or_labeled" / "labeled.csv",
        input_path / "labeled.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve labeled.csv from {input_path}. Expected {candidates[0]} or {candidates[1]}."
    )


def load_image_classification_dataset(labeled_data_path: str) -> pd.DataFrame:
    labeled_csv = _resolve_labeled_csv_path(labeled_data_path)
    df = pd.read_csv(labeled_csv)
    missing = [column for column in (FILE_PATH_COLUMN, LABEL_COLUMN) if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {labeled_csv}: {missing}")
    df = df.copy()
    df[FILE_PATH_COLUMN] = df[FILE_PATH_COLUMN].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
    df[FILENAME_COLUMN] = df.get(FILENAME_COLUMN, df[FILE_PATH_COLUMN].map(lambda value: Path(str(value)).name))
    df[IMAGE_ID_COLUMN] = df.get(IMAGE_ID_COLUMN, df[FILE_PATH_COLUMN].map(_stable_image_id))

    missing_files = [path for path in df[FILE_PATH_COLUMN] if not Path(path).exists()]
    if missing_files:
        sample = missing_files[:5]
        raise FileNotFoundError(f"Some image files referenced by {labeled_csv} do not exist: {sample}")

    extensions = {Path(path).suffix.lower() for path in df[FILE_PATH_COLUMN]}
    unsupported = sorted(extension for extension in extensions if extension and extension not in SUPPORTED_IMAGE_EXTENSIONS)
    if unsupported:
        logger.warning("Dataset contains unsupported-looking file extensions: %s", unsupported)
    return df


def _can_stratify(df: pd.DataFrame, n_selected: int) -> bool:
    if df.empty or LABEL_COLUMN not in df.columns:
        return False
    class_counts = df[LABEL_COLUMN].value_counts()
    if len(class_counts) < 2:
        return False
    if class_counts.min() < 2:
        return False
    remaining = len(df) - n_selected
    n_classes = len(class_counts)
    return n_selected >= n_classes and remaining >= n_classes


def _split_dataframe(df: pd.DataFrame, n_selected: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_selected = max(0, min(int(n_selected), len(df)))
    if n_selected == 0:
        return df.iloc[0:0].copy(), df.reset_index(drop=True).copy()
    if n_selected >= len(df):
        return df.reset_index(drop=True).copy(), df.iloc[0:0].copy()
    stratify = df[LABEL_COLUMN] if _can_stratify(df, n_selected) else None
    selected, remaining = train_test_split(
        df,
        train_size=n_selected,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return selected.reset_index(drop=True), remaining.reset_index(drop=True)


def _normalize_test_count(df: pd.DataFrame, test_size: float | int) -> int:
    n_rows = len(df)
    if n_rows <= 2:
        return 1
    n_classes = max(1, int(df[LABEL_COLUMN].nunique()))
    requested = int(round(n_rows * test_size)) if isinstance(test_size, float) and test_size < 1 else int(test_size)
    requested = max(1, requested)
    max_allowed = max(1, n_rows - 1)
    if n_rows >= (2 * n_classes):
        requested = max(requested, n_classes)
        max_allowed = min(max_allowed, n_rows - n_classes)
    return max(1, min(requested, max_allowed))


def prepare_image_splits(
    df: pd.DataFrame,
    initial_size: int = DEFAULT_INITIAL_SIZE,
    test_size: float | int = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("At least 3 labeled examples are required to prepare active-learning splits.")
    test_count = _normalize_test_count(df, test_size)
    test_df, remainder_df = _split_dataframe(df, test_count, random_state)
    if remainder_df.empty:
        raise ValueError("No samples remain for train/pool after test split.")
    max_initial = max(1, len(remainder_df) - 1) if len(remainder_df) > 1 else 1
    initial_count = min(int(initial_size), max_initial)
    labeled_df, pool_df = _split_dataframe(remainder_df, initial_count, random_state + 1)
    if labeled_df.empty:
        raise ValueError("Initial labeled split is empty after split preparation.")
    return labeled_df.reset_index(drop=True), pool_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _link_or_copy_image(source: str | Path, target: str | Path, link_mode: str = DEFAULT_LINK_MODE) -> None:
    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if link_mode == "symlink":
        try:
            target_path.symlink_to(source_path.resolve())
            return
        except OSError:
            logger.warning("Falling back to copy for %s -> %s", source_path, target_path)
    shutil.copy2(source_path, target_path)


def _materialize_yolo_classification_split(
    subset_df: pd.DataFrame,
    output_dir: str | Path,
    link_mode: str,
) -> None:
    subset_dir = Path(output_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)
    for _, row in subset_df.iterrows():
        label = str(row[LABEL_COLUMN])
        source_path = Path(str(row[FILE_PATH_COLUMN]))
        destination = subset_dir / label / source_path.name
        _link_or_copy_image(source_path, destination, link_mode=link_mode)


def build_yolo_classification_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str | Path,
    link_mode: str = DEFAULT_LINK_MODE,
) -> Path:
    dataset_root = Path(output_dir)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    _materialize_yolo_classification_split(train_df, dataset_root / "train", link_mode=link_mode)
    _materialize_yolo_classification_split(val_df, dataset_root / "val", link_mode=link_mode)
    return dataset_root


class YOLOClassificationBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_IMAGE_MODEL_PATH,
        *,
        epochs: int = DEFAULT_EPOCHS,
        imgsz: int = DEFAULT_IMGSZ,
        train_batch: int = DEFAULT_TRAIN_BATCH,
        predict_batch: int = DEFAULT_TRAIN_BATCH,
        device: str | int | None = None,
        link_mode: str = DEFAULT_LINK_MODE,
    ) -> None:
        self.model_path = model_path
        self.epochs = int(epochs)
        self.imgsz = int(imgsz)
        self.train_batch = int(train_batch)
        self.predict_batch = int(predict_batch)
        self.device = device
        self.link_mode = link_mode

    def _load_model(self, weights_path: str | None = None) -> Any:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ultralytics is required for image active learning.") from exc
        load_path = weights_path or self.model_path
        return YOLO(load_path, task="classify")

    def train(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        classes: list[str],
        work_dir: str | Path,
        iteration_name: str,
    ) -> dict[str, Any]:
        dataset_root = build_yolo_classification_dataset(
            train_df=train_df,
            val_df=val_df,
            output_dir=Path(work_dir) / "datasets" / iteration_name,
            link_mode=self.link_mode,
        )
        model = self._load_model()
        model.train(
            data=str(dataset_root),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.train_batch,
            project=str(Path(work_dir) / "models"),
            name=iteration_name,
            exist_ok=True,
            verbose=False,
            device=self.device,
        )
        save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", Path(work_dir) / "models" / iteration_name))
        best_weights = save_dir / "weights" / "best.pt"
        last_weights = save_dir / "weights" / "last.pt"
        weights_path = best_weights if best_weights.exists() else last_weights if last_weights.exists() else Path(self.model_path)
        trained_model = self._load_model(str(weights_path))
        return {
            "model": trained_model,
            "model_dir": str(save_dir),
            "weights_path": str(weights_path),
            "dataset_root": str(dataset_root),
            "classes": classes,
        }

    def predict_proba(self, model: Any, image_paths: Iterable[str], classes: list[str]) -> np.ndarray:
        image_path_list = [str(path) for path in image_paths]
        if not image_path_list:
            return np.zeros((0, len(classes)), dtype=float)
        results = model.predict(
            source=image_path_list,
            imgsz=self.imgsz,
            batch=self.predict_batch,
            verbose=False,
            device=self.device,
        )
        names = getattr(model, "names", None)
        if not names and results:
            names = getattr(results[0], "names", None)
        if isinstance(names, dict):
            model_labels = [str(names[index]) for index in sorted(names.keys())]
        elif isinstance(names, list):
            model_labels = [str(label) for label in names]
        else:
            model_labels = list(classes)

        probabilities: list[np.ndarray] = []
        for result in results:
            probs = getattr(result, "probs", None)
            if probs is None:
                raise RuntimeError("YOLO classification result does not contain probability scores.")
            raw = _tensor_to_numpy(getattr(probs, "data", probs)).astype(float).reshape(-1)
            probability_by_label = {
                model_labels[index]: float(raw[index])
                for index in range(min(len(model_labels), len(raw)))
            }
            aligned = np.asarray([probability_by_label.get(label, 0.0) for label in classes], dtype=float)
            total = aligned.sum()
            if total > 0:
                aligned = aligned / total
            probabilities.append(aligned)
        return np.vstack(probabilities)


def _evaluate_probabilities(y_true: Iterable[str], probabilities: np.ndarray, classes: list[str]) -> dict[str, Any]:
    y_true_list = [str(label) for label in y_true]
    if len(y_true_list) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "predictions": []}
    predicted_indices = probabilities.argmax(axis=1) if len(probabilities) else np.zeros(len(y_true_list), dtype=int)
    predicted_labels = [classes[int(index)] for index in predicted_indices]
    return {
        "accuracy": float(accuracy_score(y_true_list, predicted_labels)),
        "f1_macro": float(f1_score(y_true_list, predicted_labels, average="macro", zero_division=0)),
        "predictions": predicted_labels,
    }


def _confidence_scores(probabilities: np.ndarray) -> np.ndarray:
    if len(probabilities) == 0:
        return np.zeros(0, dtype=float)
    return 1.0 - probabilities.max(axis=1)


def _entropy_scores(probabilities: np.ndarray) -> np.ndarray:
    if len(probabilities) == 0:
        return np.zeros(0, dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def _select_query_order(probabilities: np.ndarray, strategy: str, random_state: int) -> np.ndarray:
    normalized = strategy.strip().lower()
    if normalized in {"confidence", "least-confidence", "least_confidence"}:
        scores = _confidence_scores(probabilities)
        return np.argsort(-scores)
    if normalized == "entropy":
        scores = _entropy_scores(probabilities)
        return np.argsort(-scores)
    if normalized == "random":
        rng = np.random.default_rng(random_state)
        order = np.arange(len(probabilities))
        rng.shuffle(order)
        return order
    raise ValueError(f"Unsupported query strategy: {strategy}")


def query_pool_indices(
    probabilities: np.ndarray,
    *,
    strategy: str,
    batch_size: int,
    random_state: int,
) -> list[int]:
    if batch_size <= 0 or len(probabilities) == 0:
        return []
    order = _select_query_order(probabilities, strategy=strategy, random_state=random_state)
    return [int(index) for index in order[: min(batch_size, len(order))]]


def _labelstudio_classification_record(
    row: pd.Series,
    *,
    image_reference: str,
    model_version: str,
) -> dict[str, Any]:
    suggested_label = str(row.get(SUGGESTED_LABEL_COLUMN) or "").strip()
    predictions: list[dict[str, Any]] = []
    if suggested_label:
        predictions = [
            {
                "model_version": model_version,
                "result": [
                    {
                        "id": f"pred-{row[IMAGE_ID_COLUMN]}",
                        "type": "choices",
                        "from_name": "label",
                        "to_name": "image",
                        "value": {"choices": [suggested_label]},
                        "score": float(row.get(CONFIDENCE_COLUMN, 0.0) or 0.0),
                    }
                ],
            }
        ]
    return {
        "data": {
            "image": image_reference,
            "image_id": str(row[IMAGE_ID_COLUMN]),
            "file_path": str(row[FILE_PATH_COLUMN]),
            "filename": str(row[FILENAME_COLUMN]),
        },
        "predictions": predictions,
    }


def export_labelstudio_classification_batch(
    selected_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    iteration: int,
    strategy: str,
    model_version: str,
    base_image_url: str = DEFAULT_BASE_IMAGE_URL,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "uncertain_manifest.csv"
    labelstudio_path = output_root / "labelstudio_import.json"
    export_df = selected_df.copy()
    export_df.to_csv(manifest_path, index=False)

    tasks = []
    for _, row in export_df.iterrows():
        image_reference = f"{base_image_url}{row[FILE_PATH_COLUMN]}" if base_image_url else str(row[FILE_PATH_COLUMN])
        tasks.append(
            _labelstudio_classification_record(
                row,
                image_reference=image_reference,
                model_version=model_version,
            )
        )
    labelstudio_path.write_text(json.dumps(_safe_json(tasks), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "iteration": iteration,
        "strategy": strategy,
        "manifest_path": str(manifest_path),
        "labelstudio_import_path": str(labelstudio_path),
        "selected_count": int(len(export_df)),
    }


def _extract_choice_label(result_item: dict[str, Any]) -> str:
    value = result_item.get("value", {})
    if not isinstance(value, dict):
        return ""
    choices = value.get("choices")
    if isinstance(choices, list) and choices:
        return str(choices[0])
    return ""


def import_labelstudio_labels(export_path: str) -> pd.DataFrame:
    input_path = Path(export_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Label Studio export does not exist: {input_path}")
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        normalized = pd.DataFrame()
        if "image_id" in df.columns:
            normalized[IMAGE_ID_COLUMN] = df["image_id"].astype(str)
        if FILE_PATH_COLUMN in df.columns:
            normalized[FILE_PATH_COLUMN] = df[FILE_PATH_COLUMN].astype(str)
        label_column = next((column for column in ("label", HUMAN_LABEL_COLUMN, "choice", "choices") if column in df.columns), "")
        if not label_column:
            raise ValueError(f"Could not infer human label column from {input_path}")
        normalized[HUMAN_LABEL_COLUMN] = df[label_column].astype(str)
        return normalized

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Label Studio JSON export must be a list of tasks.")
    rows: list[dict[str, Any]] = []
    for task in payload:
        if not isinstance(task, dict):
            continue
        data = task.get("data", {}) if isinstance(task.get("data"), dict) else {}
        image_id = str(data.get("image_id") or "")
        file_path = str(data.get(FILE_PATH_COLUMN) or "")
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            results = annotation.get("result", []) if isinstance(annotation, dict) else []
            if not isinstance(results, list):
                continue
            for result_item in results:
                if not isinstance(result_item, dict):
                    continue
                label = _extract_choice_label(result_item)
                if label:
                    rows.append(
                        {
                            IMAGE_ID_COLUMN: image_id,
                            FILE_PATH_COLUMN: file_path,
                            HUMAN_LABEL_COLUMN: label,
                        }
                    )
                    break
            else:
                continue
            break
    return pd.DataFrame(rows)


def apply_label_feedback(
    selected_df: pd.DataFrame,
    *,
    labelstudio_export_path: str = "",
) -> pd.DataFrame:
    updated = selected_df.copy()
    if not labelstudio_export_path:
        return updated
    imported_df = import_labelstudio_labels(labelstudio_export_path)
    if imported_df.empty:
        raise ValueError(f"No human labels found in Label Studio export: {labelstudio_export_path}")
    if IMAGE_ID_COLUMN in imported_df.columns and imported_df[IMAGE_ID_COLUMN].astype(str).str.len().gt(0).any():
        updated = updated.merge(
            imported_df[[IMAGE_ID_COLUMN, HUMAN_LABEL_COLUMN]].drop_duplicates(subset=[IMAGE_ID_COLUMN]),
            on=IMAGE_ID_COLUMN,
            how="left",
        )
    else:
        updated = updated.merge(
            imported_df[[FILE_PATH_COLUMN, HUMAN_LABEL_COLUMN]].drop_duplicates(subset=[FILE_PATH_COLUMN]),
            on=FILE_PATH_COLUMN,
            how="left",
        )
    if HUMAN_LABEL_COLUMN not in updated.columns:
        raise ValueError(f"No matched human labels found for selected batch from {labelstudio_export_path}")
    matched_mask = updated[HUMAN_LABEL_COLUMN].notna()
    if not matched_mask.any():
        raise ValueError(f"Label Studio export did not match any selected samples: {labelstudio_export_path}")
    updated.loc[matched_mask, LABEL_COLUMN] = updated.loc[matched_mask, HUMAN_LABEL_COLUMN].astype(str)
    return updated.drop(columns=[HUMAN_LABEL_COLUMN], errors="ignore")


def _lookup_manual_labels_path(manual_labels_by_iteration: dict[str, Any], strategy: str, iteration: int) -> str:
    if not manual_labels_by_iteration:
        return ""
    strategy_specific = manual_labels_by_iteration.get(strategy, manual_labels_by_iteration)
    if not isinstance(strategy_specific, dict):
        return ""
    return str(strategy_specific.get(str(iteration)) or strategy_specific.get(iteration) or "")


def _summarize_selected_batch(
    selected_df: pd.DataFrame,
    probabilities: np.ndarray,
    classes: list[str],
    strategy: str,
    iteration: int,
) -> pd.DataFrame:
    if selected_df.empty:
        return selected_df.copy()
    result = selected_df.copy().reset_index(drop=True)
    max_indices = probabilities.argmax(axis=1)
    result[SUGGESTED_LABEL_COLUMN] = [classes[int(index)] for index in max_indices]
    result[CONFIDENCE_COLUMN] = probabilities.max(axis=1)
    result["iteration"] = int(iteration)
    result["strategy"] = strategy
    result["uncertainty_score"] = (
        _confidence_scores(probabilities)
        if strategy != "entropy"
        else _entropy_scores(probabilities)
    )
    return result


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
        plt.plot(history_df["n_labeled"], history_df["f1_macro"], marker="o", label=strategy)
    plt.title("Active Learning Learning Curve")
    plt.xlabel("Number of labeled samples")
    plt.ylabel("F1 macro")
    plt.grid(True, alpha=0.3)
    if histories_by_strategy:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()
    return str(output_file)


def compute_sample_savings(
    strategy_history: list[dict[str, Any]],
    baseline_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not strategy_history or not baseline_history:
        return {"baseline_target_f1": None, "matched_n_labeled": None, "baseline_n_labeled": None, "saved_examples": None}
    baseline_target = max(float(item["f1_macro"]) for item in baseline_history)
    baseline_n_labeled = min(
        int(item["n_labeled"])
        for item in baseline_history
        if float(item["f1_macro"]) >= baseline_target
    )
    matched_n_labeled = next(
        (
            int(item["n_labeled"])
            for item in strategy_history
            if float(item["f1_macro"]) >= baseline_target
        ),
        None,
    )
    if matched_n_labeled is None:
        return {
            "baseline_target_f1": baseline_target,
            "baseline_n_labeled": baseline_n_labeled,
            "matched_n_labeled": None,
            "saved_examples": None,
            "relative_savings": None,
        }
    saved_examples = baseline_n_labeled - matched_n_labeled
    relative = (saved_examples / baseline_n_labeled) if baseline_n_labeled > 0 else None
    return {
        "baseline_target_f1": baseline_target,
        "baseline_n_labeled": baseline_n_labeled,
        "matched_n_labeled": matched_n_labeled,
        "saved_examples": saved_examples,
        "relative_savings": relative,
    }


def _run_image_strategy_cycle(
    *,
    strategy: str,
    task_description: str,
    labeled_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    test_df: pd.DataFrame,
    classes: list[str],
    backend: Any,
    run_dir: str | Path,
    n_iterations: int,
    batch_size: int,
    random_state: int,
    base_image_url: str,
    manual_labels_by_iteration: dict[str, Any],
) -> dict[str, Any]:
    strategy_dir = Path(run_dir) / "strategies" / strategy
    strategy_dir.mkdir(parents=True, exist_ok=True)
    current_labeled = labeled_df.copy().reset_index(drop=True)
    current_pool = pool_df.copy().reset_index(drop=True)
    history: list[dict[str, Any]] = []
    iteration_artifacts: list[dict[str, Any]] = []
    last_model_dir = ""
    model_version = getattr(backend, "model_path", DEFAULT_IMAGE_MODEL_PATH)

    for iteration in range(n_iterations + 1):
        n_labeled_before_query = int(len(current_labeled))
        pool_before_query = int(len(current_pool))
        train_result = backend.train(
            train_df=current_labeled,
            val_df=test_df,
            classes=classes,
            work_dir=strategy_dir,
            iteration_name=f"iteration_{iteration}",
        )
        model = train_result["model"]
        last_model_dir = str(train_result["model_dir"])
        model_version = str(train_result.get("weights_path") or model_version)
        test_image_paths = test_df[FILE_PATH_COLUMN].tolist() if not test_df.empty else []
        test_probabilities = backend.predict_proba(model, test_image_paths, classes)
        metrics = _evaluate_probabilities(test_df[LABEL_COLUMN].tolist(), test_probabilities, classes)
        selected_count = 0
        if iteration < n_iterations and not current_pool.empty:
            pool_probabilities = backend.predict_proba(model, current_pool[FILE_PATH_COLUMN].tolist(), classes)
            selected_indices = query_pool_indices(
                pool_probabilities,
                strategy=strategy,
                batch_size=min(batch_size, len(current_pool)),
                random_state=random_state + iteration,
            )
            selected_count = len(selected_indices)
            if selected_indices:
                selected_subset = current_pool.iloc[selected_indices].reset_index(drop=True)
                selected_probabilities = pool_probabilities[selected_indices]
                selected_manifest = _summarize_selected_batch(
                    selected_subset,
                    selected_probabilities,
                    classes=classes,
                    strategy=strategy,
                    iteration=iteration + 1,
                )
                iteration_dir = strategy_dir / f"iteration_{iteration + 1}"
                export_payload = export_labelstudio_classification_batch(
                    selected_manifest,
                    output_dir=iteration_dir,
                    iteration=iteration + 1,
                    strategy=strategy,
                    model_version=Path(model_version).name,
                    base_image_url=base_image_url,
                )
                manual_labels_path = _lookup_manual_labels_path(manual_labels_by_iteration, strategy, iteration + 1)
                updated_selected = apply_label_feedback(
                    selected_subset,
                    labelstudio_export_path=manual_labels_path,
                )
                current_labeled = pd.concat([current_labeled, updated_selected], ignore_index=True)
                current_pool = current_pool.drop(index=current_pool.index[selected_indices]).reset_index(drop=True)
                iteration_artifacts.append(
                    {
                        **export_payload,
                        "manual_labels_path": manual_labels_path,
                    }
                )
        history.append(
            {
                "iteration": int(iteration),
                "strategy": strategy,
                "task_description": task_description,
                "n_labeled": n_labeled_before_query,
                "pool_remaining": pool_before_query,
                "selected_count": int(selected_count),
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
            }
        )
        if current_pool.empty:
            break

    history_json, history_csv = _save_history(history, Path(run_dir) / "reports", strategy)
    final_labeled_path = strategy_dir / "final_labeled.csv"
    final_pool_path = strategy_dir / "remaining_pool.csv"
    current_labeled.to_csv(final_labeled_path, index=False)
    current_pool.to_csv(final_pool_path, index=False)
    return {
        "strategy": strategy,
        "history": history,
        "history_json": history_json,
        "history_csv": history_csv,
        "final_model_dir": last_model_dir,
        "final_labeled_csv": str(final_labeled_path),
        "remaining_pool_csv": str(final_pool_path),
        "iteration_artifacts": iteration_artifacts,
        "labelstudio_import_path": iteration_artifacts[-1]["labelstudio_import_path"] if iteration_artifacts else "",
        "uncertain_manifest_path": iteration_artifacts[-1]["manifest_path"] if iteration_artifacts else "",
    }


def image_classification_active_learning_impl(
    *,
    task_description: str,
    labeled_data_path: str,
    config: dict[str, Any] | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    resolved_config = {
        "artifacts_dir": DEFAULT_AL_ARTIFACTS_DIR,
        "initial_size": DEFAULT_INITIAL_SIZE,
        "n_iterations": DEFAULT_N_ITERATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "test_size": DEFAULT_TEST_SIZE,
        "random_state": DEFAULT_RANDOM_STATE,
        "model_path": DEFAULT_IMAGE_MODEL_PATH,
        "base_image_url": DEFAULT_BASE_IMAGE_URL,
        "strategies": list(DEFAULT_IMAGE_STRATEGIES),
        "epochs": DEFAULT_EPOCHS,
        "imgsz": DEFAULT_IMGSZ,
        "train_batch": DEFAULT_TRAIN_BATCH,
        "predict_batch": DEFAULT_TRAIN_BATCH,
        "link_mode": DEFAULT_LINK_MODE,
        "manual_labels_by_iteration": {},
    }
    if config:
        resolved_config.update(config)

    dataset_df = load_image_classification_dataset(labeled_data_path)
    labeled_csv = _resolve_labeled_csv_path(labeled_data_path)
    classes = sorted(dataset_df[LABEL_COLUMN].astype(str).unique().tolist())
    run_id = resolved_config.get("run_id") or make_run_id(_slugify(Path(labeled_csv).stem))
    run_dir = ensure_al_run_layout(Path(str(resolved_config["artifacts_dir"])) / run_id)
    input_manifest_path = Path(run_dir) / "reports" / "input_manifest.csv"
    dataset_df.to_csv(input_manifest_path, index=False)

    initial_labeled_df, pool_df, test_df = prepare_image_splits(
        dataset_df,
        initial_size=int(resolved_config["initial_size"]),
        test_size=resolved_config["test_size"],
        random_state=int(resolved_config["random_state"]),
    )
    initial_labeled_df.to_csv(Path(run_dir) / "splits" / "initial_labeled.csv", index=False)
    pool_df.to_csv(Path(run_dir) / "splits" / "pool.csv", index=False)
    test_df.to_csv(Path(run_dir) / "splits" / "test.csv", index=False)

    effective_backend = backend or YOLOClassificationBackend(
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
        strategy_result = _run_image_strategy_cycle(
            strategy=strategy,
            task_description=task_description,
            labeled_df=initial_labeled_df.copy(),
            pool_df=pool_df.copy(),
            test_df=test_df.copy(),
            classes=classes,
            backend=effective_backend,
            run_dir=run_dir,
            n_iterations=int(resolved_config["n_iterations"]),
            batch_size=int(resolved_config["batch_size"]),
            random_state=int(resolved_config["random_state"]),
            base_image_url=str(resolved_config["base_image_url"]),
            manual_labels_by_iteration=resolved_config.get("manual_labels_by_iteration") or {},
        )
        strategy_results[strategy] = strategy_result
        history_paths[strategy] = strategy_result["history_json"]

    learning_curve_path = plot_learning_curves(
        {strategy: result["history"] for strategy, result in strategy_results.items()},
        Path(run_dir) / "curves" / "learning_curve.png",
    )
    summary_payload = {
        "task_description": task_description,
        "labeled_data_path": str(labeled_csv),
        "run_dir": str(run_dir),
        "classes": classes,
        "input_rows": int(len(dataset_df)),
        "initial_rows": int(len(initial_labeled_df)),
        "pool_rows": int(len(pool_df)),
        "test_rows": int(len(test_df)),
        "strategies": strategies,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
    }
    if "confidence" in strategy_results and "random" in strategy_results:
        summary_payload["sample_savings_vs_random"] = compute_sample_savings(
            strategy_results["confidence"]["history"],
            strategy_results["random"]["history"],
        )
    summary_path = _write_json(Path(run_dir) / "reports" / "summary.json", summary_payload)
    primary_strategy = "confidence" if "confidence" in strategy_results else strategies[0]
    primary_result = strategy_results[primary_strategy]
    return {
        "modality": "image",
        "implemented": True,
        "task_description": task_description,
        "run_dir": str(run_dir),
        "labeled_csv": str(labeled_csv),
        "classes": classes,
        "strategy_results": strategy_results,
        "history_paths": history_paths,
        "learning_curve_path": learning_curve_path,
        "labelstudio_import_path": primary_result["labelstudio_import_path"],
        "uncertain_manifest_path": primary_result["uncertain_manifest_path"],
        "final_model_dir": primary_result["final_model_dir"],
        "summary_path": summary_path,
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
        "implemented": False,
        "task_description": task_description,
        "labeled_data_path": labeled_data_path,
        "config": config or {},
        "message": "Table active learning backend is planned but not implemented in v1.",
    }


@tool
def image_classification_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Run an image classification active-learning loop on a labeled artifact CSV or run directory.

    Args:
        task_description: Natural-language description of the classification task.
        labeled_data_path: Path to annotation_artifacts/<run_id>/ or directly to labeled.csv.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON with AL artifacts, histories, learning-curve plot path, and Label Studio export paths.
    """
    logger.info("image_classification_active_learning called: labeled_data_path=%s", labeled_data_path)
    try:
        return _json_success(
            image_classification_active_learning_impl(
                task_description=task_description,
                labeled_data_path=labeled_data_path,
                config=_read_jsonish(config_json),
            )
        )
    except Exception as exc:
        logger.exception("image_classification_active_learning failed")
        return _json_error(str(exc))


@tool
def table_classification_active_learning(
    task_description: str,
    labeled_data_path: str,
    config_json: str = "",
) -> str:
    """
    Placeholder tool for future tabular active-learning support.

    Args:
        task_description: Natural-language description of the classification task.
        labeled_data_path: Path to labeled tabular data.
        config_json: Optional JSON string or path to a JSON config file.

    Returns:
        JSON payload with implemented=false for the current v1 stub.
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
