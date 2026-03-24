from __future__ import annotations

import json
import logging
import math
import os
import shutil
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pandas as pd
from smolagents import tool

from data_quality_tools_agent.reporting import ensure_run_layout, make_run_id, render_notebook_from_artifacts, write_placeholder_png


logger = logging.getLogger(__name__)

os.environ.setdefault("MPLBACKEND", "Agg")

if find_spec("matplotlib") is not None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        logger.warning("Failed to force matplotlib backend to Agg.", exc_info=True)

SUPPORTED_FORMATS = {"csv", "parquet"}
OUTLIER_METHODS = {"iqr", "zscore"}
PROFILE_ASPECTS = {"schema", "numeric", "categorical", "summary", "all"}
CORRELATION_METHODS = {"pearson", "spearman", "kendall"}
MISSING_STRATEGIES = {"median", "mode", "drop_rows"}
DUPLICATE_STRATEGIES = {"drop", "keep"}
OUTLIER_STRATEGIES = {"clip_iqr", "remove_iqr", "none"}
MAX_COLUMNS_IN_RESPONSE = 50
MAX_TOP_VALUES = 10
MAX_RARE_LABELS = 20
MAX_SAMPLE_ROWS = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
IMAGE_HASH_FUNCTIONS = {"phash", "dhash", "ahash", "whash"}


def _safe_json(value: Any) -> Any:
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
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_json(item) for item in value]
    if pd.isna(value):
        return None
    return value


def _json_success(payload: dict[str, Any]) -> str:
    return json.dumps({"success": True, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _json_error(message: str, **payload: Any) -> str:
    return json.dumps({"success": False, "error": message, **_safe_json(payload)}, ensure_ascii=False, separators=(",", ":"))


def _normalize_format(input_path: str, input_format: str = "") -> str:
    fmt = input_format.strip().lower().lstrip(".") if input_format else Path(input_path).suffix.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported table format '{fmt}'. Supported formats: {sorted(SUPPORTED_FORMATS)}")
    return fmt


@lru_cache(maxsize=3)
def _read_table_cached(input_path: str, input_format: str, mtime_ns: int, size: int) -> tuple[pd.DataFrame, str]:
    del mtime_ns, size
    path = Path(input_path)
    fmt = _normalize_format(str(path), input_format)
    if fmt == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return df, fmt


def _read_table(input_path: str, input_format: str = "") -> tuple[pd.DataFrame, str]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    stat = path.stat()
    df, fmt = _read_table_cached(str(path.resolve()), input_format, stat.st_mtime_ns, stat.st_size)
    if df.empty:
        logger.warning("Table %s is empty (shape=%s)", input_path, df.shape)
    return df.copy(deep=True), fmt


def _clear_table_cache() -> None:
    _read_table_cached.cache_clear()


def _write_table(df: pd.DataFrame, output_path: str, output_format: str = "") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_format(str(path), output_format or path.suffix)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return str(path)


def _write_json_artifact(payload: dict[str, Any], output_path: str) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(_safe_json(payload), handle, ensure_ascii=False, indent=2)
    return str(output)


def _read_json_artifact(path: str) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {artifact_path}")
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {artifact_path}")
    return payload


def _parse_json_arg(payload: str, argument_name: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Argument '{argument_name}' must be valid JSON.") from exc


def _rows_with_missing(df: pd.DataFrame) -> int:
    return int(df.isna().any(axis=1).sum())


def _duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["number"]).columns)


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object", "string", "category", "bool"]).columns)


def _datetime_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)


def _limited_columns(columns: list[str], max_columns: int) -> tuple[list[str], bool]:
    return columns[:max_columns], len(columns) > max_columns


def _table_summary_dict(df: pd.DataFrame) -> dict[str, Any]:
    sample = df.head(MAX_SAMPLE_ROWS).where(pd.notna(df.head(MAX_SAMPLE_ROWS)), None).to_dict(orient="records")
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(map(str, df.columns)),
        "memory_estimate_bytes": int(df.memory_usage(deep=True).sum()),
        "sample_rows": sample,
    }


def _profile_schema_dict(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(map(str, df.columns)),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "non_null_counts": {str(col): int(df[col].notna().sum()) for col in df.columns},
        "nullable_columns": [str(col) for col in df.columns if df[col].isna().any()],
        "numeric_columns": _numeric_columns(df),
        "categorical_columns": _categorical_columns(df),
        "datetime_columns": _datetime_columns(df),
    }


def _profile_numeric_dict(df: pd.DataFrame, max_columns: int = MAX_COLUMNS_IN_RESPONSE) -> dict[str, Any]:
    if df.empty:
        return {"profiles": {}, "total_numeric_columns": 0, "shown": 0, "truncated": False}
    columns, truncated = _limited_columns(_numeric_columns(df), max_columns)
    profiles: dict[str, Any] = {}
    for column in columns:
        series = df[column].dropna()
        if series.empty:
            profiles[column] = {"count": 0}
            continue
        profiles[column] = {
            "count": int(series.count()),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
            "zeros_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
            "quantiles": {
                "q05": float(series.quantile(0.05)),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
            },
        }
    return {
        "profiles": profiles,
        "total_numeric_columns": len(_numeric_columns(df)),
        "shown": len(columns),
        "truncated": truncated,
    }


def _profile_categorical_dict(df: pd.DataFrame, max_columns: int = MAX_COLUMNS_IN_RESPONSE) -> dict[str, Any]:
    if df.empty:
        return {"profiles": {}, "total_categorical_columns": 0, "shown": 0, "truncated": False}
    columns, truncated = _limited_columns(_categorical_columns(df), max_columns)
    profiles: dict[str, Any] = {}
    for column in columns:
        series = df[column].fillna("__MISSING__")
        counts = series.value_counts(dropna=False)
        profiles[column] = {
            "count": int(series.count()),
            "cardinality": int(series.nunique(dropna=False)),
            "top_values": {str(key): int(val) for key, val in counts.head(MAX_TOP_VALUES).items()},
            "rare_labels": [str(key) for key, val in counts.items() if int(val) == 1][:MAX_RARE_LABELS],
        }
    return {
        "profiles": profiles,
        "total_categorical_columns": len(_categorical_columns(df)),
        "shown": len(columns),
        "truncated": truncated,
    }


def _outlier_details(df: pd.DataFrame, method: str = "iqr", max_columns: int = MAX_COLUMNS_IN_RESPONSE) -> dict[str, Any]:
    method = (method or "iqr").lower()
    if method not in OUTLIER_METHODS:
        raise ValueError(f"Unsupported outlier method '{method}'. Allowed values: {sorted(OUTLIER_METHODS)}")

    if df.empty:
        return {
            "method": method,
            "numeric_columns": [],
            "total_numeric_columns": 0,
            "truncated": False,
            "by_column": {},
            "total_outlier_cells": 0,
        }

    all_numeric = _numeric_columns(df)
    columns, truncated = _limited_columns(all_numeric, max_columns)
    by_column: dict[str, Any] = {}
    total = 0

    for column in columns:
        series = df[column].dropna()
        if series.empty:
            by_column[column] = {"count": 0, "ratio": 0.0, "method": method}
            continue
        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[column] < lower) | (df[column] > upper)
            bounds = {"lower": lower, "upper": upper}
        else:
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if math.isclose(std, 0.0):
                mask = pd.Series(False, index=df.index)
            else:
                mask = ((df[column] - mean).abs() / std) > 3.0
            bounds = {"mean": mean, "std": std, "z_threshold": 3.0}
        count = int(mask.fillna(False).sum())
        total += count
        by_column[column] = {
            "count": count,
            "ratio": round(count / max(len(df), 1), 6),
            "method": method,
            "bounds": bounds,
        }

    return {
        "method": method,
        "numeric_columns": columns,
        "total_numeric_columns": len(all_numeric),
        "truncated": truncated,
        "by_column": by_column,
        "total_outlier_cells": total,
    }


def _imbalance_details(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the table.")
    counts = df[target_column].fillna("__MISSING__").value_counts(dropna=False)
    ratios = (counts / max(len(df), 1)).round(6)
    dominance_ratio = float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() else float("inf")
    return {
        "target_column": target_column,
        "class_counts": {str(key): int(val) for key, val in counts.items()},
        "class_ratios": {str(key): float(val) for key, val in ratios.items()},
        "dominance_ratio": dominance_ratio,
        "is_imbalanced": dominance_ratio > 1.5 if len(counts) > 1 else False,
    }


def _quality_report(
    df: pd.DataFrame,
    target_column: str = "",
    outlier_method: str = "iqr",
    max_outlier_columns: int | None = MAX_COLUMNS_IN_RESPONSE,
) -> dict[str, Any]:
    missing_by_column = df.isna().sum()
    missing_ratio_by_column = df.isna().mean().round(6)
    duplicate_rows = _duplicate_rows(df)
    actual_max_outlier_columns = (
        max(len(_numeric_columns(df)), 1) if max_outlier_columns is None else max_outlier_columns
    )
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing": {
            "total_missing_cells": int(missing_by_column.sum()),
            "rows_with_missing": _rows_with_missing(df),
            "by_column": {str(key): int(val) for key, val in missing_by_column.items()},
            "missing_ratio_by_column": {str(key): float(val) for key, val in missing_ratio_by_column.items()},
        },
        "duplicates": {
            "duplicate_rows": duplicate_rows,
            "duplicate_ratio": round(duplicate_rows / max(len(df), 1), 6),
        },
        "outliers": _outlier_details(df, outlier_method, max_columns=actual_max_outlier_columns),
        "imbalance": _imbalance_details(df, target_column) if target_column else None,
    }


def _correlations_dict(
    df: pd.DataFrame,
    method: str = "pearson",
    top_n: int = 20,
    max_columns: int = min(MAX_COLUMNS_IN_RESPONSE, 30),
) -> dict[str, Any]:
    method = (method or "pearson").lower()
    if method not in CORRELATION_METHODS:
        raise ValueError(f"Unsupported correlation method '{method}'. Allowed values: {sorted(CORRELATION_METHODS)}")
    numeric_columns = _numeric_columns(df)
    columns, truncated = _limited_columns(numeric_columns, max_columns)
    if len(columns) < 2:
        return {
            "method": method,
            "matrix": {},
            "top_pairs": [],
            "total_numeric_columns": len(numeric_columns),
            "shown_numeric_columns": len(columns),
            "truncated": truncated,
        }

    corr = df[columns].corr(method=method).fillna(0.0)
    pairs: list[dict[str, Any]] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pairs.append(
                {
                    "col_a": columns[i],
                    "col_b": columns[j],
                    "correlation": round(float(corr.iloc[i, j]), 4),
                }
            )
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {
        "method": method,
        "matrix": {
            str(row_key): {str(col_key): round(float(value), 4) for col_key, value in row.items()}
            for row_key, row in corr.to_dict().items()
        },
        "top_pairs": pairs[:top_n],
        "total_numeric_columns": len(numeric_columns),
        "shown_numeric_columns": len(columns),
        "truncated": truncated,
    }


def _suggest_dtypes_dict(df: pd.DataFrame, sample_size: int = 200) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)
        recommendation = dtype
        reason = "Current dtype looks appropriate."

        if dtype in {"object", "string"}:
            non_null = series.dropna().astype(str).head(sample_size)
            lowered = non_null.str.lower()
            if not non_null.empty:
                numeric_candidate = pd.to_numeric(non_null, errors="coerce")
                datetime_candidate = pd.to_datetime(non_null, errors="coerce")
                datetime_ratio = datetime_candidate.notna().sum() / len(non_null)
                bool_values = set(lowered.unique())

                if numeric_candidate.notna().all():
                    recommendation = "float64" if (numeric_candidate % 1 != 0).any() else "int64"
                    reason = "All sampled non-null values can be parsed as numbers."
                elif datetime_ratio >= 0.9 and len(non_null) >= 5:
                    recommendation = "datetime64[ns]"
                    reason = "Most sampled non-null values can be parsed as datetimes."
                elif bool_values and bool_values <= {"true", "false", "0", "1", "yes", "no"}:
                    recommendation = "bool"
                    reason = "Sampled values match a boolean-like vocabulary."
                elif series.nunique(dropna=False) <= min(50, max(len(series) // 10, 2)):
                    recommendation = "category"
                    reason = "Low-cardinality text column is a good candidate for category dtype."
        elif dtype == "float64":
            non_null = series.dropna()
            if not non_null.empty and (non_null % 1 == 0).all():
                recommendation = "Int64"
                reason = "All non-null values are whole numbers; consider nullable Int64."

        suggestions[str(column)] = {
            "current_dtype": dtype,
            "suggested_dtype": recommendation,
            "reason": reason,
        }
    return {"suggestions": suggestions}


def _apply_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy = strategy.lower()
    if strategy == "drop_rows":
        return df.dropna().reset_index(drop=True)
    result = df.copy()
    for column in result.columns:
        if result[column].isna().sum() == 0:
            continue
        if strategy == "median" and pd.api.types.is_numeric_dtype(result[column]):
            fill_value = result[column].median()
        else:
            mode = result[column].mode(dropna=True)
            fill_value = None if mode.empty else mode.iloc[0]
        if fill_value is None:
            logger.warning("Column '%s': no mode available, NaN values remain.", column)
        else:
            result[column] = result[column].fillna(fill_value)
    return result


def _apply_duplicates(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True) if strategy.lower() == "drop" else df.copy()


def _apply_outliers(df: pd.DataFrame, strategy: str, method: str) -> pd.DataFrame:
    strategy = strategy.lower()
    if strategy == "none":
        return df.copy()

    result = df.copy()
    details = _outlier_details(result, method, max_columns=max(len(_numeric_columns(result)), 1))
    if strategy == "clip_iqr":
        for column, info in details["by_column"].items():
            bounds = info.get("bounds", {})
            if "lower" in bounds and "upper" in bounds:
                result[column] = result[column].clip(lower=bounds["lower"], upper=bounds["upper"])
        return result

    if strategy == "remove_iqr":
        mask = pd.Series(False, index=result.index)
        for column, info in details["by_column"].items():
            bounds = info.get("bounds", {})
            if "lower" in bounds and "upper" in bounds:
                mask = mask | (result[column] < bounds["lower"]) | (result[column] > bounds["upper"])
            elif "mean" in bounds and "std" in bounds and bounds["std"]:
                zscores = (result[column] - bounds["mean"]).abs() / bounds["std"]
                mask = mask | (zscores > bounds.get("z_threshold", 3.0))
        return result.loc[~mask.fillna(False)].reset_index(drop=True)

    raise ValueError(f"Unsupported outlier strategy '{strategy}'. Allowed values: {sorted(OUTLIER_STRATEGIES)}")


def _comparison_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = {
        "rows": (before["rows"], after["rows"]),
        "columns": (before["columns"], after["columns"]),
        "total_missing_cells": (before["missing"]["total_missing_cells"], after["missing"]["total_missing_cells"]),
        "rows_with_missing": (before["missing"]["rows_with_missing"], after["missing"]["rows_with_missing"]),
        "duplicate_rows": (before["duplicates"]["duplicate_rows"], after["duplicates"]["duplicate_rows"]),
        "total_outlier_cells": (before["outliers"]["total_outlier_cells"], after["outliers"]["total_outlier_cells"]),
    }
    if before.get("imbalance") and after.get("imbalance"):
        metrics["dominance_ratio"] = (before["imbalance"]["dominance_ratio"], after["imbalance"]["dominance_ratio"])
    rows = []
    for metric, (before_value, after_value) in metrics.items():
        delta = after_value - before_value if isinstance(before_value, (int, float)) else None
        rows.append({"metric": metric, "before": before_value, "after": after_value, "delta": delta})
    return rows


def _reduction_ratio(before: float, after: float) -> float:
    if before <= 0:
        return 1.0 if after <= 0 else 0.0
    ratio = (before - after) / before
    return max(0.0, min(1.0, float(ratio)))


def _strategy_score_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_map = {row["metric"]: row for row in rows}
    before_rows = float(metric_map.get("rows", {}).get("before", 0.0) or 0.0)
    after_rows = float(metric_map.get("rows", {}).get("after", 0.0) or 0.0)
    row_retention = after_rows / before_rows if before_rows else 1.0

    missing_reduction = _reduction_ratio(
        float(metric_map.get("total_missing_cells", {}).get("before", 0.0) or 0.0),
        float(metric_map.get("total_missing_cells", {}).get("after", 0.0) or 0.0),
    )
    duplicate_reduction = _reduction_ratio(
        float(metric_map.get("duplicate_rows", {}).get("before", 0.0) or 0.0),
        float(metric_map.get("duplicate_rows", {}).get("after", 0.0) or 0.0),
    )
    outlier_reduction = _reduction_ratio(
        float(metric_map.get("total_outlier_cells", {}).get("before", 0.0) or 0.0),
        float(metric_map.get("total_outlier_cells", {}).get("after", 0.0) or 0.0),
    )

    if "dominance_ratio" in metric_map:
        before_dom = float(metric_map["dominance_ratio"]["before"] or 0.0)
        after_dom = float(metric_map["dominance_ratio"]["after"] or 0.0)
        if before_dom <= 0:
            imbalance_preservation = 1.0
        else:
            imbalance_penalty = max(after_dom - before_dom, 0.0) / max(before_dom, after_dom, 1.0)
            imbalance_preservation = max(0.0, 1.0 - imbalance_penalty)
        score = (
            0.35 * row_retention
            + 0.25 * missing_reduction
            + 0.15 * duplicate_reduction
            + 0.15 * outlier_reduction
            + 0.10 * imbalance_preservation
        )
    else:
        imbalance_preservation = None
        score = 0.40 * row_retention + 0.30 * missing_reduction + 0.20 * duplicate_reduction + 0.10 * outlier_reduction

    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "row_retention": round(row_retention, 6),
        "components": {
            "missing_reduction_ratio": round(missing_reduction, 6),
            "duplicate_reduction_ratio": round(duplicate_reduction, 6),
            "outlier_reduction_ratio": round(outlier_reduction, 6),
            "imbalance_preservation": round(imbalance_preservation, 6) if imbalance_preservation is not None else None,
        },
    }


def _maybe_plot_available() -> bool:
    return find_spec("matplotlib") is not None


def _save_report_content(content: str, output_path: str, report_format: str = "") -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fmt = report_format.strip().lower().lstrip(".") if report_format else output.suffix.lower().lstrip(".")
    if fmt == "json":
        payload = _parse_json_arg(content, "content")
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        output.write_text(content, encoding="utf-8")
    return str(output)


def _ensure_image_hash_dependencies() -> tuple[Any, Any]:
    if find_spec("PIL") is None:
        raise ImportError("Pillow is required for image deduplication.")
    if find_spec("imagehash") is None:
        raise ImportError("imagehash is required for image deduplication.")
    from PIL import Image
    import imagehash

    return Image, imagehash


def _collect_images_by_class(input_dir: str) -> dict[str, list[Path]]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path must be a directory with one subfolder per class: {root}")

    images_by_class: dict[str, list[Path]] = {}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        files = [path for path in sorted(subdir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if files:
            images_by_class[subdir.name] = files
    return images_by_class


def _compute_image_hashes(
    image_paths: list[Path],
    *,
    hash_func_name: str = "phash",
    hash_size: int = 16,
) -> tuple[list[tuple[Path, Any]], list[dict[str, str]]]:
    Image, imagehash = _ensure_image_hash_dependencies()
    hash_func = getattr(imagehash, {"phash": "phash", "dhash": "dhash", "ahash": "average_hash", "whash": "whash"}[hash_func_name])
    results: list[tuple[Path, Any]] = []
    unreadable: list[dict[str, str]] = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                image_hash = hash_func(img.convert("RGB"), hash_size=hash_size)
            results.append((img_path, image_hash))
        except Exception as exc:
            logger.warning("Failed to read image %s: %s", img_path, exc)
            unreadable.append({"path": str(img_path), "error": str(exc)})

    return results, unreadable


def _find_unique_images(
    hashed_images: list[tuple[Path, Any]],
    *,
    threshold: int = 8,
) -> tuple[list[Path], list[Path]]:
    unique: list[tuple[Path, Any]] = []
    duplicates: list[Path] = []

    for path, current_hash in hashed_images:
        is_duplicate = False
        for _, existing_hash in unique:
            if (current_hash - existing_hash) <= threshold:
                is_duplicate = True
                break
        if is_duplicate:
            duplicates.append(path)
        else:
            unique.append((path, current_hash))

    return [path for path, _ in unique], duplicates


def _copy_unique_images(unique_paths: list[Path], output_dir: Path, class_name: str) -> list[str]:
    target_dir = output_dir / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for source_path in unique_paths:
        destination = target_dir / source_path.name
        counter = 1
        while destination.exists():
            destination = target_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
            counter += 1
        shutil.copy2(source_path, destination)
        copied_paths.append(str(destination))
    return copied_paths


def _deduplicate_image_dataset(
    input_dir: str,
    output_dir: str,
    *,
    hash_func_name: str = "phash",
    hash_size: int = 16,
    threshold: int = 8,
    dry_run: bool = False,
    report_output_path: str = "",
    duplicates_output_path: str = "",
) -> dict[str, Any]:
    hash_func_name = hash_func_name.strip().lower()
    if hash_func_name not in IMAGE_HASH_FUNCTIONS:
        raise ValueError(
            f"Unsupported image hash function '{hash_func_name}'. Allowed values: {sorted(IMAGE_HASH_FUNCTIONS)}"
        )
    if hash_size <= 0:
        raise ValueError("hash_size must be positive.")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    images_by_class = _collect_images_by_class(input_dir)
    if not images_by_class:
        raise ValueError(f"No class subdirectories with supported image files were found in: {input_dir}")

    output_root = Path(output_dir)
    report_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    unreadable_rows: list[dict[str, Any]] = []
    total_before = 0
    total_after = 0
    total_removed = 0

    for class_name, image_paths in images_by_class.items():
        hashed_images, unreadable = _compute_image_hashes(
            image_paths,
            hash_func_name=hash_func_name,
            hash_size=hash_size,
        )
        unique_paths, duplicate_paths = _find_unique_images(hashed_images, threshold=threshold)
        if not dry_run:
            _copy_unique_images(unique_paths, output_root, class_name)

        class_before = len(image_paths)
        class_after = len(unique_paths)
        class_removed = len(duplicate_paths)
        total_before += class_before
        total_after += class_after
        total_removed += class_removed

        report_rows.append(
            {
                "class": class_name,
                "before": class_before,
                "after": class_after,
                "removed": class_removed,
                "unreadable": len(unreadable),
            }
        )
        duplicate_rows.extend(
            {
                "class": class_name,
                "duplicate_path": str(path),
                "filename": path.name,
            }
            for path in duplicate_paths
        )
        unreadable_rows.extend({"class": class_name, **item} for item in unreadable)

    payload = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "hash_function": hash_func_name,
        "hash_size": int(hash_size),
        "threshold": int(threshold),
        "dry_run": bool(dry_run),
        "class_reports": report_rows,
        "duplicates": duplicate_rows,
        "unreadable_files": unreadable_rows,
        "total_before": int(total_before),
        "total_after": int(total_after),
        "total_removed": int(total_removed),
        "duplicate_ratio": round(total_removed / max(total_before, 1), 6),
    }
    if report_output_path:
        payload["report_output_path"] = _write_json_artifact(payload, report_output_path)
    if duplicates_output_path:
        duplicates_df = pd.DataFrame(duplicate_rows or [], columns=["class", "duplicate_path", "filename"])
        payload["duplicates_output_path"] = _write_table(duplicates_df, duplicates_output_path)
    return payload


def _plot_single(output_path: Path, plot_fn, *args) -> str:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_fn(output_path, *args)
        return str(output_path)
    except Exception as exc:
        logger.warning("Plot %s failed: %s", output_path.name, exc)
        write_placeholder_png(output_path)
        return str(output_path)


def _plot_missingness(output_path: Path, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    df.isna().sum().plot(kind="bar", ax=ax, title="Missing values by column")
    ax.set_ylabel("Missing count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_duplicates(output_path: Path, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    duplicates = _duplicate_rows(df)
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series({"unique_rows": max(len(df) - duplicates, 0), "duplicate_rows": duplicates}).plot(
        kind="bar", ax=ax, title="Duplicate row summary"
    )
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_outliers(output_path: Path, df: pd.DataFrame, outlier_method: str) -> None:
    import matplotlib.pyplot as plt

    outlier_info = _outlier_details(df, outlier_method)
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series({col: info["count"] for col, info in outlier_info["by_column"].items()}).plot(
        kind="bar", ax=ax, title=f"Outlier counts ({outlier_method})"
    )
    ax.set_ylabel("Outlier count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_correlation_heatmap(output_path: Path, df: pd.DataFrame, corr_info: dict[str, Any] | None = None) -> None:
    import matplotlib.pyplot as plt

    if corr_info is None:
        corr_info = _correlations_dict(df)
    if not corr_info["matrix"]:
        raise ValueError("No correlation matrix available.")
    corr_df = pd.DataFrame(corr_info["matrix"])
    n = len(corr_df.columns)
    size = max(6, n * 0.4)
    fontsize = max(4, 10 - n // 5)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index, fontsize=fontsize)
    ax.set_title("Correlation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_class_balance(output_path: Path, df: pd.DataFrame, target_column: str) -> None:
    import matplotlib.pyplot as plt

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the table.")
    fig, ax = plt.subplots(figsize=(8, 4))
    df[target_column].fillna("__MISSING__").value_counts(dropna=False).plot(
        kind="bar", ax=ax, title=f"Class balance: {target_column}"
    )
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_distribution(output_path: Path, df: pd.DataFrame, column: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    series = df[column].dropna()
    series.plot(kind="hist", bins=20, ax=ax1, title=f"Distribution: {column}")
    ax1.set_xlabel(column)
    ax2.boxplot(series)
    ax2.set_title(f"Box plot: {column}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_dashboard_outputs(input_path: str, output_dir: str, target_column: str = "", input_format: str = "", outlier_method: str = "iqr") -> dict[str, Any]:
    df, _ = _read_table(input_path, input_format)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []

    if _maybe_plot_available():
        corr_info = _correlations_dict(df)
        figure_paths.append(_plot_single(output_directory / "missingness.png", _plot_missingness, df))
        figure_paths.append(_plot_single(output_directory / "duplicates.png", _plot_duplicates, df))
        figure_paths.append(_plot_single(output_directory / "outliers.png", _plot_outliers, df, outlier_method))
        if corr_info["matrix"]:
            figure_paths.append(
                _plot_single(output_directory / "correlation_heatmap.png", _plot_correlation_heatmap, df, corr_info)
            )
        if target_column:
            figure_paths.append(_plot_single(output_directory / "class_balance.png", _plot_class_balance, df, target_column))
        backend = "matplotlib"
    else:
        for name in ("missingness.png", "duplicates.png", "outliers.png", "correlation_heatmap.png"):
            path = write_placeholder_png(output_directory / name)
            figure_paths.append(str(path))
        if target_column:
            path = write_placeholder_png(output_directory / "class_balance.png")
            figure_paths.append(str(path))
        backend = "placeholder"

    return {"figure_paths": figure_paths, "plotting_backend": backend}


def _distribution_outputs(
    input_path: str,
    output_dir: str,
    columns: list[str] | None = None,
    max_columns: int = 12,
    input_format: str = "",
) -> dict[str, Any]:
    df, _ = _read_table(input_path, input_format)
    numeric_columns = _numeric_columns(df)
    selected = columns or numeric_columns
    valid_columns = [column for column in selected if column in numeric_columns][:max_columns]
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not valid_columns:
        return {"figure_paths": [], "plotting_backend": "none", "columns": []}

    figure_paths: list[str] = []
    if _maybe_plot_available():
        for column in valid_columns:
            figure_paths.append(_plot_single(output_directory / f"{column}_dist.png", _plot_distribution, df, column))
        backend = "matplotlib"
    else:
        for column in valid_columns:
            figure_paths.append(str(write_placeholder_png(output_directory / f"{column}_dist.png")))
        backend = "placeholder"

    return {"figure_paths": figure_paths, "plotting_backend": backend, "columns": valid_columns}


@tool
def suggest_dtypes(input_path: str, input_format: str = "", output_path: str = "") -> str:
    """
    Suggest better dtypes for memory and correctness.

    Args:
        input_path: Path to CSV or Parquet file.
        input_format: Optional format override.
        output_path: Optional path to save the suggestions JSON.

    Returns:
        JSON with per-column current_dtype, suggested_dtype, reason, and optional output_path.
    """
    logger.info("suggest_dtypes called: path=%s", input_path)
    try:
        df, fmt = _read_table(input_path, input_format)
        payload = {"input_path": input_path, "format": fmt, **_suggest_dtypes_dict(df)}
        if output_path:
            payload["output_path"] = _write_json_artifact(payload, output_path)
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def validate_and_load_table(input_path: str, input_format: str = "", output_path: str = "") -> str:
    """
    Check if a table exists and is readable, then return a compact overview.

    Args:
        input_path: Path to CSV or Parquet file.
        input_format: "csv" or "parquet". Empty = infer from extension.
        output_path: Optional path to save the overview JSON.

    Returns:
        JSON with exists, format, readable, shape, columns, dtypes, sample_rows, and optional output_path.
    """
    logger.info("validate_and_load_table called: path=%s format=%s", input_path, input_format)
    path = Path(input_path)
    if not path.exists():
        return _json_error(f"Missing file: {path}", exists=False, readable=False, is_supported=False, format="")
    try:
        df, fmt = _read_table(str(path), input_format)
        payload = {
            "exists": True,
            "readable": True,
            "is_supported": True,
            "input_path": str(path),
            "format": fmt,
            **_table_summary_dict(df),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }
        if output_path:
            payload["output_path"] = _write_json_artifact(payload, output_path)
        return _json_success(payload)
    except Exception as exc:
        fmt = path.suffix.lstrip(".")
        return _json_error(str(exc), exists=True, readable=False, is_supported=fmt in SUPPORTED_FORMATS, format=fmt)


@tool
def profile_table(
    input_path: str,
    aspect: str = "all",
    input_format: str = "",
    max_columns: int = MAX_COLUMNS_IN_RESPONSE,
    output_path: str = "",
) -> str:
    """
    Profile a table. aspect = "schema", "numeric", "categorical", "summary", or "all".

    Args:
        input_path: Path to CSV or Parquet file.
        aspect: Which profile section to compute.
        input_format: Optional format override.
        max_columns: Maximum number of columns to include in wide per-column outputs.
        output_path: Optional path to save the profile JSON.

    Returns:
        JSON with the requested profile sections and optional output_path.
    """
    logger.info("profile_table called: path=%s aspect=%s", input_path, aspect)
    try:
        df, fmt = _read_table(input_path, input_format)
        aspect = (aspect or "all").lower()
        if aspect not in PROFILE_ASPECTS:
            raise ValueError(f"Unsupported aspect '{aspect}'. Allowed values: {sorted(PROFILE_ASPECTS)}")

        payload: dict[str, Any] = {"input_path": input_path, "format": fmt, "aspect": aspect}
        if aspect in {"schema", "all"}:
            payload["schema"] = _profile_schema_dict(df)
        if aspect in {"numeric", "all"}:
            payload["numeric"] = _profile_numeric_dict(df, max_columns=max_columns)
        if aspect in {"categorical", "all"}:
            payload["categorical"] = _profile_categorical_dict(df, max_columns=max_columns)
        if aspect in {"summary", "all"}:
            payload["summary"] = _table_summary_dict(df)

        if output_path:
            payload["output_path"] = _write_json_artifact(payload, output_path)
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def compute_correlations(
    input_path: str,
    method: str = "pearson",
    top_n: int = 20,
    input_format: str = "",
    output_path: str = "",
) -> str:
    """
    Compute pairwise correlations between numeric columns.

    Args:
        input_path: Path to CSV or Parquet file.
        method: "pearson", "spearman", or "kendall".
        top_n: Number of strongest absolute pairs to return.
        input_format: Optional format override.
        output_path: Optional path to save the correlation JSON.

    Returns:
        JSON with top_pairs, matrix, truncation info, and optional output_path.
    """
    logger.info("compute_correlations called: path=%s method=%s", input_path, method)
    try:
        df, fmt = _read_table(input_path, input_format)
        payload = {"input_path": input_path, "format": fmt, **_correlations_dict(df, method=method, top_n=top_n)}
        if output_path:
            payload["output_path"] = _write_json_artifact(payload, output_path)
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def detect_all_issues(
    input_path: str,
    target_column: str = "",
    outlier_method: str = "iqr",
    input_format: str = "",
    output_path: str = "",
) -> str:
    """
    Run all core quality checks in one step.

    Args:
        input_path: Path to CSV or Parquet file.
        target_column: Optional target column for class imbalance.
        outlier_method: "iqr" or "zscore".
        input_format: Optional format override.
        output_path: Optional path to save the quality report JSON.

    Returns:
        JSON with missing, duplicates, outliers, imbalance, rows, columns, and optional output_path.
    """
    logger.info("detect_all_issues called: path=%s target=%s method=%s", input_path, target_column, outlier_method)
    try:
        df, fmt = _read_table(input_path, input_format)
        payload = {
            "input_path": input_path,
            "format": fmt,
            **_quality_report(df, target_column, outlier_method, max_outlier_columns=MAX_COLUMNS_IN_RESPONSE),
        }
        if output_path:
            payload["output_path"] = _write_json_artifact(payload, output_path)
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def apply_cleaning_plan(
    input_path: str,
    strategy_json: str,
    output_path: str,
    input_format: str = "",
    target_column: str = "",
    outlier_method: str = "iqr",
) -> str:
    """
    Apply one full cleaning plan in fixed order: missing, duplicates, outliers.

    Args:
        input_path: Path to CSV or Parquet file.
        strategy_json: JSON with keys missing, duplicates, outliers.
        output_path: Path where the cleaned table must be written.
        input_format: Optional format override.
        target_column: Optional target column name for metadata.
        outlier_method: "iqr" or "zscore".

    Returns:
        JSON with output_path, rows_before, rows_after, strategy, and cleaning_summary.
    """
    logger.info("apply_cleaning_plan called: path=%s output=%s", input_path, output_path)
    try:
        strategy = _parse_json_arg(strategy_json, "strategy_json")
        missing_strategy = str(strategy.get("missing", "mode")).lower()
        duplicate_strategy = str(strategy.get("duplicates", "keep")).lower()
        outlier_strategy = str(strategy.get("outliers", "none")).lower()

        if missing_strategy not in MISSING_STRATEGIES:
            raise ValueError(f"Unsupported missing strategy '{missing_strategy}'. Allowed values: {sorted(MISSING_STRATEGIES)}")
        if duplicate_strategy not in DUPLICATE_STRATEGIES:
            raise ValueError(f"Unsupported duplicate strategy '{duplicate_strategy}'. Allowed values: {sorted(DUPLICATE_STRATEGIES)}")
        if outlier_strategy not in OUTLIER_STRATEGIES:
            raise ValueError(f"Unsupported outlier strategy '{outlier_strategy}'. Allowed values: {sorted(OUTLIER_STRATEGIES)}")

        df, _ = _read_table(input_path, input_format)
        cleaned = _apply_missing(df, missing_strategy)
        cleaned = _apply_duplicates(cleaned, duplicate_strategy)
        cleaned = _apply_outliers(cleaned, outlier_strategy, outlier_method)
        saved_path = _write_table(cleaned, output_path)
        payload = {
            "output_path": saved_path,
            "strategy": {
                "missing": missing_strategy,
                "duplicates": duplicate_strategy,
                "outliers": outlier_strategy,
            },
            "rows_before": int(len(df)),
            "rows_after": int(len(cleaned)),
            "cleaning_summary": {
                "target_column": target_column,
                "outlier_method": outlier_method,
                "row_retention": round(len(cleaned) / max(len(df), 1), 6),
            },
        }
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def compare_before_after(
    before_path: str,
    after_path: str,
    target_column: str = "",
    outlier_method: str = "iqr",
    output_json_path: str = "",
    output_csv_path: str = "",
) -> str:
    """
    Compare quality before and after cleaning and compute a normalized score.

    Args:
        before_path: Path to the original table.
        after_path: Path to the cleaned table.
        target_column: Optional target column for imbalance-aware scoring.
        outlier_method: "iqr" or "zscore".
        output_json_path: Optional path to save the comparison JSON.
        output_csv_path: Optional path to save the comparison CSV.

    Returns:
        JSON with comparison rows, score, row_retention, and optional output paths.
    """
    logger.info("compare_before_after called: before=%s after=%s", before_path, after_path)
    try:
        before_df, _ = _read_table(before_path)
        after_df, _ = _read_table(after_path)
        full_before_max = max(len(_numeric_columns(before_df)), 1)
        full_after_max = max(len(_numeric_columns(after_df)), 1)
        before_report = _quality_report(before_df, target_column, outlier_method, max_outlier_columns=full_before_max)
        after_report = _quality_report(after_df, target_column, outlier_method, max_outlier_columns=full_after_max)
        rows = _comparison_rows(before_report, after_report)
        score_payload = _strategy_score_from_rows(rows)
        payload = {
            "comparison": rows,
            **score_payload,
        }
        if output_csv_path:
            output = Path(output_csv_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output, index=False)
            payload["output_csv_path"] = str(output)
        if output_json_path:
            full_payload = {"before_report": before_report, "after_report": after_report, **payload}
            payload["output_json_path"] = _write_json_artifact(full_payload, output_json_path)
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))


@tool
def select_best_strategy(strategy_summaries_json: str) -> str:
    """
    Pick the best strategy from a list of compact strategy summaries.

    Args:
        strategy_summaries_json: JSON list. Each item needs strategy_name and score.

    Returns:
        JSON with chosen_strategy, chosen_index, reason, and ranked_strategies.
    """
    logger.info("select_best_strategy called")
    try:
        items = _parse_json_arg(strategy_summaries_json, "strategy_summaries_json")
        if not isinstance(items, list) or not items:
            raise ValueError("strategy_summaries_json must be a non-empty JSON list.")
        ranked = sorted(
            items,
            key=lambda item: (
                float(item.get("score", float("-inf"))),
                float(item.get("row_retention", 0.0)),
                str(item.get("strategy_name", "")),
            ),
            reverse=True,
        )
        chosen = ranked[0]
        return _json_success(
            {
                "chosen_strategy": chosen.get("strategy_name"),
                "chosen_index": int(chosen.get("strategy_index", 0)),
                "reason": "Highest normalized quality score with row-retention tie breaking.",
                "ranked_strategies": ranked,
            }
        )
    except Exception as exc:
        return _json_error(str(exc))


@tool
def save_report(content: str, output_path: str, report_format: str = "") -> str:
    """
    Save JSON, CSV, or Markdown content to disk.

    Args:
        content: Report content as text. JSON must be valid JSON text.
        output_path: Destination path.
        report_format: "json", "csv", or "md". Empty = infer from extension.

    Returns:
        JSON with output_path and saved_format.
    """
    logger.info("save_report called: output=%s", output_path)
    try:
        saved_path = _save_report_content(content, output_path, report_format)
        fmt = report_format.strip().lower().lstrip(".") if report_format else Path(output_path).suffix.lower().lstrip(".")
        return _json_success({"output_path": saved_path, "saved_format": fmt})
    except Exception as exc:
        return _json_error(str(exc))


@tool
def save_cleaned_table(source_path: str, output_path: str, input_format: str = "") -> str:
    """
    Copy a cleaned CSV or Parquet table to its final destination.

    Args:
        source_path: Path to a cleaned CSV or Parquet file.
        output_path: Final output path.
        input_format: Optional format override for reading the source file.

    Returns:
        JSON with output_path, format, rows, and columns.
    """
    logger.info("save_cleaned_table called: source=%s output=%s", source_path, output_path)
    try:
        df, _ = _read_table(source_path, input_format)
        saved_path = _write_table(df, output_path)
        return _json_success(
            {
                "output_path": saved_path,
                "format": Path(saved_path).suffix.lstrip("."),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            }
        )
    except Exception as exc:
        return _json_error(str(exc))


@tool
def plot_quality_dashboard(
    input_path: str,
    output_dir: str,
    target_column: str = "",
    input_format: str = "",
    outlier_method: str = "iqr",
) -> str:
    """
    Create the core quality dashboard plots in one directory.

    Args:
        input_path: Path to CSV or Parquet file.
        output_dir: Directory where plots should be written.
        target_column: Optional target column for class-balance plot.
        input_format: Optional format override.
        outlier_method: "iqr" or "zscore".

    Returns:
        JSON with figure_paths and plotting_backend.
    """
    logger.info("plot_quality_dashboard called: path=%s dir=%s", input_path, output_dir)
    try:
        return _json_success(_plot_dashboard_outputs(input_path, output_dir, target_column, input_format, outlier_method))
    except Exception as exc:
        return _json_error(str(exc))


@tool
def plot_distributions(
    input_path: str,
    output_dir: str,
    columns_json: str = "",
    max_columns: int = 12,
    input_format: str = "",
) -> str:
    """
    Plot histograms and box plots for numeric columns.

    Args:
        input_path: Path to CSV or Parquet file.
        output_dir: Directory where plots should be written.
        columns_json: Optional JSON list of numeric columns to plot.
        max_columns: Maximum number of columns to render.
        input_format: Optional format override.

    Returns:
        JSON with figure_paths, plotted columns, and plotting_backend.
    """
    logger.info("plot_distributions called: path=%s dir=%s", input_path, output_dir)
    try:
        columns = _parse_json_arg(columns_json, "columns_json") if columns_json else None
        if columns is not None and not isinstance(columns, list):
            raise ValueError("columns_json must be a JSON list of column names.")
        return _json_success(_distribution_outputs(input_path, output_dir, columns, max_columns, input_format))
    except Exception as exc:
        return _json_error(str(exc))


@tool
def render_quality_notebook(
    notebook_output_path: str,
    summary_report_path: str = "",
    profile_path: str = "",
    correlation_path: str = "",
    comparison_paths_json: str = "",
    figure_paths_json: str = "",
    decision_path: str = "",
    dataset_description: str = "",
    chosen_strategy_json: str = "",
) -> str:
    """
    Render the final audit notebook from saved artifacts.

    Args:
        notebook_output_path: Destination path for the notebook.
        summary_report_path: Optional path to the quality report JSON.
        profile_path: Optional path to the profile JSON.
        correlation_path: Optional path to the correlation JSON.
        comparison_paths_json: Optional JSON list of comparison artifact paths.
        figure_paths_json: Optional JSON list of figure paths.
        decision_path: Optional path to a Markdown decision summary.
        dataset_description: Optional notebook intro text.
        chosen_strategy_json: Optional JSON object describing the selected strategy.

    Returns:
        JSON with notebook_output_path.
    """
    logger.info("render_quality_notebook called: output=%s", notebook_output_path)
    try:
        comparison_paths = _parse_json_arg(comparison_paths_json, "comparison_paths_json") if comparison_paths_json else []
        figure_paths = _parse_json_arg(figure_paths_json, "figure_paths_json") if figure_paths_json else []
        chosen_strategy = _parse_json_arg(chosen_strategy_json, "chosen_strategy_json") if chosen_strategy_json else {}

        output_path = render_notebook_from_artifacts(
            notebook_output_path=notebook_output_path,
            summary_report_path=summary_report_path,
            profile_path=profile_path,
            correlation_path=correlation_path,
            comparison_paths=comparison_paths,
            figure_paths=figure_paths,
            decision_path=decision_path,
            dataset_description=dataset_description or "Dataset quality analysis run.",
            chosen_strategy=chosen_strategy,
        )
        return _json_success({"notebook_output_path": output_path})
    except Exception as exc:
        return _json_error(str(exc))


@tool
def prepare_run_dir(base_dir: str, run_label: str) -> str:
    """
    Create a standard artifact directory layout for one run.

    Args:
        base_dir: Base directory for all runs.
        run_label: Short label like detect, fix, compare, or full_audit.

    Returns:
        JSON with run_id, run_dir, reports_dir, cleaned_dir, figures_dir, notebook_dir, summary_dir.
    """
    logger.info("prepare_run_dir called: base_dir=%s label=%s", base_dir, run_label)
    try:
        run_id = f"{run_label}_{make_run_id()}"
        run_dir = ensure_run_layout(Path(base_dir) / run_id)
        return _json_success(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "reports_dir": str(run_dir / "reports"),
                "cleaned_dir": str(run_dir / "cleaned"),
                "figures_dir": str(run_dir / "figures"),
                "notebook_dir": str(run_dir / "notebook"),
                "summary_dir": str(run_dir / "summary"),
            }
        )
    except Exception as exc:
        return _json_error(str(exc))


@tool
def deduplicate_image_dataset(
    input_dir: str,
    output_dir: str,
    hash_func_name: str = "phash",
    hash_size: int = 16,
    threshold: int = 8,
    dry_run: bool = False,
    report_output_path: str = "",
    duplicates_output_path: str = "",
) -> str:
    """
    Remove near-duplicate images inside a folder-organized classification dataset.

    Args:
        input_dir: Root folder containing one subdirectory per class.
        output_dir: Output folder where unique images will be copied.
        hash_func_name: Image hash method: phash, dhash, ahash, or whash.
        hash_size: Hash resolution passed to imagehash.
        threshold: Maximum Hamming distance treated as a duplicate.
        dry_run: If True, only compute the report without copying images.
        report_output_path: Optional JSON report artifact path.
        duplicates_output_path: Optional CSV path listing removed duplicates.

    Returns:
        JSON with per-class and total deduplication statistics plus artifact paths.
    """
    logger.info(
        "deduplicate_image_dataset called: input=%s output=%s hash=%s threshold=%s dry_run=%s",
        input_dir,
        output_dir,
        hash_func_name,
        threshold,
        dry_run,
    )
    try:
        payload = _deduplicate_image_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            hash_func_name=hash_func_name,
            hash_size=hash_size,
            threshold=threshold,
            dry_run=dry_run,
            report_output_path=report_output_path,
            duplicates_output_path=duplicates_output_path,
        )
        return _json_success(payload)
    except Exception as exc:
        return _json_error(str(exc))
