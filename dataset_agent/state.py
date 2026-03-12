from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dataset_agent.models import CollectionManifest, DatasetQuerySpec


def normalize_class_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def class_variants(value: str) -> set[str]:
    base = normalize_class_name(value)
    if not base:
        return set()
    variants = {base}
    if base.endswith("ies") and len(base) > 3:
        variants.add(base[:-3] + "y")
    if base.endswith("s") and len(base) > 2:
        variants.add(base[:-1])
    else:
        variants.add(base + "s")
    return variants


def clean_class_list(classes: list[str]) -> list[str]:
    unique = []
    seen = set()
    for item in classes:
        normalized = normalize_class_name(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def extract_classes_from_query(query: str) -> list[str]:
    text = query.strip()
    if not text:
        return []

    patterns = [
        r"classes?\s*[:=-]\s*([^.;\n]+)",
        r"types?\s*[:=-]\s*([^.;\n]+)",
        r"categories?\s*[:=-]\s*([^.;\n]+)",
        r"(?:of|for)\s+([a-zA-Z0-9_\-\s,]+?)\s+(?:dataset|datasets)",
    ]

    chunks: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            chunks.append(match.group(1))

    if not chunks:
        return []

    classes: list[str] = []
    for chunk in chunks:
        parts = re.split(r",|/|\band\b|\bor\b", chunk, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip(" .:-_")
            if len(cleaned) >= 2:
                classes.append(cleaned)

    return clean_class_list(classes)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "item"


def ensure_runtime_context(context: Any) -> dict[str, Any]:
    if context.custom_context is None:
        context.custom_context = {}
    if not isinstance(context.custom_context, dict):
        context.custom_context = json.loads(json.dumps(context.custom_context))

    state = context.custom_context
    state.setdefault("query_spec", {})
    state.setdefault("runtime_config", {})
    state.setdefault("candidates", [])
    state.setdefault("decisions", [])
    state.setdefault("selected_datasets", [])
    state.setdefault("web_collected_assets", [])
    state.setdefault("covered_classes", [])
    state.setdefault("missing_classes", [])
    state.setdefault("notes", [])
    state.setdefault("manifest_path", "manifest.json")
    state.setdefault("data_dir", "data")
    return state


def append_note(state: dict[str, Any], message: str) -> None:
    notes = state.setdefault("notes", [])
    notes.append(message)


def upsert_candidate(state: dict[str, Any], candidate: dict[str, Any]) -> None:
    candidates = state.setdefault("candidates", [])
    key = (candidate.get("source"), candidate.get("dataset_id"))
    for idx, existing in enumerate(candidates):
        if (existing.get("source"), existing.get("dataset_id")) == key:
            candidates[idx] = candidate
            return
    candidates.append(candidate)


def build_manifest(state: dict[str, Any]) -> CollectionManifest:
    query_spec = DatasetQuerySpec(**state.get("query_spec", {}))
    return CollectionManifest(
        query_spec=query_spec,
        runtime_config=state.get("runtime_config", {}),
        selected_datasets=state.get("selected_datasets", []),
        rejected_datasets=state.get("decisions", []),
        web_collected_assets=state.get("web_collected_assets", []),
        covered_classes=state.get("covered_classes", []),
        missing_classes=state.get("missing_classes", []),
        notes=state.get("notes", []),
    )


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
