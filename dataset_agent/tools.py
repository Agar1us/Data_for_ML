from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi, snapshot_download
from pydantic import Field

from dataset_agent.models import DatasetCandidate, FilterDecision, WebCollectedAsset
from dataset_agent.state import (
    append_note,
    build_manifest,
    class_variants,
    clean_class_list,
    ensure_dir,
    ensure_runtime_context,
    normalize_class_name,
    slugify,
    upsert_candidate,
)
from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.agent_definition import SearchConfig
from sgr_agent_core.services.tavily_search import TavilySearchService

logger = logging.getLogger(__name__)

IMAGE_HINTS = {
    "image",
    "images",
    "photo",
    "vision",
    "computer vision",
    "image classification",
    "object detection",
    "segmentation",
}
TEXT_HINTS = {
    "text",
    "nlp",
    "natural language",
    "language",
    "sentiment",
    "classification:text",
    "text classification",
    "qa",
    "question answering",
}
OTHER_HINTS = {
    "audio",
    "speech",
    "music",
    "video",
    "tabular",
    "time series",
    "timeseries",
}

LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "английский": "en",
    "англ": "en",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    "русский": "ru",
    "рус": "ru",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "испанский": "es",
    "fr": "fr",
    "fre": "fr",
    "french": "fr",
    "французский": "fr",
    "de": "de",
    "ger": "de",
    "german": "de",
    "немецкий": "de",
    "it": "it",
    "italian": "it",
    "итальянский": "it",
    "pt": "pt",
    "por": "pt",
    "portuguese": "pt",
    "португальский": "pt",
    "zh": "zh",
    "chi": "zh",
    "chinese": "zh",
    "китайский": "zh",
    "ja": "ja",
    "japanese": "ja",
    "японский": "ja",
    "ko": "ko",
    "korean": "ko",
    "корейский": "ko",
    "ar": "ar",
    "arabic": "ar",
    "арабский": "ar",
    "hi": "hi",
    "hindi": "hi",
    "хинди": "hi",
    "tr": "tr",
    "turkish": "tr",
    "турецкий": "tr",
    "uk": "uk",
    "ukrainian": "uk",
    "украинский": "uk",
    "pl": "pl",
    "polish": "pl",
    "польский": "pl",
    "nl": "nl",
    "dutch": "nl",
    "голландский": "nl",
}


def _to_json_compatible(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v, _seen) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item, _seen) for item in value]

    obj_id = id(value)
    if obj_id in _seen:
        return f"<recursive:{type(value).__name__}>"
    _seen.add(obj_id)
    try:
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(mode="python")
                return _to_json_compatible(dumped, _seen)
            except Exception:
                pass
        if hasattr(value, "to_dict"):
            try:
                dumped = value.to_dict()
                return _to_json_compatible(dumped, _seen)
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            public = {
                key: val
                for key, val in vars(value).items()
                if not key.startswith("_") and not callable(val)
            }
            if public:
                return _to_json_compatible(public, _seen)
    finally:
        _seen.remove(obj_id)

    return str(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.isdigit():
            return int(cleaned)
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb)$", cleaned.lower())
        if match:
            number = float(match.group(1))
            unit = match.group(2)
            scale = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}[unit]
            return int(number * scale)
    return None


def _iter_dict_items(data: Any):
    if isinstance(data, dict):
        for key, value in data.items():
            yield key, value
            yield from _iter_dict_items(value)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_dict_items(item)


def _extract_examples(metadata: dict[str, Any] | None) -> int | None:
    if not metadata:
        return None

    keys = {
        "num_examples",
        "examples",
        "n_examples",
        "num_rows",
        "rows",
        "train_examples",
        "validation_examples",
        "test_examples",
        "total_rows",
    }
    values: list[int] = []
    for key, value in _iter_dict_items(metadata):
        if str(key).lower() in keys:
            int_value = _safe_int(value)
            if int_value is not None:
                values.append(int_value)

    if values:
        # Splits are often provided separately. Summing gives a better total estimate.
        return sum(values)
    return None


def _extract_size_bytes(metadata: dict[str, Any] | None) -> int | None:
    if not metadata:
        return None

    keys = {
        "dataset_size",
        "dataset_bytes",
        "size_bytes",
        "sizeinbytes",
        "size",
        "download_size",
        "totalbytes",
        "total_bytes",
    }
    candidates: list[int] = []
    for key, value in _iter_dict_items(metadata):
        if str(key).lower() in keys:
            int_value = _safe_int(value)
            if int_value is not None:
                candidates.append(int_value)

    if candidates:
        return max(candidates)
    return None


def _extract_license(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    for key, value in _iter_dict_items(metadata):
        if "license" in str(key).lower() and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_classes(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []

    class_key_hints = {"classes", "class_names", "labels", "label_names", "categories", "category_names"}
    collected: list[str] = []

    for key, value in _iter_dict_items(metadata):
        key_norm = str(key).lower()
        if key_norm in class_key_hints:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        collected.append(item.strip())
            elif isinstance(value, dict):
                for item in value.keys():
                    if isinstance(item, str) and item.strip():
                        collected.append(item.strip())
        if key_norm == "tags" and isinstance(value, list):
            for tag in value:
                if not isinstance(tag, str):
                    continue
                tag_clean = tag.strip()
                if tag_clean.startswith("label:"):
                    collected.append(tag_clean.split(":", 1)[1])
                elif tag_clean.startswith("class:"):
                    collected.append(tag_clean.split(":", 1)[1])

    return clean_class_list(collected)


def _required_classes(state: dict[str, Any]) -> list[str]:
    classes = state.get("query_spec", {}).get("requested_classes", [])
    return clean_class_list(classes)


def _normalize_language_token(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Zа-яА-Я]+", " ", value.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _resolve_language_code(value: str) -> str | None:
    token = _normalize_language_token(value)
    if not token:
        return None
    if token in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[token]
    for part in token.split():
        if part in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[part]
    for part in token.split():
        if len(part) < 3:
            continue
        for alias, code in LANGUAGE_ALIASES.items():
            if len(alias) < 3:
                continue
            if part.startswith(alias) or alias.startswith(part):
                return code
    if re.fullmatch(r"[a-z]{2,3}", token):
        return token
    return None


def _extract_requested_language(query: str) -> tuple[str | None, str | None]:
    patterns = [
        r"language\s*[:=-]\s*([a-zA-Zа-яА-Я\- ]+)",
        r"in\s+([a-zA-Z\-]+)\s+language",
        r"([a-zA-Z\-]+)\s+language\s+dataset",
        r"язык\s*[:=-]\s*([a-zA-Zа-яА-Я\- ]+)",
        r"на\s+([a-zA-Zа-яА-Я\-]+)\s+языке",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        raw_value = match.group(1).strip()
        code = _resolve_language_code(raw_value)
        if code:
            return code, raw_value
    return None, None


def _extract_language_codes(metadata: dict[str, Any] | None) -> set[str]:
    payload = _to_json_compatible(metadata or {})
    detected: set[str] = set()
    for key, value in _iter_dict_items(payload):
        key_norm = str(key).lower()
        if key_norm in {"language", "languages", "lang"}:
            values = value if isinstance(value, list) else [value]
            for item in values:
                code = _resolve_language_code(str(item))
                if code:
                    detected.add(code)
        if key_norm == "tags" and isinstance(value, list):
            for item in value:
                tag = str(item).strip().lower()
                if tag.startswith("language:"):
                    code = _resolve_language_code(tag.split(":", 1)[1])
                    if code:
                        detected.add(code)
    return detected


def _requested_modalities(state: dict[str, Any]) -> list[str]:
    modalities = state.get("query_spec", {}).get("modalities", [])
    if not modalities:
        modalities = state.get("runtime_config", {}).get("modalities", [])
    clean: list[str] = []
    for item in modalities:
        value = str(item).strip().lower()
        if value in {"image", "text"} and value not in clean:
            clean.append(value)
    return clean or ["image", "text"]


def _detect_modality(metadata: dict[str, Any] | None, candidate_text: str = "") -> str:
    payload = _to_json_compatible(metadata or {})
    blob = json.dumps(payload, ensure_ascii=False).lower() + " " + candidate_text.lower()

    image_hit = any(term in blob for term in IMAGE_HINTS)
    text_hit = any(term in blob for term in TEXT_HINTS)
    other_hit = any(term in blob for term in OTHER_HINTS)

    if image_hit and text_hit:
        return "multimodal"
    if image_hit:
        return "image"
    if text_hit:
        return "text"
    if other_hit:
        return "other"
    return "unknown"


def _modality_matches(candidate_modality: str, requested_modalities: list[str]) -> bool:
    if candidate_modality in requested_modalities:
        return True
    if candidate_modality == "multimodal":
        return bool(requested_modalities)
    if candidate_modality == "unknown":
        # Keep unknown datasets as potential candidates to avoid false negatives.
        return True
    return False


def _refresh_class_coverage_from_selected(state: dict[str, Any]) -> None:
    required_classes = _required_classes(state)
    selected = state.get("selected_datasets", []) or []
    covered: set[str] = set()
    for candidate in selected:
        for class_name in candidate.get("classes", []) or []:
            normalized = normalize_class_name(str(class_name))
            if normalized:
                covered.add(normalized)
    state["covered_classes"] = sorted(covered)
    state["missing_classes"] = [cls for cls in required_classes if cls not in covered]


def _has_required_classes(candidate_classes: list[str], required: list[str]) -> tuple[bool, list[str]]:
    if not required:
        return True, []

    candidate_norm = clean_class_list(candidate_classes)
    candidate_variants: set[str] = set(candidate_norm)
    for cls in candidate_norm:
        candidate_variants.update(class_variants(cls))

    missing = []
    for req in required:
        req_variants = class_variants(req)
        if candidate_variants.isdisjoint(req_variants):
            missing.append(req)

    return len(missing) == 0, missing


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    downloads = candidate.get("downloads") or -1
    examples = candidate.get("examples") or -1
    size_bytes = candidate.get("size_bytes") or -1
    return (downloads, examples, size_bytes)


def _dataset_subdir_name(source: str, dataset_id: str) -> str:
    return f"{source}__{slugify(dataset_id)}"


class SearchHuggingFaceDatasetsTool(BaseTool):
    """Search Hugging Face datasets and register normalized candidates."""

    tool_name: ClassVar[str] = "search_huggingface_datasets_tool"

    reasoning: str = Field(description="Why HF search is needed now")
    query: str | None = Field(default=None, description="Optional query override")
    max_results: int = Field(default=20, ge=1, le=100)

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        search_query = self.query or state.get("query_spec", {}).get("query", "")
        if not search_query:
            logger.info("Tool %s skipped: empty query", self.tool_name)
            return "HF search skipped: empty query."

        requested_modalities = _requested_modalities(state)
        requested_language_code, requested_language_raw = _extract_requested_language(search_query)
        apply_text_language_filter = "text" in requested_modalities and requested_language_code is not None

        logger.info(
            "Tool %s started | query=%r | max_results=%s | requested_modalities=%s | requested_language=%s",
            self.tool_name,
            search_query,
            self.max_results,
            requested_modalities,
            requested_language_code or "none",
        )
        api = HfApi(token=os.getenv("HF_TOKEN"))
        existing = len(state["candidates"])
        added = 0
        skipped_language = 0

        list_kwargs: dict[str, Any] = {
            "search": search_query,
            "limit": self.max_results,
            "full": True,
        }
        if apply_text_language_filter:
            list_kwargs["language"] = requested_language_code

        try:
            datasets = api.list_datasets(**list_kwargs)
        except Exception as exc:
            if apply_text_language_filter:
                append_note(
                    state,
                    f"HF language filter failed for language='{requested_language_code}', retrying without language filter: {exc}",
                )
                logger.warning(
                    "Tool %s language-filtered search failed, retrying unfiltered | language=%s | error=%s",
                    self.tool_name,
                    requested_language_code,
                    exc,
                )
                try:
                    datasets = api.list_datasets(search=search_query, limit=self.max_results, full=True)
                except Exception as retry_exc:
                    append_note(state, f"HF search failed: {retry_exc}")
                    logger.exception("Tool %s failed during HF search retry", self.tool_name)
                    return f"HF search failed: {retry_exc}"
            else:
                append_note(state, f"HF search failed: {exc}")
                logger.exception("Tool %s failed during HF search", self.tool_name)
                return f"HF search failed: {exc}"

        for ds in datasets:
            ds_id = getattr(ds, "id", None)
            if not ds_id:
                continue

            card_data = getattr(ds, "cardData", None) or {}
            metadata = {
                "card_data": card_data,
                "tags": list(getattr(ds, "tags", []) or []),
                "likes": getattr(ds, "likes", None),
                "last_modified": str(getattr(ds, "last_modified", "") or ""),
            }
            classes = _extract_classes(card_data)
            if not classes:
                classes = _extract_classes(metadata)
            candidate_languages = _extract_language_codes(metadata)
            if candidate_languages:
                metadata["detected_languages"] = sorted(candidate_languages)
            if apply_text_language_filter and candidate_languages and requested_language_code not in candidate_languages:
                skipped_language += 1
                continue

            candidate = DatasetCandidate(
                source="huggingface",
                dataset_id=ds_id,
                title=ds_id,
                url=f"https://huggingface.co/datasets/{ds_id}",
                modality=_detect_modality(metadata, candidate_text=f"{ds_id} {getattr(ds, 'description', '') or ''}"),
                description=getattr(ds, "description", None),
                classes=classes,
                downloads=_safe_int(getattr(ds, "downloads", None)),
                examples=_extract_examples(card_data),
                size_bytes=_extract_size_bytes(card_data),
                license=_extract_license(card_data),
                metadata=metadata,
            ).model_dump(mode="json")

            upsert_candidate(state, candidate)
            added += 1

        result = (
            f"HF search complete. Query='{search_query}'. "
            f"Processed={added}, skipped_language={skipped_language}, "
            f"language_filter={requested_language_code if apply_text_language_filter else 'none'}, "
            f"total_candidates={len(state['candidates'])}, previous={existing}."
        )
        if apply_text_language_filter:
            append_note(
                state,
                f"HF text-language filter applied: language='{requested_language_raw or requested_language_code}' ({requested_language_code})",
            )
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class SearchKaggleDatasetsTool(BaseTool):
    """Search Kaggle datasets and register normalized candidates."""

    tool_name: ClassVar[str] = "search_kaggle_datasets_tool"

    reasoning: str = Field(description="Why Kaggle search is needed now")
    query: str | None = Field(default=None, description="Optional query override")
    max_results: int = Field(default=20, ge=1, le=100)

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        search_query = self.query or state.get("query_spec", {}).get("query", "")
        if not search_query:
            logger.info("Tool %s skipped: empty query", self.tool_name)
            return "Kaggle search skipped: empty query."

        logger.info("Tool %s started | query=%r | max_results=%s", self.tool_name, search_query, self.max_results)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except BaseException as exc:
            message = f"Kaggle import failed: {exc}"
            append_note(state, message)
            logger.exception("Tool %s failed while importing Kaggle API", self.tool_name)
            return message

        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            append_note(state, f"Kaggle authentication failed: {exc}")
            logger.exception("Tool %s failed during Kaggle authentication", self.tool_name)
            return f"Kaggle search failed: {exc}"

        try:
            raw_results = api.dataset_list(search=search_query)
        except Exception as exc:
            append_note(state, f"Kaggle search failed: {exc}")
            logger.exception("Tool %s failed during Kaggle dataset_list", self.tool_name)
            return f"Kaggle search failed: {exc}"

        processed = 0
        for ds in raw_results[: self.max_results]:
            dataset_ref = getattr(ds, "ref", None)
            if not dataset_ref:
                continue

            metadata = {
                "title": getattr(ds, "title", None),
                "subtitle": getattr(ds, "subtitle", None),
                "size": getattr(ds, "size", None),
                "last_updated": str(getattr(ds, "lastUpdated", "") or ""),
                "tags": list(getattr(ds, "tags", []) or []),
            }

            view_payload: dict[str, Any] = {}
            try:
                dataset_view = api.dataset_view(dataset_ref)
                normalized_view = _to_json_compatible(dataset_view)
                if isinstance(normalized_view, dict):
                    view_payload = normalized_view
                metadata["dataset_view"] = normalized_view
            except Exception as exc:
                metadata["dataset_view_error"] = str(exc)

            metadata = _to_json_compatible(metadata)
            classes = _extract_classes(metadata)
            examples = _extract_examples(view_payload) or _extract_examples(metadata)
            size_bytes = _safe_int(getattr(ds, "totalBytes", None)) or _extract_size_bytes(view_payload)
            license_value = _extract_license(view_payload) or _extract_license(metadata)

            candidate = DatasetCandidate(
                source="kaggle",
                dataset_id=dataset_ref,
                title=getattr(ds, "title", dataset_ref),
                url=f"https://www.kaggle.com/datasets/{dataset_ref}",
                modality=_detect_modality(
                    metadata,
                    candidate_text=f"{getattr(ds, 'title', '') or ''} {getattr(ds, 'subtitle', '') or ''}",
                ),
                description=getattr(ds, "subtitle", None),
                classes=classes,
                downloads=_safe_int(getattr(ds, "downloadCount", None)),
                examples=examples,
                size_bytes=size_bytes,
                license=license_value,
                metadata=metadata,
            ).model_dump(mode="json")

            upsert_candidate(state, candidate)
            processed += 1

        result = (
            f"Kaggle search complete. Query='{search_query}'. "
            f"Processed={processed}, total_candidates={len(state['candidates'])}."
        )
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class FilterDatasetCandidatesTool(BaseTool):
    """Filter candidate datasets by class coverage and quality thresholds."""

    tool_name: ClassVar[str] = "filter_dataset_candidates_tool"

    reasoning: str = Field(description="Why filtering is triggered now")

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        runtime = state.get("runtime_config", {})
        required_classes = _required_classes(state)
        requested_modalities = _requested_modalities(state)

        min_downloads = int(runtime.get("min_downloads", 10))
        min_examples = int(runtime.get("min_examples", 100))
        max_dataset_bytes = int(runtime.get("max_dataset_bytes", 0))
        max_datasets = int(runtime.get("max_datasets", 20))

        candidates: list[dict[str, Any]] = state.get("candidates", [])
        logger.info(
            "Tool %s started | candidates=%s | required_classes=%s | requested_modalities=%s | min_downloads=%s | min_examples=%s | max_dataset_bytes=%s | max_datasets=%s",
            self.tool_name,
            len(candidates),
            required_classes,
            requested_modalities,
            min_downloads,
            min_examples,
            max_dataset_bytes,
            max_datasets,
        )
        accepted_pool: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []

        for candidate in candidates:
            reasons: list[str] = []
            downloads = candidate.get("downloads")
            examples = candidate.get("examples")
            size_bytes = candidate.get("size_bytes")
            classes = candidate.get("classes", [])
            modality = str(candidate.get("modality", "unknown"))

            if downloads is None or int(downloads) < min_downloads:
                reasons.append(f"downloads<{min_downloads}")
            if examples is None or int(examples) < min_examples:
                reasons.append(f"examples<{min_examples}")
            if max_dataset_bytes > 0 and size_bytes is not None and int(size_bytes) > max_dataset_bytes:
                reasons.append(f"size>{max_dataset_bytes}")
            if not _modality_matches(modality, requested_modalities):
                reasons.append(f"modality_not_requested:{modality}")

            classes_ok, missing = _has_required_classes(classes, required_classes)
            if not classes_ok:
                reasons.append("missing_required_classes")

            if reasons:
                decisions.append(
                    FilterDecision(
                        source=candidate["source"],
                        dataset_id=candidate["dataset_id"],
                        accepted=False,
                        reasons=reasons,
                        missing_classes=missing,
                    ).model_dump(mode="json")
                )
                continue

            accepted_pool.append(candidate)

        accepted_pool.sort(key=_candidate_sort_key, reverse=True)
        accepted = accepted_pool[:max_datasets]
        for dropped in accepted_pool[max_datasets:]:
            decisions.append(
                FilterDecision(
                    source=dropped["source"],
                    dataset_id=dropped["dataset_id"],
                    accepted=False,
                    reasons=[f"max_datasets_cap={max_datasets}"],
                    missing_classes=[],
                ).model_dump(mode="json")
            )

        covered = set()
        for candidate in accepted:
            for class_name in candidate.get("classes", []):
                covered.add(normalize_class_name(class_name))

        missing_classes = [cls for cls in required_classes if cls not in covered]
        state["selected_datasets"] = accepted
        state["decisions"] = decisions
        state["covered_classes"] = sorted(covered)
        state["missing_classes"] = missing_classes

        result = (
            f"Filtering complete. candidates={len(candidates)}, accepted={len(accepted)}, "
            f"rejected={len(decisions)}, missing_classes={missing_classes}."
        )
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class DownloadSelectedDatasetsTool(BaseTool):
    """Download accepted datasets into separate source-prefixed subdirectories."""

    tool_name: ClassVar[str] = "download_selected_datasets_tool"

    reasoning: str = Field(description="Why dataset downloads are performed now")

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        runtime = state.get("runtime_config", {})
        max_datasets = int(runtime.get("max_datasets", 20))
        requested_modalities = _requested_modalities(state)
        data_dir = ensure_dir(state.get("data_dir", "data"))

        selected = state.get("selected_datasets", [])
        if not selected:
            candidates = state.get("candidates", [])
            candidate_pool = [
                candidate
                for candidate in candidates
                if _modality_matches(str(candidate.get("modality", "unknown")), requested_modalities)
            ]
            candidate_pool.sort(key=_candidate_sort_key, reverse=True)
            selected = candidate_pool[:max(1, max_datasets)]
            state["selected_datasets"] = selected
            logger.info(
                "Tool %s selected datasets without verification | requested_modalities=%s | selected=%s from candidates=%s",
                self.tool_name,
                requested_modalities,
                len(selected),
                len(candidates),
            )
        if not selected:
            logger.info("Tool %s skipped: no candidate datasets to download", self.tool_name)
            _refresh_class_coverage_from_selected(state)
            return "No candidate datasets to download."

        logger.info(
            "Tool %s started | selected=%s | data_dir=%s",
            self.tool_name,
            len(selected),
            data_dir,
        )
        kaggle_api: Any = None
        KaggleApiClass = None
        if any(item.get("source") == "kaggle" for item in selected):
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi as KaggleApiClass
            except BaseException as exc:
                append_note(state, f"Kaggle import failed: {exc}")
        success = 0
        failed = 0

        for candidate in selected:
            source = candidate.get("source")
            dataset_id = candidate.get("dataset_id")
            if not source or not dataset_id:
                failed += 1
                continue

            target_dir = data_dir / _dataset_subdir_name(source, dataset_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Tool %s working | source=%s | dataset=%s | target_dir=%s",
                self.tool_name,
                source,
                dataset_id,
                target_dir,
            )

            try:
                if source == "huggingface":
                    snapshot_download(
                        repo_id=dataset_id,
                        repo_type="dataset",
                        local_dir=str(target_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        token=os.getenv("HF_TOKEN"),
                    )
                elif source == "kaggle":
                    if kaggle_api is None:
                        if KaggleApiClass is None:
                            candidate.setdefault("metadata", {})["download_error"] = "kaggle import unavailable"
                            failed += 1
                            continue
                        kaggle_api = KaggleApiClass()
                        kaggle_api.authenticate()
                    kaggle_api.dataset_download_files(dataset=dataset_id, path=str(target_dir), unzip=True, quiet=True)
                else:
                    candidate.setdefault("metadata", {})["download_error"] = f"unsupported source: {source}"
                    failed += 1
                    continue

                candidate["local_dir"] = str(target_dir)
                success += 1
                logger.info("Tool %s dataset completed | source=%s | dataset=%s", self.tool_name, source, dataset_id)
            except Exception as exc:
                candidate.setdefault("metadata", {})["download_error"] = str(exc)
                append_note(state, f"Download failed for {source}:{dataset_id}: {exc}")
                logger.warning(
                    "Tool %s dataset failed | source=%s | dataset=%s | error=%s",
                    self.tool_name,
                    source,
                    dataset_id,
                    exc,
                )
                failed += 1

        state["selected_datasets"] = selected
        _refresh_class_coverage_from_selected(state)
        result = f"Dataset download complete. success={success}, failed={failed}, target_dir='{data_dir}'."
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class TavilyFallbackCollectionTool(BaseTool):
    """Collect class-specific web images with Tavily when HF/Kaggle coverage is insufficient."""

    tool_name: ClassVar[str] = "tavily_fallback_collection_tool"

    reasoning: str = Field(description="Why Tavily fallback is needed now")

    def _extract_image_links(self, page_url: str, timeout: int = 15) -> list[str]:
        response = requests.get(page_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        links: list[str] = []
        for tag in soup.find_all("img"):
            raw = tag.get("src") or ""
            if not raw:
                continue
            url = urljoin(page_url, raw)
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                continue
            links.append(url)

        return links

    def _download_image(self, image_url: str, destination: Path, timeout: int = 20) -> bool:
        response = requests.get(image_url, stream=True, timeout=timeout)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if not content_type.startswith("image/"):
            return False

        extension = Path(urlparse(image_url).path).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            extension = ".jpg"

        file_path = destination.with_suffix(extension)
        with open(file_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)
        return True

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        _refresh_class_coverage_from_selected(state)
        missing_classes = state.get("missing_classes", [])
        requested_modalities = _requested_modalities(state)
        if "image" not in requested_modalities:
            logger.info("Tool %s skipped: image modality not requested", self.tool_name)
            return "Tavily fallback skipped: requested modalities do not include image."
        if not missing_classes:
            logger.info("Tool %s skipped: no missing classes", self.tool_name)
            return "Tavily fallback skipped: no missing classes."

        search_config = config.search
        if search_config is None:
            tavily_key = os.getenv("SGR__SEARCH__TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")
            if tavily_key:
                search_config = SearchConfig(tavily_api_key=tavily_key)

        if search_config is None or not search_config.tavily_api_key:
            message = "Tavily fallback skipped: Tavily search config or API key is missing."
            append_note(state, message)
            logger.info("Tool %s skipped: Tavily API key missing", self.tool_name)
            return message

        runtime = state.get("runtime_config", {})
        per_class_cap = int(runtime.get("max_web_images_per_class", 300))
        tavily_results = int(runtime.get("tavily_max_results", 5))
        data_dir = ensure_dir(state.get("data_dir", "data"))
        query_text = state.get("query_spec", {}).get("query", "")

        service = TavilySearchService(search_config)
        assets = state.get("web_collected_assets", [])
        logger.info(
            "Tool %s started | missing_classes=%s | per_class_cap=%s | tavily_max_results=%s",
            self.tool_name,
            missing_classes,
            per_class_cap,
            tavily_results,
        )

        total_downloaded = 0
        for class_name in missing_classes:
            class_dir = ensure_dir(data_dir / f"web__{slugify(class_name)}")
            downloaded_for_class = 0
            search_query = f"{query_text} {class_name} image dataset"
            logger.info(
                "Tool %s class-start | class=%s | query=%r | target_dir=%s",
                self.tool_name,
                class_name,
                search_query,
                class_dir,
            )

            try:
                sources = await service.search(
                    query=search_query,
                    max_results=tavily_results,
                    include_raw_content=False,
                )
            except Exception as exc:
                append_note(state, f"Tavily search failed for class '{class_name}': {exc}")
                logger.warning(
                    "Tool %s class-search-failed | class=%s | error=%s",
                    self.tool_name,
                    class_name,
                    exc,
                )
                continue
            logger.info(
                "Tool %s class-search-done | class=%s | sources=%s",
                self.tool_name,
                class_name,
                len(sources),
            )

            for source in sources:
                if downloaded_for_class >= per_class_cap:
                    break

                try:
                    image_links = self._extract_image_links(source.url)
                except Exception as exc:
                    append_note(state, f"Page parse failed for '{source.url}': {exc}")
                    continue

                for image_url in image_links:
                    if downloaded_for_class >= per_class_cap:
                        break

                    filename_stub = f"{downloaded_for_class + 1:05d}-{slugify(urlparse(source.url).netloc)}"
                    destination = class_dir / filename_stub
                    try:
                        saved = self._download_image(image_url, destination)
                    except Exception as exc:
                        append_note(state, f"Image download failed '{image_url}': {exc}")
                        continue

                    if not saved:
                        continue

                    saved_path = str(next(class_dir.glob(f"{filename_stub}.*"), destination))
                    asset = WebCollectedAsset(
                        class_name=class_name,
                        query=search_query,
                        page_url=source.url,
                        image_url=image_url,
                        local_path=saved_path,
                    ).model_dump(mode="json")
                    assets.append(asset)
                    downloaded_for_class += 1
                    total_downloaded += 1
            logger.info(
                "Tool %s class-finished | class=%s | downloaded=%s",
                self.tool_name,
                class_name,
                downloaded_for_class,
            )

        state["web_collected_assets"] = assets
        result = (
            f"Tavily fallback complete. missing_classes={missing_classes}, "
            f"downloaded_assets={total_downloaded}."
        )
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class WriteCollectionManifestTool(BaseTool):
    """Write a complete JSON manifest with selected/rejected/fallback provenance."""

    tool_name: ClassVar[str] = "write_collection_manifest_tool"

    reasoning: str = Field(description="Why manifest persistence is needed now")

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        logger.info("Tool %s started", self.tool_name)
        manifest = build_manifest(state)

        manifest_path = Path(state.get("manifest_path", "manifest.json"))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        state["manifest_written"] = str(manifest_path)

        selected = len(manifest.selected_datasets)
        rejected = len(manifest.rejected_datasets)
        web_assets = len(manifest.web_collected_assets)
        result = (
            f"Manifest written to '{manifest_path}'. "
            f"selected={selected}, rejected={rejected}, web_assets={web_assets}."
        )
        logger.info("Tool %s finished | %s", self.tool_name, result)
        return result


class DumpRuntimeStateTool(BaseTool):
    """Debug helper tool for inspecting runtime state (optional)."""

    tool_name: ClassVar[str] = "dump_runtime_state_tool"

    reasoning: str = Field(description="Why runtime state dump is needed")

    async def __call__(self, context, config, **_) -> str:
        state = ensure_runtime_context(context)
        logger.info("Tool %s invoked", self.tool_name)
        return json.dumps(state, ensure_ascii=False)[:12000]
