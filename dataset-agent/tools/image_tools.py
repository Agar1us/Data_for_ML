from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from smolagents import tool

from parsers.yandex_images import Parser
from tools.path_utils import data_root, resolve_data_output_dir
from tools.runtime import logs_root, run_log_dir, yandex_headless, yandex_manual_captcha_timeout, yandex_profile_dir


_SEARCH_CALL_CACHE: dict[tuple[str, str, str, str, bool, str], str] = {}


def _infer_extension(url: str, content_type: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        if path.endswith(ext):
            return ext.lstrip(".")
    if "png" in content_type:
        return "png"
    if "webp" in content_type:
        return "webp"
    if "gif" in content_type:
        return "gif"
    return "jpg"


def _cache_key(
    query: str,
    save_dir: str,
    size: str | None,
    image_type: str | None,
    headless: bool,
    profile_dir: str | None,
) -> tuple[str, str, str, str, bool, str]:
    return (
        query.strip().casefold(),
        resolve_data_output_dir(save_dir),
        (size or "").strip().casefold(),
        (image_type or "").strip().casefold(),
        bool(headless),
        str(Path(profile_dir).expanduser().resolve()) if profile_dir else "",
    )


def _slugify_query(query: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in query.strip())
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "images"


def _resolve_class_save_dir(save_dir: str, query: str) -> str:
    class_dir = Path(resolve_data_output_dir(save_dir))
    relative_parts = class_dir.resolve().relative_to(data_root()).parts
    if len(relative_parts) != 1:
        raise ValueError(
            "search_and_download_images expects save_dir to point to exactly one class directory "
            "directly under the configured collection root."
        )
    query_slug = _slugify_query(query)
    if class_dir.name.casefold() == query_slug.casefold():
        return str(class_dir)
    return str((class_dir / query_slug).resolve())


def _append_tool_log(record: dict) -> None:
    log_root = run_log_dir() or logs_root()
    log_path = log_root / "yandex_image_tool_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@tool
def search_and_download_images(
    query: str,
    limit: int,
    save_dir: str,
    size: str = "",
    image_type: str = "",
    delay: float = 6.0,
    headless: bool | None = None,
    profile_dir: str = "",
    manual_captcha_timeout: float | None = None,
) -> str:
    """
    Search and download images using the Yandex Images parser.

    Args:
        query: Search phrase for Yandex Images.
        limit: Maximum number of images to download.
        save_dir: Directory where downloaded images will be stored.
        size: Optional Yandex size filter.
        image_type: Optional Yandex image type filter.
        delay: Delay between page-load/scroll attempts inside the Yandex parser.
        headless: Whether to run Chromium headlessly.
        profile_dir: Optional Chromium user-data-dir for persistent cookies/session.
        manual_captcha_timeout: In non-headless mode, wait this many seconds for manual captcha solving.

    Returns:
        A compact JSON string for the agent.
    """
    effective_headless = headless if headless is not None else yandex_headless()
    effective_manual_captcha_timeout = (
        float(manual_captcha_timeout) if manual_captcha_timeout is not None else yandex_manual_captcha_timeout()
    )
    effective_profile_dir = (profile_dir or "").strip() or yandex_profile_dir()

    try:
        requested_save_dir = resolve_data_output_dir(save_dir)
        resolved_save_dir = _resolve_class_save_dir(save_dir, query)
    except ValueError as exc:
        payload = {
            "status": "error",
            "query": query,
            "save_dir": save_dir,
            "downloaded": 0,
            "failed": 0,
            "message": str(exc),
        }
        _append_tool_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "search_and_download_images",
                "cache_hit": False,
                "query": query,
                "requested_limit": int(limit),
                "save_dir": save_dir,
                "result": payload,
            }
        )
        return json.dumps(payload, ensure_ascii=False)

    key = _cache_key(
        query=query,
        save_dir=resolved_save_dir,
        size=size,
        image_type=image_type,
        headless=effective_headless,
        profile_dir=effective_profile_dir,
    )
    if key in _SEARCH_CALL_CACHE:
        cached_result = _SEARCH_CALL_CACHE[key]
        _append_tool_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": "search_and_download_images",
                "cache_hit": True,
                "query": query,
                "requested_limit": int(limit),
                "save_dir": requested_save_dir,
                "resolved_save_dir": resolved_save_dir,
                "headless": bool(effective_headless),
                "profile_dir": effective_profile_dir,
                "result": json.loads(cached_result),
            }
        )
        return cached_result

    parser = Parser(headless=effective_headless, profile_dir=effective_profile_dir or None)
    urls = parser.query_search(
        query=query,
        limit=limit,
        delay=delay,
        manual_captcha_timeout=effective_manual_captcha_timeout,
        size=size or None,
        image_type=image_type or None,
    )

    resolved_dir = Path(resolved_save_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0
    failure_details: list[dict[str, str | int]] = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    for index, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            ext = _infer_extension(url, response.headers.get("Content-Type", "").lower())
            file_path = resolved_dir / f"{index:05d}.{ext}"
            with file_path.open("wb") as handle:
                handle.write(response.content)
            downloaded += 1
        except Exception as exc:
            failed += 1
            failure_record: dict[str, str | int] = {
                "index": int(index),
                "url": url,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                failure_record["status_code"] = int(status_code)
            failure_details.append(failure_record)

    captcha_suspected = bool(parser.last_debug_info.get("captcha_suspected", False))
    status = "captcha_blocked" if captcha_suspected else ("no_results" if not urls else "ok")
    payload = {
        "status": status,
        "query": query,
        "save_dir": resolved_save_dir,
        "downloaded": int(downloaded),
        "failed": int(failed),
        "message": "Captcha was detected. Retry once manually in non-headless mode." if captcha_suspected else "Search attempt completed.",
    }
    result = json.dumps(payload, ensure_ascii=False)

    _append_tool_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": "search_and_download_images",
            "cache_hit": False,
            "query": query,
            "requested_limit": int(limit),
            "save_dir": requested_save_dir,
            "resolved_save_dir": resolved_save_dir,
            "resolved_urls": int(len(urls)),
            "headless": bool(effective_headless),
            "profile_dir": effective_profile_dir,
            "manual_captcha_timeout": float(effective_manual_captcha_timeout),
            "failed_downloads": failure_details,
            "driver_title": parser.last_debug_info.get("driver_title", ""),
            "driver_url": parser.last_debug_info.get("driver_url", ""),
            "debug_html_path": parser.last_debug_info.get("debug_html_path", ""),
            "debug_screenshot_path": parser.last_debug_info.get("debug_screenshot_path", ""),
            "captcha_suspected": captcha_suspected,
            "wait_timed_out": bool(parser.last_debug_info.get("wait_timed_out", False)),
            "manual_captcha_waited": bool(parser.last_debug_info.get("manual_captcha_waited", False)),
            "result": payload,
        }
    )
    if urls and status == "ok":
        _SEARCH_CALL_CACHE[key] = result
    return result
