from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import requests
from smolagents import tool

from parsers.yandex_images import Parser


_SEARCH_CALL_CACHE: dict[tuple[str, str, str, str, float, bool, str], str] = {}


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
    size: str,
    image_type: str,
    delay: float,
    headless: bool,
    profile_dir: str,
) -> tuple[str, str, str, str, float, bool, str]:
    return (
        query.strip().casefold(),
        os.path.abspath(save_dir),
        size.strip().casefold(),
        image_type.strip().casefold(),
        float(delay),
        bool(headless),
        os.path.abspath(profile_dir) if profile_dir else "",
    )


def _slugify_query(query: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in query.strip())
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "images"


def _resolve_class_save_dir(save_dir: str, query: str) -> str:
    root_dir = os.path.abspath(save_dir)
    class_dir_name = _slugify_query(query)
    if os.path.basename(root_dir).casefold() == class_dir_name.casefold():
        return root_dir
    return os.path.join(root_dir, class_dir_name)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


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
        A summary string with downloaded and failed image counts.
    """
    effective_headless = headless if headless is not None else _env_flag("DATASET_AGENT_YANDEX_HEADLESS", True)
    effective_manual_captcha_timeout = (
        float(manual_captcha_timeout)
        if manual_captcha_timeout is not None
        else float(os.environ.get("DATASET_AGENT_MANUAL_CAPTCHA_TIMEOUT", "0") or 0.0)
    )
    resolved_save_dir = _resolve_class_save_dir(save_dir, query)
    effective_profile_dir = profile_dir.strip() or os.environ.get("DATASET_AGENT_CHROME_PROFILE_DIR", "").strip()
    key = _cache_key(
        query,
        resolved_save_dir,
        size,
        image_type,
        delay,
        effective_headless,
        effective_profile_dir,
    )
    if key in _SEARCH_CALL_CACHE:
        return _SEARCH_CALL_CACHE[key]

    parser = Parser(
        headless=effective_headless,
        profile_dir=effective_profile_dir or None,
    )
    urls = parser.query_search(
        query=query,
        limit=limit,
        delay=delay,
        manual_captcha_timeout=effective_manual_captcha_timeout,
        size=size,
        image_type=image_type,
    )

    os.makedirs(resolved_save_dir, exist_ok=True)
    downloaded = 0
    failed = 0

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    for index, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            ext = _infer_extension(url, response.headers.get("Content-Type", "").lower())
            file_path = os.path.join(resolved_save_dir, f"{index:05d}.{ext}")
            with open(file_path, "wb") as handle:
                handle.write(response.content)
            downloaded += 1
        except Exception:
            failed += 1

    status = "ok"
    completed = True
    should_retry_same_query = False
    retry_recommended_manually = False
    suggested_retry = None
    if parser.last_debug_info.get("captcha_suspected"):
        status = "captcha_blocked"
        completed = False
        should_retry_same_query = True
        retry_recommended_manually = True
        suggested_retry = {
            "query": query,
            "limit": int(limit),
            "save_dir": save_dir,
            "size": size,
            "image_type": image_type,
            "delay": float(delay),
            "headless": False,
            "profile_dir": effective_profile_dir or os.path.join(os.path.abspath(save_dir), ".chrome_profile"),
            "manual_captcha_timeout": max(180.0, float(effective_manual_captcha_timeout)),
        }
    elif not urls:
        status = "no_results"

    payload = {
        "status": status,
        "query": query,
        "save_dir": os.path.abspath(save_dir),
        "resolved_save_dir": resolved_save_dir,
        "class_dir_name": os.path.basename(resolved_save_dir),
        "requested_limit": int(limit),
        "delay": float(delay),
        "headless": bool(effective_headless),
        "profile_dir": effective_profile_dir,
        "manual_captcha_timeout": float(effective_manual_captcha_timeout),
        "resolved_urls": int(len(urls)),
        "downloaded": int(downloaded),
        "failed": int(failed),
        "driver_title": parser.last_debug_info.get("driver_title", ""),
        "driver_url": parser.last_debug_info.get("driver_url", ""),
        "debug_html_path": parser.last_debug_info.get("debug_html_path", ""),
        "debug_screenshot_path": parser.last_debug_info.get("debug_screenshot_path", ""),
        "captcha_suspected": bool(parser.last_debug_info.get("captcha_suspected", False)),
        "wait_timed_out": bool(parser.last_debug_info.get("wait_timed_out", False)),
        "manual_captcha_waited": bool(parser.last_debug_info.get("manual_captcha_waited", False)),
        "completed": completed,
        "should_retry_same_query": should_retry_same_query,
        "retry_recommended_manually": retry_recommended_manually,
        "suggested_retry_args": suggested_retry,
        "message": (
            "Captcha was detected. Re-run the same tool with headless disabled, a persistent profile_dir, "
            "and enough manual_captcha_timeout so the user can solve the captcha in the opened browser."
            if status == "captcha_blocked"
            else "Search attempt completed. Do not retry the same query in this run; "
            "reuse these results or move on to metadata/final reporting."
        ),
    }
    result = json.dumps(payload, ensure_ascii=False)
    if urls and status == "ok":
        _SEARCH_CALL_CACHE[key] = result
    return result
