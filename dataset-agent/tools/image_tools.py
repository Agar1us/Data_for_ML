from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
from smolagents import tool

from parsers.yandex_images import Parser


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


@tool
def search_and_download_images(
    query: str,
    limit: int,
    save_dir: str,
    size: str = "",
    image_type: str = "",
) -> str:
    """
    Search and download images using the Yandex Images parser.

    Args:
        query: Search phrase for Yandex Images.
        limit: Maximum number of images to download.
        save_dir: Directory where downloaded images will be stored.
        size: Optional Yandex size filter.
        image_type: Optional Yandex image type filter.

    Returns:
        A summary string with downloaded and failed image counts.
    """
    parser = Parser(headless=True)
    urls = parser.query_search(query=query, limit=limit, size=size, image_type=image_type)

    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0
    failed = 0

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    for index, url in enumerate(urls):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            ext = _infer_extension(url, response.headers.get("Content-Type", "").lower())
            file_path = os.path.join(save_dir, f"{index:05d}.{ext}")
            with open(file_path, "wb") as handle:
                handle.write(response.content)
            downloaded += 1
        except Exception:
            failed += 1

    return (
        f"Downloaded {downloaded}/{len(urls)} images to {save_dir}. "
        f"Failed: {failed}."
    )
