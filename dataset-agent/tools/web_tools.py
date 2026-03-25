from __future__ import annotations

import json
import os
import re
from io import StringIO
from urllib.parse import urljoin

import requests
from smolagents import tool
from tools.path_utils import data_root, resolve_data_output_path


DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}


def _data_root() -> str:
    return str(data_root())


def _resolve_output_path(path: str) -> str:
    return resolve_data_output_path(path)


def _get(url: str, *, stream: bool = False) -> requests.Response:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=60, stream=stream)
    response.raise_for_status()
    return response


@tool
def fetch_page(url: str) -> str:
    """
    Fetch an HTML page and return the visible text.

    Args:
        url: Webpage URL to fetch.

    Returns:
        The visible page text truncated to 10000 characters.
    """
    from bs4 import BeautifulSoup

    response = _get(url)
    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text[:10000]


@tool
def extract_table_from_html(url: str, table_index: int = 0) -> str:
    """
    Extract a table from a webpage and return CSV.

    Args:
        url: Webpage URL containing one or more HTML tables.
        table_index: Zero-based index of the table to extract.

    Returns:
        The selected table serialized as CSV, or an error message.
    """
    import pandas as pd

    response = _get(url)
    tables = pd.read_html(StringIO(response.text))

    if not tables:
        return "No tables found on this page."
    if table_index >= len(tables):
        return f"Table index {table_index} out of range. Found {len(tables)} tables."

    return tables[table_index].to_csv(index=False)


@tool
def extract_links_from_page(url: str, pattern: str = "") -> str:
    """
    Extract all links from a webpage, optionally filtering by regex.

    Args:
        url: Webpage URL to inspect.
        pattern: Optional regular expression used to filter links.

    Returns:
        A JSON-formatted string containing unique absolute URLs.
    """
    from bs4 import BeautifulSoup

    response = _get(url)
    soup = BeautifulSoup(response.text, "lxml")

    links = []
    for anchor in soup.find_all("a", href=True):
        href = urljoin(url, anchor["href"])
        if pattern and not re.search(pattern, href):
            continue
        links.append(href)

    return json.dumps(sorted(set(links)), indent=2, ensure_ascii=False)


@tool
def download_file(url: str, save_path: str) -> str:
    """
    Download a file from a direct URL and save it to disk.

    Args:
        url: Direct URL of the file to download.
        save_path: Target path for the downloaded file.

    Returns:
        A status string with the saved file path and size.
    """
    target_path = _resolve_output_path(save_path)
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)

    response = _get(url, stream=True)
    with open(target_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                handle.write(chunk)

    size = os.path.getsize(target_path)
    return f"Downloaded to {target_path} ({size} bytes)"
