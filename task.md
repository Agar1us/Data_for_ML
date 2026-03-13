# Dataset Collection Agent — Полный план реализации (v1)

> **Цель документа:** Этот план содержит ВСЕ необходимое для последовательной
> реализации мультиагентной системы сбора датасетов. Каждая секция — это
> конкретный шаг с готовым кодом. Выполняй секции по порядку.

---

## 1. КОНТЕКСТ ПРОЕКТА

### 1.1 Что строим
Мультиагентная система на базе smolagents. По текстовому запросу пользователя
агент находит, оценивает и скачивает данные из HuggingFace, Kaggle,
веб-сайтов и Яндекс Картинок, формируя структурированный датасет
в папке `data/`.

### 1.2 Примеры запросов
"Собери датасет из лебедей 3 видов: кликун, шипун и малый (bewick)" → Тип: изображения, источники: Яндекс Картинки + HF + Kaggle

"Собери датасет для семантического анализа: положительные, отрицательные, нейтральные" → Тип: текст, источники: HF + Kaggle + веб-парсинг

"Найди табличные данные по аренде жилья, покажи доступные датасеты" → Тип: таблицы, источники: HF + Kaggle + DuckDuckGo

text


### 1.3 Ограничения v1
- Типы данных: текст, таблицы, изображения
- Объём: до 10 000 записей/изображений (или меньше, если данных нет)
- Парсинг: только статичный HTML (bs4), с поддержкой пагинации
- Каждый источник хранится отдельно в `data/<dataset_name>/<source>/`
- Очистка/дедупликация — вне скоупа (будет отдельный агент v2)
- При повторном запуске — начинать заново
- Robots.txt не проверяется

---

## 2. СТРУКТУРА ПРОЕКТА

Создай точно эту структуру файлов:
dataset-agent/ ├── main.py ├── config.py ├── .env ├── requirements.txt ├── agents/ │ ├── init.py │ ├── orchestrator.py │ ├── search_agent.py │ ├── parser_agent.py │ └── image_agent.py ├── tools/ │ ├── init.py │ ├── huggingface_tools.py │ ├── kaggle_tools.py │ ├── web_tools.py │ ├── image_tools.py │ └── storage_tools.py ├── parsers/ │ ├── init.py │ └── yandex_images.py ├── logs/ └── data/

text


---

## 3. ЗАВИСИМОСТИ

### 3.1 requirements.txt
smolagents[openai]>=1.24.0 openai>=1.30.0 requests>=2.31.0 beautifulsoup4>=4.12.0 lxml>=5.0.0 pandas>=2.0.0 datasets>=2.18.0 kaggle>=1.6.0 selenium>=4.15.0 fake-headers>=1.0.2 tqdm>=4.66.0 python-dotenv>=1.0.0

text


### 3.2 .env
OPENAI_API_KEY=sk-... KAGGLE_USERNAME=your_username KAGGLE_KEY=your_kaggle_key HF_TOKEN=hf_...

text


---

## 4. КОНФИГУРАЦИЯ

### 4.1 Файл: `config.py`
```python
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    # LLM
    model_id: str = "gpt-5-mini"
    temperature: float = 0.2
    max_tokens: int = 16000

    # Поведение агента
    max_clarifications: int = 5
    max_search_results: int = 10
    max_steps_per_agent: int = 20

    # Данные
    data_dir: str = "data"
    logs_dir: str = "logs"
    max_dataset_size_gb: float = 5.0
    max_images_per_query: int = 10000
    max_records_per_query: int = 10000

    # Парсинг
    request_delay_sec: float = 2.0
    request_timeout_sec: float = 30.0
    yandex_parser_delay: float = 6.0
    yandex_headless: bool = True

    # Imports, разрешённые для CodeAgent
    authorized_imports: list = field(default_factory=lambda: [
        "requests", "bs4", "json", "csv", "os", "pathlib",
        "pandas", "re", "time", "urllib", "hashlib", "datetime",
    ])
5. АРХИТЕКТУРА АГЕНТОВ
5.1 Общая схема
text

                    ┌───────────────────────────┐
                    │   ORCHESTRATOR (Manager)    │
                    │   CodeAgent, GPT-5 mini     │
                    │                             │
                    │  1. Уточняющие вопросы      │
                    │  2. Планирование            │
                    │  3. Делегирование            │
                    │  4. Сбор метаданных          │
                    └─────┬──────┬──────┬─────────┘
                          │      │      │
              ┌───────────┘      │      └───────────┐
              ▼                  ▼                  ▼
    ┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
    │  SEARCH AGENT   │ │ PARSER AGENT │ │  IMAGE AGENT    │
    │                 │ │              │ │                 │
    │ DuckDuckGo      │ │ fetch_page   │ │ Яндекс Картинки │
    │ VisitWebpage    │ │ extract_table│ │ download_images │
    │ search_hf       │ │ extract_links│ │                 │
    │ search_kaggle   │ │ download_file│ │                 │
    │ download_hf     │ │ save_dataset │ │                 │
    │ download_kaggle │ │ save_metadata│ │                 │
    └─────────────────┘ └──────────────┘ └─────────────────┘
5.2 ВАЖНО: API мультиагентности smolagents
В smolagents ≥1.8.0 класс ManagedAgent удалён. Для создания подагентов используется новый API: подагент — это обычный CodeAgent с обязательными параметрами name и description. Затем он передаётся менеджеру через managed_agents=[...].

Python

# Новый API (актуальный):
sub_agent = CodeAgent(
    tools=[...],
    model=model,
    name="agent_name",           # ОБЯЗАТЕЛЬНО для managed agent
    description="Описание...",   # ОБЯЗАТЕЛЬНО для managed agent
)
manager = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[sub_agent],  # Передаём как список
)
6. ИНСТРУМЕНТЫ (TOOLS) — ПОЛНЫЙ КОД
6.1 Файл: tools/storage_tools.py
Python

import os
import json
from datetime import datetime, timezone
from smolagents import tool


@tool
def save_dataset(data: str, dataset_name: str, filename: str) -> str:
    """
    Saves collected data to a file inside data/<dataset_name>/.

    Args:
        data: The data to save as a string (CSV, JSON, or plain text content).
        dataset_name: Name of the dataset subdirectory to create or use.
        filename: Name of the file to save the data into.

    Returns:
        Full path to the saved file and its size in bytes.
    """
    dir_path = os.path.join("data", dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(data)
    size = os.path.getsize(file_path)
    return f"Saved to {file_path} ({size} bytes)"


@tool
def save_metadata(dataset_name: str, metadata_json: str) -> str:
    """
    Saves dataset metadata to data/<dataset_name>/metadata.json.

    Args:
        dataset_name: Name of the dataset subdirectory.
        metadata_json: A JSON string with metadata fields. Must include:
            source (str), date (str), license (str), num_records (int),
            description (str). Example:
            '{"source": "huggingface", "date": "2026-03-13",
              "license": "CC-BY-4.0", "num_records": 1000,
              "description": "Swan species images"}'

    Returns:
        Path to the saved metadata file.
    """
    dir_path = os.path.join("data", dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "metadata.json")

    metadata = json.loads(metadata_json)
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return f"Metadata saved to {file_path}"
6.2 Файл: tools/huggingface_tools.py
Python

import os
import json
from smolagents import tool


@tool
def search_huggingface(query: str, max_results: int = 10) -> str:
    """
    Searches for datasets on HuggingFace Hub by query string.

    Args:
        query: Search query for finding datasets (e.g. 'swan species images').
        max_results: Maximum number of dataset results to return.

    Returns:
        A formatted string listing found datasets with their id,
        description, download count, and tags.
    """
    from huggingface_hub import list_datasets

    results = []
    for i, ds in enumerate(list_datasets(search=query, sort="downloads", direction=-1)):
        if i >= max_results:
            break
        results.append({
            "id": ds.id,
            "downloads": ds.downloads,
            "tags": ds.tags[:5] if ds.tags else [],
            "description": (ds.description or "")[:200],
        })
    if not results:
        return f"No datasets found for query: '{query}'"
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def download_hf_dataset(
    dataset_id: str,
    save_dir: str,
    subset: str = "",
    split: str = ""
) -> str:
    """
    Downloads a dataset from HuggingFace Hub and saves it locally.

    Args:
        dataset_id: HuggingFace dataset identifier (e.g. 'imdb' or 'user/dataset').
        save_dir: Directory path for saving (inside data/, e.g. 'data/my_ds/hf_source').
        subset: Dataset subset/config name. Pass empty string if not needed.
        split: Dataset split to download (train/test/validation). Pass empty string for all.

    Returns:
        Path to the saved dataset directory and number of records.
    """
    from datasets import load_dataset

    os.makedirs(save_dir, exist_ok=True)

    kwargs = {"path": dataset_id}
    if subset:
        kwargs["name"] = subset
    if split:
        kwargs["split"] = split

    ds = load_dataset(**kwargs)

    if split:
        # Single split — save as CSV
        file_path = os.path.join(save_dir, f"{split}.csv")
        ds.to_csv(file_path)
        return f"Downloaded {len(ds)} records to {file_path}"
    else:
        # Multiple splits
        total = 0
        for split_name, split_ds in ds.items():
            file_path = os.path.join(save_dir, f"{split_name}.csv")
            split_ds.to_csv(file_path)
            total += len(split_ds)
        return f"Downloaded {total} records ({list(ds.keys())} splits) to {save_dir}"
6.3 Файл: tools/kaggle_tools.py
Python

import os
import json
from smolagents import tool


@tool
def search_kaggle(query: str, max_results: int = 10) -> str:
    """
    Searches for datasets on Kaggle by query string.

    Args:
        query: Search query text for Kaggle datasets.
        max_results: Maximum number of results to return.

    Returns:
        A formatted string listing found datasets with ref, title,
        size, and description.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    datasets = api.dataset_list(search=query, sort_by="hottest")

    results = []
    for ds in datasets[:max_results]:
        results.append({
            "ref": str(ds),
            "title": ds.title,
            "size": ds.totalBytes,
            "description": (ds.subtitle or "")[:200],
        })

    if not results:
        return f"No Kaggle datasets found for query: '{query}'"
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def download_kaggle_dataset(dataset_ref: str, save_dir: str) -> str:
    """
    Downloads and unzips a dataset from Kaggle.

    Args:
        dataset_ref: Kaggle dataset reference (e.g. 'username/dataset-name').
        save_dir: Directory path for saving (e.g. 'data/my_ds/kaggle_source').

    Returns:
        Path to the saved directory and list of downloaded files.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    os.makedirs(save_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_ref, path=save_dir, unzip=True)

    files = os.listdir(save_dir)
    return f"Downloaded {len(files)} files to {save_dir}: {files}"
6.4 Файл: tools/web_tools.py
Python

import os
import json
import re
import time
import requests
from smolagents import tool


@tool
def fetch_page(url: str) -> str:
    """
    Fetches an HTML page and returns its text content parsed with BeautifulSoup.

    Args:
        url: The full URL of the page to fetch.

    Returns:
        The text content of the page (stripped of scripts and styles),
        truncated to first 10000 characters if longer.
    """
    from bs4 import BeautifulSoup

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text[:10000]


@tool
def extract_table_from_html(url: str, table_index: int = 0) -> str:
    """
    Extracts a specific HTML table from a webpage and returns it as CSV.

    Args:
        url: The full URL of the page containing the table.
        table_index: Zero-based index of the table on the page to extract.

    Returns:
        The extracted table as a CSV-formatted string.
    """
    import pandas as pd

    tables = pd.read_html(url)

    if not tables:
        return "No tables found on this page."
    if table_index >= len(tables):
        return f"Table index {table_index} out of range. Found {len(tables)} tables."

    return tables[table_index].to_csv(index=False)


@tool
def extract_links_from_page(url: str, pattern: str = "") -> str:
    """
    Extracts all hyperlinks from a webpage, optionally filtering by a regex pattern.

    Args:
        url: The full URL of the page to extract links from.
        pattern: Optional regex pattern to filter links
                 (e.g. '\\.csv$' for CSV files). Empty string means no filter.

    Returns:
        A JSON list of absolute URLs found on the page.
    """
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.text, "lxml")

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        if pattern and not re.search(pattern, href):
            continue
        links.append(href)

    return json.dumps(list(set(links)), indent=2)


@tool
def download_file(url: str, save_path: str) -> str:
    """
    Downloads a file from a direct URL and saves it to disk.

    Args:
        url: Direct URL of the file to download.
        save_path: Full local file path where the file will be saved
                   (e.g. 'data/my_ds/source/file.csv').

    Returns:
        Path to the downloaded file and its size in bytes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetBot/1.0)"}
    response = requests.get(url, headers=headers, timeout=60, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size = os.path.getsize(save_path)
    return f"Downloaded to {save_path} ({size} bytes)"
6.5 Файл: tools/image_tools.py
Python

import os
import requests
from smolagents import tool


@tool
def search_and_download_images(
    query: str,
    limit: int,
    save_dir: str,
    size: str = "",
    image_type: str = ""
) -> str:
    """
    Searches and downloads images using Yandex Images parser.

    Args:
        query: Search query for images (e.g. 'лебедь кликун фото').
        limit: Maximum number of images to download.
        save_dir: Directory to save images into (e.g. 'data/swans/whooper_swan').
        size: Image size filter: 'large', 'medium', or 'small'. Empty for no filter.
        image_type: Image type filter: 'photo', 'clipart', 'lineart', 'face'.
                    Empty for no filter.

    Returns:
        Report string with number of images downloaded and save path.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from parsers.yandex_images import Parser

    parser = Parser(headless=True)

    kwargs = {"query": query, "limit": limit}
    if size:
        kwargs["size"] = size
    if image_type:
        kwargs["image_type"] = image_type

    urls = parser.query_search(**kwargs)

    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0
    failed = 0

    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            ext = url.split(".")[-1].split("?")[0][:4].lower()
            if ext not in ["jpg", "jpeg", "png", "webp", "gif"]:
                ext = "jpg"

            filepath = os.path.join(save_dir, f"{i:05d}.{ext}")
            with open(filepath, "wb") as f:
                f.write(response.content)
            downloaded += 1
        except Exception:
            failed += 1
            continue

    return (
        f"Downloaded {downloaded}/{len(urls)} images to {save_dir}. "
        f"Failed: {failed}."
    )
6.6 Файл: parsers/yandex_images.py
Скопируй сюда ПОЛНЫЙ код класса Parser из исходного описания (класс Parser с Size, Orientation, ImageType, Color, Format). Код уже предоставлен пользователем — вставить без изменений.

7. АГЕНТЫ — ПОЛНЫЙ КОД
7.1 Файл: agents/search_agent.py
Python

from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
from tools.huggingface_tools import search_huggingface, download_hf_dataset
from tools.kaggle_tools import search_kaggle, download_kaggle_dataset


def create_search_agent(model, config):
    """Создаёт Search Agent — ищет источники данных."""
    return CodeAgent(
        name="search_agent",
        description=(
            "This agent searches for datasets and data sources. "
            "Give it a search task like 'find image datasets of swans on "
            "HuggingFace and Kaggle' or 'search the web for CSV files about "
            "apartment rental prices'. It will search DuckDuckGo, visit pages, "
            "search HuggingFace Hub and Kaggle, and return a report of what "
            "it found with URLs and relevance assessments."
        ),
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            search_huggingface,
            download_hf_dataset,
            search_kaggle,
            download_kaggle_dataset,
        ],
        model=model,
        max_steps=config.max_steps_per_agent,
        additional_authorized_imports=config.authorized_imports,
        verbosity_level=2,
    )
7.2 Файл: agents/parser_agent.py
Python

from smolagents import CodeAgent
from tools.web_tools import (
    fetch_page,
    extract_table_from_html,
    extract_links_from_page,
    download_file,
)
from tools.storage_tools import save_dataset, save_metadata


def create_parser_agent(model, config):
    """Создаёт Parser Agent — извлекает данные с веб-страниц."""
    return CodeAgent(
        name="parser_agent",
        description=(
            "This agent extracts and saves data from specific web pages. "
            "Give it a URL and instructions like 'extract the table of "
            "rental prices from this page and save as CSV' or 'download "
            "all CSV links from this page'. It can handle pagination by "
            "following next-page links. It saves results to the data/ directory."
        ),
        tools=[
            fetch_page,
            extract_table_from_html,
            extract_links_from_page,
            download_file,
            save_dataset,
            save_metadata,
        ],
        model=model,
        max_steps=config.max_steps_per_agent,
        additional_authorized_imports=config.authorized_imports,
        verbosity_level=2,
    )
7.3 Файл: agents/image_agent.py
Python

from smolagents import CodeAgent
from tools.image_tools import search_and_download_images
from tools.storage_tools import save_metadata


def create_image_agent(model, config):
    """Создаёт Image Agent — ищет и скачивает изображения."""
    return CodeAgent(
        name="image_agent",
        description=(
            "This agent searches and downloads images using Yandex Images. "
            "Give it a task like 'download 500 photos of whooper swan and "
            "save to data/swans/whooper_swan'. It handles the full pipeline: "
            "search, download, and save with metadata."
        ),
        tools=[
            search_and_download_images,
            save_metadata,
        ],
        model=model,
        max_steps=config.max_steps_per_agent,
        additional_authorized_imports=config.authorized_imports,
        verbosity_level=2,
    )
7.4 Файл: agents/orchestrator.py
Python

from smolagents import CodeAgent, OpenAIModel
from config import AgentConfig
from agents.search_agent import create_search_agent
from agents.parser_agent import create_parser_agent
from agents.image_agent import create_image_agent


ORCHESTRATOR_INSTRUCTIONS = """You are a Dataset Collection Orchestrator.
Your job is to help the user collect a complete, well-organized dataset.

## Your workflow:

### Step 1: Understand the request
- Analyze the user's query
- If the request is ambiguous, ask clarifying questions (up to {max_clarifications})
- Questions should cover: data type, volume, classes/categories, preferred sources
- Ask ALL clarifying questions in ONE message as a numbered list
- If the request is already clear, skip clarification

### Step 2: Plan the collection
Think through and log your plan:
- What type of data? (images / text / table)
- What sources to try? (HuggingFace, Kaggle, web, Yandex Images)
- What search queries to use?
- What folder structure to create?

### Step 3: Delegate to specialized agents
Use your managed agents:
- **search_agent**: for searching HuggingFace, Kaggle, and the web (DuckDuckGo).
  It can also download HF and Kaggle datasets directly.
- **parser_agent**: for extracting data from specific web pages (tables, text, files).
  It saves results to disk.
- **image_agent**: for downloading images from Yandex Images.

### Step 4: Report results
After all agents finish, summarize:
- What was collected (sources, counts)
- Where it was saved (paths)
- Any issues or missing data

## Rules:
- ALWAYS log your reasoning at each step
- Save each source in a SEPARATE subdirectory under data/<dataset_name>/
- Ensure metadata.json is saved for each dataset
- If a source has no relevant data, skip it and note why
- Prefer HuggingFace and Kaggle over raw web scraping when possible
"""


def create_orchestrator(config: AgentConfig):
    """Creates the full multi-agent orchestrator system."""

    model = OpenAIModel(
        model_id=config.model_id,
        temperature=config.temperature,
    )

    search_agent = create_search_agent(model, config)
    parser_agent = create_parser_agent(model, config)
    image_agent = create_image_agent(model, config)

    orchestrator = CodeAgent(
        name="orchestrator",
        description="Main orchestrator that coordinates dataset collection.",
        tools=[],
        model=model,
        managed_agents=[search_agent, parser_agent, image_agent],
        max_steps=30,
        additional_authorized_imports=config.authorized_imports,
        instructions=ORCHESTRATOR_INSTRUCTIONS.format(
            max_clarifications=config.max_clarifications
        ),
        planning_interval=3,
        verbosity_level=2,
    )

    return orchestrator
8. ТОЧКА ВХОДА
8.1 Файл: main.py
Python

import argparse
import os
import json
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from config import AgentConfig
from agents.orchestrator import create_orchestrator


def setup_logging(config: AgentConfig, query: str):
    """Creates a log directory for this run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query[:40])
    log_dir = os.path.join(config.logs_dir, f"{timestamp}_{safe_query}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_agent_logs(agent, log_dir: str):
    """Saves the agent's step-by-step logs to a JSONL file."""
    log_path = os.path.join(log_dir, "agent_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for i, step in enumerate(agent.logs):
            entry = {"step": i, "data": str(step)}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n📝 Logs saved to {log_path}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Dataset Collection Agent — collects datasets from multiple sources"
    )
    parser.add_argument("query", type=str, help="Describe the dataset you want to collect")
    parser.add_argument("--max-clarifications", type=int, default=5,
                        help="Max number of clarifying questions (default: 5)")
    parser.add_argument("--max-results", type=int, default=10,
                        help="Max search results per source (default: 10)")
    parser.add_argument("--no-clarify", action="store_true",
                        help="Skip clarifying questions")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Base directory for saved datasets")

    args = parser.parse_args()

    config = AgentConfig(
        max_clarifications=0 if args.no_clarify else args.max_clarifications,
        max_search_results=args.max_results,
        data_dir=args.data_dir,
    )

    print(f"🚀 Starting Dataset Collection Agent")
    print(f"📋 Query: {args.query}")
    print(f"⚙️  Model: {config.model_id}")
    print("-" * 60)

    log_dir = setup_logging(config, args.query)
    orchestrator = create_orchestrator(config)

    try:
        result = orchestrator.run(args.query)
        print("\n" + "=" * 60)
        print("✅ RESULT:")
        print("=" * 60)
        print(result)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        save_agent_logs(orchestrator, log_dir)


if __name__ == "__main__":
    main()
8.2 Использование CLI
Bash

# Стандартный запрос
python main.py "Собери датасет из лебедей 3 видов: кликун, шипун и малый"

# Без уточняющих вопросов
python main.py "Найди табличные данные по аренде жилья" --no-clarify

# С ограничением вопросов
python main.py "Собери датасет для sentiment analysis" --max-clarifications 3
9. ФОРМАТ МЕТАДАННЫХ
Каждый собранный датасет должен иметь metadata.json в своём каталоге. Агент должен вызывать save_metadata для каждого источника.

Пример: data/swan_species/metadata.json
JSON

{
  "dataset_name": "swan_species",
  "user_query": "Собери датасет из лебедей 3 видов: кликун, шипун, малый",
  "created_at": "2026-03-13T15:30:00+00:00",
  "target_task": "classification",
  "data_type": "images",
  "sources": [
    {
      "name": "yandex_whooper_swan",
      "type": "yandex_images",
      "query": "лебедь кликун фото",
      "num_records": 500,
      "license": "unknown",
      "save_path": "data/swan_species/whooper_swan/"
    },
    {
      "name": "hf_bird_species",
      "type": "huggingface",
      "dataset_id": "chriamue/bird-species-dataset",
      "num_records": 200,
      "license": "CC-BY-4.0",
      "save_path": "data/swan_species/hf_bird_species/"
    }
  ],
  "total_records": 700
}
10. ПАЙПЛАЙН ВЫПОЛНЕНИЯ
Для каждого запроса агент выполняет строго такую последовательность:

text

ПОЛЬЗОВАТЕЛЬ → CLI → main.py
                        │
                        ▼
              Orchestrator.run(query)
                        │
            ┌───────────┴───────────┐
            ▼                       │
   [Нужны уточнения?]              │
      Да → ask_user ──→ ответ ─────┘
      Нет ↓
            ▼
   [Планирование: тип данных,
    источники, запросы, папки]
            │
            ▼
   [Делегирование search_agent]
    "Найди датасеты по запросу X
     на HF, Kaggle и в интернете"
            │
            ▼
   [Оценка результатов поиска]
    Для каждого найденного источника:
            │
     ┌──────┼──────┬──────────────┐
     ▼      ▼      ▼              ▼
   HF?   Kaggle? Веб-таблица?  Картинки?
     │      │      │              │
     ▼      ▼      ▼              ▼
  search  search  parser       image
  _agent  _agent  _agent       _agent
  (download) (download) (extract+save) (download)
     │      │      │              │
     └──────┴──────┴──────────────┘
                   │
                   ▼
          [save_metadata для каждого]
                   │
                   ▼
          [Финальный отчёт пользователю]
11. ПОРЯДОК РЕАЛИЗАЦИИ (ЧЕКЛИСТ)
Фаза 1: Фундамент
 Создать структуру проекта (все папки и __init__.py)
 Написать config.py (скопировать из секции 4)
 Написать requirements.txt (скопировать из секции 3)
 Создать .env с API-ключами
 pip install -r requirements.txt
Фаза 2: Инструменты
 Реализовать tools/storage_tools.py (секция 6.1)
 Реализовать tools/huggingface_tools.py (секция 6.2)
 Реализовать tools/kaggle_tools.py (секция 6.3)
 Реализовать tools/web_tools.py (секция 6.4)
 Реализовать tools/image_tools.py (секция 6.5)
 Скопировать parsers/yandex_images.py (секция 6.6)
 Протестировать каждый tool вручную
Фаза 3: Агенты
 Реализовать agents/search_agent.py (секция 7.1)
 Реализовать agents/parser_agent.py (секция 7.2)
 Реализовать agents/image_agent.py (секция 7.3)
 Реализовать agents/orchestrator.py (секция 7.4)
 Реализовать main.py (секция 8.1)