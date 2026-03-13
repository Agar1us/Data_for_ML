from __future__ import annotations

from agents import create_toolcalling_agent
from tools.huggingface_tools import download_hf_dataset, search_huggingface
from tools.kaggle_tools import download_kaggle_dataset, search_kaggle
from tools.storage_tools import save_dataset, save_metadata, write_text_artifact
from tools.web_tools import (
    download_file,
    extract_links_from_page,
    extract_table_from_html,
    fetch_page,
)


PARSER_AGENT_INSTRUCTIONS = """You extract data from known webpages and direct file URLs.

Rules:
- Use only the provided tools.
- If a site-specific tool already exists for the user's requested source, use that tool path instead of writing a custom parser.
- If the task targets Hugging Face, use search_huggingface or download_hf_dataset directly instead of webpage parsing.
- If the task targets Kaggle, use search_kaggle or download_kaggle_dataset directly instead of webpage parsing.
- If the user already provided a concrete URL or domain to inspect, work on that source directly instead of asking for web search.
- If there is no dedicated tool for the target site, you may write custom parser code inline in your code steps using authorized imports such as requests, bs4, pandas, re, json, and urllib.
- Prefer inline parser code over standalone script files.
- If you need to persist generated parser code for traceability or reuse, save it with write_text_artifact under collection_artifacts/scripts/.
- Never use raw open() inside generated agent code to create files; use save_dataset, save_metadata, download_file, or write_text_artifact instead.
- Use save_dataset and save_metadata for persisted outputs when appropriate.
"""


def create_parser_agent(model, config):
    """Create the agent responsible for extracting data from webpages."""
    return create_toolcalling_agent(
        name="parser_agent",
        description=(
            "This agent extracts and saves data from specific web pages. It can "
            "also use dedicated Hugging Face and Kaggle tools when a task is "
            "mistakenly delegated here for those sources."
        ),
        tools=[
            search_huggingface,
            download_hf_dataset,
            search_kaggle,
            download_kaggle_dataset,
            fetch_page,
            extract_table_from_html,
            extract_links_from_page,
            download_file,
            save_dataset,
            save_metadata,
            write_text_artifact,
        ],
        model=model,
        config=config,
        instructions=PARSER_AGENT_INSTRUCTIONS,
    )
