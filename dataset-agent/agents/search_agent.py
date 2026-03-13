from __future__ import annotations

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

from agents import create_toolcalling_agent
from tools.huggingface_tools import download_hf_dataset, search_huggingface
from tools.kaggle_tools import download_kaggle_dataset, search_kaggle


SEARCH_AGENT_INSTRUCTIONS = """You search for datasets and data sources.

Rules:
- Use only the provided tools.
- Do not write custom parsers or helper scripts when a source-specific tool already exists.
- For Hugging Face requests, use search_huggingface and download_hf_dataset directly.
- For Kaggle requests, use search_kaggle and download_kaggle_dataset directly.
- Use DuckDuckGoSearchTool and VisitWebpageTool only for open web discovery when the source is not already specified or when source-specific tools are insufficient.
- If the user names allowed sources, stay within that source list unless asked to broaden the search.
"""


def create_search_agent(model, config):
    """Create the agent responsible for finding candidate data sources."""
    return create_toolcalling_agent(
        name="search_agent",
        description=(
            "This agent searches for datasets and data sources. Give it tasks "
            "such as finding image datasets on Hugging Face and Kaggle or "
            "searching the web for CSV files."
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
        config=config,
        instructions=SEARCH_AGENT_INSTRUCTIONS,
    )
