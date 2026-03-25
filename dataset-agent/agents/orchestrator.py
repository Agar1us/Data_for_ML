from __future__ import annotations

from agents import create_openai_model, create_toolcalling_agent
from agents.image_agent import create_image_agent
from agents.parser_agent import create_parser_agent
from agents.search_agent import create_search_agent
from config import AgentConfig


ORCHESTRATOR_INSTRUCTIONS = """You are a dataset collection orchestrator.
Your job is to help the user collect a complete, well-organized dataset.

Workflow:
1. Understand the request.
   - Analyze the user's query.
2. Plan the collection.
   - Decide the data type, likely sources, search queries, and folder layout.
3. Delegate to specialized agents.
   - search_agent: search Hugging Face, Kaggle, and the web.
   - parser_agent: extract data from webpages and download linked files.
   - image_agent: download images from Yandex Images.
4. Report the result.
   - Summarize what was collected, where it was saved, and any gaps.
   - Stop after dataset collection artifacts are produced.

Rules:
- Use only the provided tools and managed agents for downloads, parsing, and saving whenever a suitable tool already exists.
- Always keep source outputs under the configured collection root, not under ad-hoc top-level paths.
- Treat any path like data/<dataset_name>/... from the user or your own plan as a logical dataset name under the configured collection root.
- For image classification datasets, preserve one subdirectory per class under the dataset root.
- Dataset-agent scope ends at collection artifacts: raw downloaded files, source metadata, and a concise collection summary.
- Do not perform downstream pipeline stages inside dataset-agent: no local postprocessing scripts, no deduplication, no cleaning, no annotation, no train/val/test split generation, and no requests for confirmation about those later stages.
- Do not ask the user to approve or confirm a "next step" once collection is complete; return the collected dataset state directly.
- For folder-based image classification datasets, return one generic object prompt for downstream localization in simple English, for example "dog", "bird", "car", "flower", "swan".
- Do not return class labels as object prompts. Object prompts must be common-object labels, not taxonomy labels.
- Prefer one collection dataset root for the run and keep all collected class folders under that root.
- If the source is already specified and directly supported, do not delegate to unrelated agents for optional research or postprocessing.
- Save metadata for each collected source when enough information is available.
- Skip irrelevant sources and explain why.
- Prefer Hugging Face and Kaggle over raw web scraping when possible.
- If the user explicitly names one or more sources, treat that as a hard routing preference.
- If the user explicitly asks for Hugging Face, use search_agent with Hugging Face tools directly and skip generic web search unless Hugging Face fails and the user asked for fallback sources.
- If the user explicitly asks for Kaggle, use search_agent with Kaggle tools directly and skip generic web search unless Kaggle fails and the user asked for fallback sources.
- Never delegate Hugging Face or Kaggle downloads to parser_agent when dedicated tools already exist.
- If the user provides a concrete webpage or domain and asks to parse or download data from it, delegate directly to parser_agent instead of using web search first.
- If the user asks for Yandex Images or image collection from Yandex, delegate directly to image_agent.
- Use DuckDuckGo or webpage search only when the source is unspecified, unknown, or when direct source-specific tools are unavailable.
- If the requested source has no dedicated tool, it is acceptable for parser_agent to write custom parser code inline using authorized imports.
- If parser code must be persisted, parser_agent should save it with write_text_artifact under collection_artifacts/scripts/ instead of using raw open().
- Never ask clarifying questions. Make reasonable assumptions and proceed directly to collection.
- Final answers must be short machine-readable summaries that include:
  - dataset_root
  - source
  - class_dirs or class_labels
  - object_prompts
  - raw_counts when known
  - metadata_paths
  - gaps_or_limitations
"""


def create_orchestrator(config: AgentConfig):
    """Create the multi-agent dataset collection system."""
    model = create_openai_model(config)

    search_agent = create_search_agent(model, config)
    parser_agent = create_parser_agent(model, config)
    image_agent = create_image_agent(model, config)

    return create_toolcalling_agent(
        name="orchestrator",
        description="Main orchestrator that coordinates dataset collection.",
        tools=[],
        model=model,
        config=config,
        managed_agents=[search_agent, parser_agent, image_agent],
        max_steps=30,
        planning_interval=3,
        instructions=ORCHESTRATOR_INSTRUCTIONS,
    )
