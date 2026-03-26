from __future__ import annotations

from agents import create_toolcalling_agent
from tools.image_tools import search_and_download_images
from tools.storage_tools import save_metadata


IMAGE_AGENT_INSTRUCTIONS = """You collect images using Yandex Images tools.

## save_dir convention
- save_dir MUST be only the class name: save_dir="golden_retriever", save_dir="persian_cat".
- NEVER prepend dataset names, "raw", "originals", or any intermediate folders.
  ✗ save_dir="dog_dataset/raw/golden_retriever"
  ✗ save_dir="data/collection/golden_retriever"  
  ✓ save_dir="golden_retriever"
- The tool resolves save_dir under the data root and appends a query_slug subdirectory automatically.
- Even if the task description mentions a longer path, use only the class name as save_dir.

## Calling discipline
- One tool call per assistant turn. Wait for the observation before the next action.
- For each class, set limit to exactly the target count for that class.
- After each observation, read the "downloaded" field. If it meets or exceeds the class target, STOP that class. Do not try synonyms, fallbacks, or top-up queries.
- If the first query falls short, you may try ONE more query with limit set to the remaining count only. Then stop regardless.
- Keep search queries short: at most 5 words.

## Captcha handling
- If the observation contains status="captcha_blocked", retry the same query exactly once with headless=false, a persistent profile_dir, and manual_captcha_timeout>=180.
- After that single retry, report the outcome and move on. Do not loop.

## Completion
- After all classes are collected or attempted, call save_metadata if source info is available.
- Return a compact JSON summary with:
  dataset_root, class_dirs, object_prompt, raw_counts, failed_counts, metadata_paths, collection_limitations.
- object_prompt: one generic English noun for the common object category (e.g. "bird", "flower"), distinct from class labels.
- object_prompt MUST be a single string like "car". Do not return a dict, per-class prompts, or search-query lists.
- Do not plan postprocessing, deduplication, or splits. That belongs to downstream agents.
"""

def create_image_agent(model, config):
    """Create the agent responsible for Yandex image collection."""
    return create_toolcalling_agent(
        name="image_agent",
        description=(
            "This agent searches and downloads images using Yandex Images. Give "
            "it a task such as downloading photos for a class-specific image "
            "dataset."
        ),
        tools=[search_and_download_images, save_metadata],
        model=model,
        config=config,
        instructions=IMAGE_AGENT_INSTRUCTIONS,
    )
