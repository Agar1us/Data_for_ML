from __future__ import annotations

from agents import create_toolcalling_agent
from tools.image_tools import search_and_download_images
from tools.storage_tools import save_metadata


IMAGE_AGENT_INSTRUCTIONS = """You collect images with the provided Yandex Images toolchain.

Rules:
- Use only the provided tools.
- Do not write custom scraping code when the Yandex Images tool already covers the request.
- If the user explicitly requests Yandex Images, work directly with search_and_download_images.
- For class-based image collection, call search_and_download_images at most once per class/query in a run.
- Treat one completed download attempt as final, even if the downloaded count is lower than requested.
- Do not retry the same query just to "top up" missing images unless the user explicitly asks you to retry.
- Prefer separate save_dir subdirectories per class so the final dataset stays organized.
- If the tool returns status="captcha_blocked", retry the same query exactly once in manual mode:
  call search_and_download_images again with headless=false, a persistent profile_dir, and enough manual_captcha_timeout for the user to solve captcha.
- After that one manual retry, do not loop on the same query again. Report whether the retry succeeded or remained blocked.
- Save metadata when enough source information is known.
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
