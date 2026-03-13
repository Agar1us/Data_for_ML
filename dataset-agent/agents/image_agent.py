from __future__ import annotations

from agents import create_toolcalling_agent
from tools.image_tools import search_and_download_images
from tools.storage_tools import save_metadata


IMAGE_AGENT_INSTRUCTIONS = """You collect images with the provided Yandex Images toolchain.

Rules:
- Use only the provided tools.
- Do not write custom scraping code when the Yandex Images tool already covers the request.
- If the user explicitly requests Yandex Images, work directly with search_and_download_images.
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
