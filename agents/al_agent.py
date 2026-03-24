from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents import create_openai_model, create_toolcalling_agent
from agents.al_tools import (
    image_classification_active_learning,
    image_classification_active_learning_impl,
    table_classification_active_learning,
    table_classification_active_learning_impl,
)


AL_AGENT_INSTRUCTIONS = """You orchestrate active-learning experiments.

Rules:
- Choose the tool that matches the requested modality.
- For image classification, prefer image_classification_active_learning.
- For table classification, return the current stub response via table_classification_active_learning.
- Do not invent metrics or artifact paths; rely on tool output only.
"""


@dataclass
class ALAgentConfig:
    model_id: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000
    max_steps_per_agent: int = 4


def _parse_config_json(config_json: str = "") -> dict[str, Any]:
    if not config_json:
        return {}
    candidate = Path(config_json)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(config_json)


def create_al_agent(model: Any | None = None, config: ALAgentConfig | None = None) -> Any:
    resolved_config = config or ALAgentConfig()
    resolved_model = model or create_openai_model(resolved_config)
    return create_toolcalling_agent(
        name="al_agent",
        description="Active-learning agent for modality-specific sample selection and reporting.",
        model=resolved_model,
        tools=[image_classification_active_learning, table_classification_active_learning],
        config=resolved_config,
        instructions=AL_AGENT_INSTRUCTIONS,
        verbosity_level=2,
    )


def active_learning_op(
    task_description: str,
    labeled_data_path: str,
    modality: str = "image",
    config_json: str = "",
) -> dict[str, Any]:
    resolved_config = _parse_config_json(config_json)
    normalized_modality = modality.strip().lower()
    if normalized_modality == "image":
        return image_classification_active_learning_impl(
            task_description=task_description,
            labeled_data_path=labeled_data_path,
            config=resolved_config,
        )
    if normalized_modality == "table":
        return table_classification_active_learning_impl(
            task_description=task_description,
            labeled_data_path=labeled_data_path,
            config=resolved_config,
        )
    raise ValueError(f"Unsupported modality for active_learning_op: {modality}")
