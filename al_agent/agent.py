from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from al_agent import create_openai_model, create_toolcalling_agent
from al_agent.al_tools import (
    image_detection_active_learning,
    image_detection_active_learning_impl,
    table_classification_active_learning,
    table_classification_active_learning_impl,
)


AL_AGENT_INSTRUCTIONS = """You orchestrate active-learning experiments.

Rules:
- Choose the tool that matches the requested modality.
- For image object detection, prefer image_detection_active_learning.
- For table classification, return the current stub response via table_classification_active_learning.
- Do not invent metrics or artifact paths; rely on tool output only.
- Final answer must be raw JSON only, with no prose or markdown fences.
"""

IMAGE_AL_TASK_TEMPLATE = """Run bbox-level active learning for image object detection.

Use exactly one tool call:
- task_description={task_description_json}
- labeled_data_path={labeled_data_path_json}
- config_json={config_json}

Return the tool result only.
"""

TABLE_AL_TASK_TEMPLATE = """Run active learning for table classification.

Use exactly one tool call:
- task_description={task_description_json}
- labeled_data_path={labeled_data_path_json}
- config_json={config_json}

Return the tool result only.
"""


@dataclass
class ALAgentConfig:
    model_id: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000
    max_steps_per_agent: int = 4


class ALAgent:
    def __init__(self, model: Any | None = None, config: ALAgentConfig | None = None) -> None:
        self.config = config or ALAgentConfig()
        self.agent = create_al_agent(model=model, config=self.config) if self._llm_available() else None
        self.last_result: dict[str, Any] | None = None

    @property
    def memory(self) -> Any:
        return getattr(self.agent, "memory", None)

    @property
    def name(self) -> str:
        return "al_agent"

    def run(
        self,
        *,
        task_description: str,
        labeled_data_path: str,
        modality: str = "image",
        config_json: str = "",
    ) -> dict[str, Any]:
        normalized_modality = modality.strip().lower()
        if normalized_modality == "image":
            if self.agent is None:
                result = image_detection_active_learning_impl(
                    task_description=task_description,
                    labeled_data_path=labeled_data_path,
                    config=_parse_config_json(config_json),
                )
            else:
                task_prompt = IMAGE_AL_TASK_TEMPLATE.format(
                    task_description_json=json.dumps(task_description, ensure_ascii=False),
                    labeled_data_path_json=json.dumps(labeled_data_path, ensure_ascii=False),
                    config_json=json.dumps(config_json, ensure_ascii=False),
                )
                result = self._normalize_tool_result(self.agent.run(task=task_prompt))
        elif normalized_modality == "table":
            if self.agent is None:
                result = table_classification_active_learning_impl(
                    task_description=task_description,
                    labeled_data_path=labeled_data_path,
                    config=_parse_config_json(config_json),
                )
            else:
                task_prompt = TABLE_AL_TASK_TEMPLATE.format(
                    task_description_json=json.dumps(task_description, ensure_ascii=False),
                    labeled_data_path_json=json.dumps(labeled_data_path, ensure_ascii=False),
                    config_json=json.dumps(config_json, ensure_ascii=False),
                )
                result = self._normalize_tool_result(self.agent.run(task=task_prompt))
        else:
            raise ValueError(f"Unsupported modality for active_learning_op: {modality}")
        self.last_result = result
        return result

    @staticmethod
    def _normalize_tool_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            if "answer" in result and len(result) == 1:
                return ALAgent._normalize_tool_result(result["answer"])
            payload = result
        elif isinstance(result, str):
            payload = ALAgent._parse_jsonish_result(result)
        else:
            raise TypeError(f"Unexpected AL agent result type: {type(result).__name__}")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict AL result, got: {type(payload).__name__}")
        if "answer" in payload and len(payload) == 1:
            return ALAgent._normalize_tool_result(payload["answer"])
        if payload.get("success") is False:
            raise RuntimeError(str(payload.get("error") or "AL agent tool call failed."))
        if payload.get("success") is True:
            payload = {key: value for key, value in payload.items() if key != "success"}
        return payload

    @staticmethod
    def _parse_jsonish_result(result: str) -> dict[str, Any]:
        text = result.strip()
        if not text:
            raise RuntimeError("AL agent returned an empty result.")
        candidates = [text]
        fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, str):
                nested_text = parsed.strip()
                if nested_text.startswith("{") and nested_text.endswith("}"):
                    return ALAgent._parse_jsonish_result(nested_text)
        preview = text[:240].replace("\n", "\\n")
        raise RuntimeError(f"AL agent returned a non-JSON final answer: {preview}")

    @staticmethod
    def _llm_available() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


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
        tools=[image_detection_active_learning, table_classification_active_learning],
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
    return ALAgent().run(
        task_description=task_description,
        labeled_data_path=labeled_data_path,
        modality=modality,
        config_json=config_json,
    )
