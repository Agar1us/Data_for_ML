from __future__ import annotations

import inspect
from typing import Any


def _supported_kwargs(factory: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(factory.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if accepts_var_kwargs or key in signature.parameters:
            supported[key] = value
    return supported


def create_openai_model(config: Any) -> Any:
    from smolagents import OpenAIModel

    kwargs = _supported_kwargs(
        OpenAIModel,
        {
            "model_id": getattr(config, "model_id", None),
            "temperature": getattr(config, "temperature", None),
            "max_tokens": getattr(config, "max_tokens", None),
        },
    )
    return OpenAIModel(**kwargs)


def create_toolcalling_agent(
    *,
    name: str,
    description: str,
    model: Any,
    tools: list[Any],
    config: Any,
    managed_agents: list[Any] | None = None,
    instructions: str | None = None,
    max_steps: int | None = None,
    planning_interval: int | None = None,
    verbosity_level: int = 2,
) -> Any:
    from smolagents import ToolCallingAgent

    kwargs = {
        "name": name,
        "description": description,
        "tools": tools,
        "model": model,
        "managed_agents": managed_agents,
        "max_steps": max_steps if max_steps is not None else getattr(config, "max_steps_per_agent", None),
        "planning_interval": planning_interval,
        "verbosity_level": verbosity_level,
        "instructions": instructions,
    }
    return ToolCallingAgent(**_supported_kwargs(ToolCallingAgent, kwargs))
