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


def _openai_token_kwargs(config: Any) -> dict[str, Any]:
    max_tokens = getattr(config, "max_tokens", None)
    if max_tokens is None:
        return {}
    model_id = str(getattr(config, "model_id", "") or "").strip().lower()
    if model_id.startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


class _SerialToolOpenAIModelMixin:
    def _prepare_completion_kwargs(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        completion_kwargs = super()._prepare_completion_kwargs(*args, **kwargs)
        if "tools" in completion_kwargs:
            completion_kwargs["parallel_tool_calls"] = False
        else:
            completion_kwargs.pop("parallel_tool_calls", None)
        return completion_kwargs


def create_openai_model(config: Any) -> Any:
    from smolagents import OpenAIModel

    class DatasetOpenAIModel(_SerialToolOpenAIModelMixin, OpenAIModel):
        pass

    kwargs = _supported_kwargs(
        DatasetOpenAIModel,
        {
            "model_id": config.model_id,
            "temperature": config.temperature,
            **_openai_token_kwargs(config),
        },
    )
    return DatasetOpenAIModel(**kwargs)


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
    max_tool_threads: int = 1,
) -> Any:
    from smolagents import ToolCallingAgent

    kwargs = {
        "name": name,
        "description": description,
        "tools": tools,
        "model": model,
        "managed_agents": managed_agents,
        "max_steps": max_steps if max_steps is not None else config.max_steps_per_agent,
        "planning_interval": planning_interval,
        "verbosity_level": verbosity_level,
        "instructions": instructions,
        "max_tool_threads": max_tool_threads,
    }
    return ToolCallingAgent(**_supported_kwargs(ToolCallingAgent, kwargs))
