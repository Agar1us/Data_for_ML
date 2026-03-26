from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AgentConfig:
    # LLM
    model_id: str = "gpt-5-mini"
    temperature: float = 1
    max_tokens: int = 16000

    # Agent behavior
    max_search_results: int = 10
    max_steps_per_agent: int = 20

    # Data
    data_dir: str = "data/current_run/collection"
    logs_dir: str = "data/current_run/logs"
    artifacts_dir: str = "data/current_run/collection_artifacts"
    max_dataset_size_gb: float = 5.0

    # Parsing
    yandex_headless: bool = True
    yandex_profile_dir: str = ""
    yandex_manual_captcha_timeout: float = 0.0

    def resolve_dir(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    @property
    def data_root(self) -> Path:
        return self.resolve_dir(self.data_dir)

    @property
    def logs_root(self) -> Path:
        return self.resolve_dir(self.logs_dir)

    @property
    def artifacts_root(self) -> Path:
        return self.resolve_dir(self.artifacts_dir)
