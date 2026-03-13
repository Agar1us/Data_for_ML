from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class AgentConfig:
    # LLM
    model_id: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 16000

    # Agent behavior
    max_clarifications: int = 5
    max_search_results: int = 10
    max_steps_per_agent: int = 20

    # Data
    data_dir: str = "data"
    logs_dir: str = "logs"
    artifacts_dir: str = "collection_artifacts"
    max_dataset_size_gb: float = 5.0
    max_images_per_query: int = 10000
    max_records_per_query: int = 10000

    # Parsing
    request_delay_sec: float = 2.0
    request_timeout_sec: float = 30.0
    yandex_parser_delay: float = 6.0
    yandex_headless: bool = True

    # Imports allowed for CodeAgent
    authorized_imports: list[str] = field(
        default_factory=lambda: [
            "requests",
            "bs4",
            "json",
            "csv",
            "os",
            "pathlib",
            "pandas",
            "re",
            "time",
            "urllib",
            "hashlib",
            "datetime",
        ]
    )

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
