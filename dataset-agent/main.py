from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from config import AgentConfig, PROJECT_ROOT


def setup_logging(config: AgentConfig, query: str) -> str:
    """Create a log directory for this run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(char if char.isalnum() else "_" for char in query[:40]).strip("_")
    safe_query = safe_query or "request"
    log_dir = config.logs_root / f"{timestamp}_{safe_query}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def _write_jsonl(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _memory_records(agent) -> list[dict]:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return []

    records: list[dict] = []
    system_prompt = getattr(getattr(memory, "system_prompt", None), "system_prompt", "")
    if system_prompt:
        records.append(
            {
                "record_type": "system_prompt",
                "agent_name": getattr(agent, "name", None) or getattr(agent, "agent_name", type(agent).__name__),
                "data": system_prompt,
            }
        )

    if hasattr(memory, "get_full_steps"):
        steps = memory.get_full_steps()
    else:
        raw_steps = getattr(memory, "steps", [])
        steps = [getattr(step, "dict", lambda: {"data": str(step)})() for step in raw_steps]

    for index, step in enumerate(steps):
        records.append(
            {
                "record_type": "memory_step",
                "step": index,
                "agent_name": getattr(agent, "name", None) or getattr(agent, "agent_name", type(agent).__name__),
                "data": step,
            }
        )
    return records


def save_agent_logs(agent, log_dir: str) -> None:
    """Persist the agent step log, when available."""
    log_root = Path(log_dir)
    orchestrator_log_path = log_root / "agent_log.jsonl"
    orchestrator_records = _memory_records(agent)
    _write_jsonl(orchestrator_log_path, orchestrator_records)
    print(f"Logs saved to {orchestrator_log_path}")

    managed_agents = getattr(agent, "managed_agents", {}) or {}
    for name, managed_agent in managed_agents.items():
        managed_log_path = log_root / f"{name}_log.jsonl"
        managed_records = _memory_records(managed_agent)
        _write_jsonl(managed_log_path, managed_records)
        print(f"Managed agent logs saved to {managed_log_path}")


def ensure_dependencies() -> None:
    """Raise a clear error if required runtime dependencies are missing."""
    try:
        import smolagents  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "smolagents is not installed. Install the dependencies from "
            f"{PROJECT_ROOT / 'requirements.txt'} before running the agent."
        ) from exc


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Dataset Collection Agent: collect datasets from multiple sources."
    )
    parser.add_argument("query", type=str, help="Describe the dataset to collect.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum search results per source.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/current_run/collection",
        help="Directory for saved datasets, relative to the repository root by default.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="data/current_run/logs",
        help="Directory for dataset-agent logs, relative to the repository root by default.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/current_run/collection_artifacts",
        help="Directory for dataset-agent helper artifacts, relative to the repository root by default.",
    )
    args = parser.parse_args()

    config = AgentConfig(
        max_search_results=args.max_results,
        data_dir=args.data_dir,
        logs_dir=args.logs_dir,
        artifacts_dir=args.artifacts_dir,
    )

    config.data_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    os.environ["DATASET_AGENT_DATA_DIR"] = str(config.data_root)
    os.environ["DATASET_AGENT_LOGS_DIR"] = str(config.logs_root)
    os.environ["DATASET_AGENT_ARTIFACTS_DIR"] = str(config.artifacts_root)
    os.environ["DATASET_AGENT_YANDEX_HEADLESS"] = "1" if config.yandex_headless else "0"
    os.environ["DATASET_AGENT_MANUAL_CAPTCHA_TIMEOUT"] = str(config.yandex_manual_captcha_timeout)
    if config.yandex_profile_dir:
        os.environ["DATASET_AGENT_CHROME_PROFILE_DIR"] = config.yandex_profile_dir

    try:
        ensure_dependencies()
        from agents.orchestrator import create_orchestrator
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Starting Dataset Collection Agent")
    print(f"Query: {args.query}")
    print(f"Model: {config.model_id}")
    print("-" * 60)

    log_dir = setup_logging(config, args.query)
    orchestrator = create_orchestrator(config)

    try:
        result = orchestrator.run(args.query)
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(result)
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
    finally:
        save_agent_logs(orchestrator, log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
