from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

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


def save_agent_logs(agent, log_dir: str) -> None:
    """Persist the agent step log, when available."""
    log_path = os.path.join(log_dir, "agent_log.jsonl")
    logs = getattr(agent, "logs", [])
    with open(log_path, "w", encoding="utf-8") as handle:
        for index, step in enumerate(logs):
            entry = {"step": index, "data": str(step)}
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Logs saved to {log_path}")


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
        "--max-clarifications",
        type=int,
        default=5,
        help="Maximum number of clarifying questions.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum search results per source.",
    )
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="Skip clarifying questions.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for saved datasets, relative to the project root by default.",
    )
    args = parser.parse_args()

    config = AgentConfig(
        max_clarifications=0 if args.no_clarify else args.max_clarifications,
        max_search_results=args.max_results,
        data_dir=args.data_dir,
    )

    config.data_root.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    os.environ["DATASET_AGENT_DATA_DIR"] = str(config.data_root)
    os.environ["DATASET_AGENT_LOGS_DIR"] = str(config.logs_root)
    os.environ["DATASET_AGENT_ARTIFACTS_DIR"] = str(config.artifacts_root)

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
