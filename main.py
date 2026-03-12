from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure custom tools are import-loaded before config resolves definitions.
import dataset_agent.tools  # noqa: F401
from dataset_agent.state import build_manifest, clean_class_list, ensure_dir, extract_classes_from_query
from sgr_agent_core.agent_config import GlobalConfig
from sgr_agent_core.agent_factory import AgentFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGR dataset collection agent")
    parser.add_argument("--query", required=True, help="User query for dataset collection")
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated classes override (e.g. 'mute swan, trumpeter swan')",
    )
    parser.add_argument(
        "--modalities",
        default="image,text",
        help="Comma-separated modalities to search: image,text (default: image,text)",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to SGR config file")
    parser.add_argument("--agent", default="dataset_collection_agent", help="Agent definition name")

    parser.add_argument("--data-dir", default="data", help="Directory for downloaded datasets/assets")
    parser.add_argument(
        "--manifest-path",
        default="reports/collection_manifest.json",
        help="Path to output JSON manifest",
    )

    parser.add_argument("--max-datasets", type=int, default=20, help="Maximum accepted datasets")
    parser.add_argument(
        "--max-web-images-per-class",
        type=int,
        default=300,
        help="Maximum Tavily fallback images per missing class",
    )
    parser.add_argument(
        "--tavily-max-results",
        type=int,
        default=5,
        help="Maximum Tavily search results per missing class query",
    )
    parser.add_argument(
        "--log-file",
        default="",
        help="Optional path to plain-text runtime log file (default: logs/run-<timestamp>.log)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Runtime log level for console/file output",
    )
    return parser.parse_args()


def configure_env_aliases() -> None:
    if os.getenv("OPENAI_API_KEY") and not os.getenv("SGR__LLM__API_KEY"):
        os.environ["SGR__LLM__API_KEY"] = os.environ["OPENAI_API_KEY"]
    if os.getenv("TAVILY_API_KEY") and not os.getenv("SGR__SEARCH__TAVILY_API_KEY"):
        os.environ["SGR__SEARCH__TAVILY_API_KEY"] = os.environ["TAVILY_API_KEY"]


def configure_logging(log_level: str, log_file: str | None = None) -> Path:
    logs_dir = ensure_dir("logs")
    if log_file and log_file.strip():
        log_path = Path(log_file)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"run-{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Keep network/client libraries from flooding the trace.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_path


def resolve_classes(query: str, classes_override: str) -> list[str]:
    if classes_override.strip():
        return clean_class_list([part.strip() for part in classes_override.split(",") if part.strip()])
    return extract_classes_from_query(query)


def resolve_modalities(query: str, modalities_override: str) -> list[str]:
    normalized = [part.strip().lower() for part in modalities_override.split(",") if part.strip()]
    allowed = {"image", "text"}
    requested = [item for item in normalized if item in allowed]
    if requested:
        # Preserve order but deduplicate.
        deduped: list[str] = []
        for item in requested:
            if item not in deduped:
                deduped.append(item)
        return deduped

    lower_query = query.lower()
    has_image = any(term in lower_query for term in ("image", "vision", "photo", "picture"))
    has_text = any(term in lower_query for term in ("text", "nlp", "language", "sentiment"))

    if has_image and not has_text:
        return ["image"]
    if has_text and not has_image:
        return ["text"]
    return ["image", "text"]


def build_runtime_context(args: argparse.Namespace, classes: list[str], modalities: list[str]) -> dict[str, Any]:
    return {
        "query_spec": {
            "query": args.query,
            "modalities": modalities,
            "requested_classes": classes,
        },
        "runtime_config": {
            "max_datasets": max(1, args.max_datasets),
            "max_web_images_per_class": max(1, args.max_web_images_per_class),
            "tavily_max_results": max(1, args.tavily_max_results),
            "modalities": modalities,
            "class_match_policy": "exact_plus_simple_normalization",
            "allow_all_licenses": True,
            "deduplication": "none",
        },
        "candidates": [],
        "decisions": [],
        "selected_datasets": [],
        "web_collected_assets": [],
        "covered_classes": [],
        "missing_classes": classes.copy(),
        "notes": [],
        "manifest_path": args.manifest_path,
        "data_dir": args.data_dir,
    }


def build_task_message(args: argparse.Namespace, classes: list[str], modalities: list[str]) -> str:
    classes_text = ", ".join(classes) if classes else "(not explicitly provided)"
    modalities_text = ", ".join(modalities)
    return (
        "Collect classification datasets for the query below.\n"
        f"Query: {args.query}\n"
        f"Requested classes: {classes_text}\n"
        f"Requested modalities: {modalities_text}\n"
        "Workflow constraints:\n"
        "1) Search HuggingFace datasets.\n"
        "2) Search Kaggle datasets.\n"
        "3) Select datasets directly from discovered candidates without verification checks.\n"
        "4) Download selected datasets to separate directories under data_dir.\n"
        "5) If image classes are still missing after downloaded datasets and image modality is requested, run Tavily fallback with one query per missing class and collect class images.\n"
        "6) Write final JSON manifest and then call final answer tool.\n"
        "Do not run quality/class verification filtering."
    )


def print_summary(result: str | None, runtime_state: dict[str, Any]) -> None:
    selected = runtime_state.get("selected_datasets", [])
    rejected = runtime_state.get("decisions", [])
    web_assets = runtime_state.get("web_collected_assets", [])
    missing = runtime_state.get("missing_classes", [])
    modalities = runtime_state.get("query_spec", {}).get("modalities", [])

    print("\n=== Dataset Collection Summary ===")
    print(f"Requested modalities: {', '.join(modalities) if modalities else 'auto'}")
    print(f"Selected datasets: {len(selected)}")
    print(f"Rejected datasets: {len(rejected)}")
    print(f"Web fallback assets: {len(web_assets)}")
    print(f"Missing classes: {', '.join(missing) if missing else 'none'}")
    print(f"Manifest path: {runtime_state.get('manifest_path')}")
    if result:
        print("\nFinal agent answer:")
        print(result)


def ensure_manifest(runtime_state: dict[str, Any]) -> None:
    manifest_path = Path(runtime_state.get("manifest_path", "reports/collection_manifest.json"))
    if manifest_path.exists():
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(runtime_state)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def validate_required_env() -> None:
    if not os.getenv("SGR__LLM__API_KEY"):
        raise RuntimeError("Missing API key: set OPENAI_API_KEY or SGR__LLM__API_KEY")


async def run_agent(args: argparse.Namespace) -> int:
    logger = logging.getLogger("dataset_agent.main")
    configure_env_aliases()
    validate_required_env()

    classes = resolve_classes(args.query, args.classes)
    modalities = resolve_modalities(args.query, args.modalities)
    logger.info("Parsed requested classes: %s", classes if classes else "none")
    logger.info("Parsed requested modalities: %s", modalities)

    ensure_dir(args.data_dir)
    ensure_dir(Path(args.manifest_path).parent)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = GlobalConfig.from_yaml(str(config_path))
    if args.agent not in config.agents:
        available = ", ".join(sorted(config.agents.keys()))
        raise ValueError(f"Agent '{args.agent}' not found. Available: {available}")
    logger.info("Loaded config '%s' and selected agent '%s'", config_path, args.agent)

    task_message = build_task_message(args, classes, modalities)
    messages = [{"role": "user", "content": task_message}]

    agent_def = config.agents[args.agent]
    agent = await AgentFactory.create(agent_def=agent_def, task_messages=messages)
    logger.info("Agent instance created: %s", agent.id)

    runtime_state = build_runtime_context(args, classes, modalities)
    runtime_state.setdefault("notes", []).append(
        f"run_started_at={datetime.now(timezone.utc).isoformat()}"
    )
    agent._context.custom_context = runtime_state

    logger.info("Starting agent execution loop")
    result = await agent.execute()
    logger.info("Agent execution completed with state=%s", agent._context.state)
    runtime_state = agent._context.custom_context if isinstance(agent._context.custom_context, dict) else runtime_state
    ensure_manifest(runtime_state)
    logger.info("Manifest ensured at '%s'", runtime_state.get("manifest_path"))
    print_summary(result, runtime_state)
    return 0


def main() -> int:
    args = parse_args()
    log_path = configure_logging(args.log_level, args.log_file)
    logger = logging.getLogger("dataset_agent.main")
    logger.info("Runtime logging initialized. log_file='%s'", log_path)
    try:
        return asyncio.run(run_agent(args))
    finally:
        logger.info("Run log saved to: %s", log_path)
        print(f"Run log file: {log_path}")
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
