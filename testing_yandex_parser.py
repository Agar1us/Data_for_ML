from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_AGENT_ROOT = PROJECT_ROOT / "dataset-agent"
if str(DATASET_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_AGENT_ROOT))

from parsers.yandex_images import Parser  # noqa: E402
from tools.image_tools import search_and_download_images  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "query": "bewick swan",
    "limit": 10,
    "delay": 4.0,
    "mode": "urls",
    "save_dir": "data/yandex_test",
    "size": "",
    "image_type": "",
    "headless": True,
    "chrome_bin": "",
    "chromedriver": "",
    "profile_dir": "",
    "manual_captcha_timeout": 180.0,
}


def load_config(config_path: Path | None) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if config_path is None:
        return config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a JSON object.")
    config.update(user_config)
    return config


def build_parser(config: dict[str, Any]) -> Parser:
    chrome_bin = str(config.get("chrome_bin") or "").strip()
    chromedriver = str(config.get("chromedriver") or "").strip()
    profile_dir = str(config.get("profile_dir") or "").strip()
    if chrome_bin:
        os.environ["CHROME_BIN"] = chrome_bin
    if chromedriver:
        os.environ["CHROMEDRIVER"] = chromedriver
    if profile_dir:
        os.environ["DATASET_AGENT_CHROME_PROFILE_DIR"] = profile_dir
    return Parser(
        headless=bool(config.get("headless", True)),
        chrome_binary_path=chrome_bin or None,
        chromedriver_path=chromedriver or None,
        profile_dir=profile_dir or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual runner for the Yandex Images parser via Chrome/Chromium.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--query", type=str, default=None, help="Search query for Yandex Images.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images/URLs.")
    parser.add_argument("--delay", type=float, default=None, help="Delay between page scroll requests.")
    parser.add_argument("--mode", type=str, default=None, choices=["urls", "download"], help="urls = only print found URLs, download = save images to disk.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory for downloaded images in download mode.")
    parser.add_argument("--size", type=str, default=None, help="Optional Yandex size filter.")
    parser.add_argument("--image-type", type=str, default=None, help="Optional Yandex image type filter.")
    parser.add_argument("--chrome-bin", type=str, default=None, help="Optional explicit Chrome/Chromium binary path.")
    parser.add_argument("--chromedriver", type=str, default=None, help="Optional explicit chromedriver path.")
    parser.add_argument("--profile-dir", type=str, default=None, help="Optional Chromium user-data-dir for persistent session/cookies.")
    parser.add_argument("--manual-captcha-timeout", type=float, default=None, help="In non-headless mode, seconds to wait for manual captcha solving.")
    parser.add_argument("--show-browser", action="store_true", help="Disable headless mode for debugging.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.query:
        config["query"] = args.query
    if args.limit is not None:
        config["limit"] = args.limit
    if args.delay is not None:
        config["delay"] = args.delay
    if args.mode:
        config["mode"] = args.mode
    if args.save_dir:
        config["save_dir"] = args.save_dir
    if args.size is not None:
        config["size"] = args.size
    if args.image_type is not None:
        config["image_type"] = args.image_type
    if args.chrome_bin is not None:
        config["chrome_bin"] = args.chrome_bin
    if args.chromedriver is not None:
        config["chromedriver"] = args.chromedriver
    if args.profile_dir is not None:
        config["profile_dir"] = args.profile_dir
    if args.manual_captcha_timeout is not None:
        config["manual_captcha_timeout"] = args.manual_captcha_timeout
    if args.show_browser:
        config["headless"] = False

    parser_instance = build_parser(config)
    result: dict[str, Any] = {
        "query": config["query"],
        "limit": int(config["limit"]),
        "delay": float(config["delay"]),
        "chrome_bin": parser_instance.chrome_binary_path,
        "chromedriver": parser_instance.chromedriver_path,
        "headless": bool(config["headless"]),
        "profile_dir": str(config.get("profile_dir") or ""),
        "manual_captcha_timeout": float(config.get("manual_captcha_timeout", 0.0)),
        "mode": config["mode"],
    }

    if config["mode"] == "urls":
        urls = parser_instance.query_search(
            query=str(config["query"]),
            limit=int(config["limit"]),
            delay=float(config["delay"]),
            manual_captcha_timeout=float(config.get("manual_captcha_timeout", 0.0)),
            size=str(config.get("size") or "") or None,
            image_type=str(config.get("image_type") or "") or None,
        )
        result["found_count"] = len(urls)
        result["urls"] = urls
    else:
        save_dir = Path(str(config["save_dir"]))
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = search_and_download_images(
            query=str(config["query"]),
            limit=int(config["limit"]),
            save_dir=str(save_dir),
            size=str(config.get("size") or ""),
            image_type=str(config.get("image_type") or ""),
            delay=float(config["delay"]),
            headless=bool(config["headless"]),
            profile_dir=str(config.get("profile_dir") or ""),
            manual_captcha_timeout=float(config.get("manual_captcha_timeout", 0.0)),
        )
        result["save_dir"] = str(save_dir.resolve())
        try:
            result["summary"] = json.loads(summary)
        except json.JSONDecodeError:
            result["summary"] = summary

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
