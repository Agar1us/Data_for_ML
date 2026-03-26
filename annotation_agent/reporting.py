from __future__ import annotations

from pathlib import Path


def ensure_run_layout(base_dir: str | Path) -> Path:
    run_dir = Path(base_dir)
    for subdir in ("reports", "summary"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir
