from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def make_run_id(prefix: str = "annotation") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def ensure_run_layout(base_dir: str | Path) -> Path:
    run_dir = Path(base_dir)
    for subdir in ("reports", "summary", "cleaned_or_labeled", "manual_review", "masks"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir
