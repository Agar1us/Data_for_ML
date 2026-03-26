from __future__ import annotations

from pathlib import Path


_DEFAULTS = {
    "data_root": str(Path("data").resolve()),
    "logs_root": str((Path("data") / "logs").resolve()),
    "artifacts_root": str(Path("collection_artifacts").resolve()),
    "run_log_dir": "",
    "yandex_headless": True,
    "yandex_manual_captcha_timeout": 0.0,
    "yandex_profile_dir": "",
}

_RUNTIME = _DEFAULTS.copy()


def clear_runtime_context() -> None:
    _RUNTIME.clear()
    _RUNTIME.update(_DEFAULTS)


def set_runtime_context(
    *,
    data_root: str | Path,
    logs_root: str | Path,
    artifacts_root: str | Path,
    run_log_dir: str | Path = "",
    yandex_headless: bool = True,
    yandex_manual_captcha_timeout: float = 0.0,
    yandex_profile_dir: str = "",
) -> None:
    _RUNTIME.update(
        {
            "data_root": str(Path(data_root).resolve()),
            "logs_root": str(Path(logs_root).resolve()),
            "artifacts_root": str(Path(artifacts_root).resolve()),
            "run_log_dir": str(Path(run_log_dir).resolve()) if str(run_log_dir).strip() else "",
            "yandex_headless": bool(yandex_headless),
            "yandex_manual_captcha_timeout": float(yandex_manual_captcha_timeout),
            "yandex_profile_dir": str(yandex_profile_dir).strip(),
        }
    )


def data_root() -> Path:
    return Path(str(_RUNTIME["data_root"])).resolve()


def logs_root() -> Path:
    return Path(str(_RUNTIME["logs_root"])).resolve()


def artifacts_root() -> Path:
    return Path(str(_RUNTIME["artifacts_root"])).resolve()


def run_log_dir() -> Path | None:
    value = str(_RUNTIME.get("run_log_dir", "")).strip()
    return Path(value).resolve() if value else None


def yandex_headless() -> bool:
    return bool(_RUNTIME["yandex_headless"])


def yandex_manual_captcha_timeout() -> float:
    return float(_RUNTIME["yandex_manual_captcha_timeout"])


def yandex_profile_dir() -> str:
    return str(_RUNTIME["yandex_profile_dir"]).strip()
