from __future__ import annotations

from pathlib import Path

from tools.runtime import data_root as runtime_data_root


_LEGACY_RELATIVE_PREFIXES = {
    ("data",),
    ("current_run",),
    ("collection",),
    ("logs",),
    ("collection_artifacts",),
}


def _relative_parts(path: str) -> tuple[str, ...]:
    candidate = Path(path)
    return tuple(part for part in candidate.parts if part not in {"", "."})


def _assert_no_legacy_prefix(parts: tuple[str, ...]) -> None:
    lowered = tuple(part.casefold() for part in parts)
    for prefix in _LEGACY_RELATIVE_PREFIXES:
        prefix_len = len(prefix)
        if lowered[:prefix_len] == prefix:
            joined = "/".join(parts)
            raise ValueError(
                f"Legacy-relative path prefixes are not allowed: '{joined}'. "
                "Pass only a path relative to the configured collection root."
            )


def _assert_within_root(target: Path, root: Path) -> Path:
    resolved = target.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{resolved}' is outside the configured collection root '{root}'."
        ) from exc
    return resolved


def data_root() -> Path:
    return runtime_data_root()


def resolve_data_output_dir(path: str) -> str:
    root = runtime_data_root()
    if not str(path or "").strip():
        return str(root)

    candidate = Path(path)
    if candidate.is_absolute():
        return str(_assert_within_root(candidate, root))

    parts = _relative_parts(path)
    if any(part == ".." for part in parts):
        raise ValueError("Parent-directory traversal is not allowed in collection paths.")
    _assert_no_legacy_prefix(parts)
    return str(_assert_within_root(root.joinpath(*parts), root))


def resolve_data_output_path(path: str) -> str:
    return resolve_data_output_dir(path)
