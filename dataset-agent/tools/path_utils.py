from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.getenv("DATASET_AGENT_DATA_DIR", "data")).resolve()


def _clean_parts(path: str) -> tuple[str, ...]:
    parts: list[str] = []
    for part in Path(os.path.normpath(path)).parts:
        if part in {"", ".", os.sep}:
            continue
        if part == "..":
            continue
        parts.append(part)
    return tuple(parts)


def _candidate_prefixes(root: Path) -> list[tuple[str, ...]]:
    tail = root.parts[-3:] if len(root.parts) >= 3 else root.parts
    sequences: set[tuple[str, ...]] = {("data",)}

    for start in range(len(tail)):
        for end in range(start + 1, len(tail) + 1):
            sequences.add(tuple(tail[start:end]))

    return sorted(sequences, key=len, reverse=True)


def _strip_known_prefix(parts: tuple[str, ...], prefixes: list[tuple[str, ...]]) -> tuple[str, ...]:
    for prefix in prefixes:
        prefix_len = len(prefix)
        if prefix_len == 0 or len(parts) < prefix_len:
            continue
        for index in range(0, len(parts) - prefix_len + 1):
            if parts[index : index + prefix_len] == prefix:
                return parts[index + prefix_len :]
    return parts


def _relative_parts_for_root(path: str, root: Path) -> tuple[str, ...]:
    if not path or path.strip() in {"", "."}:
        return ()

    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.resolve()
        try:
            return candidate.relative_to(root).parts
        except ValueError:
            stripped = _strip_known_prefix(_clean_parts(str(candidate)), _candidate_prefixes(root))
            return stripped[-1:] if stripped == _clean_parts(str(candidate)) else stripped

    return _strip_known_prefix(_clean_parts(path), _candidate_prefixes(root))


def resolve_data_output_dir(path: str) -> str:
    root = data_root()
    relative_parts = _relative_parts_for_root(path, root)
    target = root.joinpath(*relative_parts) if relative_parts else root
    return str(target.resolve())


def resolve_data_output_path(path: str) -> str:
    return resolve_data_output_dir(path)
