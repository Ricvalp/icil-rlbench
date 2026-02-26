"""Utilities for working with RLBench segmentation label maps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Set, Tuple

MASK_LABEL_MAP_FILENAME = "mask_to_label.json"


def resolve_mask_ids(
    variation_dir: Path,
    mask_names: Iterable[str],
    *,
    map_filename: str = MASK_LABEL_MAP_FILENAME,
) -> Tuple[Set[int], Set[str], str | None]:
    """Return handles matching ``mask_names`` and report unresolved entries."""

    names = {name for name in mask_names if name}
    map_path = variation_dir / map_filename
    if not map_path.is_file():
        return set(), names, "label map file not found"
    try:
        with map_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return set(), names, str(exc)

    handle_to_name = {int(key): value for key, value in data.items()}
    matched = {handle for handle, name in handle_to_name.items() if name in names}
    found_names = {name for name in handle_to_name.values() if name in names}
    missing = names - found_names
    return matched, missing, None

