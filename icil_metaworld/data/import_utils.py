from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


def import_metaworld() -> Any:
    try:
        return importlib.import_module('metaworld')
    except ModuleNotFoundError as first_exc:
        repo_root = Path(__file__).resolve().parents[2]
        local_root = repo_root / 'Metaworld'
        if local_root.is_dir() and str(local_root) not in sys.path:
            sys.path.insert(0, str(local_root))
        try:
            return importlib.import_module('metaworld')
        except ModuleNotFoundError:
            raise first_exc
