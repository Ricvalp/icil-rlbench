from __future__ import annotations

from typing import Any

from .import_utils import import_metaworld
from .observation_filter import normalize_env_name


def get_policy_class(task_name: str) -> type[Any] | None:
    task_name = normalize_env_name(task_name)
    try:
        import_metaworld()
        from metaworld.policies import ENV_POLICY_MAP
    except Exception as exc:
        raise ImportError(
            'Could not import MetaWorld scripted policies. Ensure metaworld 3.x and mujoco are installed.'
        ) from exc
    return ENV_POLICY_MAP.get(task_name, None)


def make_policy(task_name: str, *, require: bool = True) -> Any | None:
    cls = get_policy_class(task_name)
    if cls is None:
        if require:
            raise KeyError(f'No MetaWorld scripted policy found for task {task_name!r}.')
        return None
    return cls()
