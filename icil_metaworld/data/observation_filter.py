from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class ObservationFilterConfig:
    variant: str = 'no_task_no_goal'
    remove_task_id: bool = True
    remove_goal: bool = True
    normalize: bool = False


def normalize_env_name(name: str) -> str:
    value = str(name).strip()
    # MetaWorld 3.0 uses v3 task names. Accept old user muscle memory.
    if value.endswith('-v2'):
        return value[:-3] + '-v3'
    return value


def filter_observation(obs: Any, cfg: ObservationFilterConfig) -> tuple[np.ndarray, Dict[str, Any]]:
    raw = np.asarray(obs, dtype=np.float32).reshape(-1)
    variant = str(cfg.variant).lower()
    remove_goal = bool(cfg.remove_goal)
    indices = np.arange(raw.shape[0], dtype=np.int64)
    notes: Dict[str, Any] = {
        'raw_dim': int(raw.shape[0]),
        'variant': variant,
        'removed_goal_indices': [],
        'removed_task_id_indices': [],
    }

    if variant in ('raw', 'none'):
        model = raw
    elif variant in ('no_task', 'no_task_no_goal', 'ml_no_goal'):
        keep = np.ones(raw.shape[0], dtype=bool)
        # MetaWorld v3 observations are 39D: current 18D state, previous 18D
        # state, final 3D goal slot. ML benchmarks usually zero this final
        # slot for partial observability; remove it anyway to avoid leakage and
        # keep the model interface explicit.
        if remove_goal and raw.shape[0] >= 39:
            keep[-3:] = False
            notes['removed_goal_indices'] = indices[-3:].tolist()
        model = raw[keep]
    elif variant == 'positions_only_no_goal':
        # Conservative low-dimensional diagnostic: hand/gripper + object xyz
        # for current and previous frames when using the standard 39D v3 obs.
        if raw.shape[0] >= 39:
            keep_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31], dtype=np.int64)
            keep_idx = keep_idx[keep_idx < raw.shape[0] - (3 if remove_goal else 0)]
            model = raw[keep_idx]
        else:
            model = raw
    else:
        raise ValueError(
            "obs.variant must be one of: raw, no_task, no_task_no_goal, ml_no_goal, positions_only_no_goal. "
            f'Got {cfg.variant!r}.'
        )

    if bool(cfg.normalize):
        raise NotImplementedError('Observation normalization is intentionally not implemented in the generator yet.')
    return np.asarray(model, dtype=np.float32), notes
