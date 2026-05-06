from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from absl import app, flags, logging

from .metaworld_cache import MetaWorldEpisodeStore

_CACHE_ROOT = flags.DEFINE_string('cache-root', '', 'MetaWorld cache root containing cache.h5 and index.json.')
_MAX_INSTANCES = flags.DEFINE_integer('max-instances', 20, 'Maximum task instances to print in detail; <=0 prints all.')
_TOL = flags.DEFINE_float('tol', 1e-7, 'Tolerance for considering two episodes byte/numerically identical.')
_JSON_OUT = flags.DEFINE_string('json-out', '', 'Optional path for full JSON diagnostic output.')


def _episode_arrays(store: MetaWorldEpisodeStore, task_name: str, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    length = int(store.episode_length(task_name, int(episode_id)))
    rows = np.arange(length, dtype=np.int64)
    item = store.load_episode_slices(task_name, int(episode_id), rows)
    return np.asarray(item['obs_model'], dtype=np.float32), np.asarray(item['action'], dtype=np.float32)


def diagnose(cache_root: str, *, tol: float, max_instances: int) -> Dict[str, Any]:
    store = MetaWorldEpisodeStore(cache_root, keep_open_per_worker=True, preload_to_memory=False)
    rows: List[Dict[str, Any]] = []
    total_instances = 0
    comparable_instances = 0
    identical_instances = 0
    try:
        for task_name, instance_id in store.list_task_instance_keys():
            total_instances += 1
            episode_ids = store.list_episode_ids(task_name, task_instance_id=int(instance_id)).astype(np.int64)
            row: Dict[str, Any] = {
                'task_name': str(task_name),
                'task_instance_id': int(instance_id),
                'num_episodes': int(episode_ids.shape[0]),
                'episode_ids': [int(v) for v in episode_ids.tolist()],
                'all_pairs_identical': None,
                'max_obs_abs_diff': None,
                'max_action_abs_diff': None,
                'max_goal_abs_diff': None,
                'max_first_obs_abs_diff': None,
            }
            if episode_ids.shape[0] >= 2:
                comparable_instances += 1
                ref_obs, ref_action = _episode_arrays(store, task_name, int(episode_ids[0]))
                max_obs = 0.0
                max_action = 0.0
                max_goal = 0.0
                max_first_obs = 0.0
                all_identical = True
                for episode_id in episode_ids[1:].tolist():
                    obs, action = _episode_arrays(store, task_name, int(episode_id))
                    if obs.shape != ref_obs.shape or action.shape != ref_action.shape:
                        all_identical = False
                        max_obs = float('inf')
                        max_action = float('inf')
                        max_goal = float('inf')
                        max_first_obs = float('inf')
                        break
                    obs_diff = float(np.max(np.abs(obs - ref_obs))) if obs.size else 0.0
                    action_diff = float(np.max(np.abs(action - ref_action))) if action.size else 0.0
                    goal_diff = (
                        float(np.max(np.abs(obs[:, -3:] - ref_obs[:, -3:])))
                        if obs.ndim == 2 and obs.shape[-1] >= 3 and ref_obs.shape == obs.shape
                        else float('nan')
                    )
                    first_obs_diff = float(np.max(np.abs(obs[0] - ref_obs[0]))) if obs.ndim >= 2 and obs.shape[0] else 0.0
                    max_obs = max(max_obs, obs_diff)
                    max_action = max(max_action, action_diff)
                    max_goal = max(max_goal, goal_diff)
                    max_first_obs = max(max_first_obs, first_obs_diff)
                    if obs_diff > float(tol) or action_diff > float(tol):
                        all_identical = False
                row['all_pairs_identical'] = bool(all_identical)
                row['max_obs_abs_diff'] = max_obs
                row['max_action_abs_diff'] = max_action
                row['max_goal_abs_diff'] = max_goal
                row['max_first_obs_abs_diff'] = max_first_obs
                if all_identical:
                    identical_instances += 1
            rows.append(row)
    finally:
        store.close()
    return {
        'cache_root': str(Path(cache_root).expanduser()),
        'tol': float(tol),
        'total_instances': int(total_instances),
        'comparable_instances': int(comparable_instances),
        'identical_instances': int(identical_instances),
        'different_instances': int(comparable_instances - identical_instances),
        'rows': rows if int(max_instances) <= 0 else rows[: int(max_instances)],
        'all_rows': rows,
    }


def main(argv=None):
    del argv
    if not _CACHE_ROOT.value:
        raise ValueError('--cache-root is required.')
    result = diagnose(_CACHE_ROOT.value, tol=float(_TOL.value), max_instances=int(_MAX_INSTANCES.value))
    logging.info(
        'same-instance cache diagnostic: total=%d comparable=%d identical=%d different=%d',
        result['total_instances'],
        result['comparable_instances'],
        result['identical_instances'],
        result['different_instances'],
    )
    for row in result['rows']:
        logging.info('%s', json.dumps(row, sort_keys=True))
    if _JSON_OUT.value:
        path = Path(_JSON_OUT.value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        logging.info('wrote diagnostic JSON: %s', path)


if __name__ == '__main__':
    app.run(main)
