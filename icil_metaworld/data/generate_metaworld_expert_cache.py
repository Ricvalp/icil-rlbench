from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm

from .import_utils import import_metaworld
from .observation_filter import ObservationFilterConfig, filter_observation, normalize_env_name
from .scripted_policies import make_policy

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/metaworld_generate_cache.py',
    help_string='Path to ml_collections MetaWorld cache-generation config.',
)


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(x) for x in value if str(x))


def _reset_env(env: Any, *, seed: int) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        out = env.reset(seed=int(seed))
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return np.asarray(obs, dtype=np.float32), dict(info or {})


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
    else:
        raise RuntimeError(f'Unsupported MetaWorld env.step return format: {type(out).__name__} len={len(out)}')
    return (
        np.asarray(obs, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        dict(info or {}),
    )


def _make_benchmark(metaworld: Any, benchmark_name: str, task_name: str, seed: int) -> Any:
    name = str(benchmark_name).upper()
    task_name = normalize_env_name(task_name)
    if name == 'ML1':
        return metaworld.ML1(task_name, seed=int(seed))
    if name == 'MT1':
        return metaworld.MT1(task_name, seed=int(seed))
    if name in ('ML10', 'ML25', 'ML45', 'MT10', 'MT25', 'MT50'):
        return getattr(metaworld, name)(seed=int(seed))
    raise ValueError(
        'metaworld.benchmark must be one of ML1, ML10, ML25, ML45, MT1, MT10, MT25, MT50. '
        f'Got {benchmark_name!r}.'
    )


def _selected_task_names(cfg: ConfigDict, benchmark: Any, *, split: str) -> Tuple[str, ...]:
    configured = tuple(normalize_env_name(x) for x in _as_tuple(getattr(cfg.metaworld, 'task_names', ())))
    classes = benchmark.train_classes if split == 'train' else benchmark.test_classes
    available = tuple(str(name) for name in classes.keys())
    if configured:
        missing = sorted(set(configured) - set(available))
        if missing:
            raise ValueError(f'Requested MetaWorld task_names not present in {cfg.metaworld.benchmark}/{split}: {missing}')
        selected = configured
    else:
        selected = available
    limit_tasks = int(getattr(cfg.debug, 'limit_tasks', 0))
    if limit_tasks > 0:
        selected = selected[:limit_tasks]
    return selected


def _task_entries(cfg: ConfigDict, metaworld: Any) -> List[tuple[str, Any, List[Any]]]:
    split = str(getattr(cfg.metaworld, 'train_or_test', 'train')).lower()
    if split not in ('train', 'test'):
        raise ValueError(f'metaworld.train_or_test must be train or test, got {split!r}.')
    requested = tuple(normalize_env_name(x) for x in _as_tuple(getattr(cfg.metaworld, 'task_names', ())))
    benchmark_name = str(getattr(cfg.metaworld, 'benchmark', 'ML1')).upper()

    if benchmark_name in ('ML1', 'MT1') and len(requested) > 1:
        benchmarks = [_make_benchmark(metaworld, benchmark_name, name, int(cfg.seed) + i) for i, name in enumerate(requested)]
    else:
        first_task = requested[0] if requested else 'button-press-v3'
        benchmarks = [_make_benchmark(metaworld, benchmark_name, first_task, int(cfg.seed))]

    entries: List[tuple[str, Any, List[Any]]] = []
    for benchmark in benchmarks:
        classes = benchmark.train_classes if split == 'train' else benchmark.test_classes
        tasks_all = benchmark.train_tasks if split == 'train' else benchmark.test_tasks
        if benchmark_name in ('ML1', 'MT1') and requested:
            selected_names = tuple(str(name) for name in classes.keys())
        else:
            selected_names = _selected_task_names(cfg, benchmark, split=split)
        for task_name in selected_names:
            task_instances = [task for task in tasks_all if str(task.env_name) == str(task_name)]
            limit_instances = int(getattr(cfg.debug, 'limit_instances', 0))
            configured_instances = int(getattr(cfg.metaworld, 'num_task_instances_per_task', 0))
            if configured_instances > 0:
                task_instances = task_instances[:configured_instances]
            if limit_instances > 0:
                task_instances = task_instances[:limit_instances]
            if not task_instances:
                raise RuntimeError(f'No MetaWorld task instances found for {task_name!r} in split {split!r}.')
            entries.append((str(task_name), classes[task_name], task_instances))
    return entries


def _safe_success(info: Dict[str, Any]) -> bool:
    value = info.get('success', False)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _rollout_expert_episode(
    *,
    env: Any,
    policy: Any,
    seed: int,
    max_path_length: int,
    obs_cfg: ObservationFilterConfig,
    clip_action: bool,
) -> Dict[str, Any]:
    obs, _ = _reset_env(env, seed=seed)
    obs_raw: List[np.ndarray] = []
    obs_model: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    success: List[bool] = []
    terminated: List[bool] = []
    truncated: List[bool] = []
    filter_notes: Dict[str, Any] = {}

    for _ in range(int(max_path_length)):
        model_obs, notes = filter_observation(obs, obs_cfg)
        if not filter_notes:
            filter_notes = notes
        action = np.asarray(policy.get_action(obs), dtype=np.float32).reshape(-1)
        if clip_action:
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
            action = np.clip(action, low, high).astype(np.float32)
        next_obs, reward, term, trunc, info = _step_env(env, action)
        obs_raw.append(obs.astype(np.float32))
        obs_model.append(model_obs.astype(np.float32))
        actions.append(action.astype(np.float32))
        rewards.append(float(reward))
        success.append(_safe_success(info))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        obs = next_obs
        if bool(term) or bool(trunc):
            break

    if not obs_raw:
        raise RuntimeError('MetaWorld rollout produced an empty episode.')
    success_arr = np.asarray(success, dtype=np.bool_)
    term_arr = np.asarray(terminated, dtype=np.bool_)
    trunc_arr = np.asarray(truncated, dtype=np.bool_)
    return {
        'obs_raw': np.stack(obs_raw, axis=0).astype(np.float32),
        'obs_model': np.stack(obs_model, axis=0).astype(np.float32),
        'actions': np.stack(actions, axis=0).astype(np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'success': success_arr,
        'terminated': term_arr,
        'truncated': trunc_arr,
        'done': np.logical_or(term_arr, trunc_arr),
        'success_any': bool(np.any(success_arr)),
        'success_final': bool(success_arr[-1]),
        'filter_notes': filter_notes,
    }


def _write_episode(group: h5py.Group, episode: Dict[str, Any]) -> None:
    compression = 'gzip'
    group.create_dataset('obs_raw', data=episode['obs_raw'], compression=compression)
    group.create_dataset('obs_model', data=episode['obs_model'], compression=compression)
    group.create_dataset('actions', data=episode['actions'], compression=compression)
    group.create_dataset('rewards', data=episode['rewards'], compression=compression)
    group.create_dataset('success', data=episode['success'].astype(np.uint8), compression=compression)
    group.create_dataset('done', data=episode['done'].astype(np.uint8), compression=compression)
    group.create_dataset('trunc', data=episode['truncated'].astype(np.uint8), compression=compression)
    group.create_dataset('terminated', data=episode['terminated'].astype(np.uint8), compression=compression)
    group.create_dataset('truncated', data=episode['truncated'].astype(np.uint8), compression=compression)


def generate(cfg: ConfigDict) -> Path:
    metaworld = import_metaworld()
    np.random.seed(int(cfg.seed))
    cache_root = Path(str(cfg.output.cache_root)).expanduser().resolve()
    overwrite = bool(getattr(cfg.output, 'overwrite', False))
    if cache_root.exists():
        if not overwrite:
            raise FileExistsError(f'MetaWorld cache root already exists: {cache_root}. Set output.overwrite=True to replace it.')
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / 'cache.h5'
    index_path = cache_root / 'index.json'

    obs_cfg = ObservationFilterConfig(
        variant=str(getattr(cfg.obs, 'variant', 'no_task_no_goal')),
        remove_task_id=bool(getattr(cfg.obs, 'remove_task_id', True)),
        remove_goal=bool(getattr(cfg.obs, 'remove_goal', True)),
        normalize=bool(getattr(cfg.obs, 'normalize', False)),
    )
    entries = _task_entries(cfg, metaworld)
    required_successes = int(getattr(cfg.metaworld, 'num_successful_episodes_per_instance', 1))
    max_attempts = int(getattr(cfg.metaworld, 'max_attempts_per_instance', max(1, required_successes)))
    max_path_length = int(getattr(cfg.metaworld, 'max_path_length', 200))
    clip_action = bool(getattr(cfg.action, 'clip', False))
    keep_successful_only = bool(getattr(cfg.metaworld, 'keep_successful_only', True)) if hasattr(cfg.metaworld, 'keep_successful_only') else True
    skip_failed_task_instances = bool(getattr(cfg.metaworld, 'skip_failed_task_instances', False))
    debug_limit_episodes = int(getattr(cfg.debug, 'limit_episodes', 0))
    if debug_limit_episodes > 0:
        required_successes = min(required_successes, debug_limit_episodes)

    index: Dict[str, Any] = {
        'version': 1,
        'created_unix_time': time.time(),
        'cache_file': 'cache.h5',
        'benchmark': str(getattr(cfg.metaworld, 'benchmark', 'ML1')),
        'split': str(getattr(cfg.metaworld, 'train_or_test', 'train')),
        'obs': cfg.obs.to_dict(),
        'action': cfg.action.to_dict(),
        'skipped_task_instances': [],
        'tasks': {},
        'episodes': {},
    }

    global_episode_id = 0
    obs_raw_dim = -1
    obs_model_dim = -1
    action_dim = -1

    with h5py.File(cache_path, 'w') as f:
        f.attrs['version'] = 1
        f.attrs['benchmark'] = str(getattr(cfg.metaworld, 'benchmark', 'ML1'))
        f.attrs['split'] = str(getattr(cfg.metaworld, 'train_or_test', 'train'))
        f.attrs['obs_variant'] = str(getattr(cfg.obs, 'variant', 'no_task_no_goal'))
        episodes_group = f.create_group('episodes')

        for task_index, (task_name, env_cls, task_instances) in enumerate(entries):
            logging.info('Generating MetaWorld demos for %s (%d task instances)', task_name, len(task_instances))
            index['tasks'].setdefault(task_name, {'task_index': int(task_index), 'instances': {}})
            policy = make_policy(task_name, require=bool(getattr(cfg, 'require_scripted_policy', True)))
            if policy is None:
                logging.warning('Skipping %s because no scripted policy is available.', task_name)
                continue

            for task_instance_id, task in enumerate(tqdm(task_instances, desc=task_name)):
                env = env_cls()
                try:
                    env.set_task(task)
                    if bool(getattr(cfg.metaworld, 'force_goal_observable', False)):
                        env._partially_observable = False
                        try:
                            del env.sawyer_observation_space
                        except Exception:
                            pass
                    try:
                        env.action_space.seed(int(cfg.seed) + 7919 * task_index + task_instance_id)
                    except Exception:
                        pass
                    instance_episode_ids: List[int] = []
                    attempts = 0
                    successes = 0
                    while attempts < max_attempts and successes < required_successes:
                        attempts += 1
                        ep_seed = int(cfg.seed) + 1_000_003 * task_index + 10_007 * task_instance_id + attempts
                        episode = _rollout_expert_episode(
                            env=env,
                            policy=policy,
                            seed=ep_seed,
                            max_path_length=max_path_length,
                            obs_cfg=obs_cfg,
                            clip_action=clip_action,
                        )
                        if keep_successful_only and not bool(episode['success_any']):
                            continue
                        if obs_raw_dim < 0:
                            obs_raw_dim = int(episode['obs_raw'].shape[-1])
                            obs_model_dim = int(episode['obs_model'].shape[-1])
                            action_dim = int(episode['actions'].shape[-1])
                            f.attrs['obs_raw_dim'] = obs_raw_dim
                            f.attrs['obs_model_dim'] = obs_model_dim
                            f.attrs['action_dim'] = action_dim
                            f.attrs['obs_filter_notes'] = json.dumps(episode.get('filter_notes', {}))
                        if int(episode['obs_raw'].shape[-1]) != obs_raw_dim:
                            raise RuntimeError(f'obs_raw dim changed from {obs_raw_dim} to {episode["obs_raw"].shape[-1]}.')
                        if int(episode['obs_model'].shape[-1]) != obs_model_dim:
                            raise RuntimeError(f'obs_model dim changed from {obs_model_dim} to {episode["obs_model"].shape[-1]}.')
                        if int(episode['actions'].shape[-1]) != action_dim:
                            raise RuntimeError(f'action dim changed from {action_dim} to {episode["actions"].shape[-1]}.')

                        episode_id = int(global_episode_id)
                        global_episode_id += 1
                        ep_group = episodes_group.create_group(str(episode_id))
                        _write_episode(ep_group, episode)
                        ep_group.attrs['task_name'] = str(task_name)
                        ep_group.attrs['task_index'] = int(task_index)
                        ep_group.attrs['task_instance_id'] = int(task_instance_id)
                        ep_group.attrs['env_id'] = str(task_name)
                        ep_group.attrs['seed'] = int(ep_seed)
                        ep_group.attrs['episode_id'] = int(episode_id)
                        ep_group.attrs['success_any'] = bool(episode['success_any'])
                        ep_group.attrs['success_final'] = bool(episode['success_final'])
                        ep_group.attrs['length'] = int(episode['obs_raw'].shape[0])

                        index['episodes'][str(episode_id)] = {
                            'episode_id': int(episode_id),
                            'task_name': str(task_name),
                            'task_index': int(task_index),
                            'task_instance_id': int(task_instance_id),
                            'env_id': str(task_name),
                            'seed': int(ep_seed),
                            'length': int(episode['obs_raw'].shape[0]),
                            'success_any': bool(episode['success_any']),
                            'success_final': bool(episode['success_final']),
                            'attempt': int(attempts),
                        }
                        instance_episode_ids.append(episode_id)
                        successes += 1
                    if successes < required_successes:
                        message = (
                            f'Only generated {successes}/{required_successes} successful episodes for '
                            f'{task_name} instance {task_instance_id} after {attempts} attempts.'
                        )
                        if skip_failed_task_instances:
                            logging.warning('Skipping failed MetaWorld task instance: %s', message)
                            index['skipped_task_instances'].append(
                                {
                                    'task_name': str(task_name),
                                    'task_index': int(task_index),
                                    'task_instance_id': int(task_instance_id),
                                    'successes': int(successes),
                                    'required_successes': int(required_successes),
                                    'attempts': int(attempts),
                                }
                            )
                            continue
                        raise RuntimeError(
                            message
                        )
                    index['tasks'][task_name]['instances'][str(task_instance_id)] = instance_episode_ids
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

        f.attrs['num_episodes'] = int(global_episode_id)
        if global_episode_id == 0:
            raise RuntimeError('No MetaWorld episodes were generated.')

    index['num_episodes'] = int(global_episode_id)
    index['obs_raw_dim'] = int(obs_raw_dim)
    index['obs_model_dim'] = int(obs_model_dim)
    index['action_dim'] = int(action_dim)
    with index_path.open('w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, sort_keys=True)
    logging.info('Wrote MetaWorld cache: %s (%d episodes)', cache_root, global_episode_id)
    return cache_root


def main(argv=None):
    del argv
    generate(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
