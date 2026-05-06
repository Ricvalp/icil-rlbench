from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Must be set before importing MetaWorld/MuJoCo.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from diagnostics.jax_metaworld_adaptation_common import (
    adapt_memory_for_task,
    initial_memory,
    load_metaworld_policy_components,
    load_store,
    make_builder,
    make_metaworld_env_for_instance,
    numpy_batch_to_jax,
    obs_to_model_state,
    render_frame,
    reset_env,
    step_env,
    success_from_info,
    write_gif,
)
from icil_metaworld.data.metaworld_task_builder import MetaWorldMAMLTaskSpec
from icil_metaworld.data.observation_filter import normalize_env_name

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_metaworld_paper_rollout_eval.py',
    help_string='Path to ml_collections config file.',
)


def _as_auto_bool(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('', 'auto'):
            return 'auto'
        if text in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if text in ('0', 'false', 'no', 'n', 'off'):
            return False
    if value is None:
        return 'auto'
    return bool(value)


def _infer_force_goal_observable(cfg: ConfigDict, *, store: Any, data_cfg: Any, state_dim: int) -> bool:
    configured = _as_auto_bool(getattr(cfg.sim, 'force_goal_observable', 'auto'))
    if configured != 'auto':
        return bool(configured)
    obs_cfg = store.index.get('obs', {}) if isinstance(store.index, dict) else {}
    cache_keeps_goal = str(obs_cfg.get('variant', '')).lower() in ('raw', 'none') and not bool(obs_cfg.get('remove_goal', True))
    return bool(cache_keeps_goal and not bool(data_cfg.query_zero_goal) and int(state_dim) >= 39)


def _normalised_filter(values: Sequence[str] | Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    return tuple(normalize_env_name(str(v)) for v in raw if str(v).strip())


def _selected_tasks(store: Any, cfg: ConfigDict) -> List[str]:
    include = set(_normalised_filter(getattr(cfg.data, 'tasks', ())))
    exclude = set(_normalised_filter(getattr(cfg.data, 'exclude_tasks', ())))
    tasks = [str(name) for name in store.list_task_names() if (not include or str(name) in include) and str(name) not in exclude]
    if not tasks:
        raise RuntimeError('No MetaWorld tasks selected after data.tasks/data.exclude_tasks filtering.')
    return sorted(tasks)


def _task_instance_ids(store: Any, task_name: str) -> List[int]:
    return [int(instance_id) for name, instance_id in store.list_task_instance_keys(tasks=(task_name,)) if str(name) == str(task_name)]


def _episode_for_instance(store: Any, task_name: str, instance_id: int, rng: np.random.Generator) -> int:
    episode_ids = store.list_episode_ids(task_name, task_instance_id=int(instance_id))
    if episode_ids.shape[0] == 0:
        raise RuntimeError(f'No cached episodes for {task_name} instance {instance_id}.')
    return int(episode_ids[int(rng.integers(0, episode_ids.shape[0]))])


def _fixed_task_spec(
    *,
    store: Any,
    data_cfg: Any,
    task_name: str,
    query_instance_id: int,
    rng: np.random.Generator,
) -> MetaWorldMAMLTaskSpec:
    all_instances = _task_instance_ids(store, task_name)
    support_candidates = [int(v) for v in all_instances if int(v) != int(query_instance_id)]
    if len(support_candidates) < int(data_cfg.K):
        raise RuntimeError(
            f'{task_name} has only {len(support_candidates)} support candidates for query instance '
            f'{query_instance_id}; need K={int(data_cfg.K)}.'
        )
    support_instance_ids = rng.choice(np.asarray(support_candidates, dtype=np.int64), size=int(data_cfg.K), replace=False)
    support_episode_ids = [
        _episode_for_instance(store, task_name, int(instance_id), rng) for instance_id in support_instance_ids.tolist()
    ]
    query_episode_id = _episode_for_instance(store, task_name, int(query_instance_id), rng)
    return MetaWorldMAMLTaskSpec(
        task_name=str(task_name),
        task_index=int(store.task_index(task_name)),
        task_instance_id=int(query_instance_id),
        support_episode_ids=tuple(int(v) for v in support_episode_ids),
        query_episode_id=int(query_episode_id),
        support_task_instance_ids=tuple(int(v) for v in support_instance_ids.tolist()),
        query_task_instance_id=int(query_instance_id),
    )


def _wrong_family_spec(
    *,
    store: Any,
    data_cfg: Any,
    target_task_name: str,
    requested_task_name: str,
    rng: np.random.Generator,
) -> MetaWorldMAMLTaskSpec:
    if str(requested_task_name or '').strip():
        wrong_task_name = normalize_env_name(str(requested_task_name))
        if wrong_task_name == str(target_task_name):
            raise ValueError('adaptation.different_task_name must differ from the target task.')
    else:
        candidates = [name for name in store.list_task_names() if str(name) != str(target_task_name)]
        if not candidates:
            raise RuntimeError('Need at least two task families for wrong-family adaptation.')
        wrong_task_name = str(candidates[int(rng.integers(0, len(candidates)))])
    wrong_instances = _task_instance_ids(store, wrong_task_name)
    if len(wrong_instances) < int(data_cfg.K) + 1:
        raise RuntimeError(f'{wrong_task_name} has too few instances for wrong-family adaptation.')
    query_instance_id = int(wrong_instances[int(rng.integers(0, len(wrong_instances)))])
    return _fixed_task_spec(
        store=store,
        data_cfg=data_cfg,
        task_name=wrong_task_name,
        query_instance_id=query_instance_id,
        rng=rng,
    )


def _build_query_window(
    states: List[np.ndarray],
    *,
    T_obs: int,
    H: int,
    action_dim: int,
    query_zero_goal: bool,
) -> Dict[str, np.ndarray]:
    if not states:
        raise ValueError('states must be non-empty.')
    idx = np.linspace(max(0, len(states) - int(T_obs)), len(states) - 1, num=int(T_obs), dtype=np.int64)
    if idx.shape[0] < int(T_obs):
        idx = np.pad(idx, (int(T_obs) - idx.shape[0], 0), mode='edge')
    query_state = np.stack([states[int(i)] for i in idx.tolist()], axis=0).astype(np.float32)
    if bool(query_zero_goal) and query_state.shape[-1] >= 39:
        query_state = query_state.copy()
        query_state[..., -3:] = 0.0
    return {
        'query_xyz': np.zeros((1, int(T_obs), 1, 3), dtype=np.float32),
        'query_state': query_state[None],
        'query_valid': np.ones((1, int(T_obs), 1), dtype=bool),
        'target_action': np.zeros((1, int(H), int(action_dim)), dtype=np.float32),
    }


def _run_rollout(
    *,
    env: Any,
    params: Any,
    predict_fn: Any,
    memory_tokens: Any,
    state_dim: int,
    T_obs: int,
    H: int,
    action_dim: int,
    max_steps: int,
    execute_actions_per_plan: int,
    render: bool,
    frame_stride: int,
    seed: int,
    query_zero_goal: bool,
) -> Dict[str, Any]:
    obs, _ = reset_env(env, seed=seed)
    states = [obs_to_model_state(obs, state_dim=int(state_dim))]
    frames = [render_frame(env)] if bool(render) else []
    success = False
    first_success_step = None
    rewards: List[float] = []
    env_steps = 0
    error = None
    try:
        while env_steps < int(max_steps) and not success:
            query = _build_query_window(
                states,
                T_obs=int(T_obs),
                H=int(H),
                action_dim=int(action_dim),
                query_zero_goal=bool(query_zero_goal),
            )
            plan = np.asarray(predict_fn(params, numpy_batch_to_jax(query), memory_tokens), dtype=np.float32)[0]
            n_exec = min(int(execute_actions_per_plan), plan.shape[0], int(max_steps) - env_steps)
            for i in range(n_exec):
                action = np.clip(plan[i], -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = step_env(env, action)
                rewards.append(float(reward))
                env_steps += 1
                success = success_from_info(info)
                if success and first_success_step is None:
                    first_success_step = int(env_steps)
                states.append(obs_to_model_state(obs, state_dim=int(state_dim)))
                if bool(render) and (env_steps % max(1, int(frame_stride))) == 0:
                    frames.append(render_frame(env))
                if success or terminated or truncated or env_steps >= int(max_steps):
                    break
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    return {
        'success': bool(success),
        'first_success_step': first_success_step,
        'env_steps': int(env_steps),
        'return': float(np.sum(rewards)) if rewards else 0.0,
        'max_reward': float(np.max(rewards)) if rewards else 0.0,
        'error': error,
        'frames': frames,
    }


def _mean(items: Sequence[float]) -> float:
    return float(np.mean(np.asarray(items, dtype=np.float64))) if items else 0.0


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    components = load_metaworld_policy_components(cfg)
    params = components['params']
    data_cfg = components['data_cfg']
    memory_cfg = components['memory_cfg']
    predict_fn = components['predict_fn']
    checkpoint_path = components['checkpoint_path']
    state_dim = int(components['state_dim'])
    action_dim = int(components['action_dim'])

    store = load_store(cfg, components['ckpt'])
    benchmark_name = str(getattr(cfg.sim, 'benchmark', '')).strip() or str(store.index.get('benchmark', 'MT10'))
    split = str(getattr(cfg.sim, 'split', '')).strip() or str(store.index.get('split', 'train'))
    force_goal_observable = _infer_force_goal_observable(cfg, store=store, data_cfg=data_cfg, state_dim=state_dim)
    include_wrong = bool(getattr(cfg.adaptation, 'include_wrong_family', False))
    labels = ['no_adaptation', 'same_family_adaptation']
    if include_wrong:
        labels.append('wrong_family_adaptation')

    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / 'jax_metaworld_paper_rollout_eval' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    all_rows: List[Dict[str, Any]] = []
    per_task_rows: List[Dict[str, Any]] = []
    results_by_condition: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    tasks = _selected_tasks(store, cfg)
    instances_per_task = int(cfg.rollout.instances_per_task)
    require_instances = bool(getattr(cfg.rollout, 'require_instances_per_task', True))
    video_counts = {label: 0 for label in labels}

    try:
        for task_idx, task_name in enumerate(tasks):
            instance_ids = _task_instance_ids(store, task_name)
            if len(instance_ids) < instances_per_task and require_instances:
                raise RuntimeError(
                    f'{task_name} has {len(instance_ids)} cached instances; '
                    f'need rollout.instances_per_task={instances_per_task}.'
                )
            selected_instance_ids = instance_ids[: min(instances_per_task, len(instance_ids))]
            task_rows: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}

            for local_idx, query_instance_id in enumerate(selected_instance_ids):
                item_seed = seed + 100000 * task_idx + int(query_instance_id)
                rng = np.random.default_rng(item_seed)
                target_builder = make_builder(store, data_cfg, seed=item_seed + 11, task_names=(task_name,))
                target_spec = _fixed_task_spec(
                    store=store,
                    data_cfg=data_cfg,
                    task_name=task_name,
                    query_instance_id=int(query_instance_id),
                    rng=rng,
                )
                same_adaptation = adapt_memory_for_task(
                    params=params,
                    adapt_with_stats_fn=components['adapt_with_stats_fn'],
                    builder=target_builder,
                    task=target_spec,
                    memory_cfg=memory_cfg,
                    rng=rng,
                    run_dir=run_dir if (task_idx == 0 and local_idx == 0) else None,
                    stem='same_family_adaptation',
                )
                memories = {
                    'no_adaptation': initial_memory(params),
                    'same_family_adaptation': same_adaptation['memory_tokens'],
                }
                wrong_spec: Optional[MetaWorldMAMLTaskSpec] = None
                if include_wrong:
                    wrong_spec = _wrong_family_spec(
                        store=store,
                        data_cfg=data_cfg,
                        target_task_name=task_name,
                        requested_task_name=str(cfg.adaptation.different_task_name),
                        rng=rng,
                    )
                    wrong_builder = make_builder(store, data_cfg, seed=item_seed + 17, task_names=(wrong_spec.task_name,))
                    wrong_adaptation = adapt_memory_for_task(
                        params=params,
                        adapt_with_stats_fn=components['adapt_with_stats_fn'],
                        builder=wrong_builder,
                        task=wrong_spec,
                        memory_cfg=memory_cfg,
                        rng=rng,
                        run_dir=run_dir if (task_idx == 0 and local_idx == 0) else None,
                        stem='wrong_family_adaptation',
                    )
                    memories['wrong_family_adaptation'] = wrong_adaptation['memory_tokens']

                for label in labels:
                    render = bool(cfg.video.enable) and int(video_counts[label]) < int(cfg.video.max_videos_per_condition)
                    env = make_metaworld_env_for_instance(
                        task_name=task_name,
                        task_instance_id=int(query_instance_id),
                        benchmark_name=benchmark_name,
                        split=split,
                        benchmark_seed=int(getattr(cfg.sim, 'benchmark_seed', 0)),
                        camera_name=str(cfg.video.camera_name),
                        width=int(cfg.video.width),
                        height=int(cfg.video.height),
                        force_goal_observable=bool(force_goal_observable),
                    )
                    try:
                        result = _run_rollout(
                            env=env,
                            params=params,
                            predict_fn=predict_fn,
                            memory_tokens=memories[label],
                            state_dim=state_dim,
                            T_obs=int(data_cfg.T_obs),
                            H=int(data_cfg.H),
                            action_dim=action_dim,
                            max_steps=int(cfg.rollout.max_steps),
                            execute_actions_per_plan=int(cfg.rollout.execute_actions_per_plan),
                            render=render,
                            frame_stride=int(cfg.video.frame_stride),
                            seed=item_seed + 23,
                            query_zero_goal=bool(data_cfg.query_zero_goal),
                        )
                    finally:
                        env.close()
                    frames = result.pop('frames')
                    video_path = ''
                    if render and frames:
                        video_path = write_gif(
                            run_dir
                            / f'{label}_videos'
                            / f'{task_name}_instance_{int(query_instance_id):04d}.gif',
                            frames,
                            fps=int(cfg.video.fps),
                        )
                        video_counts[label] += 1
                    row = {
                        **result,
                        'checkpoint_path': str(checkpoint_path),
                        'condition': label,
                        'task_name': str(task_name),
                        'task_instance_id': int(query_instance_id),
                        'support_task_instance_ids': '|'.join(str(int(v)) for v in target_spec.support_task_instance_ids),
                        'support_episode_ids': '|'.join(str(int(v)) for v in target_spec.support_episode_ids),
                        'wrong_task_name': str(wrong_spec.task_name) if wrong_spec is not None else '',
                        'wrong_support_task_instance_ids': '|'.join(str(int(v)) for v in wrong_spec.support_task_instance_ids)
                        if wrong_spec is not None
                        else '',
                        'video_path': video_path,
                    }
                    all_rows.append(row)
                    task_rows[label].append(row)
                    results_by_condition[label].append(row)
                    logging.info(
                        '%s task=%s instance=%d condition=%s success=%s return=%.3f error=%s',
                        checkpoint_path.name,
                        task_name,
                        int(query_instance_id),
                        label,
                        row['success'],
                        row['return'],
                        row['error'],
                    )

            for label in labels:
                items = task_rows[label]
                per_task_rows.append(
                    {
                        'condition': label,
                        'task_name': str(task_name),
                        'num_instances': len(items),
                        'success_rate': _mean([float(item['success']) for item in items]),
                        'mean_return': _mean([float(item['return']) for item in items]),
                        'mean_max_reward': _mean([float(item['max_reward']) for item in items]),
                        'num_errors': sum(1 for item in items if item.get('error')),
                    }
                )

        aggregate: Dict[str, Any] = {}
        for label in labels:
            items = results_by_condition[label]
            task_items = [row for row in per_task_rows if row['condition'] == label]
            aggregate[label] = {
                'num_tasks': len(task_items),
                'num_instances': len(items),
                'micro_success_rate': _mean([float(item['success']) for item in items]),
                'macro_success_rate': _mean([float(item['success_rate']) for item in task_items]),
                'mean_return': _mean([float(item['return']) for item in items]),
                'mean_max_reward': _mean([float(item['max_reward']) for item in items]),
                'num_errors': sum(1 for item in items if item.get('error')),
            }

        summary = {
            'checkpoint_path': str(checkpoint_path),
            'cache_root': str(store.root),
            'benchmark': str(benchmark_name),
            'split': str(split),
            'force_goal_observable': bool(force_goal_observable),
            'conditions': labels,
            'tasks': tasks,
            'instances_per_task_requested': int(instances_per_task),
            'dataset': {
                'K': int(data_cfg.K),
                'T_obs': int(data_cfg.T_obs),
                'H': int(data_cfg.H),
                'stride': int(data_cfg.stride),
                'action_stride': int(data_cfg.action_stride),
                'action_representation': str(data_cfg.action_representation),
                'task_sampling': str(data_cfg.task_sampling),
                'sample_same_task_name': bool(data_cfg.sample_same_task_name),
                'sample_same_task_instance': bool(data_cfg.sample_same_task_instance),
                'allow_support_query_same_episode': bool(data_cfg.allow_support_query_same_episode),
                'support_zero_goal': bool(data_cfg.support_zero_goal),
                'query_zero_goal': bool(data_cfg.query_zero_goal),
            },
            'memory_ttt': {
                'inner_steps': int(memory_cfg.inner_steps),
                'inner_lr': float(memory_cfg.inner_lr),
                'max_grad_norm': float(memory_cfg.max_grad_norm),
                'num_queries_per_step': int(memory_cfg.num_queries_per_step),
                'num_inner_batches': int(memory_cfg.num_inner_batches),
                'first_order': bool(memory_cfg.first_order),
                'inner_loss_mode': str(memory_cfg.inner_loss_mode),
            },
            'aggregate': aggregate,
            'per_task': per_task_rows,
            'per_instance': all_rows,
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        _write_csv(
            run_dir / 'per_task.csv',
            per_task_rows,
            (
                'condition',
                'task_name',
                'num_instances',
                'success_rate',
                'mean_return',
                'mean_max_reward',
                'num_errors',
            ),
        )
        _write_csv(
            run_dir / 'per_instance.csv',
            all_rows,
            (
                'condition',
                'task_name',
                'task_instance_id',
                'success',
                'first_success_step',
                'env_steps',
                'return',
                'max_reward',
                'error',
                'support_task_instance_ids',
                'support_episode_ids',
                'wrong_task_name',
                'wrong_support_task_instance_ids',
                'video_path',
            ),
        )
        logging.info('paper rollout eval written to %s', run_dir)
        for label, values in aggregate.items():
            logging.info(
                '%s: micro_success=%.4f macro_success=%.4f n=%d tasks=%d errors=%d',
                label,
                values['micro_success_rate'],
                values['macro_success_rate'],
                values['num_instances'],
                values['num_tasks'],
                values['num_errors'],
            )
    finally:
        store.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Unexpected positional arguments.')
    evaluate(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
