from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Must be set before importing MetaWorld/MuJoCo.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax.numpy as jnp
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
    resolve_task_name,
    sample_task_spec_for_family,
    sample_wrong_family,
    set_seed,
    step_env,
    success_from_info,
    write_gif,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_metaworld_adaptation_rollout_diagnostic.py',
    help_string='Path to ml_collections config file.',
)


def _jsonable_array(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).astype(float).tolist()


def _target_reference(env: Any, obs: np.ndarray | None = None) -> np.ndarray | None:
    target = getattr(env, '_target_pos', None)
    if target is not None:
        return np.asarray(target, dtype=np.float64).reshape(-1).copy()
    if obs is not None and np.asarray(obs).shape[-1] >= 3:
        return np.asarray(obs, dtype=np.float64).reshape(-1)[-3:].copy()
    return None


def _goal_slice_from_name(name: str, *, rand_vec_dim: int, goal_dims: int) -> slice:
    name = str(name).lower()
    goal_dims = max(1, min(int(goal_dims), int(rand_vec_dim)))
    if name == 'last':
        return slice(int(rand_vec_dim) - goal_dims, int(rand_vec_dim))
    if name == 'first':
        return slice(0, goal_dims)
    if name == 'all':
        return slice(0, int(rand_vec_dim))
    raise ValueError(f'Unsupported fixed_goal_random_start_goal_slice={name!r}.')


def _sample_reset_rand_vec(env: Any, rng: np.random.Generator) -> np.ndarray:
    space = getattr(env, '_random_reset_space', None)
    if space is None:
        raise RuntimeError('MetaWorld env does not expose _random_reset_space; cannot resample starts.')
    low = np.asarray(space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(space.high, dtype=np.float64).reshape(-1)
    return rng.uniform(low, high, size=low.shape).astype(np.float64)


def _reset_env_with_fixed_goal_random_start(
    env: Any,
    *,
    seed: int,
    base_rand_vec: np.ndarray,
    goal_slice: slice,
    goal_reference: np.ndarray | None,
    validate_goal: bool,
    goal_tolerance: float,
    max_resample_calls: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    original_get_state_rand_vec = env._get_state_rand_vec
    base_rand_vec = np.asarray(base_rand_vec, dtype=np.float64).reshape(-1)
    goal_reference_arr = None if goal_reference is None else np.asarray(goal_reference, dtype=np.float64).reshape(-1)
    call_count = 0

    def _patched_get_state_rand_vec() -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count > int(max_resample_calls):
            mixed = base_rand_vec.copy()
        else:
            mixed = _sample_reset_rand_vec(env, rng)
            if mixed.shape != base_rand_vec.shape:
                raise RuntimeError(f'Resampled rand_vec shape {mixed.shape} differs from base {base_rand_vec.shape}.')
            mixed[goal_slice] = base_rand_vec[goal_slice]
        env._last_rand_vec = mixed
        return mixed

    try:
        env._get_state_rand_vec = _patched_get_state_rand_vec
        env._freeze_rand_vec = False
        obs, info = reset_env(env, seed=seed)
    finally:
        env._get_state_rand_vec = original_get_state_rand_vec
        env._freeze_rand_vec = True

    actual_rand_vec = np.asarray(getattr(env, '_last_rand_vec', base_rand_vec), dtype=np.float64).reshape(-1).copy()
    current_goal = _target_reference(env, obs)
    goal_diff = 0.0
    if goal_reference_arr is not None and current_goal is not None:
        if goal_reference_arr.shape != current_goal.shape:
            goal_diff = float('inf')
        else:
            goal_diff = float(np.max(np.abs(current_goal - goal_reference_arr)))
    if bool(validate_goal) and goal_diff > float(goal_tolerance):
        raise RuntimeError(
            'fixed_goal_random_start changed the target goal: '
            f'max_abs_diff={goal_diff:.6g}, tolerance={float(goal_tolerance):.6g}.'
        )
    info = dict(info or {})
    info['fixed_goal_random_start'] = {
        'enabled': True,
        'goal_slice_start': None if goal_slice.start is None else int(goal_slice.start),
        'goal_slice_stop': None if goal_slice.stop is None else int(goal_slice.stop),
        'resample_calls': int(call_count),
        'base_rand_vec': _jsonable_array(base_rand_vec),
        'actual_rand_vec': _jsonable_array(actual_rand_vec),
        'goal_reference': _jsonable_array(goal_reference_arr),
        'actual_goal_reference': _jsonable_array(current_goal),
        'goal_max_abs_diff': float(goal_diff),
    }
    return obs, info


def _candidate_goal_slices(name: str, *, rand_vec_dim: int, goal_dims: int) -> List[slice]:
    name = str(name).lower()
    if name != 'auto':
        return [_goal_slice_from_name(name, rand_vec_dim=rand_vec_dim, goal_dims=goal_dims)]
    candidates: List[slice] = []
    for candidate_name in ('last', 'first', 'all'):
        candidate = _goal_slice_from_name(candidate_name, rand_vec_dim=rand_vec_dim, goal_dims=goal_dims)
        if not any(candidate.start == other.start and candidate.stop == other.stop for other in candidates):
            candidates.append(candidate)
    return candidates


def _prepare_fixed_goal_random_start(env: Any, cfg: ConfigDict, *, seed: int) -> Dict[str, Any]:
    base_rand_vec = np.asarray(getattr(env, '_last_rand_vec', None), dtype=np.float64).reshape(-1)
    if base_rand_vec.size == 0:
        raise RuntimeError('MetaWorld task did not set _last_rand_vec; cannot fix goal while resampling starts.')

    env._last_rand_vec = base_rand_vec.copy()
    env._freeze_rand_vec = True
    base_obs, _ = reset_env(env, seed=seed)
    goal_reference = _target_reference(env, base_obs)

    goal_slice_name = str(getattr(cfg.metaworld, 'fixed_goal_random_start_goal_slice', 'auto'))
    goal_dims = int(getattr(cfg.metaworld, 'fixed_goal_random_start_goal_dims', 3))
    validate_goal = bool(getattr(cfg.metaworld, 'fixed_goal_random_start_validate_goal', True))
    goal_tolerance = float(getattr(cfg.metaworld, 'fixed_goal_random_start_goal_tolerance', 1e-5))
    max_resample_calls = int(getattr(cfg.metaworld, 'fixed_goal_random_start_max_resample_calls', 256))

    for i, goal_slice in enumerate(_candidate_goal_slices(goal_slice_name, rand_vec_dim=base_rand_vec.size, goal_dims=goal_dims)):
        try:
            obs, info = _reset_env_with_fixed_goal_random_start(
                env,
                seed=int(seed) + 17 + i,
                base_rand_vec=base_rand_vec,
                goal_slice=goal_slice,
                goal_reference=goal_reference,
                validate_goal=validate_goal,
                goal_tolerance=goal_tolerance,
                max_resample_calls=max_resample_calls,
            )
        except Exception as exc:
            logging.debug('Rejected fixed-goal random-start slice %s for %s: %s', goal_slice, env.__class__.__name__, exc)
            continue
        del obs
        meta = dict(info.get('fixed_goal_random_start', {}))
        actual = np.asarray(meta.get('actual_rand_vec', []), dtype=np.float64)
        outside = np.ones(base_rand_vec.shape[0], dtype=bool)
        outside[goal_slice] = False
        changed_outside = bool(np.any(np.abs(actual[outside] - base_rand_vec[outside]) > 1e-9)) if actual.shape == base_rand_vec.shape else False
        if goal_slice.stop - goal_slice.start >= base_rand_vec.size:
            changed_outside = False
        if goal_slice_name.lower() != 'auto' or changed_outside or goal_slice.stop - goal_slice.start >= base_rand_vec.size:
            return {
                'enabled': True,
                'base_rand_vec': base_rand_vec.copy(),
                'goal_slice': goal_slice,
                'goal_reference': None if goal_reference is None else goal_reference.copy(),
                'validate_goal': validate_goal,
                'goal_tolerance': goal_tolerance,
                'max_resample_calls': max_resample_calls,
            }

    logging.warning(
        'Falling back to fixed full rand_vec for %s; goal/start may not be separable for this task instance.',
        env.__class__.__name__,
    )
    return {
        'enabled': True,
        'base_rand_vec': base_rand_vec.copy(),
        'goal_slice': slice(0, base_rand_vec.size),
        'goal_reference': None if goal_reference is None else goal_reference.copy(),
        'validate_goal': validate_goal,
        'goal_tolerance': goal_tolerance,
        'max_resample_calls': max_resample_calls,
    }


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
    fixed_goal_random_start_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if fixed_goal_random_start_state is None:
        obs, _ = reset_env(env, seed=seed)
        reset_metadata = {}
    else:
        obs, reset_info = _reset_env_with_fixed_goal_random_start(
            env,
            seed=seed,
            base_rand_vec=fixed_goal_random_start_state['base_rand_vec'],
            goal_slice=fixed_goal_random_start_state['goal_slice'],
            goal_reference=fixed_goal_random_start_state.get('goal_reference'),
            validate_goal=bool(fixed_goal_random_start_state.get('validate_goal', True)),
            goal_tolerance=float(fixed_goal_random_start_state.get('goal_tolerance', 1e-5)),
            max_resample_calls=int(fixed_goal_random_start_state.get('max_resample_calls', 256)),
        )
        reset_metadata = dict(reset_info or {}).get('fixed_goal_random_start', {})
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
        'reset_metadata': reset_metadata,
        'frames': frames,
    }


def _initial_plan_diagnostics(
    *,
    env: Any,
    params: Any,
    predict_fn: Any,
    memories: Dict[str, Any],
    state_dim: int,
    T_obs: int,
    H: int,
    action_dim: int,
    seed: int,
    query_zero_goal: bool,
    fixed_goal_random_start_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if fixed_goal_random_start_state is None:
        obs, _ = reset_env(env, seed=seed)
        reset_metadata = {}
    else:
        obs, reset_info = _reset_env_with_fixed_goal_random_start(
            env,
            seed=seed,
            base_rand_vec=fixed_goal_random_start_state['base_rand_vec'],
            goal_slice=fixed_goal_random_start_state['goal_slice'],
            goal_reference=fixed_goal_random_start_state.get('goal_reference'),
            validate_goal=bool(fixed_goal_random_start_state.get('validate_goal', True)),
            goal_tolerance=float(fixed_goal_random_start_state.get('goal_tolerance', 1e-5)),
            max_resample_calls=int(fixed_goal_random_start_state.get('max_resample_calls', 256)),
        )
        reset_metadata = dict(reset_info or {}).get('fixed_goal_random_start', {})
    state = obs_to_model_state(obs, state_dim=int(state_dim))
    query = _build_query_window(
        [state],
        T_obs=int(T_obs),
        H=int(H),
        action_dim=int(action_dim),
        query_zero_goal=bool(query_zero_goal),
    )
    query_jax = numpy_batch_to_jax(query)
    plans = {
        label: np.asarray(predict_fn(params, query_jax, memory_tokens), dtype=np.float32)[0]
        for label, memory_tokens in memories.items()
    }
    labels = list(plans.keys())
    pairwise: Dict[str, Dict[str, float]] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            key = f'{a}_vs_{b}'
            first_a = plans[a][0]
            first_b = plans[b][0]
            pairwise[key] = {
                'plan_mean_abs': float(np.mean(np.abs(plans[a] - plans[b]))),
                'first_action_mean_abs': float(np.mean(np.abs(first_a - first_b))),
                'first_action_clipped_mean_abs': float(np.mean(np.abs(np.clip(first_a, -1.0, 1.0) - np.clip(first_b, -1.0, 1.0)))),
            }
    return {
        'first_actions': {label: plans[label][0].astype(float).tolist() for label in labels},
        'plan_mean_abs': {label: float(np.mean(np.abs(plan))) for label, plan in plans.items()},
        'pairwise': pairwise,
        'reset_metadata': reset_metadata,
    }


def _as_auto_bool(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('', 'auto'):
            return 'auto'
        if text in ('1', 'true', 'yes', 'y'):
            return True
        if text in ('0', 'false', 'no', 'n'):
            return False
    return bool(value)


def _infer_force_goal_observable(cfg: ConfigDict, *, store: Any, data_cfg: Any, state_dim: int) -> bool:
    configured = _as_auto_bool(getattr(cfg.sim, 'force_goal_observable', 'auto'))
    if configured != 'auto':
        return bool(configured)
    obs_cfg = store.index.get('obs', {}) if isinstance(store.index, dict) else {}
    cache_keeps_goal = str(obs_cfg.get('variant', '')).lower() in ('raw', 'none') and not bool(obs_cfg.get('remove_goal', True))
    return bool(cache_keeps_goal and not bool(data_cfg.query_zero_goal) and int(state_dim) >= 39)


def _maybe_prepare_fixed_goal_random_start(cfg: ConfigDict, env: Any, *, seed: int) -> Dict[str, Any] | None:
    if not bool(getattr(cfg.sim, 'fixed_goal_random_start', False)):
        return None
    reset_cfg = ConfigDict()
    reset_cfg.metaworld = ConfigDict()
    reset_cfg.metaworld.fixed_goal_random_start_goal_slice = str(
        getattr(cfg.sim, 'fixed_goal_random_start_goal_slice', 'auto')
    )
    reset_cfg.metaworld.fixed_goal_random_start_goal_dims = int(
        getattr(cfg.sim, 'fixed_goal_random_start_goal_dims', 3)
    )
    reset_cfg.metaworld.fixed_goal_random_start_validate_goal = bool(
        getattr(cfg.sim, 'fixed_goal_random_start_validate_goal', True)
    )
    reset_cfg.metaworld.fixed_goal_random_start_goal_tolerance = float(
        getattr(cfg.sim, 'fixed_goal_random_start_goal_tolerance', 1e-5)
    )
    reset_cfg.metaworld.fixed_goal_random_start_max_resample_calls = int(
        getattr(cfg.sim, 'fixed_goal_random_start_max_resample_calls', 256)
    )
    return _prepare_fixed_goal_random_start(env, reset_cfg, seed=seed)


def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    set_seed(seed)
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
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / 'jax_metaworld_adaptation_rollout' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    rng = np.random.default_rng(seed + 17001)
    labels = ('no_adaptation', 'same_family_adaptation', 'wrong_family_adaptation')
    results: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    try:
        for episode_idx in range(int(cfg.rollout.num_episodes)):
            target_task_name = resolve_task_name(
                store,
                str(cfg.task.name),
                rng,
                task_names=tuple(cfg.data.tasks),
                exclude_tasks=tuple(cfg.data.exclude_tasks),
            )
            wrong_task_name = sample_wrong_family(store, target_task_name, str(cfg.adaptation.different_task_name), rng)
            target_builder = make_builder(store, data_cfg, seed=seed + 1000 + episode_idx, task_names=(target_task_name,))
            wrong_builder = make_builder(store, data_cfg, seed=seed + 2000 + episode_idx, task_names=(wrong_task_name,))
            target_spec = sample_task_spec_for_family(
                store=store,
                data_cfg=data_cfg,
                task_name=target_task_name,
                seed=seed + 3000 + episode_idx,
                rng=rng,
            )
            wrong_spec = sample_task_spec_for_family(
                store=store,
                data_cfg=data_cfg,
                task_name=wrong_task_name,
                seed=seed + 4000 + episode_idx,
                rng=rng,
            )
            same_adaptation = adapt_memory_for_task(
                params=params,
                adapt_with_stats_fn=components['adapt_with_stats_fn'],
                builder=target_builder,
                task=target_spec,
                memory_cfg=memory_cfg,
                rng=rng,
                run_dir=run_dir if episode_idx == 0 else None,
                stem='same_family_adaptation',
            )
            wrong_adaptation = adapt_memory_for_task(
                params=params,
                adapt_with_stats_fn=components['adapt_with_stats_fn'],
                builder=wrong_builder,
                task=wrong_spec,
                memory_cfg=memory_cfg,
                rng=rng,
                run_dir=run_dir if episode_idx == 0 else None,
                stem='wrong_family_adaptation',
            )
            memories = {
                'no_adaptation': initial_memory(params),
                'same_family_adaptation': same_adaptation['memory_tokens'],
                'wrong_family_adaptation': wrong_adaptation['memory_tokens'],
            }
            initial_plan_diagnostics = {}
            if bool(getattr(cfg.rollout, 'log_initial_plan_deltas', True)):
                env = make_metaworld_env_for_instance(
                    task_name=target_task_name,
                    task_instance_id=int(target_spec.query_task_instance_id),
                    benchmark_name=benchmark_name,
                    split=split,
                    benchmark_seed=int(getattr(cfg.sim, 'benchmark_seed', 0)),
                    camera_name=str(cfg.video.camera_name),
                    width=int(cfg.video.width),
                    height=int(cfg.video.height),
                    force_goal_observable=bool(force_goal_observable),
                )
                try:
                    fixed_goal_random_start_state = _maybe_prepare_fixed_goal_random_start(
                        cfg,
                        env,
                        seed=seed + 4500 + episode_idx,
                    )
                    initial_plan_diagnostics = _initial_plan_diagnostics(
                        env=env,
                        params=params,
                        predict_fn=predict_fn,
                        memories=memories,
                        state_dim=state_dim,
                        T_obs=int(data_cfg.T_obs),
                        H=int(data_cfg.H),
                        action_dim=action_dim,
                        seed=seed + 5000 + episode_idx,
                        query_zero_goal=bool(data_cfg.query_zero_goal),
                        fixed_goal_random_start_state=fixed_goal_random_start_state,
                    )
                finally:
                    env.close()
            for label in labels:
                env = make_metaworld_env_for_instance(
                    task_name=target_task_name,
                    task_instance_id=int(target_spec.query_task_instance_id),
                    benchmark_name=benchmark_name,
                    split=split,
                    benchmark_seed=int(getattr(cfg.sim, 'benchmark_seed', 0)),
                    camera_name=str(cfg.video.camera_name),
                    width=int(cfg.video.width),
                    height=int(cfg.video.height),
                    force_goal_observable=bool(force_goal_observable),
                )
                try:
                    fixed_goal_random_start_state = _maybe_prepare_fixed_goal_random_start(
                        cfg,
                        env,
                        seed=seed + 4500 + episode_idx,
                    )
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
                        render=bool(cfg.video.enable),
                        frame_stride=int(cfg.video.frame_stride),
                        seed=seed + 5000 + episode_idx,
                        query_zero_goal=bool(data_cfg.query_zero_goal),
                        fixed_goal_random_start_state=fixed_goal_random_start_state,
                    )
                finally:
                    env.close()
                frames = result.pop('frames')
                video_path = ''
                if bool(cfg.video.enable) and frames:
                    video_path = write_gif(
                        run_dir / f'{label}_videos' / f'episode_{episode_idx:04d}_{target_task_name}_inst{target_spec.query_task_instance_id}.gif',
                        frames,
                        fps=int(cfg.video.fps),
                    )
                payload = {
                    **result,
                    'video_path': video_path,
                    'target_task_name': str(target_task_name),
                    'target_query_task_instance_id': int(target_spec.query_task_instance_id),
                    'target_support_task_instance_ids': [int(v) for v in target_spec.support_task_instance_ids],
                    'wrong_task_name': str(wrong_task_name),
                    'wrong_support_task_instance_ids': [int(v) for v in wrong_spec.support_task_instance_ids],
                    'initial_plan_diagnostics': initial_plan_diagnostics,
                }
                results[label].append(payload)
                logging.info('%s episode=%d success=%s return=%.3f error=%s', label, episode_idx, payload['success'], payload['return'], payload['error'])

        summary = {
            'checkpoint_path': str(checkpoint_path),
            'cache_root': str(store.root),
            'benchmark': benchmark_name,
            'split': split,
            'force_goal_observable': bool(force_goal_observable),
            'fixed_goal_random_start': bool(getattr(cfg.sim, 'fixed_goal_random_start', False)),
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
            'results': results,
            'summary': {
                label: {
                    'num_episodes': len(items),
                    'success_rate': float(np.mean([float(x['success']) for x in items])) if items else 0.0,
                    'mean_return': float(np.mean([float(x['return']) for x in items])) if items else 0.0,
                    'mean_max_reward': float(np.mean([float(x['max_reward']) for x in items])) if items else 0.0,
                }
                for label, items in results.items()
            },
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logging.info('rollout diagnostics written to %s', run_dir)
    finally:
        store.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Unexpected positional arguments.')
    evaluate(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
