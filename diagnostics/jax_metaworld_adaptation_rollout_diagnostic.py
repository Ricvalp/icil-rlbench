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


def _build_query_window(states: List[np.ndarray], *, T_obs: int, H: int, action_dim: int) -> Dict[str, np.ndarray]:
    if not states:
        raise ValueError('states must be non-empty.')
    idx = np.linspace(max(0, len(states) - int(T_obs)), len(states) - 1, num=int(T_obs), dtype=np.int64)
    if idx.shape[0] < int(T_obs):
        idx = np.pad(idx, (int(T_obs) - idx.shape[0], 0), mode='edge')
    query_state = np.stack([states[int(i)] for i in idx.tolist()], axis=0).astype(np.float32)
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
            query = _build_query_window(states, T_obs=int(T_obs), H=int(H), action_dim=int(action_dim))
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
            target_task_name = resolve_task_name(store, str(cfg.task.name), rng)
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
            for label in labels:
                env = make_metaworld_env_for_instance(
                    task_name=target_task_name,
                    task_instance_id=int(target_spec.query_task_instance_id),
                    benchmark_name=str(getattr(cfg.sim, 'benchmark', store.index.get('benchmark', 'MT10'))),
                    split=str(getattr(cfg.sim, 'split', store.index.get('split', 'train'))),
                    benchmark_seed=int(getattr(cfg.sim, 'benchmark_seed', 0)),
                    camera_name=str(cfg.video.camera_name),
                    width=int(cfg.video.width),
                    height=int(cfg.video.height),
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
                        render=bool(cfg.video.enable),
                        frame_stride=int(cfg.video.frame_stride),
                        seed=seed + 5000 + episode_idx,
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
                }
                results[label].append(payload)
                logging.info('%s episode=%d success=%s return=%.3f error=%s', label, episode_idx, payload['success'], payload['return'], payload['error'])

        summary = {
            'checkpoint_path': str(checkpoint_path),
            'cache_root': str(store.root),
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
