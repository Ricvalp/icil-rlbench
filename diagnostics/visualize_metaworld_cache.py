from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from icil_metaworld.data.metaworld_cache import MetaWorldEpisodeStore
from icil_metaworld.data.metaworld_task_builder import MetaWorldICILConfig, MetaWorldMAMLTaskSpec, MetaWorldQueryMemoryTaskBuilder
from icil_metaworld.data.observation_filter import normalize_env_name


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('This diagnostic requires matplotlib: pip install matplotlib') from exc
    return plt


def _xyz_from_obs_raw(obs_raw: np.ndarray) -> Dict[str, np.ndarray]:
    obs = np.asarray(obs_raw, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[-1] < 7:
        raise ValueError(f'Expected obs_raw [T, >=7], got {obs.shape}.')
    out = {
        'hand': obs[:, 0:3],
        'object': obs[:, 4:7],
    }
    if obs.shape[-1] >= 39:
        out['goal_slot'] = obs[:, -3:]
    return out


def _set_equal_3d(ax: Any, points: Sequence[np.ndarray], *, pad: float = 0.03) -> None:
    valid = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in points if p is not None and np.asarray(p).size > 0]
    if not valid:
        return
    pts = np.concatenate(valid, axis=0)
    finite = np.all(np.isfinite(pts), axis=-1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins)) + float(pad)
    half = max(half, 1e-3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _load_full_episode(store: MetaWorldEpisodeStore, task_name: str, episode_id: int) -> Dict[str, Any]:
    T = int(store.episode_length(task_name, int(episode_id)))
    idx = np.arange(T, dtype=np.int64)
    return store.load_episode_slices(task_name, int(episode_id), idx, load_raw_obs=True, load_full_traj=True)


def _valid_t0_bounds(store: MetaWorldEpisodeStore, cfg: MetaWorldICILConfig, task_name: str, episode_id: int) -> tuple[int, int]:
    T = int(store.episode_length(task_name, int(episode_id)))
    min_t0 = (int(cfg.T_obs) - 1) * int(cfg.stride)
    if bool(cfg.pad_short_chunks):
        max_t0 = max(min_t0, T - 2)
    else:
        max_t0 = T - 1 - int(cfg.H) * int(cfg.action_stride)
    if max_t0 < min_t0:
        raise RuntimeError(f'Episode {episode_id} is too short for T_obs={cfg.T_obs}, H={cfg.H}.')
    return min_t0, max_t0


def _indices_for_chunk(cfg: MetaWorldICILConfig, *, t0: int, episode_length: int) -> tuple[np.ndarray, np.ndarray]:
    obs_idx = int(t0) - np.arange((int(cfg.T_obs) - 1) * int(cfg.stride), -1, -int(cfg.stride), dtype=np.int64)
    act_idx = int(t0) + np.arange(1, int(cfg.H) + 1, dtype=np.int64) * int(cfg.action_stride)
    if bool(cfg.pad_short_chunks):
        act_idx = np.minimum(act_idx, int(episode_length) - 1)
    return obs_idx.astype(np.int64), act_idx.astype(np.int64)


def _sample_task(
    store: MetaWorldEpisodeStore,
    *,
    task_name: str,
    task_instance_id: int,
    K: int,
    seed: int,
) -> MetaWorldMAMLTaskSpec:
    rng = np.random.default_rng(int(seed))
    task_name = normalize_env_name(task_name) if task_name else store.list_task_names()[0]
    if int(task_instance_id) < 0:
        keys = store.list_task_instance_keys(tasks=(task_name,))
        if not keys:
            raise RuntimeError(f'No task instances found for {task_name!r}.')
        _, task_instance_id = keys[int(rng.integers(0, len(keys)))]
    episode_ids = store.list_episode_ids(task_name, task_instance_id=int(task_instance_id))
    if episode_ids.shape[0] < int(K) + 1:
        raise RuntimeError(
            f'{task_name} instance {task_instance_id} has {episode_ids.shape[0]} episodes, '
            f'but K={K} needs at least K+1 distinct episodes.'
        )
    chosen = rng.choice(episode_ids, size=int(K) + 1, replace=False).astype(np.int64)
    return MetaWorldMAMLTaskSpec(
        task_name=str(task_name),
        task_index=int(store.task_index(task_name)),
        task_instance_id=int(task_instance_id),
        support_episode_ids=tuple(int(x) for x in chosen[: int(K)].tolist()),
        query_episode_id=int(chosen[int(K)]),
    )


def _sample_chunks(
    store: MetaWorldEpisodeStore,
    cfg: MetaWorldICILConfig,
    task: MetaWorldMAMLTaskSpec,
    *,
    num_query_chunks: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 17)
    chunks: List[Dict[str, Any]] = []
    for demo_idx, episode_id in enumerate(task.support_episode_ids):
        lo, hi = _valid_t0_bounds(store, cfg, task.task_name, int(episode_id))
        t0 = int(rng.integers(lo, hi + 1))
        chunks.append({'role': 'support', 'demo_id': int(demo_idx), 'episode_id': int(episode_id), 't0': int(t0)})
    for query_idx in range(int(num_query_chunks)):
        lo, hi = _valid_t0_bounds(store, cfg, task.task_name, int(task.query_episode_id))
        t0 = int(rng.integers(lo, hi + 1))
        chunks.append({'role': 'query', 'demo_id': None, 'query_chunk_id': int(query_idx), 'episode_id': int(task.query_episode_id), 't0': int(t0)})
    return chunks


def _plot_full_episode_overview(
    *,
    store: MetaWorldEpisodeStore,
    task: MetaWorldMAMLTaskSpec,
    output_path: Path,
    action_scale: float,
    max_quivers: int,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    episode_ids = list(task.support_episode_ids) + [int(task.query_episode_id)]
    cols = min(3, len(episode_ids))
    rows = int(np.ceil(len(episode_ids) / cols))
    fig = plt.figure(figsize=(5.2 * cols, 4.4 * rows))
    all_points: List[np.ndarray] = []
    axes = []

    for i, episode_id in enumerate(episode_ids):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        axes.append(ax)
        data = _load_full_episode(store, task.task_name, int(episode_id))
        raw = data['obs_raw']
        actions = data['action']
        success = data['success']
        xyz = _xyz_from_obs_raw(raw)
        hand = xyz['hand']
        obj = xyz['object']
        all_points.extend([hand, obj])
        role = 'query' if int(episode_id) == int(task.query_episode_id) else 'support'
        color = 'tab:purple' if role == 'query' else 'tab:blue'
        ax.plot(hand[:, 0], hand[:, 1], hand[:, 2], color=color, linewidth=1.6, label='hand')
        ax.scatter(hand[0, 0], hand[0, 1], hand[0, 2], color='black', s=22, label='start')
        ax.scatter(hand[-1, 0], hand[-1, 1], hand[-1, 2], color=color, s=36, marker='x', label='end')
        ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color='tab:orange', linewidth=1.2, alpha=0.9, label='object/button')
        if np.any(success):
            sidx = int(np.argmax(success))
            ax.scatter(hand[sidx, 0], hand[sidx, 1], hand[sidx, 2], color='tab:green', s=50, marker='*', label='first success')
        q_idx = np.linspace(0, max(0, hand.shape[0] - 1), num=min(max_quivers, hand.shape[0]), dtype=np.int64)
        vec = actions[q_idx, :3] * float(action_scale)
        ax.quiver(hand[q_idx, 0], hand[q_idx, 1], hand[q_idx, 2], vec[:, 0], vec[:, 1], vec[:, 2], color='tab:red', length=1.0, normalize=False, alpha=0.55)
        ax.set_title(f'{role} episode {episode_id}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if i == 0:
            ax.legend(loc='upper left', fontsize=8)

    for ax in axes:
        _set_equal_3d(ax, all_points)
    fig.suptitle(f'MetaWorld cache full-episode overview: {task.task_name} instance {task.task_instance_id}', y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)


def _plot_support_query_chunks(
    *,
    store: MetaWorldEpisodeStore,
    cfg: MetaWorldICILConfig,
    task: MetaWorldMAMLTaskSpec,
    chunks: Sequence[Dict[str, Any]],
    output_path: Path,
    action_scale: float,
    max_quivers: int,
    dpi: int,
) -> None:
    plt = _require_matplotlib()
    cols = min(4, len(chunks))
    rows = int(np.ceil(len(chunks) / cols))
    fig = plt.figure(figsize=(5.0 * cols, 4.2 * rows))
    all_points: List[np.ndarray] = []
    axes = []

    for i, chunk in enumerate(chunks):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        axes.append(ax)
        episode_id = int(chunk['episode_id'])
        T = int(store.episode_length(task.task_name, episode_id))
        obs_idx, act_idx = _indices_for_chunk(cfg, t0=int(chunk['t0']), episode_length=T)
        full = _load_full_episode(store, task.task_name, episode_id)
        raw = full['obs_raw']
        actions = full['action']
        xyz = _xyz_from_obs_raw(raw)
        hand = xyz['hand']
        obj = xyz['object']
        obs_hand = hand[obs_idx]
        fut_hand = hand[act_idx]
        act = actions[act_idx, :3]
        all_points.extend([hand, obj, obs_hand, fut_hand])

        role = str(chunk['role'])
        main_color = 'tab:purple' if role == 'query' else 'tab:blue'
        ax.plot(hand[:, 0], hand[:, 1], hand[:, 2], color='0.80', linewidth=0.8, label='episode hand')
        ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color='tab:orange', linewidth=1.0, alpha=0.8, label='object/button')
        ax.plot(obs_hand[:, 0], obs_hand[:, 1], obs_hand[:, 2], color=main_color, linewidth=2.6, marker='o', label='T_obs')
        ax.plot(fut_hand[:, 0], fut_hand[:, 1], fut_hand[:, 2], color='tab:green', linewidth=2.0, marker='.', label='future hand')
        q_idx = np.linspace(0, max(0, act_idx.shape[0] - 1), num=min(max_quivers, act_idx.shape[0]), dtype=np.int64)
        base = hand[act_idx[q_idx]]
        vec = act[q_idx] * float(action_scale)
        ax.quiver(base[:, 0], base[:, 1], base[:, 2], vec[:, 0], vec[:, 1], vec[:, 2], color='tab:red', length=1.0, normalize=False, alpha=0.8, label='action xyz')
        ax.scatter(hand[int(chunk['t0']), 0], hand[int(chunk['t0']), 1], hand[int(chunk['t0']), 2], color='black', s=24, label='t0')
        if role == 'support':
            title = f'support demo {chunk["demo_id"]}\nep {episode_id}, t0 {chunk["t0"]}'
        else:
            title = f'query chunk {chunk["query_chunk_id"]}\nep {episode_id}, t0 {chunk["t0"]}'
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if i == 0:
            ax.legend(loc='upper left', fontsize=7)

    for ax in axes:
        _set_equal_3d(ax, all_points)
    fig.suptitle(f'Sampled support/query chunks: {task.task_name} instance {task.task_instance_id}', y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)


def _write_metadata(path: Path, *, task: MetaWorldMAMLTaskSpec, cfg: MetaWorldICILConfig, chunks: Sequence[Dict[str, Any]]) -> None:
    payload = {
        'task': {
            'task_name': task.task_name,
            'task_index': int(task.task_index),
            'task_instance_id': int(task.task_instance_id),
            'support_episode_ids': [int(x) for x in task.support_episode_ids],
            'query_episode_id': int(task.query_episode_id),
        },
        'dataset': {
            'K': int(cfg.K),
            'T_obs': int(cfg.T_obs),
            'H': int(cfg.H),
            'stride': int(cfg.stride),
            'action_stride': int(cfg.action_stride),
            'pad_short_chunks': bool(cfg.pad_short_chunks),
        },
        'chunks': list(chunks),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _selected_episode_duplicate_summary(
    store: MetaWorldEpisodeStore,
    task: MetaWorldMAMLTaskSpec,
) -> Dict[str, Any]:
    episode_ids = [int(x) for x in task.support_episode_ids] + [int(task.query_episode_id)]
    if len(episode_ids) < 2:
        return {'episode_ids': episode_ids, 'all_identical_to_first': False, 'comparisons': []}
    first = _load_full_episode(store, task.task_name, episode_ids[0])
    comparisons = []
    all_identical = True
    for episode_id in episode_ids[1:]:
        cur = _load_full_episode(store, task.task_name, episode_id)
        same_obs = bool(np.array_equal(first['obs_raw'], cur['obs_raw']))
        same_action = bool(np.array_equal(first['action'], cur['action']))
        if not (same_obs and same_action):
            all_identical = False
        comparisons.append(
            {
                'episode_id': int(episode_id),
                'obs_raw_max_abs_diff': float(np.max(np.abs(first['obs_raw'] - cur['obs_raw']))),
                'action_max_abs_diff': float(np.max(np.abs(first['action'] - cur['action']))),
                'same_obs_raw': same_obs,
                'same_action': same_action,
            }
        )
    return {
        'episode_ids': episode_ids,
        'reference_episode_id': int(episode_ids[0]),
        'all_identical_to_first': bool(all_identical),
        'comparisons': comparisons,
    }


def visualize(args: argparse.Namespace) -> None:
    cache_root = Path(str(args.cache_root or os.environ.get('ICIL_METAWORLD_CACHE_ROOT', ''))).expanduser()
    if not str(cache_root):
        raise ValueError('Provide --cache-root or set ICIL_METAWORLD_CACHE_ROOT.')
    output_dir = Path(str(args.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    store = MetaWorldEpisodeStore(cache_root, keep_open_per_worker=False)
    try:
        task_name = normalize_env_name(args.task_name) if args.task_name else store.list_task_names()[0]
        cfg = MetaWorldICILConfig(
            K=int(args.K),
            T_obs=int(args.T_obs),
            H=int(args.H),
            stride=int(args.stride),
            action_stride=int(args.action_stride),
            pad_short_chunks=bool(args.pad_short_chunks),
        )
        task = _sample_task(
            store,
            task_name=task_name,
            task_instance_id=int(args.task_instance_id),
            K=int(args.K),
            seed=int(args.seed),
        )
        # Construct once to validate the diagnostic config against the same task
        # builder used by training. Sampling below is explicit so the metadata is
        # stable and easy to inspect.
        MetaWorldQueryMemoryTaskBuilder(store, cfg=cfg, seed=int(args.seed), task_names=(task.task_name,))
        chunks = _sample_chunks(
            store,
            cfg,
            task,
            num_query_chunks=int(args.num_query_chunks),
            seed=int(args.seed),
        )
        stem = f'{task.task_name}_instance-{task.task_instance_id}_seed-{int(args.seed)}'
        overview_path = output_dir / f'{stem}.episodes_3d.png'
        chunks_path = output_dir / f'{stem}.support_query_chunks_3d.png'
        meta_path = output_dir / f'{stem}.support_query_chunks.json'
        duplicate_summary = _selected_episode_duplicate_summary(store, task)
        if bool(duplicate_summary.get('all_identical_to_first', False)):
            print(
                'WARNING: all selected support/query episodes are byte-identical in obs_raw and actions. '
                'The full-episode subplots will look identical; this usually means the MetaWorld task instance '
                'has a fixed reset and the scripted expert is deterministic.'
            )
        _plot_full_episode_overview(
            store=store,
            task=task,
            output_path=overview_path,
            action_scale=float(args.action_scale),
            max_quivers=int(args.max_quivers),
            dpi=int(args.dpi),
        )
        _plot_support_query_chunks(
            store=store,
            cfg=cfg,
            task=task,
            chunks=chunks,
            output_path=chunks_path,
            action_scale=float(args.action_scale),
            max_quivers=int(args.max_quivers),
            dpi=int(args.dpi),
        )
        _write_metadata(meta_path, task=task, cfg=cfg, chunks=chunks)
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        meta['duplicate_summary'] = duplicate_summary
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding='utf-8')
        print(f'Wrote {overview_path}')
        print(f'Wrote {chunks_path}')
        print(f'Wrote {meta_path}')
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize MetaWorld ICIL cache support/query samples in 3D.')
    parser.add_argument('--cache-root', default=os.environ.get('ICIL_METAWORLD_CACHE_ROOT', ''), help='MetaWorld cache root containing cache.h5 and index.json.')
    parser.add_argument('--output-dir', default='diagnostics/metaworld_cache_viz', help='Directory for PNG/JSON outputs.')
    parser.add_argument('--task-name', default='button-press-v3', help='Task name, e.g. button-press-v3. v2 aliases are accepted.')
    parser.add_argument('--task-instance-id', type=int, default=-1, help='Task instance/goal id. Use -1 to sample one.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--K', type=int, default=4, help='Number of support episodes to sample.')
    parser.add_argument('--T-obs', '--T_obs', dest='T_obs', type=int, default=2)
    parser.add_argument('--H', type=int, default=8, help='Action chunk horizon.')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--action-stride', type=int, default=1)
    parser.add_argument('--pad-short-chunks', action='store_true')
    parser.add_argument('--num-query-chunks', type=int, default=4, help='Number of query chunks to visualize from the query episode.')
    parser.add_argument('--action-scale', type=float, default=0.035, help='Scale applied to normalized MetaWorld action xyz vectors for display.')
    parser.add_argument('--max-quivers', type=int, default=32, help='Maximum action arrows per subplot.')
    parser.add_argument('--dpi', type=int, default=160)
    visualize(parser.parse_args())


if __name__ == '__main__':
    main()
