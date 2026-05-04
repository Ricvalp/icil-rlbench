from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Must be set before mujoco/glfw is imported by MetaWorld/Gymnasium.
os.environ.setdefault('MUJOCO_GL', 'egl')

import imageio.v2 as imageio
import numpy as np

from icil_metaworld.data.import_utils import import_metaworld
from icil_metaworld.data.metaworld_cache import MetaWorldEpisodeStore
from icil_metaworld.data.observation_filter import normalize_env_name


def _reset_env(env: Any, *, seed: int | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return np.asarray(obs, dtype=np.float32), dict(info or {})


def _step_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    out = env.step(np.asarray(action, dtype=np.float32))
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
    else:
        raise RuntimeError(f'Unsupported env.step return: {type(out).__name__} len={len(out)}')
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info or {})


def _make_benchmark(metaworld: Any, benchmark_name: str, task_name: str, seed: int) -> Any:
    benchmark_name = str(benchmark_name).upper()
    task_name = normalize_env_name(task_name)
    if benchmark_name == 'ML1':
        return metaworld.ML1(task_name, seed=int(seed))
    if benchmark_name == 'MT1':
        return metaworld.MT1(task_name, seed=int(seed))
    if benchmark_name in ('ML10', 'ML25', 'ML45', 'MT10', 'MT25', 'MT50'):
        return getattr(metaworld, benchmark_name)(seed=int(seed))
    raise ValueError(f'Unsupported MetaWorld benchmark {benchmark_name!r}.')


def _resolve_episode(store: MetaWorldEpisodeStore, *, task_name: str, task_instance_id: int, episode_id: int) -> tuple[str, int, int]:
    if episode_id >= 0:
        meta = store.episode_metadata(int(episode_id))
        return str(meta['task_name']), int(meta['task_instance_id']), int(episode_id)
    task_name = normalize_env_name(task_name) if task_name else store.list_task_names()[0]
    if task_instance_id < 0:
        keys = store.list_task_instance_keys(tasks=(task_name,))
        if not keys:
            raise RuntimeError(f'No task instances found for {task_name!r}.')
        task_name, task_instance_id = keys[0]
    episode_ids = store.list_episode_ids(task_name, task_instance_id=int(task_instance_id))
    if episode_ids.shape[0] == 0:
        raise RuntimeError(f'No episodes found for {task_name} instance {task_instance_id}.')
    return str(task_name), int(task_instance_id), int(episode_ids[0])


def _make_env_for_task(
    *,
    task_name: str,
    task_instance_id: int,
    benchmark_name: str,
    split: str,
    benchmark_seed: int,
    camera_name: str,
    width: int,
    height: int,
    force_goal_observable: bool = False,
) -> Any:
    metaworld = import_metaworld()
    benchmark = _make_benchmark(metaworld, benchmark_name, task_name, benchmark_seed)
    classes = benchmark.train_classes if str(split).lower() == 'train' else benchmark.test_classes
    tasks = benchmark.train_tasks if str(split).lower() == 'train' else benchmark.test_tasks
    task_name = normalize_env_name(task_name)
    if task_name not in classes:
        raise KeyError(f'Task {task_name!r} is not in {benchmark_name}/{split}. Available: {list(classes.keys())}')
    matching_tasks = [task for task in tasks if str(task.env_name) == str(task_name)]
    if int(task_instance_id) >= len(matching_tasks):
        raise IndexError(f'task_instance_id={task_instance_id} but {task_name} has {len(matching_tasks)} tasks in {benchmark_name}/{split}.')
    env_cls = classes[task_name]
    env = env_cls(
        render_mode='rgb_array',
        camera_name=str(camera_name) if str(camera_name) else None,
        width=int(width),
        height=int(height),
    )
    env.set_task(matching_tasks[int(task_instance_id)])
    if bool(force_goal_observable):
        env._partially_observable = False
        try:
            del env.sawyer_observation_space
        except Exception:
            pass
    return env


def _success_value(info: Dict[str, Any]) -> bool:
    value = info.get('success', False)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _annotate_frame(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    img = np.asarray(frame).copy()
    try:
        import cv2
    except ImportError:
        return img
    # Render a readable black backing box and white text.
    line_h = 18
    box_h = 8 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], box_h), (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, str(line), (8, 18 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _render(env: Any, *, flip_vertical: bool) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError('env.render() returned None. Use render_mode="rgb_array" and an offscreen backend such as MUJOCO_GL=egl.')
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f'Unexpected render frame shape: {frame.shape}')
    if bool(flip_vertical):
        frame = np.flipud(frame)
    return frame.astype(np.uint8)


def _save_video(frames: Sequence[np.ndarray], output_base: Path, formats: Sequence[str], fps: int) -> List[str]:
    paths: List[str] = []
    for fmt in formats:
        ext = str(fmt).lower().lstrip('.')
        path = output_base.with_suffix(f'.{ext}')
        if ext == 'gif':
            imageio.mimsave(path, frames, duration=1.0 / max(1, int(fps)))
        elif ext in ('mp4', 'm4v'):
            imageio.mimsave(path, frames, fps=int(fps), macro_block_size=1)
        else:
            raise ValueError(f'Unsupported video format {fmt!r}. Use gif and/or mp4.')
        paths.append(str(path))
    return paths


def replay(args: argparse.Namespace) -> None:
    cache_root = Path(str(args.cache_root or os.environ.get('ICIL_METAWORLD_CACHE_ROOT', ''))).expanduser()
    if not str(cache_root):
        raise ValueError('Provide --cache-root or set ICIL_METAWORLD_CACHE_ROOT.')
    output_dir = Path(str(args.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = MetaWorldEpisodeStore(cache_root, keep_open_per_worker=False)
    env = None
    try:
        task_name, task_instance_id, episode_id = _resolve_episode(
            store,
            task_name=str(args.task_name),
            task_instance_id=int(args.task_instance_id),
            episode_id=int(args.episode_id),
        )
        meta = store.episode_metadata(int(episode_id))
        benchmark = str(store.index.get('benchmark', args.benchmark)) if not args.benchmark else str(args.benchmark)
        split = str(store.index.get('split', args.split)) if not args.split else str(args.split)
        T = int(store.episode_length(task_name, int(episode_id)))
        idx = np.arange(T, dtype=np.int64)
        cached = store.load_episode_slices(task_name, int(episode_id), idx, load_raw_obs=True, load_full_traj=True)
        cached_obs = np.asarray(cached['obs_raw'], dtype=np.float32)
        cached_actions = np.asarray(cached['action'], dtype=np.float32)
        cached_success = np.asarray(cached['success'], dtype=bool)
        max_steps = min(T, int(args.max_steps) if int(args.max_steps) > 0 else T)

        env = _make_env_for_task(
            task_name=task_name,
            task_instance_id=int(task_instance_id),
            benchmark_name=benchmark,
            split=split,
            benchmark_seed=int(args.benchmark_seed),
            camera_name=str(args.camera_name),
            width=int(args.width),
            height=int(args.height),
            force_goal_observable=bool(args.force_goal_observable),
        )
        obs, _ = _reset_env(env, seed=int(meta.get('seed', 0)))
        reset_obs_diff = float(np.max(np.abs(obs - cached_obs[0]))) if obs.shape == cached_obs[0].shape else float('inf')
        if reset_obs_diff > float(args.obs_match_tol):
            print(
                f'WARNING: reconstructed reset obs differs from cache by max_abs={reset_obs_diff:.6g}. '
                'If this is large, pass the benchmark seed used during cache generation via --benchmark-seed.'
            )

        frames: List[np.ndarray] = []
        obs_diffs: List[float] = [reset_obs_diff]
        first_success_step = None
        frame0 = _render(env, flip_vertical=bool(args.flip_vertical))
        frames.append(
            _annotate_frame(
                frame0,
                [
                    f'{task_name} inst={task_instance_id} ep={episode_id}',
                    f'reset | cached success_any={bool(np.any(cached_success))} | obs_diff={reset_obs_diff:.3g}',
                ],
            )
        )
        for t in range(max_steps):
            action = cached_actions[t]
            obs, reward, terminated, truncated, info = _step_env(env, action)
            success = _success_value(info)
            if success and first_success_step is None:
                first_success_step = int(t)
            if t + 1 < cached_obs.shape[0] and obs.shape == cached_obs[t + 1].shape:
                obs_diff = float(np.max(np.abs(obs - cached_obs[t + 1])))
                obs_diffs.append(obs_diff)
            else:
                obs_diff = float('nan')
            if (t % max(1, int(args.frame_stride))) == 0:
                frame = _render(env, flip_vertical=bool(args.flip_vertical))
                lines = [
                    f'{task_name} inst={task_instance_id} ep={episode_id} | t={t}/{max_steps-1}',
                    f'action xyz=({action[0]:+.2f},{action[1]:+.2f},{action[2]:+.2f}) grip={action[3]:+.2f} | reward={reward:.2f} success={success}',
                    f'obs_diff_vs_cache_next={obs_diff:.3g}',
                ]
                frames.append(_annotate_frame(frame, lines))
            if bool(terminated) or bool(truncated):
                break

        stem = f'{task_name}_instance-{task_instance_id}_episode-{episode_id}_replay'
        output_base = output_dir / stem
        paths = _save_video(frames, output_base, formats=tuple(args.formats), fps=int(args.fps))
        summary = {
            'cache_root': str(cache_root),
            'task_name': task_name,
            'task_instance_id': int(task_instance_id),
            'episode_id': int(episode_id),
            'benchmark': benchmark,
            'split': split,
            'benchmark_seed': int(args.benchmark_seed),
            'camera_name': str(args.camera_name),
            'force_goal_observable': bool(args.force_goal_observable),
            'flip_vertical': bool(args.flip_vertical),
            'frames': len(frames),
            'frame_stride': int(args.frame_stride),
            'max_steps_replayed': int(max_steps),
            'cached_success_any': bool(np.any(cached_success)),
            'cached_first_success_step': int(np.argmax(cached_success)) if np.any(cached_success) else None,
            'sim_first_success_step': first_success_step,
            'reset_obs_max_abs_diff': reset_obs_diff,
            'replay_obs_max_abs_diff': float(np.nanmax(np.asarray(obs_diffs, dtype=np.float32))),
            'video_paths': paths,
        }
        summary_path = output_base.with_suffix('.json')
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')
        for path in paths:
            print(f'Wrote {path}')
        print(f'Wrote {summary_path}')
        print(f'reset_obs_max_abs_diff={reset_obs_diff:.6g} replay_obs_max_abs_diff={summary["replay_obs_max_abs_diff"]:.6g}')
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay a cached MetaWorld expert episode in the simulator and save rendered video.')
    parser.add_argument('--cache-root', default=os.environ.get('ICIL_METAWORLD_CACHE_ROOT', ''))
    parser.add_argument('--output-dir', default='diagnostics/metaworld_sim_replay')
    parser.add_argument('--task-name', default='button-press-v3')
    parser.add_argument('--task-instance-id', type=int, default=0)
    parser.add_argument('--episode-id', type=int, default=-1, help='Use a specific cache episode id. If set, task-name/task-instance-id are inferred.')
    parser.add_argument('--benchmark', default='', help='Override cache benchmark, e.g. ML1.')
    parser.add_argument('--split', default='', help='Override cache split, e.g. train/test.')
    parser.add_argument('--benchmark-seed', type=int, default=0, help='Seed used to generate MetaWorld ML/MT task instances. Default matches configs/metaworld_generate_cache.py.')
    parser.add_argument('--camera-name', default='corner')
    parser.add_argument('--force-goal-observable', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--flip-vertical', action=argparse.BooleanOptionalAction, default=True, help='Flip rendered frames vertically before saving. Default fixes MuJoCo offscreen upside-down frames.')
    parser.add_argument('--max-steps', type=int, default=0, help='0 means replay the full cached episode.')
    parser.add_argument('--frame-stride', type=int, default=2)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--formats', nargs='+', default=('gif',), choices=('gif', 'mp4'))
    parser.add_argument('--obs-match-tol', type=float, default=1e-4)
    replay(parser.parse_args())


if __name__ == '__main__':
    main()
