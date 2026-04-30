from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
import torch
from tqdm.auto import tqdm

from icil.datasets.in_context_imitation_learning.cache_variation_h5 import (
    MASK_NAME_SUBSTRINGS_TO_IGNORE,
    MASK_NAMES_TO_IGNORE,
    _build_vector,
    _filter_by_ignore_ids,
    _subsample_fixedN,
)
from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.datasets.in_context_imitation_learning.variation_store import VariationStore, build_variation_keys
from icil.models.maml.query_memory_tasks import QueryMemoryTaskBuilder
from icil.models.maml.tasks import MAMLTaskSpec
from icil_jax_query_memory.models.config_utils import build_model_config_from_raw, resolve_dtype
from icil_jax_query_memory.models.query_memory_direct_regression import QueryMemoryDirectRegressionModel
from icil_jax_query_memory.train.config import QueryMemoryMetaConfig
from icil_jax_query_memory.train.step import create_adapt_fn, create_predict_fn
from icil_jax_query_memory.utils.action_representation import decode_action_chunk_np
from icil_jax_query_memory.utils.checkpoints import load_checkpoint

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_eval_query_memory_direct_regression.py',
    help_string='Path to ml_collections config file.',
)

_CAMERAS: Tuple[str, ...] = (
    'left_shoulder',
    'right_shoulder',
    'overhead',
    'wrist',
    'front',
)


def _as_bool(v: Any) -> bool:
    return bool(v)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _sanitize_action(
    action: np.ndarray,
    *,
    normalize_quaternion: bool,
    discretize_gripper: bool,
) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    if a.shape[0] >= 7 and normalize_quaternion:
        a[3:7] = _normalize_quaternion_xyzw(a[3:7])
    if a.shape[0] >= 8:
        if discretize_gripper:
            a[7] = 1.0 if float(a[7]) > 0.5 else 0.0
        else:
            a[7] = float(np.clip(a[7], 0.0, 1.0))
    return a


def _extract_rgb_frame(obs: Any, camera: str) -> np.ndarray:
    frame = getattr(obs, f'{camera}_rgb', None)
    if frame is None:
        frame = getattr(obs, 'front_rgb', None)
    if frame is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _write_video(frames: Sequence[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext == '.mp4':
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(out_path), list(frames), fps=int(fps))
            return out_path
        except Exception as exc:
            logging.warning('MP4 export failed (%s). Falling back to GIF.', exc)
            out_path = out_path.with_suffix('.gif')
            ext = '.gif'

    if ext == '.gif':
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
            if not pil_frames:
                raise RuntimeError('No frames to write.')
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(round(1000.0 / max(1, int(fps))))),
                loop=0,
            )
            return out_path
        except Exception as exc:
            logging.warning('GIF export failed (%s). Saving raw frames npz.', exc)

    fallback = out_path.with_suffix('.npz')
    np.savez_compressed(str(fallback), frames=np.asarray(frames, dtype=np.uint8))
    return fallback


class _LiveConditioningProcessor:
    def __init__(
        self,
        *,
        task_env: Any,
        num_points: int,
        use_rgb: bool = True,
        use_mask_id: bool = False,
        seed: int = 0,
    ):
        self.task_env = task_env
        self.num_points = int(num_points)
        self.use_rgb = bool(use_rgb)
        self.use_mask_id = bool(use_mask_id)
        self.rng = np.random.default_rng(int(seed))
        self.handle_to_name: Dict[int, str] = {}

    def _update_handle_names(self, masks: Sequence[np.ndarray]) -> None:
        from rlbench.segmentation_utils import build_handle_label_map

        unresolved: set[int] = set()
        for m in masks:
            if m is None:
                continue
            arr = np.asarray(m).reshape(-1).astype(np.int64, copy=False)
            for v in np.unique(arr):
                vi = int(v)
                if vi != 0 and vi not in self.handle_to_name:
                    unresolved.add(vi)
        if not unresolved:
            return
        mapping, _ = build_handle_label_map(self.task_env, unresolved)
        for handle, name in mapping.items():
            self.handle_to_name[int(handle)] = str(name)

    def _ignore_ids(self) -> Tuple[int, ...]:
        ignore = set()
        for handle, name in self.handle_to_name.items():
            if name in MASK_NAMES_TO_IGNORE:
                ignore.add(int(handle))
                continue
            lname = str(name).lower()
            if any(token in lname for token in MASK_NAME_SUBSTRINGS_TO_IGNORE):
                ignore.add(int(handle))
        return tuple(sorted(ignore))

    def observation_to_frame(self, obs: Any) -> Dict[str, torch.Tensor]:
        merged_points: List[np.ndarray] = []
        merged_colors: List[np.ndarray] = []
        merged_masks: List[np.ndarray] = []
        mask_views: List[np.ndarray] = []

        for cam in _CAMERAS:
            pc = getattr(obs, f'{cam}_point_cloud', None)
            msk = getattr(obs, f'{cam}_mask', None)
            rgb = getattr(obs, f'{cam}_rgb', None)
            if pc is None or msk is None:
                continue

            pts = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
            masks = np.asarray(msk).reshape(-1).astype(np.int32, copy=False)
            cols = None
            if self.use_rgb:
                if rgb is None:
                    cols = np.zeros((pts.shape[0], 3), dtype=np.uint8)
                else:
                    cols = np.asarray(rgb).reshape(-1, 3).astype(np.uint8, copy=False)

            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            masks = masks[finite]
            if cols is not None:
                cols = cols[finite]

            if pts.shape[0] == 0:
                continue
            merged_points.append(pts)
            merged_masks.append(masks)
            if cols is not None:
                merged_colors.append(cols)
            mask_views.append(masks)

        if merged_points:
            pts_all = np.concatenate(merged_points, axis=0).astype(np.float32, copy=False)
            msk_all = np.concatenate(merged_masks, axis=0).astype(np.int32, copy=False)
            col_all = (
                np.concatenate(merged_colors, axis=0).astype(np.uint8, copy=False)
                if self.use_rgb and merged_colors
                else None
            )
        else:
            pts_all = np.zeros((0, 3), dtype=np.float32)
            msk_all = np.zeros((0,), dtype=np.int32)
            col_all = np.zeros((0, 3), dtype=np.uint8) if self.use_rgb else None

        self._update_handle_names(mask_views)
        keep = _filter_by_ignore_ids(msk_all, self._ignore_ids())
        pts_all = pts_all[keep]
        msk_all = msk_all[keep]
        if col_all is not None:
            col_all = col_all[keep]

        if pts_all.shape[0] == 0:
            xyz = np.zeros((self.num_points, 3), dtype=np.float32)
            valid = np.zeros((self.num_points,), dtype=bool)
            rgb = np.zeros((self.num_points, 3), dtype=np.uint8) if self.use_rgb else None
            mask_id = np.zeros((self.num_points,), dtype=np.int64) if self.use_mask_id else None
        else:
            idx = _subsample_fixedN(self.rng, int(pts_all.shape[0]), self.num_points)
            xyz = pts_all[idx].astype(np.float32, copy=False)
            valid = np.ones((self.num_points,), dtype=bool)
            rgb = col_all[idx].astype(np.uint8, copy=False) if (self.use_rgb and col_all is not None) else None
            mask_id = msk_all[idx].astype(np.int64, copy=False) if self.use_mask_id else None

        state = _build_vector(obs, ('gripper_pose', 'gripper_open')).astype(np.float32, copy=False)

        out: Dict[str, torch.Tensor] = {
            'xyz': torch.from_numpy(xyz).float(),
            'valid': torch.from_numpy(valid).bool(),
            'state': torch.from_numpy(state).float(),
        }
        if rgb is not None:
            out['rgb'] = torch.from_numpy(rgb).float() / 255.0
        if mask_id is not None:
            out['mask_id'] = torch.from_numpy(mask_id).long()
        return out


def _build_query_window(
    history: Sequence[Dict[str, torch.Tensor]],
    *,
    dataset_cfg: ICILConfig,
    query_stride_mode: str,
) -> Dict[str, torch.Tensor]:
    if not history:
        raise RuntimeError('Query history is empty.')
    last = len(history) - 1
    qstep = int(dataset_cfg.stride) if query_stride_mode == 'dataset' else 1
    idx: List[int] = []
    for i in range(dataset_cfg.T_obs):
        rel = (dataset_cfg.T_obs - 1 - i) * qstep
        idx.append(max(0, last - rel))
    frames = [history[i] for i in idx]

    out: Dict[str, torch.Tensor] = {
        'query_xyz': torch.stack([f['xyz'] for f in frames], 0).unsqueeze(0),
        'query_state': torch.stack([f['state'] for f in frames], 0).unsqueeze(0),
        'query_valid': torch.stack([f['valid'] for f in frames], 0).unsqueeze(0),
    }
    if all('mask_id' in f for f in frames):
        out['query_mask_id'] = torch.stack([f['mask_id'] for f in frames], 0).unsqueeze(0)
    if all('rgb' in f for f in frames):
        out['query_rgb'] = torch.stack([f['rgb'] for f in frames], 0).unsqueeze(0)
    return out


def _build_rlbench_env(cfg: ConfigDict, task_name: str):
    from pyrep.const import RenderMode
    from rlbench import ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.environment import Environment

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    image_size = tuple(int(x) for x in cfg.sim.image_size)
    renderer_name = str(cfg.sim.renderer).lower()
    if renderer_name == 'opengl':
        render_mode = RenderMode.OPENGL
    elif renderer_name == 'opengl3':
        render_mode = RenderMode.OPENGL3
    else:
        raise ValueError(f"Unsupported renderer '{cfg.sim.renderer}'. Use 'opengl' or 'opengl3'.")

    for cam_name in _CAMERAS:
        cam_cfg = getattr(obs_config, f'{cam_name}_camera')
        cam_cfg.image_size = image_size
        cam_cfg.depth_in_meters = False
        cam_cfg.masks_as_one_channel = True
        cam_cfg.render_mode = render_mode

    action_mode = MoveArmThenGripper(
        EndEffectorPoseViaPlanning(
            absolute_mode=True,
            collision_checking=_as_bool(cfg.sim.collision_checking),
        ),
        Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=_as_bool(cfg.sim.headless),
        arm_max_velocity=float(cfg.sim.arm_max_velocity),
        arm_max_acceleration=float(cfg.sim.arm_max_acceleration),
    )
    env.launch()
    task_class = task_file_to_task_class(task_name)
    task_env = env.get_task(task_class)
    return env, task_env


def _model_cfg_from_checkpoint_or_default(cfg: ConfigDict, ckpt: Dict[str, Any], *, compute_dtype: jnp.dtype):
    raw = None
    if isinstance(ckpt.get('config', None), dict):
        maybe = ckpt['config'].get('model', None)
        if isinstance(maybe, dict):
            raw = ConfigDict(maybe)
    if raw is None:
        raw = cfg.model
    resolved = ckpt.get('config', {}).get('resolved', {}) if isinstance(ckpt.get('config', None), dict) else {}
    state_dim = int(resolved.get('state_dim', 8))
    action_dim = int(resolved.get('action_dim', 8))
    return build_model_config_from_raw(raw, state_dim=state_dim, action_dim=action_dim, compute_dtype=compute_dtype)


def _dataset_config_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> ICILConfig:
    ckpt_dataset = {}
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}
    use_ckpt = _as_bool(getattr(cfg.dataset, 'use_checkpoint_dataset_config', True))

    def _ival(name: str, default: int) -> int:
        if use_ckpt and name in ckpt_dataset:
            return int(ckpt_dataset[name])
        return int(getattr(cfg.dataset, name, default))

    action_representation = str(getattr(cfg.dataset, 'action_representation', 'absolute'))
    if use_ckpt and 'action_representation' in ckpt_dataset:
        action_representation = str(ckpt_dataset['action_representation'])

    return ICILConfig(
        K=_ival('K', 1),
        L=_ival('L', 16),
        T_obs=_ival('T_obs', 2),
        H=_ival('H', 16),
        stride=_ival('stride', 2),
        action_representation=action_representation,
    )


def _resolve_memory_cfg(cfg: ConfigDict, ckpt: Dict[str, Any]) -> QueryMemoryMetaConfig:
    ckpt_maml = ckpt.get('config', {}).get('maml', {}) if isinstance(ckpt.get('config', None), dict) else {}

    def _ival(local_value: int, key: str, default: int) -> int:
        if int(local_value) >= 0:
            return int(local_value)
        if isinstance(ckpt_maml, dict) and key in ckpt_maml:
            return int(ckpt_maml[key])
        return int(default)

    def _fval(local_value: float, key: str, default: float) -> float:
        if float(local_value) >= 0.0:
            return float(local_value)
        if isinstance(ckpt_maml, dict) and key in ckpt_maml:
            return float(ckpt_maml[key])
        return float(default)

    inner_lr_mode = str(ckpt_maml.get('inner_lr_mode', getattr(cfg.memory_ttt, 'inner_lr_mode', 'fixed')))
    if inner_lr_mode != 'fixed':
        raise ValueError(
            'JAX query-memory v1 eval only supports fixed inner_lr_mode. '
            f'Got {inner_lr_mode!r}.'
        )
    return QueryMemoryMetaConfig(
        inner_steps=_ival(getattr(cfg.memory_ttt, 'inner_steps', -1), 'inner_steps', 1),
        inner_lr=_fval(getattr(cfg.memory_ttt, 'inner_lr', -1.0), 'inner_lr', 1e-2),
        inner_lr_mode='fixed',
        outer_lr=0.0,
        weight_decay=0.0,
        max_grad_norm=_fval(getattr(cfg.memory_ttt, 'max_grad_norm', -1.0), 'max_grad_norm', 1.0),
        num_queries_per_step=_ival(getattr(cfg.memory_ttt, 'num_queries_per_step', -1), 'num_queries_per_step', 32),
        num_query_loss_samples=1,
        num_inner_batches=_ival(getattr(cfg.memory_ttt, 'num_inner_batches', -1), 'num_inner_batches', 0),
        holdout_index=-1,
        first_order=bool(ckpt_maml.get('first_order', True)),
        reuse_diffusion_noise=False,
        grad_accum_steps=1,
    )


def _query_stride_mode_from_eval(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, 'query_stride_mode', 'dataset')).lower()
    if mode not in ('dataset', 'consecutive'):
        raise ValueError('cfg.dataset.query_stride_mode must be one of: dataset, consecutive.')
    return mode


def _resolve_use_mask_id(model_cfg) -> bool:
    return bool(model_cfg.query_encoder.use_mask_id)


def _resolve_use_rgb(model_cfg) -> bool:
    return bool(model_cfg.query_encoder.use_rgb)


def _build_cached_support_ids(*, store: VariationStore, dataset_cfg: ICILConfig, variation: int, rng: np.random.Generator) -> Tuple[int, List[int]]:
    candidates = [idx for idx, key in enumerate(store.keys) if variation < 0 or int(key.variation) == int(variation)]
    if not candidates:
        available = sorted({int(key.variation) for key in store.keys})
        raise RuntimeError(f'No cached variation found for requested variation={variation}. Available variations: {available}')
    vidx = int(candidates[0] if variation >= 0 else rng.choice(np.asarray(candidates, dtype=np.int64)))
    episode_ids = store.list_episode_ids(vidx)
    if episode_ids.shape[0] < dataset_cfg.K:
        raise RuntimeError(f'Need at least K={dataset_cfg.K} cached support episodes, got {episode_ids.shape[0]}.')
    support_ids = rng.choice(episode_ids, size=dataset_cfg.K, replace=False)
    return vidx, [int(v) for v in np.asarray(support_ids).tolist()]


def _torch_batch_to_numpy(batch: Dict[str, Any], *, use_mask_id: bool, use_rgb: bool) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {
        'query_xyz': batch['query_xyz'].cpu().numpy(),
        'query_state': batch['query_state'].cpu().numpy(),
        'query_valid': batch['query_valid'].cpu().numpy(),
        'target_action': batch['target_action'].cpu().numpy(),
    }
    if use_mask_id and 'query_mask_id' in batch:
        out['query_mask_id'] = batch['query_mask_id'].cpu().numpy()
    if use_rgb and 'query_rgb' in batch:
        out['query_rgb'] = batch['query_rgb'].cpu().numpy()
    return out


def _prepare_support_inner_batches(
    *,
    task_builder: QueryMemoryTaskBuilder,
    vidx: int,
    support_ids: Sequence[int],
    memory_cfg: MemoryMAMLConfig,
    use_mask_id: bool,
    use_rgb: bool,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    task = MAMLTaskSpec(
        vidx=int(vidx),
        support_episode_ids=tuple(int(eid) for eid in support_ids),
        query_episode_id=int(support_ids[0]),
    )
    num_inner_batches = int(memory_cfg.inner_steps) if int(memory_cfg.num_inner_batches) <= 0 else min(int(memory_cfg.num_inner_batches), int(memory_cfg.inner_steps))
    if num_inner_batches <= 0:
        return {}
    base_batches: List[Dict[str, np.ndarray]] = []
    for _ in range(num_inner_batches):
        batch = task_builder.build_support_batch(
            task,
            count=int(memory_cfg.num_queries_per_step),
            rng=rng,
            load_rgb=use_rgb,
            load_mask_id=use_mask_id,
        )
        base_batches.append(_torch_batch_to_numpy(batch, use_mask_id=use_mask_id, use_rgb=use_rgb))
    expanded = [base_batches[idx % len(base_batches)] for idx in range(int(memory_cfg.inner_steps))]
    out: Dict[str, np.ndarray] = {}
    for key in ('query_xyz', 'query_state', 'query_valid', 'target_action'):
        out[key] = np.stack([batch[key] for batch in expanded], axis=0)
    if use_mask_id and all('query_mask_id' in batch for batch in expanded):
        out['query_mask_id'] = np.stack([batch['query_mask_id'] for batch in expanded], axis=0)
    if use_rgb and all('query_rgb' in batch for batch in expanded):
        out['query_rgb'] = np.stack([batch['query_rgb'] for batch in expanded], axis=0)
    return out


def _run_eval_episode(
    *,
    episode_index: int,
    task_env: Any,
    variation: int,
    params: Any,
    predict_fn: Any,
    adapted_memory_tokens: jnp.ndarray,
    dataset_cfg: ICILConfig,
    query_stride_mode: str,
    processor: _LiveConditioningProcessor,
    cfg: ConfigDict,
    run_dir: Path,
    action_dim: int,
    use_mask_id: bool,
    use_rgb: bool,
) -> Dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    if variation >= 0:
        task_env.set_variation(int(variation))
    descriptions, obs = task_env.reset()
    del descriptions
    history: List[Dict[str, torch.Tensor]] = [processor.observation_to_frame(obs)]
    frames: List[np.ndarray] = []
    if _as_bool(cfg.video.enable):
        frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))

    success = False
    terminated = False
    error: Optional[str] = None
    env_steps = 0
    execute_actions = max(1, int(cfg.control.execute_actions_per_plan))
    max_env_steps = int(cfg.task.max_env_steps)
    pbar = tqdm(total=max_env_steps, desc=f'Episode {episode_index}', leave=False, unit='step')
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(history, dataset_cfg=dataset_cfg, query_stride_mode=query_stride_mode)
            query_batch: Dict[str, np.ndarray] = {
                'query_xyz': query['query_xyz'].cpu().numpy(),
                'query_state': query['query_state'].cpu().numpy(),
                'query_valid': query['query_valid'].cpu().numpy(),
                'target_action': np.zeros((1, int(dataset_cfg.H), int(action_dim)), dtype=np.float32),
            }
            if use_mask_id and 'query_mask_id' in query:
                query_batch['query_mask_id'] = query['query_mask_id'].cpu().numpy()
            if use_rgb and 'query_rgb' in query:
                query_batch['query_rgb'] = query['query_rgb'].cpu().numpy()

            plan = predict_fn(params, {k: jnp.asarray(v) for k, v in query_batch.items()}, adapted_memory_tokens)
            plan_np = np.asarray(plan)
            plan_np = decode_action_chunk_np(
                plan_np,
                query_state=query_batch['query_state'],
                representation=str(dataset_cfg.action_representation),
            )[0]
            n_exec = int(min(execute_actions, plan_np.shape[0], max_env_steps - env_steps))
            for i in range(n_exec):
                action = _sanitize_action(
                    plan_np[i],
                    normalize_quaternion=_as_bool(cfg.control.normalize_quaternion),
                    discretize_gripper=_as_bool(cfg.control.discretize_gripper),
                )
                try:
                    obs, reward, terminated = task_env.step(action.astype(np.float32))
                except InvalidActionError as exc:
                    error = f'InvalidActionError: {exc}'
                    terminated = True
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    terminated = True
                    break
                env_steps += 1
                pbar.update(1)
                success = bool(float(reward) > 0.5)
                history.append(processor.observation_to_frame(obs))
                if _as_bool(cfg.video.enable):
                    frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))
                if success or terminated or env_steps >= max_env_steps:
                    break
    finally:
        pbar.close()

    video_path = None
    if _as_bool(cfg.video.enable) and frames:
        video_file = run_dir / 'videos' / f'episode_{episode_index:04d}.{str(cfg.video.format).lower()}'
        video_path = str(_write_video(frames, video_file, fps=int(cfg.video.fps)))

    return {
        'episode_index': int(episode_index),
        'success': bool(success),
        'terminated': bool(terminated),
        'env_steps': int(env_steps),
        'error': error,
        'video_path': video_path,
    }


def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt = load_checkpoint(checkpoint_path)
    compute_dtype = resolve_dtype('float32')
    model_cfg = _model_cfg_from_checkpoint_or_default(cfg, ckpt, compute_dtype=compute_dtype)
    dataset_cfg = _dataset_config_from_eval_and_checkpoint(cfg, ckpt)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    memory_cfg = _resolve_memory_cfg(cfg, ckpt)
    model = QueryMemoryDirectRegressionModel(cfg=model_cfg)
    params = ckpt['params']
    adapt_fn = create_adapt_fn(
        model=model,
        inner_steps=int(memory_cfg.inner_steps),
        inner_lr=float(memory_cfg.inner_lr),
        max_grad_norm=float(memory_cfg.max_grad_norm),
        first_order=bool(memory_cfg.first_order),
    )
    predict_fn = create_predict_fn(model=model)

    use_mask_id = bool(model_cfg.query_encoder.use_mask_id)
    use_rgb = bool(model_cfg.query_encoder.use_rgb)
    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f'{task_name}_var{variation}_{run_id}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_eval_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    env = None
    task_env = None
    support_store: Optional[VariationStore] = None
    results: List[Dict[str, Any]] = []
    try:
        cache_root = Path(str(cfg.conditioning.cache_root)).expanduser().resolve()
        task_keys = build_variation_keys(cache_root, task_name)
        if not task_keys:
            raise RuntimeError(f"No cached variations found for task '{task_name}' under {cache_root}.")
        support_store = VariationStore(task_keys, keep_open_per_worker=True)
        task_builder = QueryMemoryTaskBuilder(
            store=support_store,
            cfg=dataset_cfg,
            seed=seed + 101,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )
        rng = np.random.default_rng(seed + 102)
        vidx, support_ids = _build_cached_support_ids(store=support_store, dataset_cfg=dataset_cfg, variation=variation, rng=rng)
        support_inner = _prepare_support_inner_batches(
            task_builder=task_builder,
            vidx=vidx,
            support_ids=support_ids,
            memory_cfg=memory_cfg,
            use_mask_id=use_mask_id,
            use_rgb=use_rgb,
            rng=rng,
        )
        adapted_memory_tokens = adapt_fn(params, {k: jnp.asarray(v) for k, v in support_inner.items()})

        env, task_env = _build_rlbench_env(cfg, task_name)
        processor = _LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=use_rgb,
            use_mask_id=use_mask_id,
            seed=seed + 103,
        )

        for episode_index in range(int(cfg.task.num_eval_episodes)):
            result = _run_eval_episode(
                episode_index=episode_index,
                task_env=task_env,
                variation=variation,
                params=params,
                predict_fn=predict_fn,
                adapted_memory_tokens=adapted_memory_tokens,
                dataset_cfg=dataset_cfg,
                query_stride_mode=query_stride_mode,
                processor=processor,
                cfg=cfg,
                run_dir=run_dir,
                action_dim=int(model_cfg.action_dim),
                use_mask_id=use_mask_id,
                use_rgb=use_rgb,
            )
            results.append(result)
            logging.info(
                'episode=%d success=%s terminated=%s env_steps=%d error=%s',
                int(result['episode_index']),
                bool(result['success']),
                bool(result['terminated']),
                int(result['env_steps']),
                result['error'],
            )
    finally:
        if support_store is not None:
            support_store.close()
        if env is not None:
            env.shutdown()

    summary = {
        'checkpoint_path': str(checkpoint_path),
        'task_name': task_name,
        'variation': int(variation),
        'success_rate': float(np.mean([1.0 if r['success'] else 0.0 for r in results])) if results else 0.0,
        'num_eval_episodes': int(len(results)),
        'results': results,
    }
    with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def main(argv=None):
    del argv
    evaluate(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
