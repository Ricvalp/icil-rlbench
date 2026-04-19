from __future__ import annotations

import json
import random
import uuid
import inspect
from datetime import datetime
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm

from icil.datasets.in_context_imitation_learning.cache_variation_h5 import (
    MASK_NAME_SUBSTRINGS_TO_IGNORE,
    MASK_NAMES_TO_IGNORE,
    _build_vector,
    _filter_by_ignore_ids,
    _subsample_fixedN,
)
from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)
from icil.models import (
    Policy,
    PolicyBuilderConfig,
    PolicyConfig,
    build_policy,
)
from icil.models.maml import MAMLTaskBuilder
from icil.models.common import sinusoidal_position_embedding, sinusoidal_time_embedding

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/eval_perceiver_encoder_decoder_memory_ttt.py',
    help_string='Path to ml_collections config file.',
)

_CAMERAS: Tuple[str, ...] = (
    'left_shoulder',
    'right_shoulder',
    'overhead',
    'wrist',
    'front',
)


def _build_run_name(*, task_name: str, variation: int, unique_suffix: Optional[str] = None) -> str:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]
    unique_id = str(unique_suffix) if unique_suffix else uuid.uuid4().hex[:8]
    return f'{task_name}_var{variation}_{timestamp}_{unique_id}'


def _as_bool(v: Any) -> bool:
    return bool(v)


def _maybe_init_wandb(cfg: ConfigDict, workdir: Path) -> Optional[Any]:
    if not hasattr(cfg, 'wandb') or not _as_bool(cfg.wandb.enable):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError('cfg.wandb.enable=True but wandb is not installed.') from exc

    tags = list(cfg.wandb.tags) if getattr(cfg.wandb, 'tags', None) else None
    project = str(cfg.wandb.project)
    entity = str(cfg.wandb.entity) if str(cfg.wandb.entity) else None
    name = str(cfg.wandb.name) if str(cfg.wandb.name) else None
    group = str(cfg.wandb.group) if str(cfg.wandb.group) else None
    mode = str(cfg.wandb.mode) if str(cfg.wandb.mode) else None

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        mode=mode,
        dir=str(workdir),
        config=cfg.to_dict(),
        tags=tags,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _resolve_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and str(device_str).startswith('cuda'):
        return torch.device(str(device_str))
    return torch.device('cpu')



def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict



def _load_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f'Unsupported checkpoint object type: {type(ckpt).__name__}')
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' payload is not a state_dict dictionary.")
    return ckpt if isinstance(ckpt, dict) else {}, _strip_module_prefix(state_dict)



def _infer_state_action_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        action_dim = int(state_dict['action_out.weight'].shape[0])
    except KeyError as exc:
        raise KeyError(
            "Could not infer action_dim from checkpoint. Expected key 'action_out.weight'."
        ) from exc

    state_dim_key_candidates = (
        'context_encoder.state_proj.0.weight',
        'context_encoder.demo_query_encoder.state_proj.0.weight',
        'context_encoder.demo_frame_stack.state_proj.0.weight',
        'context_encoder.demo_query_encoder.demo_frame_stack.state_proj.0.weight',
    )
    state_dim = None
    for key in state_dim_key_candidates:
        if key in state_dict:
            state_dim = int(state_dict[key].shape[1])
            break
    if state_dim is None:
        state_dim = 8
        logging.warning(
            'Could not infer state_dim from state_dict; using fallback state_dim=%d.',
            state_dim,
        )
    return state_dim, action_dim



def _dataclass_from_dict(default_obj: Any, src: Dict[str, Any]) -> Any:
    if not is_dataclass(default_obj):
        return default_obj
    kwargs: Dict[str, Any] = {}
    for f in fields(default_obj):
        default_v = getattr(default_obj, f.name)
        if is_dataclass(default_v):
            sub_src = src.get(f.name, {}) if isinstance(src, dict) else {}
            if not isinstance(sub_src, dict):
                sub_src = {}
            kwargs[f.name] = _dataclass_from_dict(default_v, sub_src)
        else:
            if isinstance(src, dict) and f.name in src:
                v = src[f.name]
                if isinstance(default_v, tuple) and isinstance(v, (list, tuple)):
                    v = tuple(v)
                kwargs[f.name] = v
            else:
                kwargs[f.name] = default_v
    return type(default_obj)(**kwargs)



def _legacy_flat_model_cfg_to_nested(model_from_ckpt: Dict[str, Any]) -> Dict[str, Any]:
    policy_field_names = {f.name for f in fields(PolicyConfig())}
    out: Dict[str, Any] = {
        'encoder_name': 'perceiver_demo_query',
        'policy': {},
        'perceiver_demo_query': {},
        'perceiver_demo_query_v2': {},
        'traj_perceiver': {},
        'traj_perceiver_v2': {},
    }
    for k, v in model_from_ckpt.items():
        if k in (
            'policy',
            'conv3d_demo_query',
            'perceiver_demo_query',
            'perceiver_demo_query_v2',
            'traj_conv3d',
            'traj_perceiver',
            'traj_perceiver_v2',
            'encoder_name',
        ):
            out[k] = v
            continue
        if k in policy_field_names:
            out['policy'][k] = v
        else:
            out['perceiver_demo_query'][k] = v
    return out



def _model_config_from_checkpoint_or_default(ckpt: Dict[str, Any]) -> PolicyBuilderConfig:
    model_from_ckpt: Dict[str, Any] = {}
    if isinstance(ckpt.get('config', None), dict):
        model_from_ckpt = ckpt['config'].get('model', {}) or {}

    defaults = PolicyBuilderConfig()
    if not isinstance(model_from_ckpt, dict) or not model_from_ckpt:
        return defaults
    if 'policy' not in model_from_ckpt:
        model_from_ckpt = _legacy_flat_model_cfg_to_nested(model_from_ckpt)
    return _dataclass_from_dict(defaults, model_from_ckpt)



def _conditioning_use_mask_id_from_eval_and_checkpoint(
    cfg: ConfigDict,
    model_cfg: PolicyBuilderConfig,
) -> bool:
    if str(model_cfg.encoder_name) == 'conv3d_demo_query':
        return bool(model_cfg.conv3d_demo_query.use_mask_id)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query':
        return bool(model_cfg.perceiver_demo_query.use_mask_id)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query_v2':
        return bool(model_cfg.perceiver_demo_query_v2.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_perceiver_v2':
        return bool(model_cfg.traj_perceiver_v2.use_mask_id)
    return _as_bool(getattr(cfg.conditioning, 'use_mask_id', True))



def _ignore_demos_from_model_cfg(model_cfg: PolicyBuilderConfig) -> bool:
    if str(model_cfg.encoder_name) == 'conv3d_demo_query':
        return bool(model_cfg.conv3d_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query':
        return bool(model_cfg.perceiver_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query_v2':
        return bool(model_cfg.perceiver_demo_query_v2.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_perceiver_v2':
        return bool(model_cfg.traj_perceiver_v2.ignore_demos)
    return False



def _resolve_data_k(cfg: ConfigDict, ckpt: Dict[str, Any]) -> int:
    configured_k = int(cfg.dataset.K)
    if configured_k > 0:
        return configured_k
    ckpt_dataset = {}
    ckpt_has_maml_cfg = False
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}
        ckpt_has_maml_cfg = isinstance(ckpt['config'].get('maml', None), dict)
    if isinstance(ckpt_dataset, dict) and int(ckpt_dataset.get('K', 0)) > 0:
        ckpt_k = int(ckpt_dataset['K'])
        return ckpt_k if ckpt_has_maml_cfg else ckpt_k + 1
    raise ValueError(
        'cfg.dataset.K=0 requires checkpoint["config"]["dataset"]["K"] > 0. '
        'Pretrain checkpoints use K_pretrain + 1; MAML checkpoints use their stored K.'
    )



def _dataset_config_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> ICILConfig:
    ckpt_dataset = {}
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}

    use_ckpt = _as_bool(getattr(cfg.dataset, 'use_checkpoint_dataset_config', True))
    resolved_k = _resolve_data_k(cfg, ckpt)

    def _ival(name: str, default: int) -> int:
        if use_ckpt and name in ckpt_dataset:
            return int(ckpt_dataset[name])
        return int(getattr(cfg.dataset, name, default))

    return ICILConfig(
        K=int(resolved_k),
        L=_ival('L', 1),
        T_obs=_ival('T_obs', 1),
        H=_ival('H', 1),
        stride=_ival('stride', 1),
    )



def _query_stride_mode_from_eval(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, 'query_stride_mode', 'dataset')).lower()
    if mode not in ('dataset', 'consecutive'):
        raise ValueError('cfg.dataset.query_stride_mode must be one of: dataset, consecutive.')
    return mode



def _support_source_from_eval(cfg: ConfigDict) -> str:
    source = str(getattr(cfg.conditioning, 'support_source', 'cache')).lower()
    if source != 'cache':
        raise ValueError(
            'Memory TTT eval currently supports cfg.conditioning.support_source="cache" only, '
            'because inner-loop diffusion supervision requires cached action trajectories.'
        )
    return source



def _support_cache_root_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> Path:
    root = str(getattr(cfg.conditioning, 'cache_root', '')).strip()
    if not root and isinstance(ckpt.get('config', None), dict):
        root = str((ckpt['config'].get('data', {}) or {}).get('cache_root', '')).strip()
    if not root:
        raise ValueError(
            'Cached support conditioning requires cfg.conditioning.cache_root or '
            'checkpoint["config"]["data"]["cache_root"].'
        )
    cache_root = Path(root).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f'Support cache root not found: {cache_root}')
    return cache_root



def _warn_if_cached_num_points_mismatch(
    *,
    task_keys: Sequence[Any],
    expected_num_points: int,
    task_name: str,
) -> None:
    detected_values: List[int] = []
    inspected_paths: List[str] = []
    for key in task_keys:
        path = Path(str(key.path)).expanduser().resolve()
        try:
            with h5py.File(path, 'r') as f:
                detected = int(f.attrs.get('N', -1))
        except Exception as exc:
            logging.warning('Could not inspect cached point count from %s: %s', path, exc)
            continue
        if detected > 0:
            detected_values.append(detected)
            inspected_paths.append(str(path))

    if not detected_values:
        return

    unique_detected = sorted(set(int(v) for v in detected_values))
    if len(unique_detected) > 1 or int(expected_num_points) not in unique_detected:
        logging.warning('============================================================')
        logging.warning(
            'POINT-COUNT MISMATCH: cfg.conditioning.num_points=%d, but cached task %r stores N=%s.',
            int(expected_num_points),
            task_name,
            unique_detected,
        )
        logging.warning(
            'Live observations will be resampled to %d points, while cached support/training data uses N=%s.',
            int(expected_num_points),
            unique_detected,
        )
        if inspected_paths:
            logging.warning('Inspected cache files: %s', inspected_paths)
        logging.warning('============================================================')



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


def _video_cameras_from_cfg(cfg: ConfigDict) -> List[str]:
    cameras_cfg = getattr(cfg.video, 'cameras', None)
    if cameras_cfg is None:
        cameras_cfg = (str(getattr(cfg.video, 'camera', 'front')),)
    elif isinstance(cameras_cfg, str):
        cameras_cfg = (cameras_cfg,)

    cameras = [str(camera) for camera in cameras_cfg if str(camera).strip()]
    if not cameras:
        raise ValueError('cfg.video.cameras must contain at least one camera when video is enabled.')

    invalid = [camera for camera in cameras if camera not in _CAMERAS]
    if invalid:
        raise ValueError(
            f'Unsupported video cameras {invalid}. Expected subset of {list(_CAMERAS)}.'
        )
    return cameras


def _video_formats_from_cfg(cfg: ConfigDict) -> List[str]:
    formats_cfg = getattr(cfg.video, 'formats', None)
    if formats_cfg is None:
        formats_cfg = (str(getattr(cfg.video, 'format', 'mp4')).lower(),)
    elif isinstance(formats_cfg, str):
        formats_cfg = (formats_cfg,)

    formats = [str(fmt).lower() for fmt in formats_cfg if str(fmt).strip()]
    if not formats:
        raise ValueError('cfg.video.formats must contain at least one format when video is enabled.')

    invalid = [fmt for fmt in formats if fmt not in {'mp4', 'gif'}]
    if invalid:
        raise ValueError(f'Unsupported video formats {invalid}. Expected subset of [\"mp4\", \"gif\"].')
    return formats


class _LiveConditioningProcessor:
    def __init__(
        self,
        *,
        task_env: Any,
        num_points: int,
        use_rgb: bool,
        use_mask_id: bool,
        seed: int,
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
        mapping, _outstanding = build_handle_label_map(self.task_env, unresolved)
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



def _to_device_tensor(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device, non_blocking=True)



def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out



def _batch_size_from_target(batch: Dict[str, Any]) -> int:
    target = batch.get('target_action', None)
    if not torch.is_tensor(target) or target.ndim < 1:
        raise ValueError("Batch must contain tensor key 'target_action' with a batch dimension.")
    return int(target.shape[0])



def _slice_batch_dim0(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    batch_size = _batch_size_from_target(batch)
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            out[key] = value[start:end]
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out



def _iter_microbatches(batch: Dict[str, Any], grad_accum_steps: int) -> Sequence[Tuple[Dict[str, Any], float]]:
    batch_size = _batch_size_from_target(batch)
    if batch_size < 1:
        raise ValueError('Memory TTT inner-loop batches must be non-empty.')

    num_chunks = min(max(1, int(grad_accum_steps)), batch_size)
    if num_chunks == 1:
        return [(batch, 1.0)]

    microbatches: List[Tuple[Dict[str, Any], float]] = []
    boundaries = np.linspace(0, batch_size, num_chunks + 1, dtype=np.int64)
    for chunk_idx in range(num_chunks):
        start = int(boundaries[chunk_idx])
        end = int(boundaries[chunk_idx + 1])
        if start >= end:
            continue
        weight = float(end - start) / float(batch_size)
        microbatches.append((_slice_batch_dim0(batch, start, end), weight))
    return microbatches



def _drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop('cond_mask_id', None)
    out.pop('query_mask_id', None)
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
        raise ValueError(f'Unsupported renderer {cfg.sim.renderer!r}. Use opengl or opengl3.')

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



def _sample_balanced_indices(total: int, *, count: int, rng: np.random.Generator) -> List[int]:
    if total < 1:
        raise ValueError(f'total must be positive, got {total}.')
    if count < 1:
        raise ValueError(f'count must be positive, got {count}.')

    out: List[int] = []
    while len(out) < count:
        perm = rng.permutation(total)
        take = min(total, count - len(out))
        out.extend(int(idx) for idx in perm[:take].tolist())
    return out

def _resolve_num_memory_inner_batches(memory_cfg: ConfigDict) -> int:
    inner_steps = int(getattr(memory_cfg, 'inner_steps', 0))
    configured = int(getattr(memory_cfg, 'num_inner_batches', 0))
    if inner_steps <= 0:
        return 0
    if configured <= 0:
        return inner_steps
    return min(configured, inner_steps)


def _memory_optimizer_name(memory_cfg: ConfigDict) -> str:
    name = str(getattr(memory_cfg, 'optimizer', 'adam')).lower()
    if name not in {'adam', 'sgd'}:
        raise ValueError("cfg.memory_ttt.optimizer must be one of: 'adam', 'sgd'.")
    return name


def _memory_decoder_prefixes(memory_cfg: ConfigDict) -> Tuple[str, ...]:
    prefixes = getattr(memory_cfg, 'decoder_param_prefixes', ('denoiser.', 'action_in.', 'action_out.', 't_mlp.'))
    if isinstance(prefixes, str):
        prefixes = tuple(part.strip() for part in prefixes.split(',') if part.strip())
    else:
        prefixes = tuple(str(part) for part in prefixes if str(part).strip())
    if not prefixes:
        raise ValueError('cfg.memory_ttt.decoder_param_prefixes must not be empty when decoder optimization is enabled.')
    return prefixes


def _select_memory_decoder_param_names(model: Policy, memory_cfg: ConfigDict) -> List[str]:
    if not _as_bool(getattr(memory_cfg, 'optimize_decoder', False)):
        return []
    prefixes = _memory_decoder_prefixes(memory_cfg)
    selected = sorted(
        name for name, _ in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    )
    if not selected:
        raise RuntimeError(f'No decoder parameters matched prefixes={prefixes}.')
    return selected


def _param_count_by_names(model: nn.Module, names: Sequence[str]) -> int:
    name_set = set(names)
    return sum(int(param.numel()) for name, param in model.named_parameters() if name in name_set)


def _select_memory_holdout_index(K: int, memory_cfg: ConfigDict, rng: np.random.Generator) -> int:
    configured = int(getattr(memory_cfg, 'holdout_index', -1))
    if configured >= 0:
        if configured >= K:
            raise ValueError(f'cfg.memory_ttt.holdout_index={configured} out of range for K={K}.')
        return configured
    return int(rng.integers(low=0, high=int(K)))


def _expand_tokens_for_batch(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tokens.shape[0] == batch_size:
        return tokens
    if tokens.shape[0] == 1:
        return tokens.expand(batch_size, -1, -1)
    raise ValueError(f'Cannot expand token batch from B={tokens.shape[0]} to B={batch_size}.')


def _expand_mask_for_batch(mask: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.expand(batch_size, -1)
    raise ValueError(f'Cannot expand mask batch from B={mask.shape[0]} to B={batch_size}.')


def _detach_optional_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.detach() if torch.is_tensor(x) else None


def _infer_num_support_demos_from_batch(batch: Dict[str, Any]) -> Optional[int]:
    for key in ('cond_xyz', 'cond_state', 'cond_traj'):
        value = batch.get(key, None)
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[1])
    return None


def _traj_support_token_count(
    policy: Policy,
    support_tokens: Optional[torch.Tensor],
    *,
    num_support_demos: Optional[int],
) -> int:
    if support_tokens is None or num_support_demos is None or int(num_support_demos) <= 0:
        return 0

    context_encoder = getattr(policy.context_encoder, 'base_encoder', policy.context_encoder)
    cfg = getattr(context_encoder, 'cfg', None)
    if cfg is None or not hasattr(cfg, 'm_traj_tokens'):
        return 0
    if not _as_bool(getattr(cfg, 'include_traj_tokens', True)):
        return 0

    count = int(num_support_demos) * int(getattr(cfg, 'm_traj_tokens'))
    if count <= 0 or count > int(support_tokens.shape[1]):
        return 0
    return count


def _single_context_from_memory_tokens(
    policy: Policy,
    *,
    support_tokens: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    query_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    num_support_demos: Optional[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    traj_count = _traj_support_token_count(
        policy,
        support_tokens,
        num_support_demos=num_support_demos,
    )
    if traj_count <= 0 or support_tokens is None:
        return policy._concat_token_groups(
            support_tokens,
            support_mask,
            query_tokens,
            query_mask,
        )

    # Trajectory encoders produce full single-context tokens as [demo, query, traj].
    # Memory-TTT optimizes support_tokens=[demo, traj], so split the trajectory suffix
    # and reinsert query tokens before it.
    demo_count = int(support_tokens.shape[1]) - int(traj_count)
    demo_tokens = support_tokens[:, :demo_count] if demo_count > 0 else None
    traj_tokens = support_tokens[:, demo_count:]

    demo_mask = None
    traj_mask = None
    if support_mask is not None:
        demo_mask = support_mask[:, :demo_count] if demo_count > 0 else None
        traj_mask = support_mask[:, demo_count:]

    ctx_tokens, ctx_mask = policy._concat_token_groups(
        demo_tokens,
        demo_mask,
        query_tokens,
        query_mask,
    )
    return policy._concat_token_groups(
        ctx_tokens,
        ctx_mask,
        traj_tokens,
        traj_mask,
    )


def _encode_context_no_grad(policy: Policy, batch: Dict[str, torch.Tensor]) -> Any:
    with torch.no_grad():
        ctx_out = policy.context_encoder(
            query_xyz=batch['query_xyz'],
            query_state=batch['query_state'],
            cond_xyz=batch.get('cond_xyz', None),
            cond_state=batch.get('cond_state', None),
            cond_traj=batch.get('cond_traj', None),
            cond_traj_mask=batch.get('cond_traj_mask', None),
            query_rgb=batch.get('query_rgb', None),
            query_mask_id=batch.get('query_mask_id', None),
            query_valid=batch.get('query_valid', None),
            cond_rgb=batch.get('cond_rgb', None),
            cond_mask_id=batch.get('cond_mask_id', None),
            cond_valid=batch.get('cond_valid', None),
        )
        return policy._resolve_context_output(ctx_out)


def _init_memory_tokens_from_batch(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[nn.Parameter, Optional[torch.Tensor]]:
    batch_dev = _to_device_batch(batch, device)
    ctx = _encode_context_no_grad(policy, batch_dev)
    if ctx.support_tokens is None:
        raise RuntimeError('Context encoder did not return support_tokens; memory TTT requires demo support tokens.')
    support_tokens = ctx.support_tokens.detach().clone()
    support_mask = _detach_optional_tensor(ctx.support_token_mask)
    if support_tokens.shape[0] != 1:
        support_tokens = support_tokens[:1].contiguous()
        if support_mask is not None:
            support_mask = support_mask[:1].contiguous()
    return nn.Parameter(support_tokens), support_mask


def _query_tokens_from_batch(
    policy: Policy,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    ctx = _encode_context_no_grad(policy, batch)
    if ctx.query_tokens is None:
        raise RuntimeError('Context encoder did not return query_tokens; memory TTT requires query tokens.')
    return ctx.query_tokens.detach(), _detach_optional_tensor(ctx.query_token_mask)


def _resolve_batch_timesteps(policy: Policy, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    x0 = batch['target_action']
    B = int(x0.shape[0])
    provided = batch.get('timesteps', None)
    if provided is None:
        return torch.randint(
            low=0,
            high=int(policy.noise_scheduler.config.num_train_timesteps),
            size=(B,),
            device=x0.device,
            dtype=torch.long,
        )
    t = provided.to(device=x0.device, dtype=torch.long)
    if t.ndim == 0:
        t = t.view(1)
    if t.shape == (1,) and B > 1:
        t = t.expand(B)
    if t.shape != (B,):
        raise ValueError(f'timesteps must have shape ({B},), got {tuple(t.shape)}.')
    return t


def _resolve_batch_noise(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    x0 = batch['target_action']
    B = int(x0.shape[0])
    provided = batch.get('noise', None)
    if provided is None:
        return torch.randn_like(x0)
    noise = provided.to(device=x0.device, dtype=x0.dtype)
    if noise.ndim == 2:
        noise = noise.unsqueeze(0)
    if noise.shape[0] == 1 and B > 1:
        noise = noise.expand(B, -1, -1)
    if noise.shape != x0.shape:
        raise ValueError(f'noise must have shape {tuple(x0.shape)}, got {tuple(noise.shape)}.')
    return noise


def _diffusion_training_target(
    policy: Policy,
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    pred_type = str(policy.noise_scheduler.config.prediction_type)
    if pred_type == 'epsilon':
        return noise
    if pred_type == 'sample':
        return x0
    if pred_type == 'v_prediction':
        if hasattr(policy.noise_scheduler, 'get_velocity'):
            return policy.noise_scheduler.get_velocity(x0, noise, t)
        alpha_t = policy.noise_scheduler.alphas_cumprod[t].sqrt().to(x0.device)
        sigma_t = (1.0 - policy.noise_scheduler.alphas_cumprod[t]).sqrt().to(x0.device)
        alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
        return alpha_t * noise - sigma_t * x0
    raise ValueError(f'Unsupported prediction type {pred_type!r}.')


def _predict_model_output_from_tokens(
    policy: Policy,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    query_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    support_tokens: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    num_support_demos: Optional[int] = None,
) -> torch.Tensor:
    B, H, _ = x_t.shape
    d = int(policy.cfg.d_model)

    query_tokens = _expand_tokens_for_batch(query_tokens.to(device=x_t.device, dtype=x_t.dtype), B)
    query_mask = _expand_mask_for_batch(
        query_mask.to(device=x_t.device, dtype=torch.bool) if query_mask is not None else None,
        B,
    )
    if support_tokens is not None:
        support_tokens = _expand_tokens_for_batch(support_tokens.to(device=x_t.device, dtype=x_t.dtype), B)
    support_mask = _expand_mask_for_batch(
        support_mask.to(device=x_t.device, dtype=torch.bool) if support_mask is not None else None,
        B,
    )

    t_emb = sinusoidal_time_embedding(t, d)
    t_cond = policy.t_mlp(t_emb)

    h = policy.action_in(x_t)
    h = h + sinusoidal_position_embedding(H, d, device=x_t.device).to(dtype=h.dtype).unsqueeze(0)
    use_dit_ckpt = bool(policy.training and policy.cfg.grad_checkpoint_dit and torch.is_grad_enabled())

    if str(policy.context_attention_mode) == 'single':
        ctx_tokens, ctx_mask = _single_context_from_memory_tokens(
            policy,
            support_tokens=support_tokens,
            support_mask=support_mask,
            query_tokens=query_tokens,
            query_mask=query_mask,
            num_support_demos=num_support_demos,
        )
        if ctx_tokens is None:
            raise RuntimeError('Memory TTT single-context decoder received no context tokens.')
        for blk in policy.denoiser:
            h = policy._apply_single_context_block(
                blk,
                h,
                t_cond,
                ctx_tokens,
                ctx_mask,
                use_checkpoint=use_dit_ckpt,
            )
    else:
        for blk in policy.denoiser:
            h = policy._apply_two_context_block(
                blk,
                h,
                t_cond,
                query_tokens,
                query_mask,
                support_tokens,
                support_mask,
                use_checkpoint=use_dit_ckpt,
            )
    return policy.action_out(h)


def _memory_diffusion_loss(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    memory_tokens: torch.Tensor,
    memory_token_mask: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    batch_dev = _to_device_batch(batch, device)
    x0 = batch_dev['target_action']
    t = _resolve_batch_timesteps(policy, batch_dev)
    noise = _resolve_batch_noise(batch_dev)
    x_t = policy.noise_scheduler.add_noise(x0, noise, t)
    query_tokens, query_mask = _query_tokens_from_batch(policy, batch_dev)
    model_out = _predict_model_output_from_tokens(
        policy,
        x_t=x_t,
        t=t,
        query_tokens=query_tokens,
        query_mask=query_mask,
        support_tokens=memory_tokens,
        support_mask=memory_token_mask,
        num_support_demos=_infer_num_support_demos_from_batch(batch_dev),
    )
    target = _diffusion_training_target(policy, x0=x0, noise=noise, t=t)
    return F.mse_loss(model_out, target)


def _query_sample_mse_with_memory_tokens(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    memory_tokens: torch.Tensor,
    memory_token_mask: Optional[torch.Tensor],
    device: torch.device,
    use_mask_id: bool,
    inference_steps: int,
    eta: float,
) -> float:
    batch_dev = _to_device_batch(batch, device)
    with torch.no_grad():
        pred = _sample_actions_with_memory_tokens(
            policy,
            adapted_support_tokens=memory_tokens,
            adapted_support_token_mask=memory_token_mask,
            cond_xyz=batch_dev.get('cond_xyz', None),
            cond_state=batch_dev.get('cond_state', None),
            cond_traj=batch_dev.get('cond_traj', None),
            cond_traj_mask=batch_dev.get('cond_traj_mask', None),
            query_xyz=batch_dev['query_xyz'],
            query_state=batch_dev['query_state'],
            action_horizon=int(batch_dev['target_action'].shape[1]),
            cond_rgb=batch_dev.get('cond_rgb', None),
            query_rgb=batch_dev.get('query_rgb', None),
            cond_mask_id=(batch_dev.get('cond_mask_id', None) if use_mask_id else None),
            query_mask_id=(batch_dev.get('query_mask_id', None) if use_mask_id else None),
            cond_valid=batch_dev.get('cond_valid', None),
            query_valid=batch_dev.get('query_valid', None),
            inference_steps=(int(inference_steps) if int(inference_steps) > 0 else None),
            eta=float(eta),
        )
    return float(
        F.mse_loss(
            pred.detach().float(),
            batch_dev['target_action'].detach().float(),
        ).detach().cpu().item()
    )


def _grad_global_norm(params: Sequence[torch.Tensor]) -> float:
    grads = [param.grad.detach() for param in params if param.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([grad.norm(2) for grad in grads]), 2).item())


def _make_memory_optimizer(
    memory_cfg: ConfigDict,
    *,
    memory_tokens: nn.Parameter,
    decoder_params: Sequence[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    param_groups: List[Dict[str, Any]] = [{'params': [memory_tokens], 'lr': float(memory_cfg.inner_lr)}]
    if decoder_params:
        param_groups.append({'params': list(decoder_params), 'lr': float(getattr(memory_cfg, 'decoder_lr', memory_cfg.inner_lr))})
    opt_name = _memory_optimizer_name(memory_cfg)
    if opt_name == 'adam':
        return torch.optim.Adam(param_groups)
    return torch.optim.SGD(param_groups, momentum=float(getattr(memory_cfg, 'sgd_momentum', 0.0)))



def _num_valid_query_t0s(task_builder: MAMLTaskBuilder, *, vidx: int, episode_id: int) -> int:
    T = int(task_builder.store.episode_length(int(vidx), int(episode_id)))
    required = 1 + ((int(task_builder.cfg.T_obs) - 1) * int(task_builder.cfg.stride))
    return max(0, T - required + 1)



def _sample_query_t0s_for_episode(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    episode_id: int,
    count: int,
    rng: np.random.Generator,
) -> List[int]:
    num_valid = _num_valid_query_t0s(task_builder, vidx=int(vidx), episode_id=int(episode_id))
    if num_valid < 1:
        raise RuntimeError(
            f'No valid query windows for vidx={vidx}, episode_id={episode_id}. '
            f'Need at least T_obs={int(task_builder.cfg.T_obs)} observed frames with stride={int(task_builder.cfg.stride)}.'
        )
    return _sample_balanced_indices(num_valid, count=count, rng=rng)



def _build_query_sample_at_t0(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    episode_id: int,
    t0: int,
    load_rgb: bool,
    load_mask_id: bool,
) -> Dict[str, Any]:
    episode_length = int(task_builder.store.episode_length(int(vidx), int(episode_id)))
    obs_idx, act_idx = task_builder._build_obs_act_indices(int(t0), episode_length=episode_length)
    q_obs = task_builder.store.load_episode_slices(
        int(vidx),
        int(episode_id),
        obs_idx,
        load_rgb=load_rgb,
        load_mask_id=load_mask_id,
        load_full_traj=False,
    )
    q_act = task_builder.store.load_episode_slices(
        int(vidx),
        int(episode_id),
        act_idx,
        load_rgb=False,
        load_mask_id=False,
        load_full_traj=False,
    )

    sample: Dict[str, Any] = {
        'query_xyz': q_obs['xyz'],
        'query_state': q_obs['state'],
        'query_valid': q_obs['valid'],
        'target_action': q_act['action'],
        'meta': {
            'vidx': int(vidx),
            'query_episode': int(episode_id),
            't0': int(t0),
        },
    }
    if load_mask_id and 'mask_id' in q_obs:
        sample['query_mask_id'] = q_obs['mask_id']
    if load_rgb and 'rgb' in q_obs:
        sample['query_rgb'] = q_obs['rgb']
    return sample



def _select_support_vidx(
    store: VariationStore,
    *,
    variation: int,
    rng: np.random.Generator,
) -> int:
    candidates = [
        idx for idx, key in enumerate(store.keys)
        if variation < 0 or int(key.variation) == int(variation)
    ]
    if not candidates:
        available = sorted({int(key.variation) for key in store.keys})
        raise RuntimeError(
            f'No cached variation found for requested variation={variation}. '
            f'Available variations for task {store.keys[0].task if store.keys else "?"!r}: {available}'
        )
    if variation >= 0:
        return int(candidates[0])
    return int(rng.choice(np.asarray(candidates, dtype=np.int64)))



def _unsqueeze_support_batch_dim(support: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in support.items():
        out[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
    return out


def _build_memory_query_batch(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    support_cond: Dict[str, Any],
    support_ids: Sequence[int],
    query_episode_id: int,
    count: int,
    rng: np.random.Generator,
    noise: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    load_rgb: bool = True,
    load_mask_id: bool = True,
) -> Dict[str, Any]:
    if count < 1:
        raise ValueError(f'count must be >= 1, got {count}.')
    sampled_t0s = _sample_query_t0s_for_episode(
        task_builder,
        vidx=int(vidx),
        episode_id=int(query_episode_id),
        count=int(count),
        rng=rng,
    )
    samples: List[Dict[str, Any]] = []
    for t0 in sampled_t0s:
        query = _build_query_sample_at_t0(
            task_builder,
            vidx=int(vidx),
            episode_id=int(query_episode_id),
            t0=int(t0),
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        query['meta'].update(
            {
                'memory_support_episodes': [int(ep_id) for ep_id in support_ids],
                'memory_query_episode': int(query_episode_id),
            }
        )
        samples.append({**support_cond, **query})
    batch = task_builder._stack_samples(samples)
    task_builder.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
    return batch


def _build_cached_memory_ttt_package(
    *,
    store: VariationStore,
    task_builder: MAMLTaskBuilder,
    dataset_cfg: ICILConfig,
    variation: int,
    rng: np.random.Generator,
    torch_generator: Optional[torch.Generator],
    device: torch.device,
    num_train_timesteps: int,
    action_dim: int,
    use_rgb: bool,
    use_mask_id: bool,
    memory_cfg: ConfigDict,
) -> Dict[str, Any]:
    if int(dataset_cfg.K) < 2:
        raise ValueError('Memory TTT requires dataset.K >= 2 so one episode can be held out.')

    vidx = _select_support_vidx(store, variation=variation, rng=rng)
    episode_ids = store.list_episode_ids(vidx)
    log_query_loss = _as_bool(getattr(memory_cfg, 'log_query_loss', False))
    log_query_sample_mse = _as_bool(getattr(memory_cfg, 'log_query_sample_mse', False))
    needs_query_batch = log_query_loss or log_query_sample_mse
    required_episodes = int(dataset_cfg.K) + (1 if needs_query_batch else 0)
    if episode_ids.shape[0] < required_episodes:
        raise RuntimeError(
            f'Need at least {required_episodes} cached episodes, got {episode_ids.shape[0]} '
            f'for task={store.keys[vidx].task} variation={store.keys[vidx].variation}. '
            f'Memory TTT support K={dataset_cfg.K}, log_query_loss={log_query_loss}, '
            f'log_query_sample_mse={log_query_sample_mse}.'
        )

    chosen_ids_np = rng.choice(episode_ids, size=required_episodes, replace=False)
    chosen_ids = [int(eid) for eid in np.asarray(chosen_ids_np).tolist()]
    support_ids = chosen_ids[: int(dataset_cfg.K)]
    extra_query_episode_id = int(chosen_ids[int(dataset_cfg.K)]) if needs_query_batch else None

    holdout_index = _select_memory_holdout_index(len(support_ids), memory_cfg, rng)
    heldout_episode_id = int(support_ids[holdout_index])
    memory_support_ids = [
        int(ep_id) for idx, ep_id in enumerate(support_ids) if int(idx) != int(holdout_index)
    ]
    if not memory_support_ids:
        raise RuntimeError('Memory TTT produced an empty memory support set.')

    memory_support = task_builder.build_conditioning_from_support_ids(
        rng,
        vidx=int(vidx),
        support_ids=memory_support_ids,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
        load_full_traj=True,
    )
    if memory_support is None:
        raise RuntimeError('Failed to build memory-support conditioning.')

    preload_batches = _as_bool(getattr(memory_cfg, 'preload_support_batches_to_device', False))
    num_inner_batches = _resolve_num_memory_inner_batches(memory_cfg)
    num_queries_per_step = int(getattr(memory_cfg, 'num_queries_per_step', 1))
    if num_queries_per_step < 1:
        raise ValueError('cfg.memory_ttt.num_queries_per_step must be >= 1.')

    batch_device = device if preload_batches else torch.device('cpu')
    batch_generator = torch_generator
    if batch_device.type == 'cpu' and torch_generator is not None:
        batch_generator = torch.Generator()
        batch_generator.manual_seed(int(torch_generator.initial_seed()))

    shared_noise = None
    shared_timesteps = None
    if _as_bool(getattr(memory_cfg, 'reuse_diffusion_noise', False)):
        shared_noise = torch.randn(
            (1, int(dataset_cfg.H), int(action_dim)),
            device=batch_device,
            dtype=torch.float32,
            generator=batch_generator,
        )
        shared_timesteps = torch.randint(
            low=0,
            high=int(num_train_timesteps),
            size=(1,),
            device=batch_device,
            dtype=torch.long,
            generator=batch_generator,
        )

    memory_init_batch = _build_memory_query_batch(
        task_builder,
        vidx=int(vidx),
        support_cond=memory_support,
        support_ids=memory_support_ids,
        query_episode_id=heldout_episode_id,
        count=1,
        rng=rng,
        noise=None,
        timesteps=None,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
    )
    memory_init_batch = _drop_mask_ids_if_disabled(memory_init_batch, use_mask_id)
    if preload_batches:
        memory_init_batch = _to_device_batch(memory_init_batch, device)

    inner_batches: List[Dict[str, Any]] = []
    prepare_pbar = None
    if num_inner_batches > 0:
        prepare_pbar = tqdm(
            total=num_inner_batches,
            desc='Memory TTT Prepare',
            leave=True,
            unit='batch',
        )
    try:
        for batch_idx in range(num_inner_batches):
            inner_batch = _build_memory_query_batch(
                task_builder,
                vidx=int(vidx),
                support_cond=memory_support,
                support_ids=memory_support_ids,
                query_episode_id=heldout_episode_id,
                count=num_queries_per_step,
                rng=rng,
                noise=shared_noise if _as_bool(getattr(memory_cfg, 'reuse_diffusion_noise', False)) else None,
                timesteps=shared_timesteps if _as_bool(getattr(memory_cfg, 'reuse_diffusion_noise', False)) else None,
                load_rgb=use_rgb,
                load_mask_id=use_mask_id,
            )
            inner_batch = _drop_mask_ids_if_disabled(inner_batch, use_mask_id)
            if preload_batches:
                inner_batch = _to_device_batch(inner_batch, device)
            inner_batches.append(inner_batch)
            if prepare_pbar is not None:
                prepare_pbar.update(1)
                prepare_pbar.set_postfix(batch=batch_idx + 1)
    finally:
        if prepare_pbar is not None:
            prepare_pbar.close()

    query_batch = None
    if needs_query_batch and extra_query_episode_id is not None:
        query_loss_count = int(getattr(memory_cfg, 'num_query_loss_samples', num_queries_per_step))
        query_loss_count = max(1, query_loss_count)
        query_batch = _build_memory_query_batch(
            task_builder,
            vidx=int(vidx),
            support_cond=memory_support,
            support_ids=memory_support_ids,
            query_episode_id=int(extra_query_episode_id),
            count=query_loss_count,
            rng=rng,
            noise=shared_noise if _as_bool(getattr(memory_cfg, 'reuse_diffusion_noise', False)) else None,
            timesteps=shared_timesteps if _as_bool(getattr(memory_cfg, 'reuse_diffusion_noise', False)) else None,
            load_rgb=use_rgb,
            load_mask_id=use_mask_id,
        )
        query_batch = _drop_mask_ids_if_disabled(query_batch, use_mask_id)
        if preload_batches:
            query_batch = _to_device_batch(query_batch, device)

    rollout_support = _unsqueeze_support_batch_dim(memory_support)
    rollout_support['meta'] = {
        'task': store.keys[vidx].task,
        'variation': int(store.keys[vidx].variation),
        'support_episodes': list(support_ids),
        'memory_support_episodes': list(memory_support_ids),
        'heldout_episode': int(heldout_episode_id),
        'holdout_index': int(holdout_index),
        'query_episode': int(extra_query_episode_id) if extra_query_episode_id is not None else None,
    }

    return {
        'vidx': int(vidx),
        'support_ids': list(support_ids),
        'memory_support_ids': list(memory_support_ids),
        'heldout_episode_id': int(heldout_episode_id),
        'holdout_index': int(holdout_index),
        'query_episode_id': int(extra_query_episode_id) if extra_query_episode_id is not None else None,
        'inner_batches': inner_batches,
        'num_inner_batches': int(num_inner_batches),
        'memory_init_batch': memory_init_batch,
        'query_batch': query_batch,
        'support_cond': rollout_support,
        'meta': rollout_support['meta'],
    }




def _apply_memory_ttt_adaptation_in_place(
    *,
    policy: Policy,
    base_state_dict: Dict[str, torch.Tensor],
    support_package: Dict[str, Any],
    decoder_param_names: Sequence[str],
    device: torch.device,
    memory_cfg: ConfigDict,
    use_mask_id: bool,
    sample_mse_inference_steps: int,
    sample_mse_eta: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    policy.load_state_dict(base_state_dict, strict=True)
    policy.to(device)
    policy.eval()

    original_requires_grad = {
        name: bool(param.requires_grad)
        for name, param in policy.named_parameters()
    }
    decoder_name_set = set(str(name) for name in decoder_param_names)

    try:
        for _, param in policy.named_parameters():
            param.requires_grad_(False)

        decoder_params: List[torch.nn.Parameter] = []
        if _as_bool(getattr(memory_cfg, 'optimize_decoder', False)):
            for name, param in policy.named_parameters():
                if name in decoder_name_set:
                    param.requires_grad_(True)
                    decoder_params.append(param)
            if not decoder_params:
                raise RuntimeError('cfg.memory_ttt.optimize_decoder=True but no decoder parameters were selected.')

        memory_tokens, memory_token_mask = _init_memory_tokens_from_batch(
            policy,
            support_package['memory_init_batch'],
            device=device,
        )
        memory_tokens = nn.Parameter(memory_tokens.detach().to(device))
        if memory_token_mask is not None:
            memory_token_mask = memory_token_mask.to(device=device, dtype=torch.bool)

        initial_memory = memory_tokens.detach().clone()
        optim_params: List[torch.nn.Parameter] = [memory_tokens] + list(decoder_params)
        optimizer = _make_memory_optimizer(
            memory_cfg,
            memory_tokens=memory_tokens,
            decoder_params=decoder_params,
        )

        inner_steps = int(getattr(memory_cfg, 'inner_steps', 0))
        inner_losses: List[float] = []
        inner_grad_norms: List[float] = []
        query_losses: List[float] = []
        query_sample_mses: List[float] = []
        memory_norms: List[float] = []
        memory_delta_norms: List[float] = []
        log_query_loss = _as_bool(getattr(memory_cfg, 'log_query_loss', False))
        log_query_sample_mse = _as_bool(getattr(memory_cfg, 'log_query_sample_mse', False))
        grad_accum_steps = int(getattr(memory_cfg, 'grad_accum_steps', 1))
        if grad_accum_steps < 1:
            raise ValueError(f'cfg.memory_ttt.grad_accum_steps must be >= 1, got {grad_accum_steps}.')

        def _append_memory_stats() -> None:
            memory_detached = memory_tokens.detach()
            memory_norms.append(float(memory_detached.norm().cpu().item()))
            memory_delta_norms.append(float((memory_detached - initial_memory).norm().cpu().item()))

        def _eval_query_loss() -> float:
            query_batch = support_package.get('query_batch', None)
            if query_batch is None:
                return 0.0
            with torch.no_grad():
                loss = _memory_diffusion_loss(
                    policy,
                    query_batch,
                    memory_tokens=memory_tokens,
                    memory_token_mask=memory_token_mask,
                    device=device,
                )
            return float(loss.detach().cpu().item())

        def _eval_query_sample_mse() -> float:
            query_batch = support_package.get('query_batch', None)
            if query_batch is None:
                return 0.0
            return _query_sample_mse_with_memory_tokens(
                policy,
                query_batch,
                memory_tokens=memory_tokens,
                memory_token_mask=memory_token_mask,
                device=device,
                use_mask_id=use_mask_id,
                inference_steps=int(sample_mse_inference_steps),
                eta=float(sample_mse_eta),
            )

        _append_memory_stats()
        if support_package.get('query_batch', None) is not None and log_query_loss:
            query_losses.append(_eval_query_loss())
        if support_package.get('query_batch', None) is not None and log_query_sample_mse:
            query_sample_mses.append(_eval_query_sample_mse())

        inner_batches = list(support_package.get('inner_batches', []))
        if inner_steps > 0 and len(inner_batches) < 1:
            raise ValueError('cfg.memory_ttt.inner_steps > 0 requires at least one prepared inner batch.')

        pbar = None
        if inner_steps > 0:
            pbar = tqdm(
                total=inner_steps,
                desc='Memory TTT Inner GD',
                leave=True,
                unit='step',
            )
        try:
            for step_idx in range(1, inner_steps + 1):
                inner_batch = inner_batches[(step_idx - 1) % len(inner_batches)]
                optimizer.zero_grad(set_to_none=True)
                if grad_accum_steps == 1:
                    loss = _memory_diffusion_loss(
                        policy,
                        inner_batch,
                        memory_tokens=memory_tokens,
                        memory_token_mask=memory_token_mask,
                        device=device,
                    )
                    loss.backward()
                    loss_value = float(loss.detach().cpu().item())
                else:
                    loss_value = 0.0
                    for micro_batch, weight in _iter_microbatches(inner_batch, grad_accum_steps):
                        micro_loss = _memory_diffusion_loss(
                            policy,
                            micro_batch,
                            memory_tokens=memory_tokens,
                            memory_token_mask=memory_token_mask,
                            device=device,
                        )
                        (micro_loss * float(weight)).backward()
                        loss_value += float(micro_loss.detach().cpu().item()) * float(weight)
                grad_norm = _grad_global_norm(optim_params)
                max_grad_norm = float(getattr(memory_cfg, 'max_grad_norm', 0.0))
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(optim_params, max_grad_norm)
                optimizer.step()

                inner_losses.append(loss_value)
                inner_grad_norms.append(float(grad_norm))
                _append_memory_stats()
                if support_package.get('query_batch', None) is not None and log_query_loss:
                    query_losses.append(_eval_query_loss())
                if support_package.get('query_batch', None) is not None and log_query_sample_mse:
                    query_sample_mses.append(_eval_query_sample_mse())

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(step=step_idx, loss=f'{loss_value:.4g}')
        finally:
            if pbar is not None:
                pbar.close()

        final_memory = memory_tokens.detach()
        memory_state = {
            'support_tokens': final_memory,
            'support_token_mask': memory_token_mask.detach() if torch.is_tensor(memory_token_mask) else None,
        }
        stats = {
            'inner_losses': inner_losses,
            'inner_grad_norms': inner_grad_norms,
            'avg_inner_loss': float(sum(inner_losses) / max(1, len(inner_losses))) if inner_losses else 0.0,
            'avg_inner_grad_norm': float(sum(inner_grad_norms) / max(1, len(inner_grad_norms)))
            if inner_grad_norms
            else 0.0,
            'query_losses': query_losses,
            'avg_query_loss': float(sum(query_losses) / max(1, len(query_losses))) if query_losses else 0.0,
            'query_sample_mses': query_sample_mses,
            'avg_query_sample_mse': float(sum(query_sample_mses) / max(1, len(query_sample_mses)))
            if query_sample_mses
            else 0.0,
            'memory_norms': memory_norms,
            'memory_delta_norms': memory_delta_norms,
            'memory_token_shape': list(final_memory.shape),
            'memory_token_count': int(final_memory.numel()),
            'initial_memory_norm': float(initial_memory.norm().cpu().item()),
            'final_memory_norm': float(final_memory.norm().cpu().item()),
            'final_memory_delta_norm': float((final_memory - initial_memory).norm().cpu().item()),
            'decoder_param_tensors': int(len(decoder_param_names)),
            'decoder_param_count': int(sum(int(param.numel()) for param in decoder_params)),
            'grad_accum_steps': int(grad_accum_steps),
            'support_ids': list(support_package['support_ids']),
            'memory_support_ids': list(support_package['memory_support_ids']),
            'heldout_episode_id': int(support_package['heldout_episode_id']),
            'holdout_index': int(support_package['holdout_index']),
            'query_episode_id': support_package.get('query_episode_id', None),
            'num_inner_batches': int(support_package.get('num_inner_batches', len(inner_batches))),
        }
        return memory_state, stats
    finally:
        for name, param in policy.named_parameters():
            param.requires_grad_(bool(original_requires_grad.get(name, True)))
        policy.eval()


def _save_inner_loss_artifacts(
    *,
    inner_losses: Sequence[float],
    query_losses: Optional[Sequence[float]] = None,
    query_sample_mses: Optional[Sequence[float]] = None,
    run_dir: Path,
    stem: str,
) -> Dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)

    losses_path = run_dir / f'{stem}.inner_losses.json'
    payload: Dict[str, Any] = {
        'inner_step': list(range(1, len(inner_losses) + 1)),
        'support_loss': [float(v) for v in inner_losses],
        # Backward-compatible alias for the original support-loss-only artifact.
        'loss': [float(v) for v in inner_losses],
    }
    if query_losses is not None and len(query_losses) > 0:
        payload['query_inner_step'] = list(range(0, len(query_losses)))
        payload['query_loss'] = [float(v) for v in query_losses]
    if query_sample_mses is not None and len(query_sample_mses) > 0:
        payload['query_sample_mse_inner_step'] = list(range(0, len(query_sample_mses)))
        payload['query_sample_mse'] = [float(v) for v in query_sample_mses]

    with losses_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    plot_path = run_dir / f'{stem}.inner_losses.png'
    try:
        import matplotlib.pyplot as plt

        xs = np.arange(1, len(inner_losses) + 1, dtype=np.int64)
        ys = np.asarray(inner_losses, dtype=np.float64)
        has_query = query_losses is not None and len(query_losses) > 0
        if has_query:
            xs_query = np.arange(0, len(query_losses), dtype=np.int64)
            ys_query = np.asarray(query_losses, dtype=np.float64)
        else:
            xs_query = np.zeros((0,), dtype=np.int64)
            ys_query = np.zeros((0,), dtype=np.float64)

        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if len(xs) > 0:
            ax.plot(xs, ys, marker='o', linewidth=1.5, label='held-out support')
        if has_query:
            ax.plot(xs_query, ys_query, marker='s', linewidth=1.5, label='extra query')
        ax.set_xlabel('Inner Step')
        ax.set_ylabel('Diffusion Loss')
        ax.set_yscale('log')
        ax.set_title('Memory TTT Inner-Loop Loss')
        ax.grid(True, which='both', alpha=0.3)
        max_x = int(max(len(inner_losses), len(query_losses) - 1 if has_query else 0))
        if max_x > 0:
            max_xticks = min(8, max_x + 1)
            xticks = np.unique(np.linspace(0, max_x, num=max_xticks, dtype=np.int64))
            ax.set_xticks(xticks.tolist())
        elif has_query or len(xs) > 0:
            ax.set_xticks([0])
        if len(xs) > 0 or has_query:
            ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional plotting dependency/runtime
        logging.warning('Failed to save memory TTT inner-loss plot to %s: %s', plot_path, exc)
        plot_path = None

    sample_mse_plot_path = None
    if query_sample_mses is not None and len(query_sample_mses) > 0:
        sample_mse_plot_path = run_dir / f'{stem}.query_sample_mse.png'
        try:
            import matplotlib.pyplot as plt

            xs_mse = np.arange(0, len(query_sample_mses), dtype=np.int64)
            ys_mse = np.asarray(query_sample_mses, dtype=np.float64)
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(xs_mse, ys_mse, marker='o', linewidth=1.5, label='extra query sample MSE')
            ax.set_xlabel('Inner Step')
            ax.set_ylabel('Sampled Action MSE')
            ax.set_yscale('log')
            ax.set_title('Memory TTT Extra Query Sample MSE')
            ax.grid(True, which='both', alpha=0.3)
            max_x = int(len(query_sample_mses) - 1)
            if max_x > 0:
                max_xticks = min(8, max_x + 1)
                xticks = np.unique(np.linspace(0, max_x, num=max_xticks, dtype=np.int64))
                ax.set_xticks(xticks.tolist())
            else:
                ax.set_xticks([0])
            ax.legend()
            fig.tight_layout()
            fig.savefig(sample_mse_plot_path, dpi=160)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - optional plotting dependency/runtime
            logging.warning('Failed to save memory TTT query sample-MSE plot to %s: %s', sample_mse_plot_path, exc)
            sample_mse_plot_path = None

    return {
        'inner_loss_json_path': str(losses_path),
        'inner_loss_plot_path': str(plot_path) if plot_path is not None else '',
        'query_sample_mse_plot_path': str(sample_mse_plot_path) if sample_mse_plot_path is not None else '',
    }


def _find_success_gif_path(result: Dict[str, Any]) -> Optional[Path]:
    video_paths = result.get('video_paths', {}) or {}
    if isinstance(video_paths, dict):
        front_paths = video_paths.get('front', None)
        if isinstance(front_paths, dict):
            gif_path = front_paths.get('gif', '')
            if gif_path:
                return Path(str(gif_path))

        for camera_paths in video_paths.values():
            if isinstance(camera_paths, dict):
                gif_path = camera_paths.get('gif', '')
                if gif_path:
                    return Path(str(gif_path))

    legacy_path = str(result.get('video_path', '') or '')
    if legacy_path.lower().endswith('.gif'):
        return Path(legacy_path)
    return None


def _maybe_log_wandb_eval_summary(
    wandb_run: Optional[Any],
    *,
    results: Sequence[Dict[str, Any]],
    success_rate: float,
    task_name: str,
    variation: int,
) -> None:
    if wandb_run is None:
        return
    try:
        import wandb
    except ImportError as exc:
        raise ImportError('wandb must be installed to log eval results.') from exc

    log_dict: Dict[str, Any] = {
        'eval/success_rate': float(success_rate),
    }

    success_result = next((result for result in results if bool(result.get('success', False))), None)
    if success_result is not None:
        gif_path = _find_success_gif_path(success_result)
        if gif_path is None:
            logging.warning(
                'wandb logging requested, but no GIF was found for the first successful episode. '
                'Keep cfg.video.enable=True and include "gif" in cfg.video.formats to log a successful rollout GIF.'
            )
        elif not gif_path.is_file():
            logging.warning('Successful GIF path does not exist on disk: %s', gif_path)
        else:
            log_dict['eval/success_gif'] = wandb.Video(
                str(gif_path),
                format='gif',
                caption=(
                    f'task={task_name} variation={variation} episode={int(success_result["episode_index"])} '
                    f'success_rate={float(success_rate):.3f}'
                ),
            )

    wandb_run.log(log_dict)


@torch.no_grad()
def _sample_actions_with_memory_tokens(
    policy: Policy,
    *,
    adapted_support_tokens: torch.Tensor,
    adapted_support_token_mask: Optional[torch.Tensor],
    cond_xyz: Optional[torch.Tensor],
    query_xyz: torch.Tensor,
    query_state: torch.Tensor,
    action_horizon: int,
    cond_state: Optional[torch.Tensor] = None,
    cond_traj: Optional[torch.Tensor] = None,
    cond_traj_mask: Optional[torch.Tensor] = None,
    cond_rgb: Optional[torch.Tensor] = None,
    query_rgb: Optional[torch.Tensor] = None,
    cond_mask_id: Optional[torch.Tensor] = None,
    query_mask_id: Optional[torch.Tensor] = None,
    cond_valid: Optional[torch.Tensor] = None,
    query_valid: Optional[torch.Tensor] = None,
    inference_steps: Optional[int] = None,
    eta: float = 0.0,
) -> torch.Tensor:
    if action_horizon < 1:
        raise ValueError('action_horizon must be >= 1.')
    if eta < 0.0:
        raise ValueError('eta must be >= 0.')

    device = query_xyz.device
    B = int(query_xyz.shape[0])
    H = int(action_horizon)
    A = int(policy.action_dim)

    query_batch: Dict[str, Any] = {
        'query_xyz': query_xyz,
        'query_state': query_state,
    }
    if cond_xyz is not None:
        query_batch['cond_xyz'] = cond_xyz
    if cond_state is not None:
        query_batch['cond_state'] = cond_state
    if cond_traj is not None:
        query_batch['cond_traj'] = cond_traj
    if cond_traj_mask is not None:
        query_batch['cond_traj_mask'] = cond_traj_mask
    if cond_rgb is not None:
        query_batch['cond_rgb'] = cond_rgb
    if query_rgb is not None:
        query_batch['query_rgb'] = query_rgb
    if cond_mask_id is not None:
        query_batch['cond_mask_id'] = cond_mask_id
    if query_mask_id is not None:
        query_batch['query_mask_id'] = query_mask_id
    if cond_valid is not None:
        query_batch['cond_valid'] = cond_valid
    if query_valid is not None:
        query_batch['query_valid'] = query_valid

    query_tokens, query_mask = _query_tokens_from_batch(policy, query_batch)
    support_tokens = adapted_support_tokens.to(device=device, dtype=query_tokens.dtype)
    support_mask = (
        adapted_support_token_mask.to(device=device, dtype=torch.bool)
        if adapted_support_token_mask is not None
        else None
    )

    scheduler = policy.noise_scheduler
    total_T = int(scheduler.config.num_train_timesteps)
    steps = policy.num_inference_steps if inference_steps is None else int(inference_steps)
    steps = max(1, min(steps, total_T))

    try:
        scheduler.set_timesteps(steps, device=device)
    except TypeError:
        scheduler.set_timesteps(steps)

    x_t = torch.randn(B, H, A, device=device)
    step_sig = inspect.signature(scheduler.step).parameters
    for t_now in scheduler.timesteps:
        t_int = int(t_now.item() if torch.is_tensor(t_now) else t_now)
        t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)
        model_out = _predict_model_output_from_tokens(
            policy,
            x_t=x_t,
            t=t_batch,
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
            num_support_demos=_infer_num_support_demos_from_batch(query_batch),
        )
        step_kwargs: Dict[str, Any] = {}
        if 'eta' in step_sig:
            step_kwargs['eta'] = float(eta)
        step_out = scheduler.step(model_out, t_now, x_t, **step_kwargs)
        if isinstance(step_out, tuple):
            x_t = step_out[0]
        else:
            x_t = step_out.prev_sample
    return x_t


def _run_eval_episode(
    *,
    episode_index: int,
    task_env: Any,
    variation: int,
    model: Policy,
    device: torch.device,
    dataset_cfg: ICILConfig,
    support_cond: Optional[Dict[str, Any]],
    ignore_demos: bool,
    query_stride_mode: str,
    processor: _LiveConditioningProcessor,
    cfg: ConfigDict,
    run_dir: Path,
    adapted_support_tokens: Optional[torch.Tensor] = None,
    adapted_support_token_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    if variation >= 0:
        task_env.set_variation(int(variation))
    descriptions, obs = task_env.reset()
    del descriptions

    history: List[Dict[str, torch.Tensor]] = [processor.observation_to_frame(obs)]
    video_cameras = _video_cameras_from_cfg(cfg) if _as_bool(cfg.video.enable) else []
    video_formats = _video_formats_from_cfg(cfg) if _as_bool(cfg.video.enable) else []
    frames_by_camera: Dict[str, List[np.ndarray]] = {camera: [] for camera in video_cameras}
    if _as_bool(cfg.video.enable):
        for camera in video_cameras:
            frames_by_camera[camera].append(_extract_rgb_frame(obs, camera))

    success = False
    terminated = False
    error: Optional[str] = None
    env_steps = 0

    execute_actions = max(1, int(cfg.control.execute_actions_per_plan))
    max_env_steps = int(cfg.task.max_env_steps)
    pbar = tqdm(
        total=max_env_steps,
        desc=f'Episode {episode_index}',
        leave=False,
        unit='step',
    )
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(
                history,
                dataset_cfg=dataset_cfg,
                query_stride_mode=query_stride_mode,
            )
            query_xyz = _to_device_tensor(query['query_xyz'], device)
            query_state = _to_device_tensor(query['query_state'], device)
            query_valid = _to_device_tensor(query.get('query_valid', None), device)
            query_mask_id = _to_device_tensor(query.get('query_mask_id', None), device)
            query_rgb = _to_device_tensor(query.get('query_rgb', None), device)

            if ignore_demos:
                B = int(query_xyz.shape[0])
                N = int(query_xyz.shape[2])
                S = int(query_state.shape[-1])
                cond_xyz = query_xyz.new_zeros((B, 1, 1, N, 3))
                cond_state = query_state.new_zeros((B, 1, 1, S))
                cond_valid = None
                cond_mask_id = None
                cond_rgb = None
                cond_traj = None
                cond_traj_mask = None
            else:
                if support_cond is None:
                    raise RuntimeError('support_cond is required when ignore_demos=False.')
                cond_xyz = _to_device_tensor(support_cond['cond_xyz'], device)
                cond_state = _to_device_tensor(support_cond['cond_state'], device)
                cond_valid = _to_device_tensor(support_cond.get('cond_valid', None), device)
                cond_mask_id = _to_device_tensor(support_cond.get('cond_mask_id', None), device)
                cond_rgb = _to_device_tensor(support_cond.get('cond_rgb', None), device)
                cond_traj = _to_device_tensor(support_cond.get('cond_traj', None), device)
                cond_traj_mask = _to_device_tensor(support_cond.get('cond_traj_mask', None), device)

            with torch.no_grad():
                if adapted_support_tokens is not None:
                    plan = _sample_actions_with_memory_tokens(
                        model,
                        adapted_support_tokens=adapted_support_tokens,
                        adapted_support_token_mask=adapted_support_token_mask,
                        cond_xyz=cond_xyz,
                        cond_state=cond_state,
                        cond_traj=cond_traj,
                        cond_traj_mask=cond_traj_mask,
                        query_xyz=query_xyz,
                        query_state=query_state,
                        action_horizon=int(dataset_cfg.H),
                        cond_rgb=cond_rgb,
                        query_rgb=query_rgb,
                        cond_mask_id=cond_mask_id,
                        query_mask_id=query_mask_id,
                        cond_valid=cond_valid,
                        query_valid=query_valid,
                        inference_steps=int(cfg.inference.inference_steps),
                        eta=float(cfg.inference.eta),
                    )
                else:
                    plan = model.sample_actions(
                        cond_xyz=cond_xyz,
                        cond_state=cond_state,
                        cond_traj=cond_traj,
                        cond_traj_mask=cond_traj_mask,
                        query_xyz=query_xyz,
                        query_state=query_state,
                        action_horizon=int(dataset_cfg.H),
                        cond_rgb=cond_rgb,
                        query_rgb=query_rgb,
                        cond_mask_id=cond_mask_id,
                        query_mask_id=query_mask_id,
                        cond_valid=cond_valid,
                        query_valid=query_valid,
                        inference_steps=int(cfg.inference.inference_steps),
                        eta=float(cfg.inference.eta),
                    )
            plan_np = plan[0].detach().cpu().numpy()
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
                    for camera in video_cameras:
                        frames_by_camera[camera].append(_extract_rgb_frame(obs, camera))

                if success or terminated or env_steps >= max_env_steps:
                    break
    finally:
        pbar.close()

    video_path = None
    video_paths: Dict[str, Dict[str, str]] = {}
    if _as_bool(cfg.video.enable):
        for camera in video_cameras:
            camera_frames = frames_by_camera.get(camera, [])
            if not camera_frames:
                continue
            camera_outputs: Dict[str, str] = {}
            for fmt in video_formats:
                video_file = run_dir / 'videos' / camera / f'episode_{episode_index:04d}.{fmt}'
                actual = _write_video(camera_frames, video_file, fps=int(cfg.video.fps))
                camera_outputs[fmt] = str(actual)
                if video_path is None:
                    video_path = str(actual)
            if camera_outputs:
                video_paths[camera] = camera_outputs

    return {
        'episode_index': int(episode_index),
        'success': bool(success),
        'terminated': bool(terminated),
        'env_steps': int(env_steps),
        'error': error,
        'video_path': video_path,
        'video_paths': video_paths,
    }


def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    device = _resolve_device(str(cfg.device))

    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt, state_dict = _load_checkpoint(checkpoint_path)
    model_cfg = _model_config_from_checkpoint_or_default(ckpt)
    dataset_cfg = _dataset_config_from_eval_and_checkpoint(cfg, ckpt)
    if not hasattr(cfg, 'memory_ttt'):
        raise ValueError('Memory TTT eval requires cfg.memory_ttt.')
    memory_cfg = cfg.memory_ttt
    use_mask_id = _conditioning_use_mask_id_from_eval_and_checkpoint(cfg, model_cfg)
    ignore_demos = _ignore_demos_from_model_cfg(model_cfg)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    support_source = _support_source_from_eval(cfg)
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)

    if ignore_demos:
        raise ValueError('Memory TTT eval requires a checkpoint whose model conditions on support demos (ignore_demos=False).')

    model = build_policy(
        model_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    decoder_param_names = _select_memory_decoder_param_names(model, memory_cfg)
    decoder_param_count = _param_count_by_names(model, decoder_param_names)
    grad_accum_steps = int(getattr(memory_cfg, 'grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f'cfg.memory_ttt.grad_accum_steps must be >= 1, got {grad_accum_steps}.')

    if action_dim != 8:
        raise ValueError(
            f'Current eval pipeline expects action_dim=8 (gripper_pose[7] + gripper_open[1]), got {action_dim}.'
        )

    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    output_root = Path(str(cfg.output.root_dir)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(cfg, output_root)
    run_suffix = str(wandb_run.id) if wandb_run is not None else None
    run_name = _build_run_name(task_name=task_name, variation=variation, unique_suffix=run_suffix)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if wandb_run is not None:
        wandb_run.name = run_name

    resolved_payload = cfg.to_dict()
    resolved_payload['dataset']['K'] = int(dataset_cfg.K)
    resolved_payload['dataset']['L'] = int(dataset_cfg.L)
    resolved_payload['dataset']['T_obs'] = int(dataset_cfg.T_obs)
    resolved_payload['dataset']['H'] = int(dataset_cfg.H)
    resolved_payload['dataset']['stride'] = int(dataset_cfg.stride)
    resolved_payload['memory_ttt']['grad_accum_steps'] = int(grad_accum_steps)
    resolved_payload['resolved'] = {
        'checkpoint_path': str(checkpoint_path),
        'run_name': run_name,
        'run_dir': str(run_dir),
        'decoder_param_names': list(decoder_param_names),
        'decoder_param_tensors': len(decoder_param_names),
        'decoder_param_count': int(decoder_param_count),
        'support_source': support_source,
    }
    config_path = run_dir / 'resolved_eval_config.json'
    with config_path.open('w', encoding='utf-8') as f:
        json.dump(resolved_payload, f, indent=2)
    if wandb_run is not None:
        wandb_run.config.update(resolved_payload, allow_val_change=True)
        wandb_run.save(str(config_path), policy='now')

    logging.info('Loading task=%r variation=%d', task_name, variation)
    logging.info('Checkpoint=%s', checkpoint_path)
    logging.info('Model cfg: encoder_name=%s | ignore_demos=%s', model_cfg.encoder_name, ignore_demos)
    logging.info('Conditioning cfg: use_mask_id=%s | support_source=%s', use_mask_id, support_source)
    logging.info('Eval query_stride_mode=%s', query_stride_mode)
    logging.info(
        'Dataset cfg: K=%d L=%d T_obs=%d H=%d stride=%d',
        dataset_cfg.K,
        dataset_cfg.L,
        dataset_cfg.T_obs,
        dataset_cfg.H,
        dataset_cfg.stride,
    )
    logging.info(
        'Memory TTT cfg: inner_steps=%d | inner_lr=%.3e | optimizer=%s | max_grad_norm=%.3f | '
        'num_queries_per_step=%d | grad_accum_steps=%d | num_inner_batches=%d | reuse_diffusion_noise=%s | '
        'optimize_decoder=%s | decoder_params=%d | preload_batches=%s | log_query_loss=%s | '
        'num_query_loss_samples=%d | log_query_sample_mse=%s',
        int(memory_cfg.inner_steps),
        float(memory_cfg.inner_lr),
        _memory_optimizer_name(memory_cfg),
        float(memory_cfg.max_grad_norm),
        int(memory_cfg.num_queries_per_step),
        int(grad_accum_steps),
        int(_resolve_num_memory_inner_batches(memory_cfg)),
        str(_as_bool(memory_cfg.reuse_diffusion_noise)),
        str(_as_bool(memory_cfg.optimize_decoder)),
        len(decoder_param_names),
        str(_as_bool(getattr(memory_cfg, 'preload_support_batches_to_device', False))),
        str(_as_bool(getattr(memory_cfg, 'log_query_loss', False))),
        int(getattr(memory_cfg, 'num_query_loss_samples', memory_cfg.num_queries_per_step)),
        str(_as_bool(getattr(memory_cfg, 'log_query_sample_mse', False))),
    )
    logging.info(
        'Memory TTT decoder params: tensors=%d | params=%s | examples=%s',
        len(decoder_param_names),
        f'{decoder_param_count:,}',
        decoder_param_names[:8],
    )

    env = None
    task_env = None
    support_store: Optional[VariationStore] = None
    results: List[Dict[str, Any]] = []
    support_cache_root: Optional[Path] = None
    current_support_package: Optional[Dict[str, Any]] = None
    current_memory_state: Optional[Dict[str, Any]] = None
    current_memory_stats: Optional[Dict[str, Any]] = None

    try:
        if variation < 0:
            raise ValueError('Cached memory TTT support conditioning requires cfg.task.variation >= 0.')
        support_cache_root = _support_cache_root_from_eval_and_checkpoint(cfg, ckpt)
        task_keys = build_variation_keys(support_cache_root, task_name)
        if not task_keys:
            raise RuntimeError(f'No cached variations found for task {task_name!r} under {support_cache_root}.')
        _warn_if_cached_num_points_mismatch(
            task_keys=task_keys,
            expected_num_points=int(cfg.conditioning.num_points),
            task_name=task_name,
        )
        support_store = VariationStore(task_keys, keep_open_per_worker=True)
        logging.info('Using cached support from %s', support_cache_root)

        env, task_env = _build_rlbench_env(cfg, task_name)
        processor = _LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=_as_bool(cfg.conditioning.use_rgb),
            use_mask_id=use_mask_id,
            seed=seed + 11,
        )
        task_builder = MAMLTaskBuilder(
            store=support_store,
            cfg=dataset_cfg,
            seed=seed + 23,
            num_tries_per_item=int(getattr(memory_cfg, 'num_tries_per_item', 100)),
        )

        rng = np.random.default_rng(seed + 17)
        for ep in range(int(cfg.task.num_eval_episodes)):
            regen = _as_bool(cfg.conditioning.regenerate_demos_each_episode)
            if current_support_package is None or regen:
                torch_seed = seed + 100_003 + ep
                torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
                torch_gen.manual_seed(torch_seed)
                current_support_package = _build_cached_memory_ttt_package(
                    store=support_store,
                    task_builder=task_builder,
                    dataset_cfg=ICILConfig(
                        K=int(dataset_cfg.K),
                        L=int(dataset_cfg.L),
                        T_obs=int(dataset_cfg.T_obs),
                        H=int(dataset_cfg.H),
                        stride=int(dataset_cfg.stride),
                    ),
                    variation=variation,
                    rng=rng,
                    torch_generator=torch_gen,
                    device=device,
                    num_train_timesteps=int(model.noise_scheduler.config.num_train_timesteps),
                    action_dim=int(action_dim),
                    use_rgb=_as_bool(cfg.conditioning.use_rgb),
                    use_mask_id=use_mask_id,
                    memory_cfg=memory_cfg,
                )
                current_memory_state, current_memory_stats = _apply_memory_ttt_adaptation_in_place(
                    policy=model,
                    base_state_dict=state_dict,
                    support_package=current_support_package,
                    decoder_param_names=decoder_param_names,
                    device=device,
                    memory_cfg=memory_cfg,
                    use_mask_id=use_mask_id,
                    sample_mse_inference_steps=int(cfg.inference.inference_steps),
                    sample_mse_eta=float(cfg.inference.eta),
                )
                current_memory_stats.update(
                    _save_inner_loss_artifacts(
                        inner_losses=current_memory_stats.get('inner_losses', []),
                        query_losses=current_memory_stats.get('query_losses', None),
                        query_sample_mses=current_memory_stats.get('query_sample_mses', None),
                        run_dir=run_dir,
                        stem=f'memory_ttt_episode_{ep:04d}',
                    )
                )
                model.eval()
                if current_memory_stats.get('query_losses', None) or current_memory_stats.get('query_sample_mses', None):
                    logging.info(
                        'Memory TTT adaptation ready for episode %d | support_episodes=%s | memory_support=%s | '
                        'heldout_episode=%s | query_episode=%s | inner_batches=%d | avg_inner_loss=%.6f | '
                        'avg_query_loss=%.6f | avg_query_sample_mse=%.6f | avg_inner_grad_norm=%.6f | '
                        'final_memory_delta_norm=%.6f',
                        ep,
                        current_memory_stats['support_ids'],
                        current_memory_stats['memory_support_ids'],
                        current_memory_stats['heldout_episode_id'],
                        current_memory_stats.get('query_episode_id', None),
                        int(current_memory_stats.get('num_inner_batches', 0)),
                        float(current_memory_stats['avg_inner_loss']),
                        float(current_memory_stats['avg_query_loss']),
                        float(current_memory_stats.get('avg_query_sample_mse', 0.0)),
                        float(current_memory_stats['avg_inner_grad_norm']),
                        float(current_memory_stats['final_memory_delta_norm']),
                    )
                else:
                    logging.info(
                        'Memory TTT adaptation ready for episode %d | support_episodes=%s | memory_support=%s | '
                        'heldout_episode=%s | inner_batches=%d | avg_inner_loss=%.6f | '
                        'avg_inner_grad_norm=%.6f | final_memory_delta_norm=%.6f',
                        ep,
                        current_memory_stats['support_ids'],
                        current_memory_stats['memory_support_ids'],
                        current_memory_stats['heldout_episode_id'],
                        int(current_memory_stats.get('num_inner_batches', 0)),
                        float(current_memory_stats['avg_inner_loss']),
                        float(current_memory_stats['avg_inner_grad_norm']),
                        float(current_memory_stats['final_memory_delta_norm']),
                    )

            res = _run_eval_episode(
                episode_index=ep,
                task_env=task_env,
                variation=variation,
                model=model,
                device=device,
                dataset_cfg=ICILConfig(
                    K=int(dataset_cfg.K),
                    L=int(dataset_cfg.L),
                    T_obs=int(dataset_cfg.T_obs),
                    H=int(dataset_cfg.H),
                    stride=int(dataset_cfg.stride),
                ),
                support_cond=current_support_package['support_cond'] if current_support_package is not None else None,
                ignore_demos=ignore_demos,
                query_stride_mode=query_stride_mode,
                processor=processor,
                cfg=cfg,
                run_dir=run_dir,
                adapted_support_tokens=(
                    current_memory_state['support_tokens'] if current_memory_state is not None else None
                ),
                adapted_support_token_mask=(
                    current_memory_state.get('support_token_mask', None) if current_memory_state is not None else None
                ),
            )
            if current_memory_stats is not None:
                res['memory_ttt'] = current_memory_stats
            results.append(res)
            logging.info(
                'Episode %d | success=%s | steps=%d%s',
                ep,
                res['success'],
                res['env_steps'],
                f' | error={res["error"]}' if res['error'] else '',
            )

        n_success = sum(1 for r in results if r['success'])
        success_rate = float(n_success) / float(max(1, len(results)))
        summary = {
            'task': task_name,
            'variation': variation,
            'checkpoint_path': str(checkpoint_path),
            'num_episodes': len(results),
            'num_success': int(n_success),
            'success_rate': success_rate,
            'decoder_param_names': list(decoder_param_names),
            'decoder_param_tensors': len(decoder_param_names),
            'decoder_param_count': int(decoder_param_count),
            'resolved_dataset': {
                'K': int(dataset_cfg.K),
                'L': int(dataset_cfg.L),
                'T_obs': int(dataset_cfg.T_obs),
                'H': int(dataset_cfg.H),
                'stride': int(dataset_cfg.stride),
            },
            'resolved_memory_ttt': {
                'inner_steps': int(memory_cfg.inner_steps),
                'inner_lr': float(memory_cfg.inner_lr),
                'optimizer': _memory_optimizer_name(memory_cfg),
                'max_grad_norm': float(memory_cfg.max_grad_norm),
                'num_queries_per_step': int(memory_cfg.num_queries_per_step),
                'grad_accum_steps': int(grad_accum_steps),
                'num_inner_batches': int(_resolve_num_memory_inner_batches(memory_cfg)),
                'reuse_diffusion_noise': _as_bool(memory_cfg.reuse_diffusion_noise),
                'preload_support_batches_to_device': _as_bool(
                    getattr(memory_cfg, 'preload_support_batches_to_device', False)
                ),
                'optimize_decoder': _as_bool(memory_cfg.optimize_decoder),
                'decoder_lr': float(getattr(memory_cfg, 'decoder_lr', memory_cfg.inner_lr)),
                'decoder_param_prefixes': list(_memory_decoder_prefixes(memory_cfg)),
                'holdout_index': int(getattr(memory_cfg, 'holdout_index', -1)),
                'log_query_loss': _as_bool(getattr(memory_cfg, 'log_query_loss', False)),
                'num_query_loss_samples': int(getattr(memory_cfg, 'num_query_loss_samples', memory_cfg.num_queries_per_step)),
                'log_query_sample_mse': _as_bool(getattr(memory_cfg, 'log_query_sample_mse', False)),
            },
            'results': results,
        }
        summary_path = run_dir / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        if wandb_run is not None:
            wandb_run.save(str(summary_path), policy='now')
            _maybe_log_wandb_eval_summary(
                wandb_run,
                results=results,
                success_rate=success_rate,
                task_name=task_name,
                variation=variation,
            )
        logging.info(
            'Evaluation complete | success=%d/%d (%.3f) | outputs=%s',
            n_success,
            len(results),
            success_rate,
            run_dir,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if support_store is not None:
            support_store.close()
        if env is not None:
            env.shutdown()


def main(argv: Sequence[str]) -> None:
    del argv
    cfg = _CONFIG.value
    evaluate(cfg)


if __name__ == '__main__':
    app.run(main)
