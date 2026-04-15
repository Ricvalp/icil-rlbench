from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.func import functional_call
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
from icil.models.maml import (
    MAMLConfig,
    MAMLTaskBuilder,
    MAMLTaskSpec,
    PolicyLossWrapper,
    copy_fast_params_into_policy,
    get_fast_param_names,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/eval_perceiver_encoder_decoder_maml.py',
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
        'traj_perceiver': {},
    }
    for k, v in model_from_ckpt.items():
        if k in ('policy', 'perceiver_demo_query', 'traj_perceiver', 'encoder_name'):
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
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.use_mask_id)
    return _as_bool(getattr(cfg.conditioning, 'use_mask_id', True))



def _ignore_demos_from_model_cfg(model_cfg: PolicyBuilderConfig) -> bool:
    if str(model_cfg.encoder_name) == 'conv3d_demo_query':
        return bool(model_cfg.conv3d_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query':
        return bool(model_cfg.perceiver_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.ignore_demos)
    return False



def _resolve_data_k(cfg: ConfigDict, ckpt: Dict[str, Any]) -> int:
    configured_k = int(cfg.dataset.K)
    if configured_k > 0:
        return configured_k
    ckpt_dataset = {}
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}
    if isinstance(ckpt_dataset, dict) and int(ckpt_dataset.get('K', 0)) > 0:
        return int(ckpt_dataset['K'])
    raise ValueError(
        'cfg.dataset.K=0 requires checkpoint["config"]["dataset"]["K"] > 0 so MAML eval can reuse the '
        'training support count.'
    )


def _resolve_maml_cfg_from_checkpoint(ckpt: Dict[str, Any], *, data_k: int) -> MAMLConfig:
    ckpt_config = ckpt.get('config', None)
    if not isinstance(ckpt_config, dict):
        raise ValueError('MAML eval requires checkpoint["config"] to be present.')

    maml_src = ckpt_config.get('maml', {}) or {}
    if not isinstance(maml_src, dict) or not maml_src:
        raise ValueError('MAML eval requires checkpoint["config"]["maml"] to be present.')

    maml_cfg = _dataclass_from_dict(MAMLConfig(), maml_src)
    resolved_outer_context_size = int(getattr(maml_cfg, 'outer_context_size', 0))
    if resolved_outer_context_size <= 0:
        resolved_payload = ckpt_config.get('resolved', {}) or {}
        if isinstance(resolved_payload, dict) and int(resolved_payload.get('outer_context_size', 0)) > 0:
            resolved_outer_context_size = int(resolved_payload['outer_context_size'])
        elif int(data_k) > 0:
            resolved_outer_context_size = int(data_k)

    if resolved_outer_context_size <= 0:
        raise ValueError('Resolved MAML outer_context_size must be positive.')
    if resolved_outer_context_size > int(data_k):
        raise ValueError(
            f'MAML outer_context_size={resolved_outer_context_size} exceeds eval data.K={int(data_k)}.'
        )
    maml_cfg.outer_context_size = int(resolved_outer_context_size)
    return maml_cfg



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
            'MAML eval currently supports cfg.conditioning.support_source="cache" only, '
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



def _resolve_maml_fast_param_names(
    model: Policy,
    ckpt: Dict[str, Any],
    maml_cfg: MAMLConfig,
) -> List[str]:
    ckpt_config = ckpt.get('config', None)
    resolved_payload = ckpt_config.get('resolved', {}) if isinstance(ckpt_config, dict) else {}
    model_param_names = {name for name, _ in model.named_parameters()}

    if isinstance(resolved_payload, dict):
        fast_names = resolved_payload.get('fast_param_names', None)
        if isinstance(fast_names, list) and fast_names:
            fast_names = [str(name) for name in fast_names]
            missing = [name for name in fast_names if name not in model_param_names]
            if missing:
                raise ValueError(
                    'Checkpoint-resolved fast_param_names do not match the loaded model. '
                    f'Examples: {missing[:8]}'
                )
            return fast_names

    logging.warning(
        'Checkpoint has no resolved fast_param_names; recomputing from checkpoint maml config.'
    )
    return get_fast_param_names(
        model,
        last_frac=float(maml_cfg.last_frac_fast),
        include_decoder_mlp=bool(getattr(maml_cfg, 'include_decoder_mlp_fast', True)),
        include_ada=bool(maml_cfg.include_ada_fast),
        include_final_norm=bool(maml_cfg.include_final_norm_fast),
        include_input_projections=bool(getattr(maml_cfg, 'include_input_projections_fast', False)),
        include_output_head=bool(getattr(maml_cfg, 'include_output_head_fast', False)),
        include_diffusion_conditioning=bool(getattr(maml_cfg, 'include_diffusion_conditioning_fast', False)),
    )



def _count_params_by_name(model: Policy, names: Sequence[str]) -> int:
    name_set = set(names)
    return sum(int(param.numel()) for name, param in model.named_parameters() if name in name_set)



def _sample_loo_indices(K: int, *, num_loo_per_task: int, rng: np.random.Generator) -> List[int]:
    if K < 1:
        raise ValueError(f'K must be positive, got {K}.')
    if num_loo_per_task < 1:
        raise ValueError(f'num_loo_per_task must be positive, got {num_loo_per_task}.')

    out: List[int] = []
    while len(out) < int(num_loo_per_task):
        perm = rng.permutation(K)
        take = min(K, int(num_loo_per_task) - len(out))
        out.extend(int(idx) for idx in perm[:take].tolist())
    return out



def _clip_grads_in_list(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    if max_norm <= 0.0:
        return grads
    valid_grads = [grad for grad in grads if grad is not None]
    if not valid_grads:
        return grads
    total_norm = torch.norm(torch.stack([grad.norm(2) for grad in valid_grads]), 2)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-6)
    return [grad * scale if grad is not None else None for grad in grads]



def _grad_list_global_norm(grads: Sequence[torch.Tensor]) -> float:
    valid_grads = [grad.detach() for grad in grads if grad is not None]
    if not valid_grads:
        return 0.0
    return float(torch.norm(torch.stack([grad.norm(2) for grad in valid_grads]), 2).item())



def _prefix_param_names(names: Sequence[str], prefix: str = 'policy.') -> List[str]:
    return [f'{prefix}{name}' for name in names]



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



def _build_cached_maml_package(
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
    adapt_cfg: ConfigDict,
) -> Dict[str, Any]:
    vidx = _select_support_vidx(store, variation=variation, rng=rng)
    episode_ids = store.list_episode_ids(vidx)
    log_query_loss = _as_bool(getattr(adapt_cfg, 'log_query_loss', False))
    required_episodes = int(dataset_cfg.K) + (1 if log_query_loss else 0)
    if episode_ids.shape[0] < required_episodes:
        raise RuntimeError(
            f'Need at least {required_episodes} cached episodes, got {episode_ids.shape[0]} '
            f'for task={store.keys[vidx].task} variation={store.keys[vidx].variation}. '
            f'MAML eval support K={dataset_cfg.K}, log_query_loss={log_query_loss}.'
        )

    chosen_ids_np = rng.choice(episode_ids, size=required_episodes, replace=False)
    chosen_ids = [int(eid) for eid in np.asarray(chosen_ids_np).tolist()]
    support_ids = chosen_ids[: int(dataset_cfg.K)]
    query_episode_id = int(chosen_ids[int(dataset_cfg.K)]) if log_query_loss else None
    holdout_indices = _sample_loo_indices(
        len(support_ids),
        num_loo_per_task=int(adapt_cfg.num_loo_per_task),
        rng=rng,
    )
    preload_support_batches = _as_bool(getattr(adapt_cfg, 'preload_support_batches_to_device', False))
    support_batch_device = device if preload_support_batches else torch.device('cpu')
    support_batch_generator = torch_generator
    if support_batch_device.type == 'cpu' and torch_generator is not None:
        support_batch_generator = torch.Generator()
        support_batch_generator.manual_seed(int(torch_generator.initial_seed()))

    shared_noise = None
    shared_timesteps = None
    if _as_bool(adapt_cfg.reuse_diffusion_noise):
        shared_noise = torch.randn(
            (1, int(dataset_cfg.H), int(action_dim)),
            device=support_batch_device,
            dtype=torch.float32,
            generator=support_batch_generator,
        )
        shared_timesteps = torch.randint(
            low=0,
            high=int(num_train_timesteps),
            size=(1,),
            device=support_batch_device,
            dtype=torch.long,
            generator=support_batch_generator,
        )

    task_spec = MAMLTaskSpec(
        vidx=int(vidx),
        support_episode_ids=tuple(support_ids),
        query_episode_id=int(query_episode_id) if query_episode_id is not None else int(support_ids[0]),
    )

    support_batches: List[Dict[str, Any]] = []
    prepare_pbar = None
    if int(adapt_cfg.inner_steps) > 0:
        prepare_pbar = tqdm(
            total=int(adapt_cfg.inner_steps),
            desc='MAML Prepare',
            leave=True,
            unit='step',
        )
    try:
        for step_idx in range(int(adapt_cfg.inner_steps)):
            support_batch = task_builder.build_support_batch_loo_cached(
                task_spec,
                holdout_indices=holdout_indices,
                rng=rng,
                noise=shared_noise if _as_bool(adapt_cfg.reuse_diffusion_noise) else None,
                timesteps=shared_timesteps if _as_bool(adapt_cfg.reuse_diffusion_noise) else None,
                load_rgb=use_rgb,
                load_mask_id=use_mask_id,
            )
            support_batch = _drop_mask_ids_if_disabled(support_batch, use_mask_id)
            if preload_support_batches:
                support_batch = _to_device_batch(support_batch, device)
            support_batches.append(support_batch)
            if prepare_pbar is not None:
                prepare_pbar.update(1)
                prepare_pbar.set_postfix(step=step_idx + 1)
    finally:
        if prepare_pbar is not None:
            prepare_pbar.close()

    rollout_support_ids = list(support_ids)
    outer_context_size = int(adapt_cfg.outer_context_size)
    if outer_context_size > 0 and outer_context_size < len(rollout_support_ids):
        keep = np.sort(rng.choice(len(rollout_support_ids), size=outer_context_size, replace=False))
        rollout_support_ids = [rollout_support_ids[int(idx)] for idx in keep.tolist()]

    rollout_support = task_builder.build_conditioning_from_support_ids(
        rng,
        vidx=int(vidx),
        support_ids=rollout_support_ids,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
        load_full_traj=True,
    )
    if rollout_support is None:
        raise RuntimeError(
            f'Failed to build rollout support conditioning for task={store.keys[vidx].task} '
            f'variation={store.keys[vidx].variation}.'
        )
    rollout_support = _unsqueeze_support_batch_dim(rollout_support)
    rollout_support['meta'] = {
        'task': store.keys[vidx].task,
        'variation': int(store.keys[vidx].variation),
        'support_episodes': list(support_ids),
        'rollout_context_episodes': list(rollout_support_ids),
        'query_episode': int(query_episode_id) if query_episode_id is not None else None,
        'holdout_indices': list(holdout_indices),
    }

    query_batch = None
    if log_query_loss:
        query_noise = shared_noise
        query_timesteps = shared_timesteps
        if query_noise is None:
            query_noise = torch.randn(
                (1, int(dataset_cfg.H), int(action_dim)),
                device=device,
                dtype=torch.float32,
                generator=torch_generator,
            )
        if query_timesteps is None:
            query_timesteps = torch.randint(
                low=0,
                high=int(num_train_timesteps),
                size=(1,),
                device=device,
                dtype=torch.long,
                generator=torch_generator,
            )
        query_task = MAMLTaskSpec(
            vidx=int(vidx),
            support_episode_ids=tuple(rollout_support_ids),
            query_episode_id=int(query_episode_id),
        )
        query_batch = task_builder.build_query_batch(
            query_task,
            rng=rng,
            noise=query_noise,
            timesteps=query_timesteps,
            load_rgb=use_rgb,
            load_mask_id=use_mask_id,
        )
        query_batch = _to_device_batch(query_batch, device)
        query_batch = _drop_mask_ids_if_disabled(query_batch, use_mask_id)

    return {
        'vidx': int(vidx),
        'support_ids': list(support_ids),
        'rollout_support_ids': list(rollout_support_ids),
        'query_episode_id': int(query_episode_id) if query_episode_id is not None else None,
        'holdout_indices': list(holdout_indices),
        'support_batches': support_batches,
        'query_batch': query_batch,
        'support_cond': rollout_support,
        'meta': rollout_support['meta'],
    }



def _adapt_fast_params_with_stats(
    loss_wrapper: PolicyLossWrapper,
    *,
    support_batches: Sequence[Dict[str, Any]],
    fast_names_wrapped: Sequence[str],
    device: torch.device,
    inner_lr: float,
    max_grad_norm: float,
    query_batch: Optional[Dict[str, Any]] = None,
    progress_desc: str = 'MAML Inner',
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    adapted_params = dict(loss_wrapper.named_parameters())
    buffers = dict(loss_wrapper.named_buffers())
    inner_losses: List[float] = []
    inner_grad_norms: List[float] = []
    query_losses: List[float] = []
    unused_fast_param_names: List[str] = []
    logged_unused_warning = False

    def _eval_query_loss(params: Dict[str, torch.Tensor]) -> float:
        if query_batch is None:
            return 0.0
        with torch.no_grad():
            loss = functional_call(loss_wrapper, (params, buffers), (query_batch,))
        return float(loss.detach().cpu().item())

    if query_batch is not None:
        query_losses.append(_eval_query_loss(adapted_params))

    pbar = None
    if len(support_batches) > 0:
        pbar = tqdm(
            total=len(support_batches),
            desc=progress_desc,
            leave=True,
            unit='step',
        )

    try:
        iterator = enumerate(support_batches, start=1)
        for step_idx, support_batch_cpu in iterator:
            support_batch = _to_device_batch(support_batch_cpu, device)
            support_loss = functional_call(loss_wrapper, (adapted_params, buffers), (support_batch,))
            fast_tensors = [adapted_params[name] for name in fast_names_wrapped]
            grads = torch.autograd.grad(
                support_loss,
                fast_tensors,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            if not logged_unused_warning:
                unused_fast_param_names = [
                    name for name, grad in zip(fast_names_wrapped, grads) if grad is None
                ]
                if unused_fast_param_names:
                    logging.warning(
                        'MAML eval selected %d fast params that were unused in the inner-loop forward. '
                        'They will be left unchanged. Examples: %s',
                        len(unused_fast_param_names),
                        unused_fast_param_names[:8],
                    )
                logged_unused_warning = True
            loss_value = float(support_loss.detach().cpu().item())
            inner_losses.append(loss_value)
            inner_grad_norms.append(_grad_list_global_norm(grads))
            grads = _clip_grads_in_list(list(grads), float(max_grad_norm))

            new_params = dict(adapted_params)
            for name, param, grad in zip(fast_names_wrapped, fast_tensors, grads):
                if grad is None:
                    continue
                new_params[name] = param - float(inner_lr) * grad
            adapted_params = new_params
            if query_batch is not None:
                query_losses.append(_eval_query_loss(adapted_params))

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    step=step_idx,
                    loss=f'{loss_value:.4g}',
                )
    finally:
        if pbar is not None:
            pbar.close()

    stats = {
        'inner_losses': inner_losses,
        'inner_grad_norms': inner_grad_norms,
        'avg_inner_loss': float(sum(inner_losses) / max(1, len(inner_losses))) if inner_losses else 0.0,
        'avg_inner_grad_norm': float(sum(inner_grad_norms) / max(1, len(inner_grad_norms)))
        if inner_grad_norms
        else 0.0,
        'query_losses': query_losses,
        'avg_query_loss': float(sum(query_losses) / max(1, len(query_losses))) if query_losses else 0.0,
        'unused_fast_params': unused_fast_param_names,
    }
    return adapted_params, stats



def _apply_maml_adaptation_in_place(
    *,
    policy: Policy,
    base_state_dict: Dict[str, torch.Tensor],
    support_package: Dict[str, Any],
    fast_names: Sequence[str],
    device: torch.device,
    maml_cfg: MAMLConfig,
) -> Dict[str, Any]:
    policy.load_state_dict(base_state_dict, strict=True)
    policy.to(device)
    loss_wrapper = PolicyLossWrapper(policy)

    was_training = policy.training
    policy.train()
    with torch.enable_grad():
        adapted_params, stats = _adapt_fast_params_with_stats(
            loss_wrapper,
            support_batches=support_package['support_batches'],
            query_batch=support_package.get('query_batch', None),
            fast_names_wrapped=_prefix_param_names(fast_names),
            device=device,
            inner_lr=float(maml_cfg.inner_lr),
            max_grad_norm=float(maml_cfg.max_grad_norm),
            progress_desc='MAML Inner GD',
        )
    copy_fast_params_into_policy(
        policy,
        adapted_params=adapted_params,
        fast_names=_prefix_param_names(fast_names),
    )
    if was_training:
        policy.train()
    else:
        policy.eval()

    stats.update(
        {
            'support_ids': list(support_package['support_ids']),
            'rollout_support_ids': list(support_package['rollout_support_ids']),
            'query_episode_id': support_package.get('query_episode_id', None),
            'holdout_indices': list(support_package['holdout_indices']),
        }
    )
    return stats


def _save_inner_loss_artifacts(
    *,
    inner_losses: Sequence[float],
    query_losses: Optional[Sequence[float]] = None,
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
            ax.plot(xs, ys, marker='o', linewidth=1.5, label='support LOO')
        if has_query:
            ax.plot(xs_query, ys_query, marker='s', linewidth=1.5, label='held-out query')
        ax.set_xlabel('Inner Step')
        ax.set_ylabel('Diffusion Loss')
        ax.set_yscale('log')
        ax.set_title('MAML Eval Inner-Loop Loss')
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
        logging.warning('Failed to save MAML inner-loss plot to %s: %s', plot_path, exc)
        plot_path = None

    return {
        'inner_loss_json_path': str(losses_path),
        'inner_loss_plot_path': str(plot_path) if plot_path is not None else '',
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
    maml_cfg = _resolve_maml_cfg_from_checkpoint(ckpt, data_k=int(dataset_cfg.K))
    use_mask_id = _conditioning_use_mask_id_from_eval_and_checkpoint(cfg, model_cfg)
    ignore_demos = _ignore_demos_from_model_cfg(model_cfg)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    support_source = _support_source_from_eval(cfg)
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)

    if ignore_demos:
        raise ValueError(
            'MAML eval requires a checkpoint whose model conditions on support demos (ignore_demos=False).'
        )

    model = build_policy(
        model_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fast_names = _resolve_maml_fast_param_names(model, ckpt, maml_cfg)
    fast_param_count = _count_params_by_name(model, fast_names)

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
    resolved_payload['checkpoint_maml'] = {
        'inner_steps': int(maml_cfg.inner_steps),
        'inner_lr': float(maml_cfg.inner_lr),
        'outer_lr': float(maml_cfg.outer_lr),
        'weight_decay': float(maml_cfg.weight_decay),
        'max_grad_norm': float(maml_cfg.max_grad_norm),
        'last_frac_fast': float(maml_cfg.last_frac_fast),
        'include_decoder_mlp_fast': bool(getattr(maml_cfg, 'include_decoder_mlp_fast', True)),
        'include_ada_fast': bool(maml_cfg.include_ada_fast),
        'include_final_norm_fast': bool(maml_cfg.include_final_norm_fast),
        'include_input_projections_fast': bool(getattr(maml_cfg, 'include_input_projections_fast', False)),
        'include_output_head_fast': bool(getattr(maml_cfg, 'include_output_head_fast', False)),
        'include_diffusion_conditioning_fast': bool(
            getattr(maml_cfg, 'include_diffusion_conditioning_fast', False)
        ),
        'num_loo_per_task': int(maml_cfg.num_loo_per_task),
        'outer_context_size': int(maml_cfg.outer_context_size),
        'reuse_diffusion_noise': bool(maml_cfg.reuse_diffusion_noise),
    }
    resolved_payload['resolved'] = {
        'checkpoint_path': str(checkpoint_path),
        'run_name': run_name,
        'run_dir': str(run_dir),
        'fast_param_names': list(fast_names),
        'fast_param_tensors': len(fast_names),
        'fast_param_count': int(fast_param_count),
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
        'MAML cfg: inner_steps=%d | inner_lr=%.3e | max_grad_norm=%.3f | outer_context_size=%d | '
        'num_loo_per_task=%d | reuse_diffusion_noise=%s | preload_support_batches_to_device=%s | log_query_loss=%s',
        int(maml_cfg.inner_steps),
        float(maml_cfg.inner_lr),
        float(maml_cfg.max_grad_norm),
        int(maml_cfg.outer_context_size),
        int(maml_cfg.num_loo_per_task),
        str(bool(maml_cfg.reuse_diffusion_noise)),
        str(_as_bool(getattr(cfg.maml_eval, 'preload_support_batches_to_device', False))),
        str(_as_bool(getattr(cfg.maml_eval, 'log_query_loss', False))),
    )
    logging.info(
        'MAML fast params: tensors=%d | params=%s | examples=%s',
        len(fast_names),
        f'{fast_param_count:,}',
        fast_names[:8],
    )

    env = None
    task_env = None
    support_store: Optional[VariationStore] = None
    results: List[Dict[str, Any]] = []
    support_cache_root: Optional[Path] = None
    current_support_package: Optional[Dict[str, Any]] = None
    current_maml_stats: Optional[Dict[str, Any]] = None

    try:
        if variation < 0:
            raise ValueError('Cached MAML support conditioning requires cfg.task.variation >= 0.')
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
            num_tries_per_item=int(getattr(cfg.maml_eval, 'num_tries_per_item', 100)),
        )

        rng = np.random.default_rng(seed + 17)
        for ep in range(int(cfg.task.num_eval_episodes)):
            regen = _as_bool(cfg.conditioning.regenerate_demos_each_episode)
            if current_support_package is None or regen:
                torch_seed = seed + 100_003 + ep
                torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
                torch_gen.manual_seed(torch_seed)
                current_support_package = _build_cached_maml_package(
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
                    adapt_cfg=ConfigDict(
                        {
                            'inner_steps': int(maml_cfg.inner_steps),
                            'num_loo_per_task': int(maml_cfg.num_loo_per_task),
                            'outer_context_size': int(maml_cfg.outer_context_size),
                            'reuse_diffusion_noise': bool(maml_cfg.reuse_diffusion_noise),
                            'preload_support_batches_to_device': _as_bool(
                                getattr(cfg.maml_eval, 'preload_support_batches_to_device', False)
                            ),
                            'log_query_loss': _as_bool(getattr(cfg.maml_eval, 'log_query_loss', False)),
                        }
                    ),
                )
                current_maml_stats = _apply_maml_adaptation_in_place(
                    policy=model,
                    base_state_dict=state_dict,
                    support_package=current_support_package,
                    fast_names=fast_names,
                    device=device,
                    maml_cfg=maml_cfg,
                )
                current_maml_stats.update(
                    _save_inner_loss_artifacts(
                        inner_losses=current_maml_stats.get('inner_losses', []),
                        query_losses=current_maml_stats.get('query_losses', None),
                        run_dir=run_dir,
                        stem=f'maml_episode_{ep:04d}',
                    )
                )
                model.eval()
                if current_maml_stats.get('query_losses', None):
                    logging.info(
                        'MAML adaptation ready for episode %d | support_episodes=%s | rollout_context=%s | '
                        'query_episode=%s | holdout_indices=%s | avg_inner_loss=%.6f | '
                        'avg_query_loss=%.6f | avg_inner_grad_norm=%.6f',
                        ep,
                        current_maml_stats['support_ids'],
                        current_maml_stats['rollout_support_ids'],
                        current_maml_stats.get('query_episode_id', None),
                        current_maml_stats['holdout_indices'],
                        float(current_maml_stats['avg_inner_loss']),
                        float(current_maml_stats['avg_query_loss']),
                        float(current_maml_stats['avg_inner_grad_norm']),
                    )
                else:
                    logging.info(
                        'MAML adaptation ready for episode %d | support_episodes=%s | rollout_context=%s | '
                        'holdout_indices=%s | avg_inner_loss=%.6f | avg_inner_grad_norm=%.6f',
                        ep,
                        current_maml_stats['support_ids'],
                        current_maml_stats['rollout_support_ids'],
                        current_maml_stats['holdout_indices'],
                        float(current_maml_stats['avg_inner_loss']),
                        float(current_maml_stats['avg_inner_grad_norm']),
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
            )
            if current_maml_stats is not None:
                res['maml'] = current_maml_stats
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
            'fast_param_names': list(fast_names),
            'fast_param_tensors': len(fast_names),
            'fast_param_count': int(fast_param_count),
            'resolved_dataset': {
                'K': int(dataset_cfg.K),
                'L': int(dataset_cfg.L),
                'T_obs': int(dataset_cfg.T_obs),
                'H': int(dataset_cfg.H),
                'stride': int(dataset_cfg.stride),
            },
            'resolved_maml': {
                'outer_context_size': int(maml_cfg.outer_context_size),
                'inner_steps': int(maml_cfg.inner_steps),
                'inner_lr': float(maml_cfg.inner_lr),
                'max_grad_norm': float(maml_cfg.max_grad_norm),
                'last_frac_fast': float(maml_cfg.last_frac_fast),
                'include_decoder_mlp_fast': bool(getattr(maml_cfg, 'include_decoder_mlp_fast', True)),
                'include_ada_fast': bool(maml_cfg.include_ada_fast),
                'include_final_norm_fast': bool(maml_cfg.include_final_norm_fast),
                'include_input_projections_fast': bool(getattr(maml_cfg, 'include_input_projections_fast', False)),
                'include_output_head_fast': bool(getattr(maml_cfg, 'include_output_head_fast', False)),
                'include_diffusion_conditioning_fast': bool(
                    getattr(maml_cfg, 'include_diffusion_conditioning_fast', False)
                ),
                'num_loo_per_task': int(maml_cfg.num_loo_per_task),
                'reuse_diffusion_noise': bool(maml_cfg.reuse_diffusion_noise),
                'preload_support_batches_to_device': _as_bool(
                    getattr(cfg.maml_eval, 'preload_support_batches_to_device', False)
                ),
                'log_query_loss': _as_bool(getattr(cfg.maml_eval, 'log_query_loss', False)),
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
