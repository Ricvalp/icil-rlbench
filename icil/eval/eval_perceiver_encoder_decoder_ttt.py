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
import torch.nn.functional as F
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
from icil.models.policies.config_utils import inherit_missing_encoder_attention_backend
from icil.models.maml import MAMLTaskBuilder, MAMLTaskSpec, PolicyLossWrapper, copy_fast_params_into_policy

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/eval_perceiver_encoder_decoder_ttt.py',
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
        'perceiver_demo_query_supernode_v2': {},
        'traj_perceiver': {},
        'traj_perceiver_v2': {},
        'traj_supernode_perceiver_v2': {},
    }
    for k, v in model_from_ckpt.items():
        if k in (
            'policy',
            'conv3d_demo_query',
            'perceiver_demo_query',
            'perceiver_demo_query_v2',
            'perceiver_demo_query_supernode_v2',
            'traj_conv3d',
            'traj_perceiver',
            'traj_perceiver_v2',
            'traj_supernode_perceiver_v2',
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
    model_from_ckpt = inherit_missing_encoder_attention_backend(model_from_ckpt)
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
    if str(model_cfg.encoder_name) == 'perceiver_demo_query_supernode_v2':
        return bool(model_cfg.perceiver_demo_query_supernode_v2.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_perceiver_v2':
        return bool(model_cfg.traj_perceiver_v2.use_mask_id)
    if str(model_cfg.encoder_name) == 'traj_supernode_perceiver_v2':
        return bool(model_cfg.traj_supernode_perceiver_v2.use_mask_id)
    return _as_bool(getattr(cfg.conditioning, 'use_mask_id', True))



def _ignore_demos_from_model_cfg(model_cfg: PolicyBuilderConfig) -> bool:
    if str(model_cfg.encoder_name) == 'conv3d_demo_query':
        return bool(model_cfg.conv3d_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query':
        return bool(model_cfg.perceiver_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query_v2':
        return bool(model_cfg.perceiver_demo_query_v2.ignore_demos)
    if str(model_cfg.encoder_name) == 'perceiver_demo_query_supernode_v2':
        return bool(model_cfg.perceiver_demo_query_supernode_v2.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_conv3d':
        return bool(model_cfg.traj_conv3d.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_perceiver':
        return bool(model_cfg.traj_perceiver.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_perceiver_v2':
        return bool(model_cfg.traj_perceiver_v2.ignore_demos)
    if str(model_cfg.encoder_name) == 'traj_supernode_perceiver_v2':
        return bool(model_cfg.traj_supernode_perceiver_v2.ignore_demos)
    return False



def _resolve_data_k(cfg: ConfigDict, ckpt: Dict[str, Any]) -> int:
    configured_k = int(cfg.dataset.K)
    if configured_k > 0:
        return configured_k
    ckpt_dataset = {}
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}
    if isinstance(ckpt_dataset, dict) and int(ckpt_dataset.get('K', 0)) > 0:
        return int(ckpt_dataset['K']) + 1
    raise ValueError(
        'cfg.dataset.K=0 requires checkpoint["config"]["dataset"]["K"] > 0 so TTT can use K_pretrain + 1.'
    )



def _resolve_outer_context_size(cfg: ConfigDict, *, data_k: int, ckpt: Dict[str, Any]) -> int:
    configured = int(cfg.ttt.outer_context_size)
    if configured > 0:
        resolved = configured
    else:
        resolved = 0
        if isinstance(ckpt.get('config', None), dict):
            dataset_cfg = ckpt['config'].get('dataset', {}) or {}
            if isinstance(dataset_cfg, dict) and int(dataset_cfg.get('K', 0)) > 0:
                resolved = int(dataset_cfg['K'])
        if resolved <= 0:
            resolved = int(data_k)
    if resolved <= 0:
        raise ValueError('Resolved outer_context_size must be positive.')
    if resolved > data_k:
        raise ValueError(f'outer_context_size={resolved} exceeds data.K={data_k}.')
    return resolved



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
            'TTT eval currently supports cfg.conditioning.support_source="cache" only, '
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
        raise ValueError('Inner-loop support batches must be non-empty.')

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



def _denoiser_ada_prefixes(block_idx: int) -> List[str]:
    return [
        f'denoiser.{block_idx}.adaln1.to_scale_shift.',
        f'denoiser.{block_idx}.adaln2.to_scale_shift.',
        f'denoiser.{block_idx}.adaln3.to_scale_shift.',
        f'denoiser.{block_idx}.adaln_q.to_scale_shift.',
        f'denoiser.{block_idx}.adaln_s.to_scale_shift.',
    ]



def _select_ttt_fast_param_names(model: Policy, ttt_cfg: ConfigDict) -> List[str]:
    if not hasattr(model, 'denoiser'):
        raise AttributeError('Policy has no attribute denoiser.')

    prefixes: List[str] = []
    n_blocks = len(model.denoiser)
    if n_blocks <= 0:
        raise ValueError('Policy denoiser is empty.')

    last_frac = float(ttt_cfg.last_frac_fast)
    if last_frac <= 0.0:
        num_blocks = 1
    else:
        num_blocks = max(1, int(round(float(n_blocks) * last_frac)))
        num_blocks = min(num_blocks, n_blocks)
    start_idx = n_blocks - num_blocks

    for block_idx in range(start_idx, n_blocks):
        if _as_bool(ttt_cfg.include_decoder_mlp_fast):
            prefixes.append(f'denoiser.{block_idx}.mlp.')
        if _as_bool(ttt_cfg.include_ada_fast):
            prefixes.extend(_denoiser_ada_prefixes(block_idx))
        if _as_bool(ttt_cfg.include_decoder_self_attention_fast):
            prefixes.append(f'denoiser.{block_idx}.self_attn.')
        if _as_bool(ttt_cfg.include_decoder_cross_attention_fast):
            prefixes.append(f'denoiser.{block_idx}.cross_attn.')
            prefixes.append(f'denoiser.{block_idx}.cross_attn_q.')
            prefixes.append(f'denoiser.{block_idx}.cross_attn_s.')

    if _as_bool(ttt_cfg.include_encoder_fast):
        prefixes.append('context_encoder.')
    if _as_bool(ttt_cfg.include_input_projections_fast):
        prefixes.append('action_in.')
    if _as_bool(ttt_cfg.include_output_head_fast):
        prefixes.append('action_out.')
    if _as_bool(ttt_cfg.include_diffusion_conditioning_fast):
        prefixes.append('t_mlp.')

    if _as_bool(ttt_cfg.include_final_norm_fast):
        logging.info('cfg.ttt.include_final_norm_fast=True, but current Policy has no final norm block; ignoring.')

    if not prefixes:
        raise ValueError('No TTT fast-parameter groups were enabled in cfg.ttt.')

    param_names = [name for name, _ in model.named_parameters()]
    selected = sorted({name for name in param_names if any(name.startswith(prefix) for prefix in prefixes)})
    if not selected:
        raise RuntimeError('No parameters matched the requested TTT fast-parameter selection.')
    return selected



def _count_params_by_name(model: Policy, names: Sequence[str]) -> int:
    name_set = set(names)
    return sum(int(param.numel()) for name, param in model.named_parameters() if name in name_set)



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



def _sample_loo_indices(K: int, *, num_loo_per_task: int, rng: np.random.Generator) -> List[int]:
    return _sample_balanced_indices(K, count=num_loo_per_task, rng=rng)


def _resolve_num_support_batches_loo(ttt_cfg: ConfigDict) -> int:
    inner_steps = int(getattr(ttt_cfg, 'inner_steps', 0))
    configured = int(getattr(ttt_cfg, 'num_support_batches_loo', 0))
    if inner_steps <= 0:
        return 0
    if configured <= 0:
        return inner_steps
    return min(configured, inner_steps)



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



def _build_support_batch_loo(
    task_builder: MAMLTaskBuilder,
    task: MAMLTaskSpec,
    *,
    holdout_indices: Sequence[int],
    rng: np.random.Generator,
    noise: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    load_rgb: bool = True,
    load_mask_id: bool = True,
) -> Dict[str, Any]:
    if not holdout_indices:
        raise ValueError('holdout_indices must be non-empty.')

    support_ids = list(task.support_episode_ids)
    holdout_t0s: List[Optional[int]] = [None] * len(holdout_indices)
    positions_by_holdout: Dict[int, List[int]] = {}
    for pos, holdout_idx in enumerate(holdout_indices):
        holdout_idx_i = int(holdout_idx)
        if holdout_idx_i < 0 or holdout_idx_i >= len(support_ids):
            raise IndexError(f'holdout_idx={holdout_idx_i} out of range for K={len(support_ids)}')
        positions_by_holdout.setdefault(holdout_idx_i, []).append(pos)

    for holdout_idx, positions in positions_by_holdout.items():
        heldout_episode_id = int(support_ids[int(holdout_idx)])
        sampled_t0s = _sample_query_t0s_for_episode(
            task_builder,
            vidx=int(task.vidx),
            episode_id=heldout_episode_id,
            count=len(positions),
            rng=rng,
        )
        for pos, t0 in zip(positions, sampled_t0s):
            holdout_t0s[pos] = int(t0)

    samples: List[Dict[str, Any]] = []
    for holdout_idx, t0 in zip(holdout_indices, holdout_t0s):
        holdout_idx_i = int(holdout_idx)
        heldout_episode_id = int(support_ids[holdout_idx_i])
        kept_support_ids = [
            int(ep_id) for idx, ep_id in enumerate(support_ids) if idx != holdout_idx_i
        ]
        support = task_builder.build_conditioning_from_support_ids(
            rng,
            vidx=int(task.vidx),
            support_ids=kept_support_ids,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
            load_full_traj=True,
        )
        if support is None:
            raise RuntimeError('Failed to build support conditioning for inner-loop adaptation.')
        if t0 is None:
            raise RuntimeError(
                f'Internal error while sampling query window for heldout episode {heldout_episode_id}.'
            )
        query = _build_query_sample_at_t0(
            task_builder,
            vidx=int(task.vidx),
            episode_id=heldout_episode_id,
            t0=int(t0),
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        query['meta'].update(
            {
                'holdout_index': holdout_idx_i,
                'heldout_support_episode': heldout_episode_id,
                'support_episodes': kept_support_ids,
                'task_query_episode': int(task.query_episode_id),
            }
        )
        samples.append({**support, **query})

    batch = task_builder._stack_samples(samples)
    task_builder.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
    return batch



def _build_query_batch_at_t0s(
    task_builder: MAMLTaskBuilder,
    task: MAMLTaskSpec,
    *,
    support_ids: Sequence[int],
    count: int,
    rng: np.random.Generator,
    noise: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    load_rgb: bool = True,
    load_mask_id: bool = True,
) -> Dict[str, Any]:
    if count < 1:
        raise ValueError(f'count must be >= 1, got {count}.')

    support = task_builder.build_conditioning_from_support_ids(
        rng,
        vidx=int(task.vidx),
        support_ids=[int(ep_id) for ep_id in support_ids],
        load_rgb=load_rgb,
        load_mask_id=load_mask_id,
        load_full_traj=True,
    )
    if support is None:
        raise RuntimeError('Failed to build extra-query support conditioning.')

    sampled_t0s = _sample_query_t0s_for_episode(
        task_builder,
        vidx=int(task.vidx),
        episode_id=int(task.query_episode_id),
        count=int(count),
        rng=rng,
    )
    samples: List[Dict[str, Any]] = []
    for t0 in sampled_t0s:
        query = _build_query_sample_at_t0(
            task_builder,
            vidx=int(task.vidx),
            episode_id=int(task.query_episode_id),
            t0=int(t0),
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        query['meta'].update(
            {
                'support_episodes': [int(ep_id) for ep_id in support_ids],
                'task_query_episode': int(task.query_episode_id),
            }
        )
        samples.append({**support, **query})

    batch = task_builder._stack_samples(samples)
    task_builder.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
    return batch


@torch.no_grad()
def _sample_actions_from_batch(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    use_mask_id: bool,
    inference_steps: int,
    eta: float,
) -> torch.Tensor:
    was_training = policy.training
    policy.eval()
    try:
        return policy.sample_actions(
            cond_xyz=batch.get('cond_xyz', None),
            cond_state=batch.get('cond_state', None),
            cond_traj=batch.get('cond_traj', None),
            cond_traj_mask=batch.get('cond_traj_mask', None),
            query_xyz=batch['query_xyz'],
            query_state=batch['query_state'],
            action_horizon=int(batch['target_action'].shape[1]),
            cond_rgb=batch.get('cond_rgb', None),
            query_rgb=batch.get('query_rgb', None),
            cond_mask_id=(batch.get('cond_mask_id', None) if use_mask_id else None),
            query_mask_id=(batch.get('query_mask_id', None) if use_mask_id else None),
            cond_valid=batch.get('cond_valid', None),
            query_valid=batch.get('query_valid', None),
            inference_steps=(int(inference_steps) if int(inference_steps) > 0 else None),
            eta=float(eta),
        )
    finally:
        if was_training:
            policy.train()


def _query_sample_mse_from_batch(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    use_mask_id: bool,
    inference_steps: int,
    eta: float,
) -> float:
    pred = _sample_actions_from_batch(
        policy,
        batch,
        use_mask_id=use_mask_id,
        inference_steps=int(inference_steps),
        eta=float(eta),
    )
    return float(
        F.mse_loss(
            pred.detach().float(),
            batch['target_action'].detach().float(),
        ).detach().cpu().item()
    )



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



def _build_cached_ttt_package(
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
    ttt_cfg: ConfigDict,
) -> Dict[str, Any]:
    vidx = _select_support_vidx(store, variation=variation, rng=rng)
    episode_ids = store.list_episode_ids(vidx)
    log_query_loss = _as_bool(getattr(ttt_cfg, 'log_query_loss', False))
    log_query_sample_mse = _as_bool(getattr(ttt_cfg, 'log_query_sample_mse', False))
    needs_query_batch = log_query_loss or log_query_sample_mse
    required_episodes = int(dataset_cfg.K) + (1 if needs_query_batch else 0)
    if episode_ids.shape[0] < required_episodes:
        raise RuntimeError(
            f'Need at least {required_episodes} cached episodes, got {episode_ids.shape[0]} '
            f'for task={store.keys[vidx].task} variation={store.keys[vidx].variation}. '
            f'TTT support K={dataset_cfg.K}, log_query_loss={log_query_loss}, '
            f'log_query_sample_mse={log_query_sample_mse}.'
        )

    chosen_ids_np = rng.choice(episode_ids, size=required_episodes, replace=False)
    chosen_ids = [int(eid) for eid in np.asarray(chosen_ids_np).tolist()]
    support_ids = chosen_ids[: int(dataset_cfg.K)]
    query_episode_id = int(chosen_ids[int(dataset_cfg.K)]) if needs_query_batch else None
    holdout_indices = _sample_loo_indices(
        len(support_ids),
        num_loo_per_task=int(ttt_cfg.num_loo_per_task),
        rng=rng,
    )
    preload_support_batches = _as_bool(getattr(ttt_cfg, 'preload_support_batches_to_device', False))
    num_support_batches = _resolve_num_support_batches_loo(ttt_cfg)
    support_batch_device = device if preload_support_batches else torch.device('cpu')
    support_batch_generator = torch_generator
    if support_batch_device.type == 'cpu' and torch_generator is not None:
        support_batch_generator = torch.Generator()
        support_batch_generator.manual_seed(int(torch_generator.initial_seed()))

    shared_noise = None
    shared_timesteps = None
    if _as_bool(ttt_cfg.reuse_diffusion_noise):
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
    if num_support_batches > 0:
        prepare_pbar = tqdm(
            total=num_support_batches,
            desc='TTT Prepare',
            leave=True,
            unit='step',
        )
    try:
        for batch_idx in range(num_support_batches):
            support_batch = _build_support_batch_loo(
                task_builder,
                task_spec,
                holdout_indices=holdout_indices,
                rng=rng,
                noise=shared_noise if _as_bool(ttt_cfg.reuse_diffusion_noise) else None,
                timesteps=shared_timesteps if _as_bool(ttt_cfg.reuse_diffusion_noise) else None,
                load_rgb=use_rgb,
                load_mask_id=use_mask_id,
            )
            support_batch = _drop_mask_ids_if_disabled(support_batch, use_mask_id)
            if preload_support_batches:
                support_batch = _to_device_batch(support_batch, device)
            support_batches.append(support_batch)
            if prepare_pbar is not None:
                prepare_pbar.update(1)
                prepare_pbar.set_postfix(batch=batch_idx + 1)
    finally:
        if prepare_pbar is not None:
            prepare_pbar.close()

    rollout_support_ids = list(support_ids)
    outer_context_size = int(ttt_cfg.outer_context_size)
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
    if needs_query_batch:
        query_loss_count = int(getattr(ttt_cfg, 'num_query_loss_samples', 1))
        query_loss_count = max(1, query_loss_count)
        query_noise = shared_noise
        query_timesteps = shared_timesteps
        if query_noise is None:
            query_noise = torch.randn(
                (query_loss_count, int(dataset_cfg.H), int(action_dim)),
                device=device,
                dtype=torch.float32,
                generator=torch_generator,
            )
        if query_timesteps is None:
            query_timesteps = torch.randint(
                low=0,
                high=int(num_train_timesteps),
                size=(query_loss_count,),
                device=device,
                dtype=torch.long,
                generator=torch_generator,
            )
        query_task = MAMLTaskSpec(
            vidx=int(vidx),
            support_episode_ids=tuple(rollout_support_ids),
            query_episode_id=int(query_episode_id),
        )
        query_batch = _build_query_batch_at_t0s(
            task_builder,
            query_task,
            support_ids=rollout_support_ids,
            count=query_loss_count,
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
        'num_support_batches': int(num_support_batches),
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
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    grad_accum_steps: int = 1,
    query_batch: Optional[Dict[str, Any]] = None,
    log_query_loss: bool = False,
    log_query_sample_mse: bool = False,
    use_mask_id: bool = False,
    sample_mse_inference_steps: int = 0,
    sample_mse_eta: float = 0.0,
    progress_desc: str = 'TTT Inner',
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if int(grad_accum_steps) < 1:
        raise ValueError(f'grad_accum_steps must be >= 1, got {grad_accum_steps}.')
    if int(inner_steps) < 0:
        raise ValueError(f'inner_steps must be >= 0, got {inner_steps}.')

    adapted_params = dict(loss_wrapper.named_parameters())
    buffers = dict(loss_wrapper.named_buffers())
    inner_losses: List[float] = []
    inner_grad_norms: List[float] = []
    query_losses: List[float] = []
    query_sample_mses: List[float] = []
    unused_fast_param_names: List[str] = []
    logged_unused_warning = False

    def _eval_query_loss(params: Dict[str, torch.Tensor]) -> float:
        if query_batch is None:
            return 0.0
        with torch.no_grad():
            loss = functional_call(loss_wrapper, (params, buffers), (query_batch,))
        return float(loss.detach().cpu().item())

    def _eval_query_sample_mse(params: Dict[str, torch.Tensor]) -> float:
        if query_batch is None:
            return 0.0
        copy_fast_params_into_policy(
            loss_wrapper.policy,
            adapted_params=params,
            fast_names=fast_names_wrapped,
        )
        return _query_sample_mse_from_batch(
            loss_wrapper.policy,
            query_batch,
            use_mask_id=use_mask_id,
            inference_steps=int(sample_mse_inference_steps),
            eta=float(sample_mse_eta),
        )

    if query_batch is not None and log_query_loss:
        query_losses.append(_eval_query_loss(adapted_params))
    if query_batch is not None and log_query_sample_mse:
        query_sample_mses.append(_eval_query_sample_mse(adapted_params))

    pbar = None
    if int(inner_steps) > 0:
        pbar = tqdm(
            total=int(inner_steps),
            desc=progress_desc,
            leave=True,
            unit='step',
        )

    try:
        if int(inner_steps) > 0 and len(support_batches) < 1:
            raise ValueError('inner_steps > 0 requires at least one prepared support batch.')
        for step_idx in range(1, int(inner_steps) + 1):
            support_batch_cpu = support_batches[(step_idx - 1) % len(support_batches)]
            fast_tensors = [adapted_params[name] for name in fast_names_wrapped]
            if int(grad_accum_steps) == 1:
                support_batch = _to_device_batch(support_batch_cpu, device)
                support_loss = functional_call(loss_wrapper, (adapted_params, buffers), (support_batch,))
                grads = torch.autograd.grad(
                    support_loss,
                    fast_tensors,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
                loss_value = float(support_loss.detach().cpu().item())
            else:
                grads_accum: List[Optional[torch.Tensor]] = [None for _ in fast_tensors]
                loss_value = 0.0
                for micro_batch_cpu, weight in _iter_microbatches(
                    support_batch_cpu,
                    int(grad_accum_steps),
                ):
                    micro_batch = _to_device_batch(micro_batch_cpu, device)
                    micro_loss = functional_call(loss_wrapper, (adapted_params, buffers), (micro_batch,))
                    micro_grads = torch.autograd.grad(
                        micro_loss * float(weight),
                        fast_tensors,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    loss_value += float(micro_loss.detach().cpu().item()) * float(weight)
                    for grad_idx, grad in enumerate(micro_grads):
                        if grad is None:
                            continue
                        if grads_accum[grad_idx] is None:
                            grads_accum[grad_idx] = grad
                        else:
                            grads_accum[grad_idx] = grads_accum[grad_idx] + grad
                grads = tuple(grads_accum)
            if not logged_unused_warning:
                unused_fast_param_names = [
                    name for name, grad in zip(fast_names_wrapped, grads) if grad is None
                ]
                if unused_fast_param_names:
                    logging.warning(
                        'TTT selected %d fast params that were unused in the inner-loop forward. '
                        'They will be left unchanged. Examples: %s',
                        len(unused_fast_param_names),
                        unused_fast_param_names[:8],
                    )
                logged_unused_warning = True
            inner_losses.append(loss_value)
            inner_grad_norms.append(_grad_list_global_norm(grads))
            grads = _clip_grads_in_list(list(grads), float(max_grad_norm))

            new_params = dict(adapted_params)
            for name, param, grad in zip(fast_names_wrapped, fast_tensors, grads):
                if grad is None:
                    continue
                new_params[name] = param - float(inner_lr) * grad
            adapted_params = new_params
            if query_batch is not None and log_query_loss:
                query_losses.append(_eval_query_loss(adapted_params))
            if query_batch is not None and log_query_sample_mse:
                query_sample_mses.append(_eval_query_sample_mse(adapted_params))

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
        'query_sample_mses': query_sample_mses,
        'avg_query_sample_mse': float(sum(query_sample_mses) / max(1, len(query_sample_mses)))
        if query_sample_mses
        else 0.0,
        'grad_accum_steps': int(grad_accum_steps),
    }
    return adapted_params, stats



def _apply_ttt_adaptation_in_place(
    *,
    policy: Policy,
    base_state_dict: Dict[str, torch.Tensor],
    support_package: Dict[str, Any],
    fast_names: Sequence[str],
    device: torch.device,
    ttt_cfg: ConfigDict,
    use_mask_id: bool,
    sample_mse_inference_steps: int,
    sample_mse_eta: float,
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
            inner_steps=int(ttt_cfg.inner_steps),
            inner_lr=float(ttt_cfg.inner_lr),
            max_grad_norm=float(ttt_cfg.max_grad_norm),
            grad_accum_steps=int(getattr(ttt_cfg, 'grad_accum_steps', 1)),
            log_query_loss=_as_bool(getattr(ttt_cfg, 'log_query_loss', False)),
            log_query_sample_mse=_as_bool(getattr(ttt_cfg, 'log_query_sample_mse', False)),
            use_mask_id=use_mask_id,
            sample_mse_inference_steps=int(sample_mse_inference_steps),
            sample_mse_eta=float(sample_mse_eta),
            progress_desc='TTT Inner GD',
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
            'num_support_batches': int(support_package.get('num_support_batches', len(support_package['support_batches']))),
        }
    )
    return stats


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
            ax.plot(xs, ys, marker='o', linewidth=1.5, label='support LOO')
        if has_query:
            ax.plot(xs_query, ys_query, marker='s', linewidth=1.5, label='held-out query')
        ax.set_xlabel('Inner Step')
        ax.set_ylabel('Diffusion Loss')
        ax.set_yscale('log')
        ax.set_title('TTT Inner-Loop Loss')
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
        logging.warning('Failed to save TTT inner-loss plot to %s: %s', plot_path, exc)
        plot_path = None

    sample_mse_plot_path = None
    if query_sample_mses is not None and len(query_sample_mses) > 0:
        sample_mse_plot_path = run_dir / f'{stem}.query_sample_mse.png'
        try:
            import matplotlib.pyplot as plt

            xs_mse = np.arange(0, len(query_sample_mses), dtype=np.int64)
            ys_mse = np.asarray(query_sample_mses, dtype=np.float64)
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            ax.plot(xs_mse, ys_mse, marker='o', linewidth=1.5, label='held-out query sample MSE')
            ax.set_xlabel('Inner Step')
            ax.set_ylabel('Sampled Action MSE')
            ax.set_yscale('log')
            ax.set_title('TTT Held-Out Query Sample MSE')
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
            logging.warning('Failed to save TTT query sample-MSE plot to %s: %s', sample_mse_plot_path, exc)
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
    resolved_outer_context_size = _resolve_outer_context_size(cfg, data_k=int(dataset_cfg.K), ckpt=ckpt)
    use_mask_id = _conditioning_use_mask_id_from_eval_and_checkpoint(cfg, model_cfg)
    ignore_demos = _ignore_demos_from_model_cfg(model_cfg)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    support_source = _support_source_from_eval(cfg)
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)

    if ignore_demos:
        raise ValueError('TTT eval requires a checkpoint whose model conditions on support demos (ignore_demos=False).')

    model = build_policy(
        model_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fast_names = _select_ttt_fast_param_names(model, cfg.ttt)
    fast_param_count = _count_params_by_name(model, fast_names)
    grad_accum_steps = int(getattr(cfg.ttt, 'grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f'cfg.ttt.grad_accum_steps must be >= 1, got {grad_accum_steps}.')

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
    resolved_payload['ttt']['outer_context_size'] = int(resolved_outer_context_size)
    resolved_payload['ttt']['grad_accum_steps'] = int(grad_accum_steps)
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
        'TTT cfg: inner_steps=%d | inner_lr=%.3e | max_grad_norm=%.3f | outer_context_size=%d | '
        'num_loo_per_task=%d | grad_accum_steps=%d | num_support_batches_loo=%d | reuse_diffusion_noise=%s | '
        'preload_support_batches_to_device=%s | log_query_loss=%s | num_query_loss_samples=%d | '
        'log_query_sample_mse=%s',
        int(cfg.ttt.inner_steps),
        float(cfg.ttt.inner_lr),
        float(cfg.ttt.max_grad_norm),
        int(resolved_outer_context_size),
        int(cfg.ttt.num_loo_per_task),
        int(grad_accum_steps),
        int(_resolve_num_support_batches_loo(cfg.ttt)),
        str(_as_bool(cfg.ttt.reuse_diffusion_noise)),
        str(_as_bool(getattr(cfg.ttt, 'preload_support_batches_to_device', False))),
        str(_as_bool(getattr(cfg.ttt, 'log_query_loss', False))),
        int(getattr(cfg.ttt, 'num_query_loss_samples', 1)),
        str(_as_bool(getattr(cfg.ttt, 'log_query_sample_mse', False))),
    )
    logging.info(
        'TTT fast params: tensors=%d | params=%s | examples=%s',
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
    current_ttt_stats: Optional[Dict[str, Any]] = None

    try:
        if variation < 0:
            raise ValueError('Cached TTT support conditioning requires cfg.task.variation >= 0.')
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
            num_tries_per_item=int(getattr(cfg.ttt, 'num_tries_per_item', 100)),
        )

        rng = np.random.default_rng(seed + 17)
        for ep in range(int(cfg.task.num_eval_episodes)):
            regen = _as_bool(cfg.conditioning.regenerate_demos_each_episode)
            if current_support_package is None or regen:
                torch_seed = seed + 100_003 + ep
                torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
                torch_gen.manual_seed(torch_seed)
                current_support_package = _build_cached_ttt_package(
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
                    ttt_cfg=ConfigDict(
                        {
                            'inner_steps': int(cfg.ttt.inner_steps),
                            'num_loo_per_task': int(cfg.ttt.num_loo_per_task),
                            'grad_accum_steps': int(grad_accum_steps),
                            'num_support_batches_loo': int(_resolve_num_support_batches_loo(cfg.ttt)),
                            'outer_context_size': int(resolved_outer_context_size),
                            'reuse_diffusion_noise': _as_bool(cfg.ttt.reuse_diffusion_noise),
                            'preload_support_batches_to_device': _as_bool(
                                getattr(cfg.ttt, 'preload_support_batches_to_device', False)
                            ),
                            'log_query_loss': _as_bool(getattr(cfg.ttt, 'log_query_loss', False)),
                            'num_query_loss_samples': int(getattr(cfg.ttt, 'num_query_loss_samples', 1)),
                            'log_query_sample_mse': _as_bool(getattr(cfg.ttt, 'log_query_sample_mse', False)),
                        }
                    ),
                )
                current_ttt_stats = _apply_ttt_adaptation_in_place(
                    policy=model,
                    base_state_dict=state_dict,
                    support_package=current_support_package,
                    fast_names=fast_names,
                    device=device,
                    ttt_cfg=cfg.ttt,
                    use_mask_id=use_mask_id,
                    sample_mse_inference_steps=int(cfg.inference.inference_steps),
                    sample_mse_eta=float(cfg.inference.eta),
                )
                current_ttt_stats.update(
                    _save_inner_loss_artifacts(
                        inner_losses=current_ttt_stats.get('inner_losses', []),
                        query_losses=current_ttt_stats.get('query_losses', None),
                        query_sample_mses=current_ttt_stats.get('query_sample_mses', None),
                        run_dir=run_dir,
                        stem=f'ttt_episode_{ep:04d}',
                    )
                )
                model.eval()
                if current_ttt_stats.get('query_losses', None) or current_ttt_stats.get('query_sample_mses', None):
                    logging.info(
                        'TTT adaptation ready for episode %d | support_episodes=%s | rollout_context=%s | '
                        'query_episode=%s | holdout_indices=%s | support_batches=%d | avg_inner_loss=%.6f | '
                        'avg_query_loss=%.6f | avg_query_sample_mse=%.6f | avg_inner_grad_norm=%.6f',
                        ep,
                        current_ttt_stats['support_ids'],
                        current_ttt_stats['rollout_support_ids'],
                        current_ttt_stats.get('query_episode_id', None),
                        current_ttt_stats['holdout_indices'],
                        int(current_ttt_stats.get('num_support_batches', 0)),
                        float(current_ttt_stats['avg_inner_loss']),
                        float(current_ttt_stats['avg_query_loss']),
                        float(current_ttt_stats.get('avg_query_sample_mse', 0.0)),
                        float(current_ttt_stats['avg_inner_grad_norm']),
                    )
                else:
                    logging.info(
                        'TTT adaptation ready for episode %d | support_episodes=%s | rollout_context=%s | '
                        'holdout_indices=%s | support_batches=%d | avg_inner_loss=%.6f | avg_inner_grad_norm=%.6f',
                        ep,
                        current_ttt_stats['support_ids'],
                        current_ttt_stats['rollout_support_ids'],
                        current_ttt_stats['holdout_indices'],
                        int(current_ttt_stats.get('num_support_batches', 0)),
                        float(current_ttt_stats['avg_inner_loss']),
                        float(current_ttt_stats['avg_inner_grad_norm']),
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
            if current_ttt_stats is not None:
                res['ttt'] = current_ttt_stats
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
            'resolved_ttt': {
                'outer_context_size': int(resolved_outer_context_size),
                'inner_steps': int(cfg.ttt.inner_steps),
                'inner_lr': float(cfg.ttt.inner_lr),
                'max_grad_norm': float(cfg.ttt.max_grad_norm),
                'num_loo_per_task': int(cfg.ttt.num_loo_per_task),
                'grad_accum_steps': int(grad_accum_steps),
                'num_support_batches_loo': int(_resolve_num_support_batches_loo(cfg.ttt)),
                'reuse_diffusion_noise': _as_bool(cfg.ttt.reuse_diffusion_noise),
                'preload_support_batches_to_device': _as_bool(
                    getattr(cfg.ttt, 'preload_support_batches_to_device', False)
                ),
                'log_query_loss': _as_bool(getattr(cfg.ttt, 'log_query_loss', False)),
                'num_query_loss_samples': int(getattr(cfg.ttt, 'num_query_loss_samples', 1)),
                'log_query_sample_mse': _as_bool(getattr(cfg.ttt, 'log_query_sample_mse', False)),
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
