from __future__ import annotations

import json
import random
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm

from icil.action_representation import decode_action_chunk
from icil.datasets.in_context_imitation_learning.cache_variation_h5 import (
    MASK_NAME_SUBSTRINGS_TO_IGNORE,
    MASK_NAMES_TO_IGNORE,
    _build_vector,
    _filter_by_ignore_ids,
    _subsample_fixedN,
)
from icil.datasets.in_context_imitation_learning.icil_datasets import (
    ICILConfig,
    ICILSamplerCore,
)
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

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/eval_single_task_perceiver.py",
    help_string="Path to ml_collections config file.",
)

_CAMERAS: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "overhead",
    "wrist",
    "front",
)


def _as_bool(v: Any) -> bool:
    return bool(v)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and str(device_str).startswith("cuda"):
        return torch.device(str(device_str))
    return torch.device("cpu")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Accept bare state_dict checkpoints.
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(ckpt).__name__}")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' payload is not a state_dict dictionary.")
    return ckpt if isinstance(ckpt, dict) else {}, _strip_module_prefix(state_dict)


def _infer_state_action_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        action_dim = int(state_dict["action_out.weight"].shape[0])
    except KeyError as exc:
        raise KeyError(
            "Could not infer action_dim from checkpoint. Expected key 'action_out.weight'."
        ) from exc

    state_dim_key_candidates = (
        "context_encoder.state_proj.0.weight",
        "context_encoder.demo_query_encoder.state_proj.0.weight",
        "context_encoder.demo_frame_stack.state_proj.0.weight",
        "context_encoder.demo_query_encoder.demo_frame_stack.state_proj.0.weight",
    )
    state_dim = None
    for key in state_dim_key_candidates:
        if key in state_dict:
            state_dim = int(state_dict[key].shape[1])
            break
    if state_dim is None:
        # For non-perceiver context encoders, this key may not exist.
        # Keep a practical default for RLBench gripper pose+open.
        state_dim = 8
        logging.warning(
            "Could not infer state_dim from state_dict; using fallback state_dim=%d.",
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
    # Legacy flat configs were perceiver-only.
    out: Dict[str, Any] = {
        "encoder_name": "perceiver_demo_query",
        "policy": {},
        "perceiver_demo_query": {},
        "perceiver_demo_query_v2": {},
        "perceiver_demo_query_supernode_v2": {},
        "traj_perceiver": {},
        "traj_perceiver_v2": {},
        "traj_supernode_perceiver_v2": {},
    }
    for k, v in model_from_ckpt.items():
        if k in (
            "policy",
            "conv3d_demo_query",
            "perceiver_demo_query",
            "perceiver_demo_query_v2",
            "perceiver_demo_query_supernode_v2",
            "traj_conv3d",
            "traj_perceiver",
            "traj_perceiver_v2",
            "traj_supernode_perceiver_v2",
            "encoder_name",
        ):
            out[k] = v
            continue
        if k in policy_field_names:
            out["policy"][k] = v
        else:
            out["perceiver_demo_query"][k] = v
    return out


def _model_config_from_checkpoint_or_default(
    ckpt: Dict[str, Any],
) -> PolicyBuilderConfig:
    model_from_ckpt: Dict[str, Any] = {}
    if isinstance(ckpt.get("config", None), dict):
        model_from_ckpt = ckpt["config"].get("model", {}) or {}

    defaults = PolicyBuilderConfig()
    if not isinstance(model_from_ckpt, dict) or not model_from_ckpt:
        return defaults
    if "policy" not in model_from_ckpt:
        model_from_ckpt = _legacy_flat_model_cfg_to_nested(model_from_ckpt)
    model_from_ckpt = inherit_missing_encoder_attention_backend(model_from_ckpt)
    return _dataclass_from_dict(defaults, model_from_ckpt)


def _conditioning_use_mask_id_from_eval_and_checkpoint(
    cfg: ConfigDict,
    model_cfg: PolicyBuilderConfig,
) -> bool:
    if str(model_cfg.encoder_name) == "conv3d_demo_query":
        return bool(model_cfg.conv3d_demo_query.use_mask_id)
    if str(model_cfg.encoder_name) == "perceiver_demo_query":
        return bool(model_cfg.perceiver_demo_query.use_mask_id)
    if str(model_cfg.encoder_name) == "perceiver_demo_query_v2":
        return bool(model_cfg.perceiver_demo_query_v2.use_mask_id)
    if str(model_cfg.encoder_name) == "perceiver_demo_query_supernode_v2":
        return bool(model_cfg.perceiver_demo_query_supernode_v2.use_mask_id)
    if str(model_cfg.encoder_name) == "traj_conv3d":
        return bool(model_cfg.traj_conv3d.use_mask_id)
    if str(model_cfg.encoder_name) == "traj_perceiver":
        return bool(model_cfg.traj_perceiver.use_mask_id)
    if str(model_cfg.encoder_name) == "traj_perceiver_v2":
        return bool(model_cfg.traj_perceiver_v2.use_mask_id)
    if str(model_cfg.encoder_name) == "traj_supernode_perceiver_v2":
        return bool(model_cfg.traj_supernode_perceiver_v2.use_mask_id)
    return _as_bool(getattr(cfg.conditioning, "use_mask_id", True))


def _ignore_demos_from_model_cfg(model_cfg: PolicyBuilderConfig) -> bool:
    if str(model_cfg.encoder_name) == "conv3d_demo_query":
        return bool(model_cfg.conv3d_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == "perceiver_demo_query":
        return bool(model_cfg.perceiver_demo_query.ignore_demos)
    if str(model_cfg.encoder_name) == "perceiver_demo_query_v2":
        return bool(model_cfg.perceiver_demo_query_v2.ignore_demos)
    if str(model_cfg.encoder_name) == "perceiver_demo_query_supernode_v2":
        return bool(model_cfg.perceiver_demo_query_supernode_v2.ignore_demos)
    if str(model_cfg.encoder_name) == "traj_conv3d":
        return bool(model_cfg.traj_conv3d.ignore_demos)
    if str(model_cfg.encoder_name) == "traj_perceiver":
        return bool(model_cfg.traj_perceiver.ignore_demos)
    if str(model_cfg.encoder_name) == "traj_perceiver_v2":
        return bool(model_cfg.traj_perceiver_v2.ignore_demos)
    if str(model_cfg.encoder_name) == "traj_supernode_perceiver_v2":
        return bool(model_cfg.traj_supernode_perceiver_v2.ignore_demos)
    return False


def _dataset_config_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> ICILConfig:
    ckpt_dataset = {}
    if isinstance(ckpt.get("config", None), dict):
        ckpt_dataset = ckpt["config"].get("dataset", {}) or {}

    use_ckpt = _as_bool(getattr(cfg.dataset, "use_checkpoint_dataset_config", True))

    def _ival(name: str, default: int) -> int:
        if use_ckpt and name in ckpt_dataset:
            return int(ckpt_dataset[name])
        return int(getattr(cfg.dataset, name, default))

    action_representation = str(getattr(cfg.dataset, "action_representation", "absolute"))
    if use_ckpt and "action_representation" in ckpt_dataset:
        action_representation = str(ckpt_dataset["action_representation"])

    return ICILConfig(
        K=_ival("K", 1),
        L=_ival("L", 1),
        T_obs=_ival("T_obs", 1),
        H=_ival("H", 1),
        stride=_ival("stride", 1),
        action_representation=action_representation,
    )


def _query_stride_mode_from_eval(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, "query_stride_mode", "dataset")).lower()
    if mode not in ("dataset", "consecutive"):
        raise ValueError("cfg.dataset.query_stride_mode must be one of: dataset, consecutive.")
    return mode


def _support_source_from_eval(cfg: ConfigDict) -> str:
    source = str(getattr(cfg.conditioning, "support_source", "cache")).lower()
    if source not in ("cache", "live"):
        raise ValueError("cfg.conditioning.support_source must be one of: cache, live.")
    return source


def _support_cache_root_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> Path:
    root = str(getattr(cfg.conditioning, "cache_root", "")).strip()
    if not root and isinstance(ckpt.get("config", None), dict):
        root = str((ckpt["config"].get("data", {}) or {}).get("cache_root", "")).strip()
    if not root:
        raise ValueError(
            "Cached support conditioning requires cfg.conditioning.cache_root or "
            "checkpoint['config']['data']['cache_root']."
        )
    cache_root = Path(root).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Support cache root not found: {cache_root}")
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
            with h5py.File(path, "r") as f:
                detected = int(f.attrs.get("N", -1))
        except Exception as exc:
            logging.warning("Could not inspect cached point count from %s: %s", path, exc)
            continue
        if detected > 0:
            detected_values.append(detected)
            inspected_paths.append(str(path))

    if not detected_values:
        return

    unique_detected = sorted(set(int(v) for v in detected_values))
    if len(unique_detected) > 1 or int(expected_num_points) not in unique_detected:
        logging.warning("============================================================")
        logging.warning(
            "POINT-COUNT MISMATCH: cfg.conditioning.num_points=%d, but cached task '%s' stores N=%s.",
            int(expected_num_points),
            task_name,
            unique_detected,
        )
        logging.warning(
            "Live observations will be resampled to %d points, while cached support/training data uses N=%s.",
            int(expected_num_points),
            unique_detected,
        )
        if inspected_paths:
            logging.warning("Inspected cache files: %s", inspected_paths)
        logging.warning("============================================================")


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
    frame = getattr(obs, f"{camera}_rgb", None)
    if frame is None:
        frame = getattr(obs, "front_rgb", None)
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

    if ext == ".mp4":
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(out_path), list(frames), fps=int(fps))
            return out_path
        except Exception as exc:
            logging.warning("MP4 export failed (%s). Falling back to GIF.", exc)
            out_path = out_path.with_suffix(".gif")
            ext = ".gif"

    if ext == ".gif":
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
            if not pil_frames:
                raise RuntimeError("No frames to write.")
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(round(1000.0 / max(1, int(fps))))),
                loop=0,
            )
            return out_path
        except Exception as exc:
            logging.warning("GIF export failed (%s). Saving raw frames npz.", exc)

    fallback = out_path.with_suffix(".npz")
    np.savez_compressed(str(fallback), frames=np.asarray(frames, dtype=np.uint8))
    return fallback


class _LiveConditioningProcessor:
    """
    Converts live RLBench observations into the same point/state representation used by caching.
    """

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
            pc = getattr(obs, f"{cam}_point_cloud", None)
            msk = getattr(obs, f"{cam}_mask", None)
            rgb = getattr(obs, f"{cam}_rgb", None)
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

        state = _build_vector(obs, ("gripper_pose", "gripper_open")).astype(np.float32, copy=False)

        out: Dict[str, torch.Tensor] = {
            "xyz": torch.from_numpy(xyz).float(),
            "valid": torch.from_numpy(valid).bool(),
            "state": torch.from_numpy(state).float(),
        }
        if rgb is not None:
            out["rgb"] = torch.from_numpy(rgb).float() / 255.0
        if mask_id is not None:
            out["mask_id"] = torch.from_numpy(mask_id).long()
        return out


def _build_live_support_conditioning(
    demos: Sequence[Any],
    *,
    dataset_cfg: ICILConfig,
    processor: _LiveConditioningProcessor,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    if len(demos) < dataset_cfg.K:
        raise ValueError(f"Need at least K={dataset_cfg.K} demos, got {len(demos)}.")

    sampler = ICILSamplerCore(store=None, cfg=dataset_cfg, seed=int(rng.integers(1 << 31)))

    cond_xyz: List[torch.Tensor] = []
    cond_state: List[torch.Tensor] = []
    cond_valid: List[torch.Tensor] = []
    cond_mask: List[torch.Tensor] = []
    cond_rgb: List[torch.Tensor] = []
    all_have_mask = True
    all_have_rgb = True

    for demo in demos[: dataset_cfg.K]:
        frames = [processor.observation_to_frame(obs) for obs in demo]
        if not frames:
            raise RuntimeError("Received an empty demo while building conditioning.")
        keyframes = sampler._sample_keyframes(len(frames), dataset_cfg.L, rng)

        picked = [frames[int(i)] for i in keyframes]
        cond_xyz.append(torch.stack([f["xyz"] for f in picked], 0))
        cond_state.append(torch.stack([f["state"] for f in picked], 0))
        cond_valid.append(torch.stack([f["valid"] for f in picked], 0))

        if all("mask_id" in f for f in picked):
            cond_mask.append(torch.stack([f["mask_id"] for f in picked], 0))
        else:
            all_have_mask = False
        if all("rgb" in f for f in picked):
            cond_rgb.append(torch.stack([f["rgb"] for f in picked], 0))
        else:
            all_have_rgb = False

    out: Dict[str, torch.Tensor] = {
        "cond_xyz": torch.stack(cond_xyz, 0).unsqueeze(0),    # [1,K,L,N,3]
        "cond_state": torch.stack(cond_state, 0).unsqueeze(0),  # [1,K,L,S]
        "cond_valid": torch.stack(cond_valid, 0).unsqueeze(0),  # [1,K,L,N]
    }
    # Trajectory branch compatibility: use keyframe state sequence as support trajectory fallback.
    out["cond_traj"] = sampler._encode_support_traj(out["cond_state"])  # [1,K,L,S]
    out["cond_traj_mask"] = torch.ones(
        out["cond_state"].shape[:3],
        dtype=torch.bool,
    )  # [1,K,L]
    if all_have_mask and len(cond_mask) == dataset_cfg.K:
        out["cond_mask_id"] = torch.stack(cond_mask, 0).unsqueeze(0)  # [1,K,L,N]
    if all_have_rgb and len(cond_rgb) == dataset_cfg.K:
        out["cond_rgb"] = torch.stack(cond_rgb, 0).unsqueeze(0)  # [1,K,L,N,3]
    return out


def _unsqueeze_support_batch_dim(support: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in support.items():
        out[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
    return out


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
            f"No cached variation found for requested variation={variation}. "
            f"Available variations for task '{store.keys[0].task if store.keys else '?'}': {available}"
        )
    if variation >= 0:
        return int(candidates[0])
    return int(rng.choice(np.asarray(candidates, dtype=np.int64)))


def _build_cached_support_conditioning(
    *,
    store: VariationStore,
    dataset_cfg: ICILConfig,
    variation: int,
    rng: np.random.Generator,
    use_rgb: bool,
    use_mask_id: bool,
) -> Dict[str, Any]:
    vidx = _select_support_vidx(store, variation=variation, rng=rng)
    episode_ids = store.list_episode_ids(vidx)
    if episode_ids.shape[0] < dataset_cfg.K:
        raise RuntimeError(
            f"Need at least K={dataset_cfg.K} cached support episodes, got {episode_ids.shape[0]} "
            f"for task={store.keys[vidx].task} variation={store.keys[vidx].variation}."
        )
    support_ids = rng.choice(episode_ids, size=dataset_cfg.K, replace=False)
    sampler = ICILSamplerCore(store=store, cfg=dataset_cfg, seed=int(rng.integers(1 << 31)))
    support = sampler.build_support_conditioning(
        rng,
        vidx=vidx,
        support_ids=support_ids,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
        load_full_traj=True,
    )
    if support is None:
        raise RuntimeError(
            f"Failed to build cached support conditioning for task={store.keys[vidx].task} "
            f"variation={store.keys[vidx].variation}."
        )
    support = _unsqueeze_support_batch_dim(support)
    support["meta"] = {
        "task": store.keys[vidx].task,
        "variation": int(store.keys[vidx].variation),
        "support_episodes": [int(eid) for eid in np.asarray(support_ids).tolist()],
    }
    return support


def _build_query_window(
    history: Sequence[Dict[str, torch.Tensor]],
    *,
    dataset_cfg: ICILConfig,
    query_stride_mode: str,
) -> Dict[str, torch.Tensor]:
    if not history:
        raise RuntimeError("Query history is empty.")
    last = len(history) - 1
    qstep = int(dataset_cfg.stride) if query_stride_mode == "dataset" else 1
    idx: List[int] = []
    for i in range(dataset_cfg.T_obs):
        rel = (dataset_cfg.T_obs - 1 - i) * qstep
        idx.append(max(0, last - rel))
    frames = [history[i] for i in idx]

    out: Dict[str, torch.Tensor] = {
        "query_xyz": torch.stack([f["xyz"] for f in frames], 0).unsqueeze(0),      # [1,T_obs,N,3]
        "query_state": torch.stack([f["state"] for f in frames], 0).unsqueeze(0),  # [1,T_obs,S]
        "query_valid": torch.stack([f["valid"] for f in frames], 0).unsqueeze(0),  # [1,T_obs,N]
    }
    if all("mask_id" in f for f in frames):
        out["query_mask_id"] = torch.stack([f["mask_id"] for f in frames], 0).unsqueeze(0)  # [1,T_obs,N]
    if all("rgb" in f for f in frames):
        out["query_rgb"] = torch.stack([f["rgb"] for f in frames], 0).unsqueeze(0)  # [1,T_obs,N,3]
    return out


def _to_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device, non_blocking=True)


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
    if renderer_name == "opengl":
        render_mode = RenderMode.OPENGL
    elif renderer_name == "opengl3":
        render_mode = RenderMode.OPENGL3
    else:
        raise ValueError(f"Unsupported renderer '{cfg.sim.renderer}'. Use 'opengl' or 'opengl3'.")

    for cam_name in _CAMERAS:
        cam_cfg = getattr(obs_config, f"{cam_name}_camera")
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
    del descriptions  # unused

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
    pbar = tqdm(
        total=max_env_steps,
        desc=f"Episode {episode_index}",
        leave=False,
        unit="step",
    )
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(
                history,
                dataset_cfg=dataset_cfg,
                query_stride_mode=query_stride_mode,
            )
            query_xyz = _to_device(query["query_xyz"], device)
            query_state = _to_device(query["query_state"], device)
            query_valid = _to_device(query.get("query_valid", None), device)
            query_mask_id = _to_device(query.get("query_mask_id", None), device)
            query_rgb = _to_device(query.get("query_rgb", None), device)

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
                    raise RuntimeError("support_cond is required when ignore_demos=False.")
                cond_xyz = _to_device(support_cond["cond_xyz"], device)
                cond_state = _to_device(support_cond["cond_state"], device)
                cond_valid = _to_device(support_cond.get("cond_valid", None), device)
                cond_mask_id = _to_device(support_cond.get("cond_mask_id", None), device)
                cond_rgb = _to_device(support_cond.get("cond_rgb", None), device)
                cond_traj = _to_device(support_cond.get("cond_traj", None), device)
                cond_traj_mask = _to_device(support_cond.get("cond_traj_mask", None), device)

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
            plan = decode_action_chunk(
                plan,
                query_state=query_state,
                representation=str(dataset_cfg.action_representation),
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
                    error = f"InvalidActionError: {exc}"
                    terminated = True
                    break
                except Exception as exc:  # pragma: no cover - defensive path in runtime envs
                    error = f"{type(exc).__name__}: {exc}"
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
    if _as_bool(cfg.video.enable) and len(frames) > 0:
        video_file = run_dir / "videos" / f"episode_{episode_index:04d}.{str(cfg.video.format).lower()}"
        actual = _write_video(frames, video_file, fps=int(cfg.video.fps))
        video_path = str(actual)

    return {
        "episode_index": int(episode_index),
        "success": bool(success),
        "terminated": bool(terminated),
        "env_steps": int(env_steps),
        "error": error,
        "video_path": video_path,
    }


def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    device = _resolve_device(str(cfg.device))

    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt, state_dict = _load_checkpoint(checkpoint_path, device)
    model_cfg = _model_config_from_checkpoint_or_default(ckpt)
    dataset_cfg = _dataset_config_from_eval_and_checkpoint(cfg, ckpt)
    use_mask_id = _conditioning_use_mask_id_from_eval_and_checkpoint(cfg, model_cfg)
    ignore_demos = _ignore_demos_from_model_cfg(model_cfg)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    support_source = _support_source_from_eval(cfg)
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)

    model = build_policy(
        model_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if action_dim != 8:
        raise ValueError(
            f"Current eval pipeline expects action_dim=8 (gripper_pose[7] + gripper_open[1]), got {action_dim}."
        )

    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f"{task_name}_var{variation}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_eval_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logging.info("Loading task='%s' variation=%d", task_name, variation)
    logging.info("Checkpoint=%s", checkpoint_path)
    logging.info("Model cfg: encoder_name=%s | ignore_demos=%s", model_cfg.encoder_name, ignore_demos)
    logging.info("Conditioning cfg: use_mask_id=%s", use_mask_id)
    logging.info("Conditioning support_source=%s", support_source)
    logging.info("Eval query_stride_mode=%s", query_stride_mode)
    logging.info(
        "Dataset cfg: K=%d L=%d T_obs=%d H=%d stride=%d",
        dataset_cfg.K,
        dataset_cfg.L,
        dataset_cfg.T_obs,
        dataset_cfg.H,
        dataset_cfg.stride,
    )

    env = None
    task_env = None
    support_store: Optional[VariationStore] = None
    results: List[Dict[str, Any]] = []
    support_cond: Optional[Dict[str, Any]] = None
    cache_root: Optional[Path] = None
    task_keys: Optional[List[Any]] = None

    try:
        if support_source == "cache":
            if variation < 0:
                raise ValueError("Cached support conditioning requires cfg.task.variation >= 0.")
            cache_root = _support_cache_root_from_eval_and_checkpoint(cfg, ckpt)
            task_keys = build_variation_keys(cache_root, task_name)
            if not task_keys:
                raise RuntimeError(f"No cached variations found for task '{task_name}' under {cache_root}.")
            _warn_if_cached_num_points_mismatch(
                task_keys=task_keys,
                expected_num_points=int(cfg.conditioning.num_points),
                task_name=task_name,
            )
            if not ignore_demos:
                support_store = VariationStore(task_keys, keep_open_per_worker=True)
                logging.info("Using cached support from %s", cache_root)

        env, task_env = _build_rlbench_env(cfg, task_name)
        processor = _LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=_as_bool(cfg.conditioning.use_rgb),
            use_mask_id=use_mask_id,
            seed=seed + 11,
        )

        rng = np.random.default_rng(seed + 17)
        for ep in range(int(cfg.task.num_eval_episodes)):
            if not ignore_demos:
                regen = _as_bool(cfg.conditioning.regenerate_demos_each_episode)
                if support_cond is None or regen:
                    if support_source == "cache":
                        if support_store is None:
                            raise RuntimeError("support_store is required for cached support conditioning.")
                        support_cond = _build_cached_support_conditioning(
                            store=support_store,
                            dataset_cfg=dataset_cfg,
                            variation=variation,
                            rng=rng,
                            use_rgb=_as_bool(cfg.conditioning.use_rgb),
                            use_mask_id=use_mask_id,
                        )
                        logging.info(
                            "Built cached support conditioning from variation=%d episodes=%s.",
                            int(support_cond["meta"]["variation"]),
                            support_cond["meta"]["support_episodes"],
                        )
                    else:
                        if variation >= 0:
                            task_env.set_variation(variation)
                        demos = task_env.get_demos(amount=int(dataset_cfg.K), live_demos=True)
                        support_cond = _build_live_support_conditioning(
                            demos,
                            dataset_cfg=dataset_cfg,
                            processor=processor,
                            rng=rng,
                        )
                        logging.info("Built live support conditioning from %d demos.", dataset_cfg.K)
            else:
                support_cond = None

            res = _run_eval_episode(
                episode_index=ep,
                task_env=task_env,
                variation=variation,
                model=model,
                device=device,
                dataset_cfg=dataset_cfg,
                support_cond=support_cond,
                ignore_demos=ignore_demos,
                query_stride_mode=query_stride_mode,
                processor=processor,
                cfg=cfg,
                run_dir=run_dir,
            )
            results.append(res)
            logging.info(
                "Episode %d | success=%s | steps=%d%s",
                ep,
                res["success"],
                res["env_steps"],
                f" | error={res['error']}" if res["error"] else "",
            )

        n_success = sum(1 for r in results if r["success"])
        success_rate = float(n_success) / float(max(1, len(results)))
        summary = {
            "task": task_name,
            "variation": variation,
            "checkpoint_path": str(checkpoint_path),
            "num_episodes": len(results),
            "num_success": int(n_success),
            "success_rate": success_rate,
            "results": results,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logging.info(
            "Evaluation complete | success=%d/%d (%.3f) | outputs=%s",
            n_success,
            len(results),
            success_rate,
            run_dir,
        )
    finally:
        if support_store is not None:
            support_store.close()
        if env is not None:
            env.shutdown()


def main(argv: Sequence[str]) -> None:
    del argv
    cfg = _CONFIG.value
    evaluate(cfg)


if __name__ == "__main__":
    app.run(main)
