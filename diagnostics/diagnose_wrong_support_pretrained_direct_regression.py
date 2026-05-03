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
import torch.nn.functional as F
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
    DirectRegressionPolicy,
    PolicyBuilderConfig,
    build_direct_regression_policy,
)
from icil.models.policies.config_utils import inherit_missing_encoder_attention_backend

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/diagnose_wrong_support_pretrained_direct_regression.py",
    help_string="Path to ml_collections diagnostic config file.",
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
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint object type: {type(ckpt).__name__}")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model payload is not a state_dict dictionary.")
    return ckpt if isinstance(ckpt, dict) else {}, _strip_module_prefix(state_dict)


def _infer_state_action_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        action_dim = int(state_dict["action_out.weight"].shape[0])
    except KeyError as exc:
        raise KeyError("Could not infer action_dim; expected checkpoint key 'action_out.weight'.") from exc

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
        state_dim = 8
        logging.warning("Could not infer state_dim from checkpoint; using fallback state_dim=8.")
    return state_dim, action_dim


def _dataclass_from_dict(default_obj: Any, src: Dict[str, Any]) -> Any:
    if not is_dataclass(default_obj):
        return default_obj
    kwargs: Dict[str, Any] = {}
    for f in fields(default_obj):
        default_v = getattr(default_obj, f.name)
        if is_dataclass(default_v):
            sub_src = src.get(f.name, {}) if isinstance(src, dict) else {}
            kwargs[f.name] = _dataclass_from_dict(default_v, sub_src if isinstance(sub_src, dict) else {})
        else:
            if isinstance(src, dict) and f.name in src:
                v = src[f.name]
                if isinstance(default_v, tuple) and isinstance(v, (list, tuple)):
                    v = tuple(v)
                kwargs[f.name] = v
            else:
                kwargs[f.name] = default_v
    return type(default_obj)(**kwargs)


def _model_config_from_checkpoint_or_default(ckpt: Dict[str, Any]) -> PolicyBuilderConfig:
    model_from_ckpt: Dict[str, Any] = {}
    if isinstance(ckpt.get("config", None), dict):
        model_from_ckpt = ckpt["config"].get("model", {}) or {}
    defaults = PolicyBuilderConfig()
    if not isinstance(model_from_ckpt, dict) or not model_from_ckpt:
        return defaults
    model_from_ckpt = inherit_missing_encoder_attention_backend(model_from_ckpt)
    return _dataclass_from_dict(defaults, model_from_ckpt)


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


def _conditioning_use_mask_id_from_eval_and_checkpoint(
    cfg: ConfigDict,
    model_cfg: PolicyBuilderConfig,
) -> bool:
    name = str(model_cfg.encoder_name)
    if name == "conv3d_demo_query":
        return bool(model_cfg.conv3d_demo_query.use_mask_id)
    if name == "perceiver_demo_query":
        return bool(model_cfg.perceiver_demo_query.use_mask_id)
    if name == "perceiver_demo_query_v2":
        return bool(model_cfg.perceiver_demo_query_v2.use_mask_id)
    if name == "perceiver_demo_query_supernode_v2":
        return bool(model_cfg.perceiver_demo_query_supernode_v2.use_mask_id)
    if name == "traj_conv3d":
        return bool(model_cfg.traj_conv3d.use_mask_id)
    if name == "traj_perceiver":
        return bool(model_cfg.traj_perceiver.use_mask_id)
    if name == "traj_perceiver_v2":
        return bool(model_cfg.traj_perceiver_v2.use_mask_id)
    if name == "traj_supernode_perceiver_v2":
        return bool(model_cfg.traj_supernode_perceiver_v2.use_mask_id)
    return _as_bool(getattr(cfg.conditioning, "use_mask_id", True))


def _ignore_demos_from_model_cfg(model_cfg: PolicyBuilderConfig) -> bool:
    name = str(model_cfg.encoder_name)
    if name == "conv3d_demo_query":
        return bool(model_cfg.conv3d_demo_query.ignore_demos)
    if name == "perceiver_demo_query":
        return bool(model_cfg.perceiver_demo_query.ignore_demos)
    if name == "perceiver_demo_query_v2":
        return bool(model_cfg.perceiver_demo_query_v2.ignore_demos)
    if name == "perceiver_demo_query_supernode_v2":
        return bool(model_cfg.perceiver_demo_query_supernode_v2.ignore_demos)
    if name == "traj_conv3d":
        return bool(model_cfg.traj_conv3d.ignore_demos)
    if name == "traj_perceiver":
        return bool(model_cfg.traj_perceiver.ignore_demos)
    if name == "traj_perceiver_v2":
        return bool(model_cfg.traj_perceiver_v2.ignore_demos)
    if name == "traj_supernode_perceiver_v2":
        return bool(model_cfg.traj_supernode_perceiver_v2.ignore_demos)
    return False


def _support_cache_root_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> Path:
    root = str(getattr(cfg.conditioning, "cache_root", "")).strip()
    if not root and isinstance(ckpt.get("config", None), dict):
        root = str((ckpt["config"].get("data", {}) or {}).get("cache_root", "")).strip()
    if not root:
        raise ValueError("Set cfg.conditioning.cache_root or use a checkpoint with config.data.cache_root.")
    cache_root = Path(root).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Support cache root not found: {cache_root}")
    return cache_root


def _query_stride_mode_from_eval(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, "query_stride_mode", "consecutive")).lower()
    if mode not in ("dataset", "consecutive"):
        raise ValueError("cfg.dataset.query_stride_mode must be one of: dataset, consecutive.")
    return mode


def _warn_if_cached_num_points_mismatch(
    *,
    task_keys: Sequence[Any],
    expected_num_points: int,
    task_name: str,
) -> None:
    detected_values: List[int] = []
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
    if not detected_values:
        return
    unique_detected = sorted(set(int(v) for v in detected_values))
    if len(unique_detected) > 1 or int(expected_num_points) not in unique_detected:
        logging.warning(
            "Point-count mismatch for task=%s: cfg.conditioning.num_points=%d, cached N=%s.",
            task_name,
            int(expected_num_points),
            unique_detected,
        )


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
            f"No cached variation found for requested variation={variation}. Available={available}."
        )
    return int(candidates[0]) if variation >= 0 else int(rng.choice(np.asarray(candidates, dtype=np.int64)))


def _unsqueeze_support_batch_dim(support: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in support.items():
        out[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
    return out


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


def _select_task_vidx(store: VariationStore, *, task_name: str, variation: int) -> int:
    candidates = [
        i for i, key in enumerate(store.keys)
        if str(key.task) == str(task_name) and (variation < 0 or int(key.variation) == int(variation))
    ]
    if not candidates:
        available = sorted((str(key.task), int(key.variation)) for key in store.keys)
        raise RuntimeError(f"No cached target data for task={task_name} variation={variation}. Available={available}")
    return int(candidates[0])


def _stack_query_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    required = ("query_xyz", "query_state", "query_valid", "target_action")
    batch: Dict[str, Any] = {k: torch.stack([s[k] for s in samples], 0) for k in required}
    optional = ("query_rgb", "query_mask_id")
    for key in optional:
        if all(key in s for s in samples):
            batch[key] = torch.stack([s[key] for s in samples], 0)
    batch["meta"] = [s.get("meta", {}) for s in samples]
    return batch


def _build_target_query_batch(
    *,
    store: VariationStore,
    dataset_cfg: ICILConfig,
    task_name: str,
    variation: int,
    batch_size: int,
    seed: int,
    num_tries_per_item: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    vidx = _select_task_vidx(store, task_name=task_name, variation=variation)
    sampler = ICILSamplerCore(
        store=store,
        cfg=dataset_cfg,
        seed=int(seed),
        num_tries_per_item=int(num_tries_per_item),
    )
    samples: List[Dict[str, Any]] = []
    for _ in range(max(1, int(batch_size))):
        sample = None
        for _try in range(max(1, int(num_tries_per_item))):
            sample = sampler._build_one_sample(rng, vidx=vidx)
            if sample is not None:
                break
        if sample is None:
            raise RuntimeError(
                f"Failed to sample target query item for task={task_name} variation={variation}."
            )
        samples.append(sample)
    return _stack_query_samples(samples)


def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _expand_support_to_batch(support: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in support.items():
        if torch.is_tensor(v):
            t = v.to(device, non_blocking=True)
            if t.dim() > 0 and int(t.shape[0]) == 1:
                t = t.expand(int(batch_size), *t.shape[1:])
            out[k] = t
        else:
            out[k] = v
    return out


@torch.no_grad()
def _predict_action_chunk(
    model: DirectRegressionPolicy,
    *,
    support: Dict[str, Any],
    query_batch: Dict[str, Any],
    use_mask_id: bool,
    action_horizon: Optional[int] = None,
) -> torch.Tensor:
    if action_horizon is None:
        if "target_action" not in query_batch:
            raise KeyError("query_batch has no target_action; pass action_horizon explicitly.")
        action_horizon = int(query_batch["target_action"].shape[1])
    return model.sample_actions(
        cond_xyz=support["cond_xyz"],
        cond_state=support["cond_state"],
        cond_traj=support.get("cond_traj", None),
        cond_traj_mask=support.get("cond_traj_mask", None),
        query_xyz=query_batch["query_xyz"],
        query_state=query_batch["query_state"],
        action_horizon=int(action_horizon),
        cond_rgb=support.get("cond_rgb", None),
        query_rgb=query_batch.get("query_rgb", None),
        cond_mask_id=(support.get("cond_mask_id", None) if use_mask_id else None),
        query_mask_id=(query_batch.get("query_mask_id", None) if use_mask_id else None),
        cond_valid=support.get("cond_valid", None),
        query_valid=query_batch.get("query_valid", None),
    )


def _mse_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    xyz_dim = min(3, int(target.shape[-1]))
    return {
        "mse": float(F.mse_loss(pred, target, reduction="mean").detach().cpu()),
        "l1": float(F.l1_loss(pred, target, reduction="mean").detach().cpu()),
        "xyz_mse": float(F.mse_loss(pred[..., :xyz_dim], target[..., :xyz_dim], reduction="mean").detach().cpu()),
    }


def _mean_dict(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {k: float(np.mean([float(x[k]) for x in items])) for k in keys}


def _run_mse_diagnostic(
    *,
    model: DirectRegressionPolicy,
    target_store: VariationStore,
    correct_support: Dict[str, Any],
    wrong_support: Dict[str, Any],
    dataset_cfg: ICILConfig,
    cfg: ConfigDict,
    device: torch.device,
    use_mask_id: bool,
) -> Dict[str, Any]:
    correct_metrics: List[Dict[str, float]] = []
    wrong_metrics: List[Dict[str, float]] = []
    deltas: List[float] = []

    for bidx in tqdm(range(int(cfg.mse.num_batches)), desc="MSE diagnostic", unit="batch"):
        query = _build_target_query_batch(
            store=target_store,
            dataset_cfg=dataset_cfg,
            task_name=str(cfg.task.name),
            variation=int(cfg.task.variation),
            batch_size=int(cfg.mse.batch_size),
            seed=int(cfg.seed) + 1009 + bidx,
            num_tries_per_item=int(cfg.mse.num_tries_per_item),
        )
        query = _to_device_batch(query, device)
        B = int(query["target_action"].shape[0])
        correct = _expand_support_to_batch(correct_support, B, device)
        wrong = _expand_support_to_batch(wrong_support, B, device)

        pred_correct = _predict_action_chunk(model, support=correct, query_batch=query, use_mask_id=use_mask_id)
        pred_wrong = _predict_action_chunk(model, support=wrong, query_batch=query, use_mask_id=use_mask_id)
        target = query["target_action"]

        correct_metrics.append(_mse_metrics(pred_correct, target))
        wrong_metrics.append(_mse_metrics(pred_wrong, target))
        deltas.append(float(F.mse_loss(pred_wrong, pred_correct, reduction="mean").detach().cpu()))

    correct_mean = _mean_dict(correct_metrics)
    wrong_mean = _mean_dict(wrong_metrics)
    return {
        "num_batches": int(cfg.mse.num_batches),
        "batch_size": int(cfg.mse.batch_size),
        "correct_support": correct_mean,
        "wrong_support": wrong_mean,
        "wrong_minus_correct": {
            k: float(wrong_mean[k] - correct_mean[k])
            for k in sorted(correct_mean.keys() & wrong_mean.keys())
        },
        "pred_wrong_vs_pred_correct_mse": float(np.mean(deltas)) if deltas else None,
        "per_batch_correct": correct_metrics,
        "per_batch_wrong": wrong_metrics,
    }


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
            arr = np.asarray(m).reshape(-1).astype(np.int64, copy=False)
            for v in np.unique(arr):
                vi = int(v)
                if vi != 0 and vi not in self.handle_to_name:
                    unresolved.add(vi)
        if unresolved:
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
        for cam in _CAMERAS:
            pc = getattr(obs, f"{cam}_point_cloud", None)
            msk = getattr(obs, f"{cam}_mask", None)
            rgb = getattr(obs, f"{cam}_rgb", None)
            if pc is None or msk is None:
                continue
            pts = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
            masks = np.asarray(msk).reshape(-1).astype(np.int32, copy=False)
            cols = np.asarray(rgb).reshape(-1, 3).astype(np.uint8, copy=False) if self.use_rgb and rgb is not None else None
            finite = np.isfinite(pts).all(axis=1)
            merged_points.append(pts[finite])
            merged_masks.append(masks[finite])
            if cols is not None:
                merged_colors.append(cols[finite])
        if merged_points:
            pts_all = np.concatenate(merged_points, axis=0).astype(np.float32, copy=False)
            msk_all = np.concatenate(merged_masks, axis=0).astype(np.int32, copy=False)
            col_all = np.concatenate(merged_colors, axis=0).astype(np.uint8, copy=False) if self.use_rgb and merged_colors else None
        else:
            pts_all = np.zeros((0, 3), dtype=np.float32)
            msk_all = np.zeros((0,), dtype=np.int32)
            col_all = None
        self._update_handle_names(merged_masks)
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
            rgb = col_all[idx].astype(np.uint8, copy=False) if self.use_rgb and col_all is not None else None
            mask_id = msk_all[idx].astype(np.int64, copy=False) if self.use_mask_id else None
        out: Dict[str, torch.Tensor] = {
            "xyz": torch.from_numpy(xyz).float(),
            "valid": torch.from_numpy(valid).bool(),
            "state": torch.from_numpy(_build_vector(obs, ("gripper_pose", "gripper_open")).astype(np.float32, copy=False)).float(),
        }
        if rgb is not None:
            out["rgb"] = torch.from_numpy(rgb).float() / 255.0
        if mask_id is not None:
            out["mask_id"] = torch.from_numpy(mask_id).long()
        return out


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
    idx = [max(0, last - (dataset_cfg.T_obs - 1 - i) * qstep) for i in range(dataset_cfg.T_obs)]
    frames = [history[i] for i in idx]
    out: Dict[str, torch.Tensor] = {
        "query_xyz": torch.stack([f["xyz"] for f in frames], 0).unsqueeze(0),
        "query_state": torch.stack([f["state"] for f in frames], 0).unsqueeze(0),
        "query_valid": torch.stack([f["valid"] for f in frames], 0).unsqueeze(0),
    }
    if all("mask_id" in f for f in frames):
        out["query_mask_id"] = torch.stack([f["mask_id"] for f in frames], 0).unsqueeze(0)
    if all("rgb" in f for f in frames):
        out["query_rgb"] = torch.stack([f["rgb"] for f in frames], 0).unsqueeze(0)
    return out


def _to_device_tensor(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    return None if t is None else t.to(device, non_blocking=True)


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
    render_mode = RenderMode.OPENGL if str(cfg.sim.renderer).lower() == "opengl" else RenderMode.OPENGL3
    for cam_name in _CAMERAS:
        cam_cfg = getattr(obs_config, f"{cam_name}_camera")
        cam_cfg.image_size = image_size
        cam_cfg.depth_in_meters = False
        cam_cfg.masks_as_one_channel = True
        cam_cfg.render_mode = render_mode
    action_mode = MoveArmThenGripper(
        EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=_as_bool(cfg.sim.collision_checking)),
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
    return env, env.get_task(task_file_to_task_class(task_name))


def _run_support_rollout(
    *,
    episode_index: int,
    task_env: Any,
    variation: int,
    model: DirectRegressionPolicy,
    device: torch.device,
    dataset_cfg: ICILConfig,
    support_cond: Dict[str, Any],
    support_label: str,
    query_stride_mode: str,
    processor: _LiveConditioningProcessor,
    cfg: ConfigDict,
    run_dir: Path,
    use_mask_id: bool,
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
    support = _expand_support_to_batch(support_cond, 1, device)
    pbar = tqdm(
        total=max_env_steps,
        desc=f"{support_label} episode {episode_index}",
        leave=False,
        unit="step",
    )
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(history, dataset_cfg=dataset_cfg, query_stride_mode=query_stride_mode)
            query = _to_device_batch(query, device)
            with torch.no_grad():
                plan = _predict_action_chunk(
                    model,
                    support=support,
                    query_batch=query,
                    use_mask_id=use_mask_id,
                    action_horizon=int(dataset_cfg.H),
                )
            plan = decode_action_chunk(
                plan,
                query_state=query["query_state"],
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
                except Exception as exc:
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
    if _as_bool(cfg.video.enable) and frames:
        video_file = run_dir / f"{support_label}_videos" / f"episode_{episode_index:04d}.{str(cfg.video.format).lower()}"
        video_path = str(_write_video(frames, video_file, fps=int(cfg.video.fps)))

    return {
        "support_label": str(support_label),
        "episode_index": int(episode_index),
        "success": bool(success),
        "terminated": bool(terminated),
        "env_steps": int(env_steps),
        "error": error,
        "video_path": video_path,
    }


def _run_rollout_diagnostic(
    *,
    model: DirectRegressionPolicy,
    correct_support: Dict[str, Any],
    wrong_support: Dict[str, Any],
    dataset_cfg: ICILConfig,
    cfg: ConfigDict,
    device: torch.device,
    run_dir: Path,
    use_mask_id: bool,
) -> Dict[str, Any]:
    env = None
    correct_results: List[Dict[str, Any]] = []
    wrong_results: List[Dict[str, Any]] = []
    try:
        env, task_env = _build_rlbench_env(cfg, str(cfg.task.name))
        processor = _LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=_as_bool(cfg.conditioning.use_rgb),
            use_mask_id=use_mask_id,
            seed=int(cfg.seed) + 501,
        )
        query_stride_mode = _query_stride_mode_from_eval(cfg)
        for support_label, support_cond, results in (
            ("correct_support", correct_support, correct_results),
            ("wrong_support", wrong_support, wrong_results),
        ):
            for ep in range(int(cfg.task.num_rollout_episodes)):
                res = _run_support_rollout(
                    episode_index=ep,
                    task_env=task_env,
                    variation=int(cfg.task.variation),
                    model=model,
                    device=device,
                    dataset_cfg=dataset_cfg,
                    support_cond=support_cond,
                    support_label=support_label,
                    query_stride_mode=query_stride_mode,
                    processor=processor,
                    cfg=cfg,
                    run_dir=run_dir,
                    use_mask_id=use_mask_id,
                )
                results.append(res)
                logging.info(
                    "%s rollout %d | success=%s | steps=%d",
                    support_label,
                    ep,
                    res["success"],
                    res["env_steps"],
                )
    finally:
        if env is not None:
            env.shutdown()

    def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n_success = sum(1 for r in results if r["success"])
        return {
            "num_episodes": len(results),
            "num_success": int(n_success),
            "success_rate": float(n_success) / float(max(1, len(results))),
            "results": list(results),
        }

    correct_summary = _summarize(correct_results)
    wrong_summary = _summarize(wrong_results)
    return {
        "correct_support": correct_summary,
        "wrong_support": wrong_summary,
        "wrong_minus_correct_success_rate": (
            float(wrong_summary["success_rate"]) - float(correct_summary["success_rate"])
        ),
    }


def diagnose(cfg: ConfigDict) -> None:
    _set_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.device))
    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt, state_dict = _load_checkpoint(checkpoint_path, device)
    model_cfg = _model_config_from_checkpoint_or_default(ckpt)
    dataset_cfg = _dataset_config_from_eval_and_checkpoint(cfg, ckpt)
    use_mask_id = _conditioning_use_mask_id_from_eval_and_checkpoint(cfg, model_cfg)
    ignore_demos = _ignore_demos_from_model_cfg(model_cfg)
    if ignore_demos:
        raise ValueError(
            "This checkpoint is configured with ignore_demos=True, so wrong-support diagnostics are not meaningful."
        )
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)
    if int(model_cfg.direct_regression.horizon) != int(dataset_cfg.H):
        raise ValueError(
            f"Model horizon ({model_cfg.direct_regression.horizon}) != dataset H ({dataset_cfg.H})."
        )

    model = build_direct_regression_policy(model_cfg, state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if str(cfg.task.name) == str(cfg.wrong_support.task_name) and int(cfg.task.variation) == int(cfg.wrong_support.variation):
        raise ValueError("Wrong support must come from a different task/variation than the target query task.")

    cache_root = _support_cache_root_from_eval_and_checkpoint(cfg, ckpt)
    target_keys = build_variation_keys(cache_root, str(cfg.task.name))
    wrong_keys = build_variation_keys(cache_root, str(cfg.wrong_support.task_name))
    if not target_keys:
        raise RuntimeError(f"No cached target task found under {cache_root}: {cfg.task.name}")
    if not wrong_keys:
        raise RuntimeError(f"No cached wrong-support task found under {cache_root}: {cfg.wrong_support.task_name}")
    _warn_if_cached_num_points_mismatch(
        task_keys=target_keys,
        expected_num_points=int(cfg.conditioning.num_points),
        task_name=str(cfg.task.name),
    )
    _warn_if_cached_num_points_mismatch(
        task_keys=wrong_keys,
        expected_num_points=int(cfg.conditioning.num_points),
        task_name=str(cfg.wrong_support.task_name),
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{cfg.task.name}_var{int(cfg.task.variation)}"
        f"_wrong_{cfg.wrong_support.task_name}_var{int(cfg.wrong_support.variation)}_{run_id}"
    )
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / "wrong_support_pretrained_direct_regression" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logging.info("Checkpoint=%s", checkpoint_path)
    logging.info("Target task=%s variation=%d", cfg.task.name, int(cfg.task.variation))
    logging.info("Wrong support task=%s variation=%d", cfg.wrong_support.task_name, int(cfg.wrong_support.variation))
    logging.info(
        "Dataset cfg: K=%d L=%d T_obs=%d H=%d stride=%d action=%s",
        dataset_cfg.K,
        dataset_cfg.L,
        dataset_cfg.T_obs,
        dataset_cfg.H,
        dataset_cfg.stride,
        dataset_cfg.action_representation,
    )
    logging.info("Model encoder=%s use_mask_id=%s", model_cfg.encoder_name, use_mask_id)

    rng = np.random.default_rng(int(cfg.seed) + 17)
    target_store = VariationStore(target_keys, keep_open_per_worker=True)
    wrong_store = VariationStore(wrong_keys, keep_open_per_worker=True)
    correct_support = _build_cached_support_conditioning(
        store=target_store,
        dataset_cfg=dataset_cfg,
        variation=int(cfg.task.variation),
        rng=rng,
        use_rgb=_as_bool(cfg.conditioning.use_rgb),
        use_mask_id=use_mask_id,
    )
    wrong_support = _build_cached_support_conditioning(
        store=wrong_store,
        dataset_cfg=dataset_cfg,
        variation=int(cfg.wrong_support.variation),
        rng=rng,
        use_rgb=_as_bool(cfg.conditioning.use_rgb),
        use_mask_id=use_mask_id,
    )

    summary: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "target": {"task": str(cfg.task.name), "variation": int(cfg.task.variation)},
        "correct_support": correct_support.get("meta", {}),
        "wrong_support": wrong_support.get("meta", {}),
        "dataset": {
            "K": int(dataset_cfg.K),
            "L": int(dataset_cfg.L),
            "T_obs": int(dataset_cfg.T_obs),
            "H": int(dataset_cfg.H),
            "stride": int(dataset_cfg.stride),
            "action_representation": str(dataset_cfg.action_representation),
        },
        "model": {
            "encoder_name": str(model_cfg.encoder_name),
            "use_mask_id": bool(use_mask_id),
        },
    }

    if _as_bool(cfg.mse.enable):
        mse_summary = _run_mse_diagnostic(
            model=model,
            target_store=target_store,
            correct_support=correct_support,
            wrong_support=wrong_support,
            dataset_cfg=dataset_cfg,
            cfg=cfg,
            device=device,
            use_mask_id=use_mask_id,
        )
        summary["mse_diagnostic"] = mse_summary
        with (run_dir / "mse_summary.json").open("w", encoding="utf-8") as f:
            json.dump(mse_summary, f, indent=2)
        logging.info("Correct support MSE: %s", mse_summary["correct_support"])
        logging.info("Wrong support MSE:   %s", mse_summary["wrong_support"])
        logging.info("Wrong-correct delta: %s", mse_summary["wrong_minus_correct"])

    if int(cfg.task.num_rollout_episodes) > 0:
        rollout_summary = _run_rollout_diagnostic(
            model=model,
            correct_support=correct_support,
            wrong_support=wrong_support,
            dataset_cfg=dataset_cfg,
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            use_mask_id=use_mask_id,
        )
        summary["wrong_support_rollout"] = rollout_summary
        with (run_dir / "rollout_summary.json").open("w", encoding="utf-8") as f:
            json.dump(rollout_summary, f, indent=2)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Diagnostics written to %s", run_dir)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Unexpected positional arguments.")
    diagnose(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)
