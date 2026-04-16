from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.profiler import ProfilerActivity

from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)
from icil.models import PolicyBuilderConfig
from icil.models.policies.config_utils import build_policy_builder_config_from_configdict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_bool(v: Any) -> bool:
    return bool(v)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def unwrap_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    return batch_list[0]


def drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop("cond_mask_id", None)
    out.pop("query_mask_id", None)
    return out


def discover_cached_tasks(cache_root: Path) -> List[str]:
    tasks: List[str] = []
    if not cache_root.is_dir():
        return tasks
    for p in sorted(cache_root.iterdir()):
        if p.is_dir() and any(p.glob("variation*.h5")):
            tasks.append(p.name)
    return tasks


def build_store(
    cache_root: Path,
    tasks: Sequence[str],
    keep_open_per_worker: bool,
) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    selected_tasks = list(tasks) if tasks else discover_cached_tasks(cache_root)
    if not selected_tasks:
        raise RuntimeError(f"No tasks found in cache root: {cache_root}")

    keys = []
    missing_tasks = []
    for task in selected_tasks:
        task_keys = build_variation_keys(cache_root, task)
        if not task_keys:
            missing_tasks.append(task)
            continue
        keys.extend(task_keys)

    if missing_tasks:
        missing_csv = ", ".join(sorted(missing_tasks))
        raise RuntimeError(f"No variation*.h5 files found for tasks: {missing_csv}")
    if not keys:
        raise RuntimeError(f"No variation*.h5 files found under {cache_root}")

    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected_tasks


def infer_dims(store: VariationStore) -> Tuple[int, int]:
    for vidx in range(len(store)):
        episode_ids = store.list_episode_ids(vidx)
        if episode_ids.shape[0] == 0:
            continue
        episode_id = int(episode_ids[0])
        T = int(store.episode_length(vidx, episode_id))
        if T <= 0:
            continue
        sample = store.load_episode_slices(
            vidx=vidx,
            episode_id=episode_id,
            t_idx=np.asarray([0], dtype=np.int64),
            load_rgb=False,
            load_mask_id=False,
        )
        state_dim = int(sample["state"].shape[-1])
        action_dim = int(sample["action"].shape[-1])
        return state_dim, action_dim
    raise RuntimeError("Could not infer state/action dims from cache (no non-empty episodes found).")


def build_model_cfg(cfg: ConfigDict) -> PolicyBuilderConfig:
    return build_policy_builder_config_from_configdict(cfg, as_bool=as_bool)


def resolve_use_mask_id(train_model_cfg: ConfigDict) -> bool:
    encoder_name = str(getattr(train_model_cfg, "encoder_name", "perceiver_demo_query"))
    if encoder_name == "traj_perceiver_v2":
        return as_bool(getattr(getattr(train_model_cfg, "traj_perceiver_v2", ConfigDict()), "use_mask_id", True))
    if encoder_name == "perceiver_demo_query_v2":
        return as_bool(getattr(getattr(train_model_cfg, "perceiver_demo_query_v2", ConfigDict()), "use_mask_id", True))
    if encoder_name == "traj_perceiver":
        return as_bool(getattr(getattr(train_model_cfg, "traj_perceiver", ConfigDict()), "use_mask_id", True))
    if encoder_name == "traj_conv3d":
        return as_bool(getattr(getattr(train_model_cfg, "traj_conv3d", ConfigDict()), "use_mask_id", True))
    if encoder_name == "conv3d_demo_query":
        return as_bool(getattr(getattr(train_model_cfg, "conv3d_demo_query", ConfigDict()), "use_mask_id", True))
    return as_bool(
        getattr(getattr(train_model_cfg, "perceiver_demo_query", ConfigDict()), "use_mask_id", True)
    )


def resolve_trace_path(profile_cfg: ConfigDict, *, rank: Optional[int] = None) -> Path:
    output_dir = Path(str(profile_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_filename = str(profile_cfg.trace_filename)
    if rank is None:
        return output_dir / trace_filename
    trace_path = output_dir / trace_filename
    return trace_path.with_name(f"{trace_path.stem}_rank{rank}{trace_path.suffix}")


def build_profiler_activities(device: torch.device, profile_cfg: ConfigDict) -> List[ProfilerActivity]:
    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda" and as_bool(getattr(profile_cfg, "trace_cuda", True)):
        activities.append(ProfilerActivity.CUDA)
    return activities


def export_memory_timeline_artifacts(
    prof: Any,
    *,
    trace_path: Path,
    device: torch.device,
    cuda_device_index: Optional[int] = None,
) -> None:
    if device.type == "cuda":
        resolved_cuda_index = cuda_device_index
        if resolved_cuda_index is None:
            resolved_cuda_index = device.index
        if resolved_cuda_index is None:
            resolved_cuda_index = torch.cuda.current_device()
        memory_device = f"cuda:{int(resolved_cuda_index)}"
    else:
        memory_device = "cpu"
    memory_json_path = trace_path.with_suffix(".memory.json")
    memory_html_path = trace_path.with_suffix(".memory.html")
    memory_png_path = trace_path.with_suffix(".memory.png")
    memory_json_exported = False

    try:
        prof.export_memory_timeline(str(memory_json_path), device=memory_device)
        memory_json_exported = True
    except Exception as exc:  # pragma: no cover
        print(f"[profile] warning: failed to export memory JSON timeline: {exc}")

    try:
        prof.export_memory_timeline(str(memory_html_path), device=memory_device)
    except Exception as exc:  # pragma: no cover
        print(f"[profile] warning: failed to export memory HTML timeline: {exc}")

    if memory_json_exported:
        try:
            render_memory_timeline_png(memory_json_path, memory_png_path)
        except Exception as exc:  # pragma: no cover
            print(f"[profile] warning: failed to render memory PNG timeline: {exc}")


def render_memory_timeline_png(memory_json_path: Path, output_png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render memory timeline PNG.") from exc

    with memory_json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("Unexpected memory timeline JSON format; expected [times, sizes].")

    times = np.asarray(payload[0], dtype=np.float64)
    sizes = np.asarray(payload[1], dtype=np.float64)
    if times.ndim != 1 or sizes.ndim != 2:
        raise ValueError("Invalid memory timeline shapes; expected times[1D], sizes[2D].")
    if sizes.shape[0] != times.shape[0]:
        raise ValueError("Mismatched memory timeline lengths between times and sizes.")
    if times.shape[0] == 0 or sizes.shape[1] < 2:
        raise ValueError("Memory timeline is empty or missing category columns.")

    times_ms = (times - times[0]) / 1e3
    sizes_gib = sizes / float(1024 ** 3)

    labels = [
        "PARAMETER",
        "OPTIMIZER_STATE",
        "INPUT",
        "TEMPORARY",
        "ACTIVATION",
        "GRADIENT",
        "AUTOGRAD_DETAIL",
        "UNKNOWN",
    ]
    n_categories = min(len(labels), int(sizes_gib.shape[1] - 1))
    layers = [sizes_gib[:, idx + 1] for idx in range(n_categories)]

    fig = plt.figure(figsize=(14, 6), dpi=120)
    plt.stackplot(times_ms, *layers, labels=labels[:n_categories], alpha=0.85)
    plt.xlabel("Time (ms)")
    plt.ylabel("Memory (GiB)")
    plt.title(memory_json_path.name)
    plt.legend(loc="upper left", ncol=4, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_png_path)
    plt.close(fig)
