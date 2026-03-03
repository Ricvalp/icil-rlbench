#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from absl import app
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from icil.datasets.in_context_imitation_learning.icil_datasets import (
    ICILConfig,
    ICILPretrainBatchIterable,
)
from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)
from icil.models.perceiver_encoder_decoder import (
    ICILPerceiverDiffusionPolicy,
    ModelConfig,
)

_TRAIN_CONFIG = config_flags.DEFINE_config_file(
    "train_config",
    default="configs/pretrain_perceiver_encoder_decoder.py",
    help_string="Path to the base pretraining ml_collections config file.",
)

_PROFILE_CONFIG = config_flags.DEFINE_config_file(
    "profile_config",
    default="configs/profile_pretrain_perceiver_encoder_decoder.py",
    help_string="Path to the profiling ml_collections config file.",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_bool(v: Any) -> bool:
    return bool(v)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _unwrap_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    # DataLoader uses batch_size=1 because dataset already yields batched dicts.
    return batch_list[0]


def _drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop("cond_mask_id", None)
    out.pop("query_mask_id", None)
    return out


def _discover_cached_tasks(cache_root: Path) -> List[str]:
    tasks: List[str] = []
    if not cache_root.is_dir():
        return tasks
    for p in sorted(cache_root.iterdir()):
        if p.is_dir() and any(p.glob("variation*.h5")):
            tasks.append(p.name)
    return tasks


def _build_store(cache_root: Path, tasks: Sequence[str], keep_open_per_worker: bool) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    selected_tasks = list(tasks) if tasks else _discover_cached_tasks(cache_root)
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


def _infer_dims(store: VariationStore) -> Tuple[int, int]:
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


def _build_model_cfg(cfg: ConfigDict) -> ModelConfig:
    return ModelConfig(
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        m_frame_tokens=int(cfg.m_frame_tokens),
        frame_tokenizer_layers=int(cfg.frame_tokenizer_layers),
        M_demo_latents=int(cfg.M_demo_latents),
        demo_perceiver_layers=int(cfg.demo_perceiver_layers),
        ignore_demos=_as_bool(getattr(cfg, "ignore_demos", False)),
        denoiser_layers=int(cfg.denoiser_layers),
        denoiser_mlp_mult=int(cfg.denoiser_mlp_mult),
        dropout=float(cfg.dropout),
        mask_hash_buckets=int(cfg.mask_hash_buckets),
        use_mask_id=_as_bool(getattr(cfg, "use_mask_id", True)),
        role_embed_max_K=int(cfg.role_embed_max_K),
        role_embed_max_L=int(cfg.role_embed_max_L),
        role_embed_max_Tobs=int(cfg.role_embed_max_Tobs),
        rgb_alpha_init=float(getattr(cfg, "rgb_alpha_init", 1.0)),
        diffusion_T=int(cfg.diffusion_T),
    )


def _resolve_trace_path(profile_cfg: ConfigDict) -> Path:
    output_dir = Path(str(profile_cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_filename = str(profile_cfg.trace_filename)
    return output_dir / trace_filename


def _build_profiler_activities(device: torch.device, profile_cfg: ConfigDict) -> List[ProfilerActivity]:
    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda" and _as_bool(getattr(profile_cfg, "trace_cuda", True)):
        activities.append(ProfilerActivity.CUDA)
    return activities


def _render_memory_timeline_png(memory_json_path: Path, output_png_path: Path) -> None:
    """
    Render torch profiler memory timeline JSON ([times, sizes]) to a PNG.
    """
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

    # Timestamps are in microseconds in torch profiler memory timeline.
    times_ms = (times - times[0]) / 1e3
    sizes_gib = sizes / float(1024 ** 3)

    # sizes[:,0] is baseline; categories start at column 1.
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


def profile_train(train_cfg: ConfigDict, profile_cfg: ConfigDict) -> Path:
    seed = int(train_cfg.seed)
    _set_seed(seed)

    if torch.cuda.is_available() and str(train_cfg.device).startswith("cuda"):
        device = torch.device(str(train_cfg.device))
    else:
        device = torch.device("cpu")

    trace_n_steps = int(profile_cfg.trace_n_steps)
    if trace_n_steps < 1:
        raise ValueError("profile_config.trace_n_steps must be >= 1.")

    cache_root = Path(str(train_cfg.data.cache_root))
    tasks: List[str] = list(train_cfg.data.tasks) if train_cfg.data.tasks is not None else []
    store, _ = _build_store(
        cache_root=cache_root,
        tasks=tasks,
        keep_open_per_worker=_as_bool(train_cfg.data.keep_open_per_worker),
    )

    try:
        state_dim, action_dim = _infer_dims(store)

        dataset_cfg = ICILConfig(
            K=int(train_cfg.dataset.K),
            L=int(train_cfg.dataset.L),
            T_obs=int(train_cfg.dataset.T_obs),
            H=int(train_cfg.dataset.H),
            stride=int(train_cfg.dataset.stride),
        )

        train_steps = int(train_cfg.train.num_steps)
        grad_accum = int(train_cfg.train.grad_accum_steps)
        if grad_accum < 1:
            raise ValueError("train_config.train.grad_accum_steps must be >= 1.")
        if trace_n_steps > train_steps:
            raise ValueError(
                f"profile trace_n_steps={trace_n_steps} exceeds configured train num_steps={train_steps}."
            )

        total_micro_batches = train_steps * grad_accum
        pretrain_dataset = ICILPretrainBatchIterable(
            store=store,
            cfg=dataset_cfg,
            batch_size_B=int(train_cfg.train.batch_size),
            num_batches=total_micro_batches,
            seed=seed,
            num_tries_per_item=int(train_cfg.dataset.num_tries_per_item),
        )

        num_workers = int(train_cfg.data.num_workers)
        pin_memory = _as_bool(train_cfg.data.pin_memory) and device.type == "cuda"
        persistent_workers = _as_bool(train_cfg.data.persistent_workers) and num_workers > 0
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=1,
            collate_fn=_unwrap_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model = ICILPerceiverDiffusionPolicy(
            cfg=_build_model_cfg(train_cfg.model),
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.train.lr),
            betas=(float(train_cfg.train.beta1), float(train_cfg.train.beta2)),
            weight_decay=float(train_cfg.train.weight_decay),
        )

        use_amp = _as_bool(train_cfg.train.use_amp) and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        grad_clip_norm = float(train_cfg.train.grad_clip_norm)
        use_mask_id = _as_bool(getattr(train_cfg.model, "use_mask_id", True))

        model.train()
        optimizer.zero_grad(set_to_none=True)

        step = 0
        micro_count = 0
        data_iter = iter(pretrain_loader)

        activities = _build_profiler_activities(device, profile_cfg)
        trace_path = _resolve_trace_path(profile_cfg)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with profile(
            activities=activities,
            record_shapes=_as_bool(getattr(profile_cfg, "record_shapes", True)),
            profile_memory=_as_bool(getattr(profile_cfg, "profile_memory", True)),
            with_stack=_as_bool(getattr(profile_cfg, "with_stack", False)),
            with_flops=_as_bool(getattr(profile_cfg, "with_flops", False)),
        ) as prof:
            pbar = tqdm(total=trace_n_steps, desc="Profiling train steps", dynamic_ncols=True)
            try:
                while step < trace_n_steps:
                    with record_function("dataloader.next"):
                        try:
                            batch = next(data_iter)
                        except StopIteration as exc:
                            raise RuntimeError(
                                "Dataloader exhausted before reaching trace_n_steps."
                            ) from exc

                    with record_function("batch.to_device"):
                        batch = _to_device(batch, device)
                        model_batch = _drop_mask_ids_if_disabled(batch, use_mask_id)

                    with record_function("train.forward"):
                        with torch.autocast(device_type=device.type, enabled=use_amp):
                            out = model.forward_loss(model_batch)
                            loss = out["loss"] / grad_accum

                    with record_function("train.backward"):
                        scaler.scale(loss).backward()
                    micro_count += 1

                    if micro_count % grad_accum == 0:
                        with record_function("train.optimizer_step"):
                            if grad_clip_norm > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                        step += 1
                        pbar.update(1)

                    prof.step()
            finally:
                pbar.close()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(trace_path))
        memory_device = (
            f"cuda:{torch.cuda.current_device()}"
            if device.type == "cuda"
            else "cpu"
        )
        # Optional memory timeline artifacts (viewable separately from Perfetto).
        # Export JSON and HTML independently so an HTML backend/runtime issue
        # does not prevent JSON export or fail the profiling run.
        memory_json_path = trace_path.with_suffix(".memory.json")
        memory_html_path = trace_path.with_suffix(".memory.html")
        memory_png_path = trace_path.with_suffix(".memory.png")
        memory_json_exported = False

        try:
            prof.export_memory_timeline(str(memory_json_path), device=memory_device)
            memory_json_exported = True
        except Exception as exc:  # pragma: no cover - best-effort artifact export
            print(f"[profile] warning: failed to export memory JSON timeline: {exc}")

        try:
            prof.export_memory_timeline(str(memory_html_path), device=memory_device)
        except Exception as exc:  # pragma: no cover - best-effort artifact export
            print(f"[profile] warning: failed to export memory HTML timeline: {exc}")

        if memory_json_exported:
            try:
                _render_memory_timeline_png(memory_json_path, memory_png_path)
            except Exception as exc:  # pragma: no cover - best-effort artifact export
                print(f"[profile] warning: failed to render memory PNG timeline: {exc}")

        return trace_path
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    del argv
    train_cfg = _TRAIN_CONFIG.value
    profile_cfg = _PROFILE_CONFIG.value
    profile_train(train_cfg, profile_cfg)


if __name__ == "__main__":
    app.run(main)
