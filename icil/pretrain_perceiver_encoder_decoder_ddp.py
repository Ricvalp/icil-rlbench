from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

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

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/pretrain_perceiver_encoder_decoder.py",
    help_string="Path to a ml_collections config file.",
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
    # DataLoader uses batch_size=1 because dataset already yields pre-batched dicts.
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
        num_train_timesteps=int(cfg.num_train_timesteps),
        beta_start=float(getattr(cfg, "beta_start", 1e-4)),
        beta_end=float(getattr(cfg, "beta_end", 2e-2)),
        beta_schedule=str(getattr(cfg, "beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(getattr(cfg, "prediction_type", "v_prediction")),
        set_alpha_to_one=_as_bool(getattr(cfg, "set_alpha_to_one", True)),
        steps_offset=int(getattr(cfg, "steps_offset", 0)),
        num_inference_steps=(
            int(getattr(cfg, "num_inference_steps"))
            if getattr(cfg, "num_inference_steps", None) is not None
            else None
        ),
    )


def _save_checkpoint(
    ckpt_path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: ConfigDict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg.to_dict(),
        },
        ckpt_path,
    )


def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return total, trainable


def _maybe_init_wandb(cfg: ConfigDict, workdir: Path) -> Optional[Any]:
    if not hasattr(cfg, "wandb") or not _as_bool(cfg.wandb.enable):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("cfg.wandb.enable=True but wandb is not installed.") from exc

    tags = list(cfg.wandb.tags) if getattr(cfg.wandb, "tags", None) else None
    project = str(cfg.wandb.project)
    entity = str(cfg.wandb.entity) if str(cfg.wandb.entity) else None
    name = str(cfg.wandb.name) if str(cfg.wandb.name) else None
    group = str(cfg.wandb.group) if str(cfg.wandb.group) else None
    mode = str(cfg.wandb.mode) if str(cfg.wandb.mode) else None

    config_dict = cfg.to_dict()
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        mode=mode,
        dir=str(workdir),
        config=config_dict,
        tags=tags,
    )
    return run


def _resolve_run_id(wandb_run: Optional[Any]) -> str:
    if wandb_run is not None:
        return str(wandb_run.id)
    return time.strftime("local-%Y%m%d-%H%M%S")


def _plot_pred_vs_gt_3d(
    pred_x0: torch.Tensor,
    gt_x0: torch.Tensor,
    max_items: int = 4,
    *,
    include_query_pointcloud: bool = False,
    query_xyz: Optional[torch.Tensor] = None,
    query_valid: Optional[torch.Tensor] = None,
    max_query_points: int = 2048,
) -> Optional[Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    pred = pred_x0.detach().float().cpu().numpy()
    gt = gt_x0.detach().float().cpu().numpy()
    if pred.ndim != 3 or gt.ndim != 3:
        return None
    qxyz = query_xyz.detach().float().cpu().numpy() if query_xyz is not None else None
    qvalid = query_valid.detach().bool().cpu().numpy() if query_valid is not None else None

    B, H, A = pred.shape
    n = int(max(1, min(B, max_items)))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    def _xyz(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = arr[:, 0] if A >= 1 else np.zeros((H,), dtype=np.float32)
        y = arr[:, 1] if A >= 2 else np.zeros((H,), dtype=np.float32)
        z = arr[:, 2] if A >= 3 else np.zeros((H,), dtype=np.float32)
        return x, y, z

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        if include_query_pointcloud and qxyz is not None:
            # Use last observed query frame as environment context.
            if qxyz.ndim == 4:  # [B, T_obs, N, 3]
                pts = qxyz[i, -1]
                mask = (
                    qvalid[i, -1]
                    if (qvalid is not None and qvalid.ndim == 3)
                    else np.ones((pts.shape[0],), dtype=bool)
                )
            elif qxyz.ndim == 3:  # [B, N, 3]
                pts = qxyz[i]
                mask = (
                    qvalid[i]
                    if (qvalid is not None and qvalid.ndim == 2)
                    else np.ones((pts.shape[0],), dtype=bool)
                )
            else:
                pts = None
                mask = None
            if pts is not None:
                pts = pts[mask]
                if pts.shape[0] > 0:
                    pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.shape[0] > int(max_query_points) and int(max_query_points) > 0:
                    idx = np.linspace(0, pts.shape[0] - 1, int(max_query_points), dtype=np.int64)
                    pts = pts[idx]
                if pts.shape[0] > 0:
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color="lightgray",
                        s=1.5,
                        alpha=0.35,
                        label="query_pc" if i == 0 else None,
                    )

        gx, gy, gz = _xyz(gt[i])
        px, py, pz = _xyz(pred[i])
        ax.plot(gx, gy, gz, color="tab:green", linewidth=2.0, label="gt")
        ax.plot(px, py, pz, color="tab:orange", linewidth=2.0, linestyle="--", label="pred")
        ax.scatter(gx[0], gy[0], gz[0], color="tab:green", s=18)
        ax.scatter(px[0], py[0], pz[0], color="tab:orange", s=18)
        ax.set_title(f"sample {i}")
        if i == 0:
            ax.legend(loc="upper right")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.tight_layout()
    return fig


def _plot_denoising_trace_3d(
    x0_trace: torch.Tensor,
    timesteps: torch.Tensor,
    max_items: int = 2,
) -> Optional[Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    tr = x0_trace.detach().float().cpu().numpy()
    ts = timesteps.detach().cpu().numpy().astype(np.int64)
    if tr.ndim != 4:
        return None

    S, B, H, A = tr.shape
    n = int(max(1, min(B, max_items)))
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    cmap = plt.get_cmap("viridis")

    def _xyz(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = arr[:, 0] if A >= 1 else np.zeros((H,), dtype=np.float32)
        y = arr[:, 1] if A >= 2 else np.zeros((H,), dtype=np.float32)
        z = arr[:, 2] if A >= 3 else np.zeros((H,), dtype=np.float32)
        return x, y, z

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        for s in range(S):
            x, y, z = _xyz(tr[s, i])
            color = cmap(float(s) / float(max(1, S - 1)))
            alpha = 1.0 if s == (S - 1) else 0.25
            lw = 2.2 if s == (S - 1) else 1.0
            label = None
            if s == 0:
                label = f"start t={int(ts[s])}"
            elif s == (S - 1):
                label = f"final t={int(ts[s])}"
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=lw, label=label)
        ax.set_title(f"sample {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if i == 0:
            ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def _init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return distributed, rank, world_size, local_rank


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_string(value: str, src: int = 0) -> str:
    if not dist.is_initialized():
        return value
    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=src)
    return str(obj_list[0])


def _distributed_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(dist.get_world_size())
    return float(tensor.item())


class _PolicyLossWrapper(nn.Module):
    def __init__(self, policy: ICILPerceiverDiffusionPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.policy.forward_loss(batch)


def train(cfg: ConfigDict) -> None:
    distributed, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0
    store: Optional[VariationStore] = None
    wandb_run = None

    try:
        base_seed = int(cfg.seed)
        # Keep model init deterministic across ranks; data sampling seed is rank-shifted below.
        _set_seed(base_seed)

        if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
            if distributed:
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device(str(cfg.device))
        else:
            device = torch.device("cpu")

        if not is_main:
            logging.set_verbosity(logging.ERROR)

        cache_root = Path(str(cfg.data.cache_root))
        tasks: List[str] = list(cfg.data.tasks) if cfg.data.tasks is not None else []
        store, tasks_used = _build_store(
            cache_root=cache_root,
            tasks=tasks,
            keep_open_per_worker=_as_bool(cfg.data.keep_open_per_worker),
        )

        output_parent = Path(
            str(getattr(cfg, "output_parent_dir", getattr(cfg, "workdir", "output_data_playground_v3/.experiments")))
        )
        output_parent.mkdir(parents=True, exist_ok=True)

        if is_main:
            wandb_run = _maybe_init_wandb(cfg, output_parent)
        run_id = _resolve_run_id(wandb_run) if is_main else ""
        if distributed:
            run_id = _broadcast_string(run_id, src=0)
        if is_main and wandb_run is not None:
            # Always use run id as run name for deterministic checkpoint/output mapping.
            wandb_run.name = run_id

        workdir = output_parent / run_id
        workdir.mkdir(parents=True, exist_ok=True)

        ckpt_parent = Path(
            str(
                getattr(
                    cfg.train,
                    "checkpoint_parent_dir",
                    workdir.parent / "checkpoints",
                )
            )
        )
        checkpoint_dir = ckpt_parent / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if distributed:
            dist.barrier()

        if is_main:
            config_payload = cfg.to_dict()
            config_payload["runtime"] = {
                "run_id": run_id,
                "output_dir": str(workdir),
                "checkpoint_dir": str(checkpoint_dir),
                "distributed": bool(distributed),
                "world_size": int(world_size),
            }
            config_path = workdir / "config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=2)

            logging.info("Run id=%s", run_id)
            logging.info("Output dir=%s", workdir)
            logging.info("Checkpoint dir=%s", checkpoint_dir)
            logging.info("DDP=%s | world_size=%d", distributed, world_size)
            if wandb_run is not None:
                wandb_run.config.update(
                    {
                        "runtime": {
                            "run_id": run_id,
                            "output_dir": str(workdir),
                            "checkpoint_dir": str(checkpoint_dir),
                            "distributed": bool(distributed),
                            "world_size": int(world_size),
                        }
                    },
                    allow_val_change=True,
                )
                # Log exact resolved config file for reproducibility.
                wandb_run.save(str(config_path), policy="now")

        state_dim, action_dim = _infer_dims(store)
        if is_main:
            logging.info("Using cache_root=%s", cache_root)
            logging.info("Tasks=%s | variations=%d", tasks_used, len(store))
            logging.info("Inferred dims: state_dim=%d, action_dim=%d", state_dim, action_dim)

        dataset_cfg = ICILConfig(
            K=int(cfg.dataset.K),
            L=int(cfg.dataset.L),
            T_obs=int(cfg.dataset.T_obs),
            H=int(cfg.dataset.H),
            stride=int(cfg.dataset.stride),
        )

        train_steps = int(cfg.train.num_steps)
        grad_accum = int(cfg.train.grad_accum_steps)
        if grad_accum < 1:
            raise ValueError("cfg.train.grad_accum_steps must be >= 1.")
        total_micro_batches = train_steps * grad_accum

        pretrain_dataset = ICILPretrainBatchIterable(
            store=store,
            cfg=dataset_cfg,
            batch_size_B=int(cfg.train.batch_size),
            num_batches=total_micro_batches,
            seed=base_seed + rank * 1000003,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )

        num_workers = int(cfg.data.num_workers)
        pin_memory = _as_bool(cfg.data.pin_memory) and device.type == "cuda"
        persistent_workers = _as_bool(cfg.data.persistent_workers) and num_workers > 0
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=1,
            collate_fn=_unwrap_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        policy = ICILPerceiverDiffusionPolicy(
            cfg=_build_model_cfg(cfg.model),
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        wrapped = _PolicyLossWrapper(policy).to(device)
        if distributed:
            ddp_model = DDP(
                wrapped,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        else:
            ddp_model = wrapped

        policy_for_io = ddp_model.module.policy if distributed else ddp_model.policy

        n_total, n_trainable = _count_parameters(policy_for_io)
        if is_main:
            print(
                f"Model params: total={n_total:,} ({n_total / 1e6:.3f}M) | "
                f"trainable={n_trainable:,} ({n_trainable / 1e6:.3f}M)"
            )
            logging.info(
                "Model params: total=%s (%.3fM) | trainable=%s (%.3fM)",
                f"{n_total:,}",
                n_total / 1e6,
                f"{n_trainable:,}",
                n_trainable / 1e6,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "model/num_params_total": n_total,
                        "model/num_params_trainable": n_trainable,
                    },
                    step=0,
                )

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=float(cfg.train.lr),
            betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
            weight_decay=float(cfg.train.weight_decay),
        )

        use_amp = _as_bool(cfg.train.use_amp) and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        grad_clip_norm = float(cfg.train.grad_clip_norm)
        log_every = int(cfg.train.log_every)
        ckpt_every = int(cfg.train.ckpt_every)
        wandb_loss_every = int(getattr(cfg.wandb, "n_loss_steps", 0)) if wandb_run is not None else 0
        wandb_sample_every = int(getattr(cfg.wandb, "n_sample_steps", 0)) if wandb_run is not None else 0
        wandb_sample_batch = int(getattr(cfg.wandb, "sample_batch_items", 4)) if wandb_run is not None else 0
        wandb_sample_inference_steps = (
            int(getattr(cfg.wandb, "sample_inference_steps", 0)) if wandb_run is not None else 0
        )
        wandb_sample_eta = float(getattr(cfg.wandb, "sample_eta", 0.0)) if wandb_run is not None else 0.0
        wandb_sample_trace_frames = int(getattr(cfg.wandb, "sample_trace_frames", 8)) if wandb_run is not None else 0
        wandb_include_query_pc = (
            _as_bool(getattr(cfg.wandb, "include_query_pointcloud_in_x0_pred_vs_gt_3d", False))
            if wandb_run is not None
            else False
        )
        wandb_query_pc_max_points = (
            int(getattr(cfg.wandb, "query_pointcloud_max_points", 2048))
            if wandb_run is not None
            else 2048
        )
        use_mask_id = _as_bool(getattr(cfg.model, "use_mask_id", True))

        step = 0
        resume_path = str(cfg.train.resume_path) if cfg.train.resume_path is not None else ""
        if resume_path:
            ckpt = torch.load(resume_path, map_location=device)
            policy_for_io.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            step = int(ckpt["step"])
            if is_main:
                logging.info("Resumed from %s at step=%d", resume_path, step)

        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)

        log_loss = 0.0
        log_mse = 0.0
        log_count = 0
        window_start = time.time()
        micro_count = 0
        micro_loss_sum = 0.0
        micro_mse_sum = 0.0
        wb_loss_sum = 0.0
        wb_mse_sum = 0.0
        wb_count = 0

        for batch in pretrain_loader:
            if step >= train_steps:
                break

            batch = _to_device(batch, device)
            model_batch = _drop_mask_ids_if_disabled(batch, use_mask_id)
            sync_this_micro = ((micro_count + 1) % grad_accum == 0)
            sync_context = (
                nullcontext()
                if (not distributed or sync_this_micro)
                else ddp_model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = ddp_model(model_batch)
                    loss = out["loss"] / grad_accum
                scaler.scale(loss).backward()

            micro_count += 1
            micro_loss_sum += float(out["loss"].detach().cpu())
            micro_mse_sum += float(out["mse"].detach().cpu())

            if micro_count % grad_accum != 0:
                continue

            # Average local micro-batch losses to one optimizer-step scalar.
            step_loss_local = micro_loss_sum / float(grad_accum)
            step_mse_local = micro_mse_sum / float(grad_accum)
            micro_loss_sum = 0.0
            micro_mse_sum = 0.0

            # Aggregate across ranks for logging.
            step_loss = _distributed_mean(step_loss_local, device)
            step_mse = _distributed_mean(step_mse_local, device)

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main:
                log_loss += step_loss
                log_mse += step_mse
                log_count += 1
                wb_loss_sum += step_loss
                wb_mse_sum += step_mse
                wb_count += 1

                if log_every > 0 and (step % log_every == 0 or step == 1):
                    elapsed = max(1e-6, time.time() - window_start)
                    steps_per_sec = log_count / elapsed
                    avg_loss = log_loss / max(1, log_count)
                    avg_mse = log_mse / max(1, log_count)
                    lr = optimizer.param_groups[0]["lr"]
                    logging.info(
                        "step %d/%d | loss %.6f | mse %.6f | lr %.3e | %.2f step/s",
                        step,
                        train_steps,
                        avg_loss,
                        avg_mse,
                        lr,
                        steps_per_sec,
                    )
                    log_loss = 0.0
                    log_mse = 0.0
                    log_count = 0
                    window_start = time.time()

                if wandb_run is not None and wandb_loss_every > 0 and (step % wandb_loss_every == 0 or step == 1):
                    wandb_run.log(
                        {
                            "train/loss": wb_loss_sum / max(1, wb_count),
                            "train/mse": wb_mse_sum / max(1, wb_count),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/step": step,
                        },
                        step=step,
                    )
                    wb_loss_sum = 0.0
                    wb_mse_sum = 0.0
                    wb_count = 0

                if wandb_run is not None and wandb_sample_every > 0 and (step % wandb_sample_every == 0):
                    was_training = ddp_model.training
                    ddp_model.eval()
                    with torch.no_grad():
                        sample_out = policy_for_io.sample_actions(
                            cond_xyz=batch["cond_xyz"],
                            cond_state=batch["cond_state"],
                            query_xyz=batch["query_xyz"],
                            query_state=batch["query_state"],
                            action_horizon=int(batch["target_action"].shape[1]),
                            cond_rgb=batch.get("cond_rgb", None),
                            query_rgb=batch.get("query_rgb", None),
                            cond_mask_id=(batch.get("cond_mask_id", None) if use_mask_id else None),
                            query_mask_id=(batch.get("query_mask_id", None) if use_mask_id else None),
                            cond_valid=batch.get("cond_valid", None),
                            query_valid=batch.get("query_valid", None),
                            inference_steps=(
                                wandb_sample_inference_steps if wandb_sample_inference_steps > 0 else None
                            ),
                            eta=wandb_sample_eta,
                            return_trace=True,
                            trace_steps=(wandb_sample_trace_frames if wandb_sample_trace_frames > 0 else None),
                        )
                        if isinstance(sample_out, tuple):
                            pred_x0, denoise_trace = sample_out
                        else:
                            pred_x0, denoise_trace = sample_out, None
                    if was_training:
                        ddp_model.train()

                    sample_mse = float(F.mse_loss(pred_x0, batch["target_action"]).detach().cpu())
                    fig = _plot_pred_vs_gt_3d(
                        pred_x0=pred_x0,
                        gt_x0=batch["target_action"],
                        max_items=wandb_sample_batch,
                        include_query_pointcloud=wandb_include_query_pc,
                        query_xyz=batch.get("query_xyz", None),
                        query_valid=batch.get("query_valid", None),
                        max_query_points=wandb_query_pc_max_points,
                    )
                    fig_trace = None
                    if denoise_trace is not None:
                        fig_trace = _plot_denoising_trace_3d(
                            denoise_trace["x0_hat"],
                            denoise_trace["timesteps"],
                            max_items=max(1, min(2, wandb_sample_batch)),
                        )
                    log_dict: Dict[str, Any] = {
                        "samples/x0_mse": sample_mse,
                        "train/step": step,
                    }
                    if fig is not None or fig_trace is not None:
                        import wandb

                        if fig is not None:
                            log_dict["samples/x0_pred_vs_gt_3d"] = wandb.Image(fig)
                        if fig_trace is not None:
                            log_dict["samples/x0_denoising_trace_3d"] = wandb.Image(fig_trace)
                    wandb_run.log(log_dict, step=step)
                    if fig is not None or fig_trace is not None:
                        try:
                            import matplotlib.pyplot as plt

                            if fig is not None:
                                plt.close(fig)
                            if fig_trace is not None:
                                plt.close(fig_trace)
                        except Exception:
                            pass

                if ckpt_every > 0 and step % ckpt_every == 0:
                    ckpt_path = checkpoint_dir / f"step_{step:07d}.pt"
                    _save_checkpoint(
                        ckpt_path,
                        step=step,
                        model=policy_for_io,
                        optimizer=optimizer,
                        scaler=scaler,
                        cfg=cfg,
                    )
                    logging.info("Saved checkpoint: %s", ckpt_path)

        if is_main:
            final_ckpt = checkpoint_dir / "last.pt"
            _save_checkpoint(
                final_ckpt,
                step=step,
                model=policy_for_io,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
            )
            logging.info("Training complete. Final checkpoint: %s", final_ckpt)

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if store is not None:
            store.close()
        _cleanup_distributed()


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == "__main__":
    app.run(main)
