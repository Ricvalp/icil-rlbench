from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ml_collections import ConfigDict

from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)
from icil.models import (
    Conv3dDemoQueryEncoderConfig,
    PerceiverDemoQueryEncoderConfig,
    PolicyBuilderConfig,
    PolicyConfig,
    TrajConv3DConfig,
    TrajPerceiverConfig,
)



def as_bool(value: Any) -> bool:
    return bool(value)



def normalize_task_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if str(value).strip()]



def discover_cached_tasks(cache_root: Path) -> List[str]:
    tasks: List[str] = []
    if not cache_root.is_dir():
        return tasks
    for path in sorted(cache_root.iterdir()):
        if path.is_dir() and any(path.glob('variation*.h5')):
            tasks.append(path.name)
    return tasks



def resolve_selected_tasks(
    cache_root: Path,
    tasks: Sequence[str],
    exclude_tasks: Sequence[str],
) -> List[str]:
    selected_tasks = list(tasks) if tasks else discover_cached_tasks(cache_root)
    exclude_set = set(exclude_tasks)
    if exclude_set:
        selected_tasks = [task for task in selected_tasks if task not in exclude_set]
    return selected_tasks



def build_store(
    cache_root: Path,
    tasks: Sequence[str],
    exclude_tasks: Sequence[str],
    keep_open_per_worker: bool,
) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f'Cache root not found: {cache_root}')

    selected_tasks = resolve_selected_tasks(cache_root, tasks, exclude_tasks)
    if not selected_tasks:
        raise RuntimeError(
            f'No tasks remain after applying exclude_tasks under cache root: {cache_root}'
        )

    keys = []
    missing_tasks = []
    for task in selected_tasks:
        task_keys = build_variation_keys(cache_root, task)
        if not task_keys:
            missing_tasks.append(task)
            continue
        keys.extend(task_keys)

    if missing_tasks:
        missing_csv = ', '.join(sorted(missing_tasks))
        raise RuntimeError(f'No variation*.h5 files found for tasks: {missing_csv}')
    if not keys:
        raise RuntimeError(f'No variation*.h5 files found under {cache_root}')

    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected_tasks



def build_optional_store(
    cache_root: Path,
    tasks: Sequence[str],
    keep_open_per_worker: bool,
) -> Tuple[Optional[VariationStore], List[str]]:
    normalized_tasks = normalize_task_list(tasks)
    if not normalized_tasks:
        return None, []
    try:
        return build_store(
            cache_root=cache_root,
            tasks=normalized_tasks,
            exclude_tasks=(),
            keep_open_per_worker=keep_open_per_worker,
        )
    except RuntimeError:
        return None, []



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
        state_dim = int(sample['state'].shape[-1])
        action_dim = int(sample['action'].shape[-1])
        return state_dim, action_dim
    raise RuntimeError('Could not infer state/action dims from cache (no non-empty episodes found).')



def build_model_cfg(cfg: ConfigDict) -> PolicyBuilderConfig:
    policy_cfg_raw = cfg.policy
    conv3d_cfg_raw = getattr(cfg, 'conv3d_demo_query', ConfigDict())
    perceiver_cfg_raw = cfg.perceiver_demo_query
    traj_conv3d_cfg_raw = getattr(cfg, 'traj_conv3d', ConfigDict())
    traj_cfg_raw = cfg.traj_perceiver

    policy_cfg = PolicyConfig(
        d_model=int(policy_cfg_raw.d_model),
        n_heads=int(policy_cfg_raw.n_heads),
        denoiser_layers=int(policy_cfg_raw.denoiser_layers),
        denoiser_mlp_mult=int(policy_cfg_raw.denoiser_mlp_mult),
        dropout=float(policy_cfg_raw.dropout),
        grad_checkpoint_dit=as_bool(getattr(policy_cfg_raw, 'grad_checkpoint_dit', False)),
        context_attention_mode=str(getattr(policy_cfg_raw, 'context_attention_mode', 'single')),
        num_train_timesteps=int(policy_cfg_raw.num_train_timesteps),
        beta_start=float(getattr(policy_cfg_raw, 'beta_start', 1e-4)),
        beta_end=float(getattr(policy_cfg_raw, 'beta_end', 2e-2)),
        beta_schedule=str(getattr(policy_cfg_raw, 'beta_schedule', 'squaredcos_cap_v2')),
        prediction_type=str(getattr(policy_cfg_raw, 'prediction_type', 'v_prediction')),
        set_alpha_to_one=as_bool(getattr(policy_cfg_raw, 'set_alpha_to_one', True)),
        steps_offset=int(getattr(policy_cfg_raw, 'steps_offset', 0)),
        num_inference_steps=(
            int(getattr(policy_cfg_raw, 'num_inference_steps'))
            if getattr(policy_cfg_raw, 'num_inference_steps', None) is not None
            else None
        ),
    )
    perceiver_cfg = PerceiverDemoQueryEncoderConfig(
        d_model=int(perceiver_cfg_raw.d_model),
        n_heads=int(perceiver_cfg_raw.n_heads),
        m_frame_tokens=int(perceiver_cfg_raw.m_frame_tokens),
        frame_tokenizer_layers=int(perceiver_cfg_raw.frame_tokenizer_layers),
        M_demo_latents=int(perceiver_cfg_raw.M_demo_latents),
        demo_perceiver_layers=int(perceiver_cfg_raw.demo_perceiver_layers),
        mask_hash_buckets=int(perceiver_cfg_raw.mask_hash_buckets),
        use_mask_id=as_bool(getattr(perceiver_cfg_raw, 'use_mask_id', True)),
        role_embed_max_K=int(perceiver_cfg_raw.role_embed_max_K),
        role_embed_max_L=int(perceiver_cfg_raw.role_embed_max_L),
        role_embed_max_Tobs=int(perceiver_cfg_raw.role_embed_max_Tobs),
        rgb_alpha_init=float(getattr(perceiver_cfg_raw, 'rgb_alpha_init', 1.0)),
        dropout=float(perceiver_cfg_raw.dropout),
        ignore_demos=as_bool(getattr(perceiver_cfg_raw, 'ignore_demos', False)),
        compress_demo_latents=as_bool(getattr(perceiver_cfg_raw, 'compress_demo_latents', True)),
        checkpoint_demo_memory=as_bool(getattr(perceiver_cfg_raw, 'checkpoint_demo_memory', False)),
        checkpoint_build_demo_memory=as_bool(
            getattr(perceiver_cfg_raw, 'checkpoint_build_demo_memory', False)
        ),
        checkpoint_frame_tokenizer=as_bool(
            getattr(perceiver_cfg_raw, 'checkpoint_frame_tokenizer', False)
        ),
        tokenize_frames_chunked=as_bool(
            getattr(perceiver_cfg_raw, 'tokenize_frames_chunked', False)
        ),
        chunk_frames=int(getattr(perceiver_cfg_raw, 'chunk_frames', 32)),
    )
    conv3d_cfg = Conv3dDemoQueryEncoderConfig(
        d_model=int(getattr(conv3d_cfg_raw, 'd_model', policy_cfg.d_model)),
        n_heads=int(getattr(conv3d_cfg_raw, 'n_heads', policy_cfg.n_heads)),
        m_frame_tokens=int(getattr(conv3d_cfg_raw, 'm_frame_tokens', 64)),
        max_voxels=int(getattr(conv3d_cfg_raw, 'max_voxels', 4096)),
        voxel_size=float(getattr(conv3d_cfg_raw, 'voxel_size', 0.01)),
        use_learned_topk=as_bool(getattr(conv3d_cfg_raw, 'use_learned_topk', True)),
        n_mix_layers=int(getattr(conv3d_cfg_raw, 'n_mix_layers', 2)),
        M_demo_latents=int(getattr(conv3d_cfg_raw, 'M_demo_latents', 256)),
        demo_perceiver_layers=int(getattr(conv3d_cfg_raw, 'demo_perceiver_layers', 3)),
        mask_hash_buckets=int(getattr(conv3d_cfg_raw, 'mask_hash_buckets', 2048)),
        use_mask_id=as_bool(getattr(conv3d_cfg_raw, 'use_mask_id', True)),
        role_embed_max_K=int(getattr(conv3d_cfg_raw, 'role_embed_max_K', 32)),
        role_embed_max_L=int(getattr(conv3d_cfg_raw, 'role_embed_max_L', 64)),
        role_embed_max_Tobs=int(getattr(conv3d_cfg_raw, 'role_embed_max_Tobs', 16)),
        rgb_alpha_init=float(getattr(conv3d_cfg_raw, 'rgb_alpha_init', 1.0)),
        dropout=float(getattr(conv3d_cfg_raw, 'dropout', 0.0)),
        ignore_demos=as_bool(getattr(conv3d_cfg_raw, 'ignore_demos', False)),
    )
    traj_cfg = TrajPerceiverConfig(
        d_model=int(getattr(traj_cfg_raw, 'd_model', policy_cfg.d_model)),
        n_heads=int(getattr(traj_cfg_raw, 'n_heads', policy_cfg.n_heads)),
        dropout=float(getattr(traj_cfg_raw, 'dropout', 0.0)),
        m_frame_tokens=int(getattr(traj_cfg_raw, 'm_frame_tokens', 64)),
        frame_tokenizer_layers=int(getattr(traj_cfg_raw, 'frame_tokenizer_layers', 2)),
        M_demo_latents=int(getattr(traj_cfg_raw, 'M_demo_latents', 256)),
        demo_perceiver_layers=int(getattr(traj_cfg_raw, 'demo_perceiver_layers', 3)),
        mask_hash_buckets=int(getattr(traj_cfg_raw, 'mask_hash_buckets', 2048)),
        use_mask_id=as_bool(getattr(traj_cfg_raw, 'use_mask_id', True)),
        role_embed_max_K=int(getattr(traj_cfg_raw, 'role_embed_max_K', 32)),
        role_embed_max_L=int(getattr(traj_cfg_raw, 'role_embed_max_L', 64)),
        role_embed_max_Tobs=int(getattr(traj_cfg_raw, 'role_embed_max_Tobs', 16)),
        rgb_alpha_init=float(getattr(traj_cfg_raw, 'rgb_alpha_init', 1.0)),
        ignore_demos=as_bool(getattr(traj_cfg_raw, 'ignore_demos', False)),
        compress_demo_latents=as_bool(getattr(traj_cfg_raw, 'compress_demo_latents', True)),
        checkpoint_demo_memory=as_bool(getattr(traj_cfg_raw, 'checkpoint_demo_memory', False)),
        checkpoint_build_demo_memory=as_bool(
            getattr(traj_cfg_raw, 'checkpoint_build_demo_memory', False)
        ),
        checkpoint_frame_tokenizer=as_bool(
            getattr(traj_cfg_raw, 'checkpoint_frame_tokenizer', False)
        ),
        tokenize_frames_chunked=as_bool(
            getattr(traj_cfg_raw, 'tokenize_frames_chunked', False)
        ),
        chunk_frames=int(getattr(traj_cfg_raw, 'chunk_frames', 32)),
        m_traj_tokens=int(getattr(traj_cfg_raw, 'm_traj_tokens', 16)),
        traj_perceiver_layers=int(getattr(traj_cfg_raw, 'traj_perceiver_layers', getattr(traj_cfg_raw, 'n_layers', 2))),
        traj_dim=int(getattr(traj_cfg_raw, 'traj_dim', 8)),
        use_demo_id_embed=as_bool(getattr(traj_cfg_raw, 'use_demo_id_embed', True)),
        include_traj_tokens=as_bool(getattr(traj_cfg_raw, 'include_traj_tokens', True)),
        use_cond_state_as_traj_fallback=as_bool(
            getattr(traj_cfg_raw, 'use_cond_state_as_traj_fallback', True)
        ),
    )
    traj_conv3d_cfg = TrajConv3DConfig(
        d_model=int(getattr(traj_conv3d_cfg_raw, 'd_model', policy_cfg.d_model)),
        n_heads=int(getattr(traj_conv3d_cfg_raw, 'n_heads', policy_cfg.n_heads)),
        dropout=float(getattr(traj_conv3d_cfg_raw, 'dropout', 0.0)),
        m_frame_tokens=int(getattr(traj_conv3d_cfg_raw, 'm_frame_tokens', 64)),
        n_mix_layers=int(getattr(traj_conv3d_cfg_raw, 'n_mix_layers', 2)),
        max_voxels=int(getattr(traj_conv3d_cfg_raw, 'max_voxels', 4096)),
        voxel_size=float(getattr(traj_conv3d_cfg_raw, 'voxel_size', 0.01)),
        use_learned_topk=as_bool(getattr(traj_conv3d_cfg_raw, 'use_learned_topk', True)),
        M_demo_latents=int(getattr(traj_conv3d_cfg_raw, 'M_demo_latents', 256)),
        demo_perceiver_layers=int(getattr(traj_conv3d_cfg_raw, 'demo_perceiver_layers', 3)),
        mask_hash_buckets=int(getattr(traj_conv3d_cfg_raw, 'mask_hash_buckets', 2048)),
        use_mask_id=as_bool(getattr(traj_conv3d_cfg_raw, 'use_mask_id', True)),
        role_embed_max_K=int(getattr(traj_conv3d_cfg_raw, 'role_embed_max_K', 32)),
        role_embed_max_L=int(getattr(traj_conv3d_cfg_raw, 'role_embed_max_L', 64)),
        role_embed_max_Tobs=int(getattr(traj_conv3d_cfg_raw, 'role_embed_max_Tobs', 16)),
        rgb_alpha_init=float(getattr(traj_conv3d_cfg_raw, 'rgb_alpha_init', 1.0)),
        ignore_demos=as_bool(getattr(traj_conv3d_cfg_raw, 'ignore_demos', False)),
        m_traj_tokens=int(getattr(traj_conv3d_cfg_raw, 'm_traj_tokens', 16)),
        traj_perceiver_layers=int(getattr(traj_conv3d_cfg_raw, 'traj_perceiver_layers', 2)),
        traj_dim=int(getattr(traj_conv3d_cfg_raw, 'traj_dim', 8)),
        use_demo_id_embed=as_bool(getattr(traj_conv3d_cfg_raw, 'use_demo_id_embed', True)),
        include_traj_tokens=as_bool(getattr(traj_conv3d_cfg_raw, 'include_traj_tokens', True)),
        use_cond_state_as_traj_fallback=as_bool(
            getattr(traj_conv3d_cfg_raw, 'use_cond_state_as_traj_fallback', True)
        ),
    )
    return PolicyBuilderConfig(
        policy=policy_cfg,
        encoder_name=str(cfg.encoder_name),
        conv3d_demo_query=conv3d_cfg,
        perceiver_demo_query=perceiver_cfg,
        traj_conv3d=traj_conv3d_cfg,
        traj_perceiver=traj_cfg,
    )



def resolve_use_mask_id(model_cfg: ConfigDict) -> bool:
    encoder_name = str(getattr(model_cfg, 'encoder_name', 'perceiver_demo_query'))
    if encoder_name == 'traj_perceiver':
        return as_bool(getattr(getattr(model_cfg, 'traj_perceiver', ConfigDict()), 'use_mask_id', True))
    if encoder_name == 'traj_conv3d':
        return as_bool(getattr(getattr(model_cfg, 'traj_conv3d', ConfigDict()), 'use_mask_id', True))
    if encoder_name == 'conv3d_demo_query':
        return as_bool(getattr(getattr(model_cfg, 'conv3d_demo_query', ConfigDict()), 'use_mask_id', True))
    return as_bool(getattr(getattr(model_cfg, 'perceiver_demo_query', ConfigDict()), 'use_mask_id', True))



def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return total, trainable



def maybe_init_wandb(cfg: ConfigDict, workdir: Path) -> Optional[Any]:
    if not hasattr(cfg, 'wandb') or not as_bool(cfg.wandb.enable):
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

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        mode=mode,
        dir=str(workdir),
        config=cfg.to_dict(),
        tags=tags,
    )
    return run



def resolve_run_id(wandb_run: Optional[Any]) -> str:
    if wandb_run is not None:
        return str(wandb_run.id)
    return time.strftime('local-%Y%m%d-%H%M%S')



def plot_pred_vs_gt_3d(
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

    def xyz(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = arr[:, 0] if A >= 1 else np.zeros((H,), dtype=np.float32)
        y = arr[:, 1] if A >= 2 else np.zeros((H,), dtype=np.float32)
        z = arr[:, 2] if A >= 3 else np.zeros((H,), dtype=np.float32)
        return x, y, z

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        query_pts_for_limits: Optional[np.ndarray] = None
        if include_query_pointcloud and qxyz is not None:
            if qxyz.ndim == 4:
                pts = qxyz[i, -1]
                mask = qvalid[i, -1] if (qvalid is not None and qvalid.ndim == 3) else np.ones((pts.shape[0],), dtype=bool)
            elif qxyz.ndim == 3:
                pts = qxyz[i]
                mask = qvalid[i] if (qvalid is not None and qvalid.ndim == 2) else np.ones((pts.shape[0],), dtype=bool)
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
                    query_pts_for_limits = pts
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color='lightgray',
                        s=1.5,
                        alpha=0.35,
                        label='query_pc' if i == 0 else None,
                    )

        gx, gy, gz = xyz(gt[i])
        px, py, pz = xyz(pred[i])
        ax.plot(gx, gy, gz, color='tab:green', linewidth=2.0, label='gt')
        ax.plot(px, py, pz, color='tab:orange', linewidth=2.0, linestyle='--', label='pred')
        ax.scatter(gx[0], gy[0], gz[0], color='tab:green', s=18)
        ax.scatter(px[0], py[0], pz[0], color='tab:orange', s=18)
        ax.set_title(f'sample {i}')
        if i == 0:
            ax.legend(loc='upper right')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        all_pts = [
            np.stack([gx, gy, gz], axis=1),
            np.stack([px, py, pz], axis=1),
        ]
        if query_pts_for_limits is not None:
            all_pts.append(query_pts_for_limits[:, :3])
        pts_all = np.concatenate(all_pts, axis=0)
        mins = pts_all.min(axis=0)
        maxs = pts_all.max(axis=0)
        center = 0.5 * (mins + maxs)
        half_range = 0.5 * float(np.max(maxs - mins))
        if half_range < 1e-6:
            half_range = 1e-3
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)

    fig.tight_layout()
    return fig



def plot_denoising_trace_3d(
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
    cmap = plt.get_cmap('viridis')

    def xyz(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = arr[:, 0] if A >= 1 else np.zeros((H,), dtype=np.float32)
        y = arr[:, 1] if A >= 2 else np.zeros((H,), dtype=np.float32)
        z = arr[:, 2] if A >= 3 else np.zeros((H,), dtype=np.float32)
        return x, y, z

    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        for s in range(S):
            x, y, z = xyz(tr[s, i])
            color = cmap(float(s) / float(max(1, S - 1)))
            alpha = 1.0 if s == (S - 1) else 0.25
            lw = 2.2 if s == (S - 1) else 1.0
            label = None
            if s == 0:
                label = f'start t={int(ts[s])}'
            elif s == (S - 1):
                label = f'final t={int(ts[s])}'
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=lw, label=label)
        ax.set_title(f'sample {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if i == 0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    return fig
