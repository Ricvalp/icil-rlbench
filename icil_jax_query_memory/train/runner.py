from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax import jax_utils
from ml_collections import ConfigDict
import torch
from torch.utils.data import DataLoader

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.models.maml.memory_core import MemoryMAMLConfig as TorchMemoryMAMLConfig
from icil.models.maml.query_memory_tasks import QueryMemoryTaskBuilder, prepare_query_memory_task_for_meta_step
from icil.models.maml.query_memory_tasks import PreparedQueryMemoryTaskBatchIterable
from icil.models.maml.train_utils import (
    build_store as _build_store,
    infer_dims as _infer_dims,
    normalize_task_list as _normalize_task_list,
)
from icil_jax_query_memory.data.adapter import prepared_tasks_to_sharded_batch
from icil_jax_query_memory.models.config_utils import build_model_config_from_raw, resolve_dtype
from icil_jax_query_memory.models.query_memory_direct_regression import QueryMemoryDirectRegressionModel
from icil_jax_query_memory.train.config import QueryMemoryMetaConfig
from icil_jax_query_memory.train.step import create_adapt_fn, create_predict_fn, create_train_state, create_train_step
from icil_jax_query_memory.utils.action_representation import decode_action_chunk_np
from icil_jax_query_memory.utils.checkpoints import load_checkpoint, save_checkpoint


def _data_source(cfg: ConfigDict) -> str:
    return str(getattr(cfg.data, 'source', 'rlbench')).strip().lower()


def _as_bool(value: Any) -> bool:
    return bool(value)


def _shutdown_dataloader_workers(loader_iter: Any, loader: Any) -> None:
    # PyTorch usually tears workers down when the iterator is GC'ed, but JAX
    # jobs are often interrupted or fail during compilation/logging. Explicit
    # shutdown avoids orphaned spawned pt_data_worker processes keeping many GB
    # of RSS alive after the main process exits.
    seen = set()
    for obj in (loader_iter, getattr(loader, '_iterator', None)):
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        shutdown = getattr(obj, '_shutdown_workers', None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass


def _set_seed(seed: int) -> None:
    np.random.seed(seed)


def _resolve_use_mask_id(model_cfg: ConfigDict) -> bool:
    return bool(getattr(model_cfg.simple_query_point_encoder, 'use_mask_id', False))


def _resolve_use_rgb(model_cfg: ConfigDict) -> bool:
    return bool(getattr(model_cfg.simple_query_point_encoder, 'use_rgb', True))


def _resolve_dataset_cfg(cfg: ConfigDict) -> ICILConfig:
    return ICILConfig(
        K=int(cfg.dataset.K),
        L=int(cfg.dataset.L),
        T_obs=int(cfg.dataset.T_obs),
        H=int(cfg.dataset.H),
        stride=int(cfg.dataset.stride),
        action_representation=str(getattr(cfg.dataset, 'action_representation', 'absolute')),
        task_sampling=str(getattr(cfg.data, 'task_sampling', 'variation_uniform')),
        task_sampling_alpha=float(getattr(cfg.data, 'task_sampling_alpha', 0.5)),
    )


def _resolve_metaworld_dataset_cfg(cfg: ConfigDict) -> Any:
    from icil_metaworld.data.metaworld_task_builder import MetaWorldICILConfig

    return MetaWorldICILConfig(
        K=int(cfg.dataset.K),
        T_obs=int(cfg.dataset.T_obs),
        H=int(cfg.dataset.H),
        stride=int(cfg.dataset.stride),
        action_stride=int(getattr(cfg.dataset, 'action_stride', 1)),
        pad_short_chunks=bool(getattr(cfg.dataset, 'pad_short_chunks', False)),
        action_representation=str(getattr(cfg.dataset, 'action_representation', 'absolute')),
        task_sampling=str(getattr(cfg.data, 'task_sampling', 'task_instance_uniform')),
        sample_same_task_name=bool(getattr(cfg.data, 'sample_same_task_name', True)),
        sample_same_task_instance=bool(getattr(cfg.data, 'sample_same_task_instance', True)),
        allow_support_query_same_episode=bool(getattr(cfg.data, 'allow_support_query_same_episode', False)),
        support_zero_goal=bool(getattr(cfg.data, 'support_zero_goal', False)),
        query_zero_goal=bool(getattr(cfg.data, 'query_zero_goal', False)),
    )


def _resolve_memory_cfg(cfg: ConfigDict) -> QueryMemoryMetaConfig:
    grad_accum_steps = int(getattr(cfg.maml, 'grad_accum_steps', 1))
    if grad_accum_steps != 1:
        raise ValueError('JAX query-memory v1 only supports maml.grad_accum_steps=1.')
    if bool(getattr(cfg.maml, 'reuse_diffusion_noise', False)):
        raise ValueError('JAX query-memory v1 does not use diffusion noise; set maml.reuse_diffusion_noise=False.')
    inner_loss_mode = str(getattr(cfg.maml, 'inner_loss_mode', 'read')).lower()
    if inner_loss_mode not in ('read', 'write'):
        raise ValueError("maml.inner_loss_mode must be one of: 'read', 'write'.")
    if bool(getattr(cfg.maml, 'use_wrong_support_margin', False)) and bool(getattr(cfg.maml, 'use_read_improvement_margin', False)):
        raise ValueError('Set only one of maml.use_wrong_support_margin or maml.use_read_improvement_margin for clean attribution.')
    return QueryMemoryMetaConfig(
        inner_steps=int(cfg.maml.inner_steps),
        inner_lr=float(cfg.maml.inner_lr),
        inner_lr_mode=str(getattr(cfg.maml, 'inner_lr_mode', 'fixed')),
        outer_lr=float(cfg.maml.outer_lr),
        weight_decay=float(cfg.train.weight_decay),
        max_grad_norm=float(cfg.maml.max_grad_norm),
        num_queries_per_step=int(cfg.maml.num_queries_per_step),
        num_query_loss_samples=int(cfg.maml.num_query_loss_samples),
        num_inner_batches=int(getattr(cfg.maml, 'num_inner_batches', 0)),
        holdout_index=int(getattr(cfg.maml, 'holdout_index', -1)),
        first_order=bool(cfg.maml.first_order),
        reuse_diffusion_noise=False,
        grad_accum_steps=1,
        inner_loss_mode=inner_loss_mode,
        memory_layer_norm_after_update=bool(
            getattr(
                cfg.maml,
                'memory_layer_norm_after_update',
                getattr(cfg.model.query_memory_direct_regression, 'memory_layer_norm_after_update', False),
            )
        ),
        use_read_improvement_margin=bool(getattr(cfg.maml, 'use_read_improvement_margin', False)),
        read_improvement_margin=float(getattr(cfg.maml, 'read_improvement_margin', 0.0)),
        read_improvement_margin_weight=float(getattr(cfg.maml, 'read_improvement_margin_weight', 0.0)),
        log_output_delta=bool(getattr(cfg.maml, 'log_output_delta', False)),
        training_mode_metrics_only=bool(getattr(cfg.maml, 'training_mode_metrics_only', False)),
        use_wrong_support_margin=bool(getattr(cfg.maml, 'use_wrong_support_margin', False)),
        wrong_support_margin=float(getattr(cfg.maml, 'wrong_support_margin', 0.0)),
        wrong_support_margin_weight=float(getattr(cfg.maml, 'wrong_support_margin_weight', 0.0)),
        wrong_support_strategy=str(getattr(cfg.maml, 'wrong_support_strategy', 'random_different_task')),
        use_memory_contrast=bool(getattr(cfg.maml, 'use_memory_contrast', False)),
        memory_contrast_weight=float(getattr(cfg.maml, 'memory_contrast_weight', 0.0)),
        memory_contrast_temperature=float(getattr(cfg.maml, 'memory_contrast_temperature', 0.1)),
        memory_contrast_on_delta=bool(getattr(cfg.maml, 'memory_contrast_on_delta', True)),
        query_goal_dropout_rate=float(getattr(cfg.maml, 'query_goal_dropout_rate', 0.0)),
        query_goal_dropout_state_start=int(getattr(cfg.maml, 'query_goal_dropout_state_start', 36)),
        log_attention_metrics=bool(getattr(cfg.maml, 'log_attention_metrics', False)),
        goal_prediction_loss_weight=float(getattr(cfg.maml, 'goal_prediction_loss_weight', 0.0)),
        goal_prediction_loss_type=str(getattr(cfg.maml, 'goal_prediction_loss_type', 'mse')),
    )


def _infer_num_points(store, *, use_rgb: bool, use_mask_id: bool) -> int:
    for vidx in range(len(store)):
        episode_ids = store.list_episode_ids(vidx)
        if episode_ids.shape[0] == 0:
            continue
        sample = store.load_episode_slices(
            vidx=vidx,
            episode_id=int(episode_ids[0]),
            t_idx=np.asarray([0], dtype=np.int64),
            load_rgb=use_rgb,
            load_mask_id=use_mask_id,
            load_full_traj=False,
        )
        return int(sample['xyz'].shape[1])
    raise RuntimeError('Could not infer number of points from cache.')


def _maybe_init_wandb(cfg: ConfigDict, workdir: Path) -> Optional[Any]:
    if not hasattr(cfg, 'wandb') or not _as_bool(cfg.wandb.enable):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError('cfg.wandb.enable=True but wandb is not installed.') from exc

    return wandb.init(
        project=str(cfg.wandb.project),
        entity=str(cfg.wandb.entity) if str(cfg.wandb.entity) else None,
        group=str(cfg.wandb.group) if str(cfg.wandb.group) else None,
        name=str(cfg.wandb.name) if str(cfg.wandb.name) else None,
        mode=str(cfg.wandb.mode) if str(cfg.wandb.mode) else None,
        dir=str(workdir),
        config=cfg.to_dict(),
        tags=list(getattr(cfg.wandb, 'tags', ())) if getattr(cfg.wandb, 'tags', None) else None,
    )


def _resolve_run_id(wandb_run: Optional[Any]) -> str:
    if wandb_run is not None:
        return str(wandb_run.id)
    return time.strftime('local-%Y%m%d-%H%M%S')


_JAX_BATCH_KEYS = (
    'query_xyz',
    'query_state',
    'query_valid',
    'target_action',
    'query_rgb',
    'query_mask_id',
    'demo_id',
    'support_demo_id',
    'chunk_start',
    'support_chunk_start',
)


def _torch_batch_to_jax(batch: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    out: Dict[str, jnp.ndarray] = {}
    for key in _JAX_BATCH_KEYS:
        if key in batch:
            value = batch[key]
            if torch.is_tensor(value):
                out[key] = jnp.asarray(value.detach().cpu().numpy())
            elif isinstance(value, np.ndarray):
                out[key] = jnp.asarray(value)
    return out


def _plot_pred_vs_gt_3d_np(
    pred_x0: np.ndarray,
    gt_x0: np.ndarray,
    *,
    max_items: int,
) -> Optional[Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    pred = np.asarray(pred_x0, dtype=np.float32)
    gt = np.asarray(gt_x0, dtype=np.float32)
    if pred.ndim != 3 or gt.ndim != 3:
        return None

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
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        gx, gy, gz = _xyz(gt[i])
        px, py, pz = _xyz(pred[i])
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
        pts_all = np.concatenate(
            [
                np.stack([gx, gy, gz], axis=1),
                np.stack([px, py, pz], axis=1),
            ],
            axis=0,
        )
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


def _sample_logging_artifacts(
    *,
    params: Any,
    task_spec: Any,
    task_builder: QueryMemoryTaskBuilder,
    memory_cfg: QueryMemoryMetaConfig,
    adapt_fn: Any,
    predict_fn: Any,
    dataset_cfg: ICILConfig,
    use_mask_id: bool,
    use_rgb: bool,
    sample_batch_items: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[Any]]:
    logging_cfg = TorchMemoryMAMLConfig(
        inner_steps=int(memory_cfg.inner_steps),
        inner_lr=float(memory_cfg.inner_lr),
        inner_lr_mode=str(memory_cfg.inner_lr_mode),
        outer_lr=float(memory_cfg.outer_lr),
        weight_decay=float(memory_cfg.weight_decay),
        max_grad_norm=float(memory_cfg.max_grad_norm),
        num_queries_per_step=int(memory_cfg.num_queries_per_step),
        num_inner_batches=int(memory_cfg.num_inner_batches),
        num_query_loss_samples=max(1, int(sample_batch_items)),
        holdout_index=int(memory_cfg.holdout_index),
        reuse_diffusion_noise=bool(memory_cfg.reuse_diffusion_noise),
        grad_accum_steps=int(memory_cfg.grad_accum_steps),
    )
    prepared_task = prepare_query_memory_task_for_meta_step(
        task_spec,
        task_builder=task_builder,
        cfg=logging_cfg,
        device=torch.device('cpu'),
        use_mask_id=use_mask_id,
        rng=rng,
        load_rgb=use_rgb,
    )
    if not prepared_task['inner_batches']:
        return None, None
    inner_batch_np: Dict[str, np.ndarray] = {}
    sample_keys = _JAX_BATCH_KEYS
    for key in sample_keys:
        if key in prepared_task['inner_batches'][0]:
            inner_batch_np[key] = np.stack(
                [batch[key].detach().cpu().numpy() for batch in prepared_task['inner_batches']],
                axis=0,
            )
    inner_batch = _torch_batch_to_jax(inner_batch_np)
    query_batch = _torch_batch_to_jax(prepared_task['query_batch'])
    adapted_memory = adapt_fn(params, inner_batch)
    pred = predict_fn(params, query_batch, adapted_memory)
    pred_np = np.asarray(pred)
    target_np = np.asarray(query_batch['target_action'])
    sample_mse = float(np.mean(np.square(pred_np - target_np)))
    plot_pred = decode_action_chunk_np(
        pred_np,
        query_state=np.asarray(query_batch['query_state']),
        representation=str(dataset_cfg.action_representation),
    )
    plot_gt = decode_action_chunk_np(
        target_np,
        query_state=np.asarray(query_batch['query_state']),
        representation=str(dataset_cfg.action_representation),
    )
    fig = _plot_pred_vs_gt_3d_np(
        pred_x0=plot_pred,
        gt_x0=plot_gt,
        max_items=int(sample_batch_items),
    )
    return sample_mse, fig


def train(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError('No JAX devices found.')
    num_devices = len(devices)
    per_device_batch = int(cfg.train.batch_size)
    if per_device_batch < 1:
        raise ValueError('train.batch_size must be >= 1 and is interpreted as per-device task batch size.')
    global_task_batch = num_devices * per_device_batch

    source = _data_source(cfg)
    if source == 'rlbench':
        dataset_cfg = _resolve_dataset_cfg(cfg)
    elif source == 'metaworld':
        dataset_cfg = _resolve_metaworld_dataset_cfg(cfg)
    else:
        raise ValueError(f"Unsupported data.source={source!r}. Expected 'rlbench' or 'metaworld'.")
    memory_cfg = _resolve_memory_cfg(cfg)
    model_cfg_raw = cfg.model
    # Keep goal-token visibility aligned with the data masking policy for the
    # new object-centric tokenizer. Existing flat-token configs ignore these
    # fields, so this is backward-compatible.
    model_cfg_raw.query_goal_hidden = bool(getattr(cfg.data, 'query_zero_goal', False))
    model_cfg_raw.support_goal_hidden = bool(getattr(cfg.data, 'support_zero_goal', False))
    if hasattr(model_cfg_raw, 'support_encoder_memory'):
        model_cfg_raw.support_encoder_memory.goal_visible = not bool(getattr(cfg.data, 'support_zero_goal', False))
    if bool(memory_cfg.log_attention_metrics):
        model_cfg_raw.query_memory_direct_regression.log_attention_weights = True
    if float(memory_cfg.goal_prediction_loss_weight) > 0.0:
        model_cfg_raw.query_memory_direct_regression.use_goal_prediction_head = True
    use_mask_id = _resolve_use_mask_id(model_cfg_raw)
    use_rgb = _resolve_use_rgb(model_cfg_raw)

    if float(getattr(cfg.model.query_memory_direct_regression, 'dropout', 0.0)) != 0.0:
        raise ValueError('JAX query-memory v1 currently expects decoder dropout=0.0.')
    if float(getattr(cfg.model.query_memory_direct_regression, 'conditioner_dropout', 0.0)) != 0.0:
        raise ValueError('JAX query-memory v1 currently expects conditioner_dropout=0.0.')

    cache_root = Path(str(cfg.data.cache_root)).expanduser()
    tasks = _normalize_task_list(getattr(cfg.data, 'tasks', ()))
    exclude_tasks = _normalize_task_list(getattr(cfg.data, 'exclude_tasks', ()))
    if source == 'rlbench':
        store, selected_tasks = _build_store(
            cache_root=cache_root,
            tasks=tasks,
            exclude_tasks=exclude_tasks,
            keep_open_per_worker=bool(cfg.data.keep_open_per_worker),
        )
        state_dim, action_dim = _infer_dims(store)
        num_points = _infer_num_points(store, use_rgb=use_rgb, use_mask_id=use_mask_id)
    else:
        from icil_metaworld.data.metaworld_cache import MetaWorldEpisodeStore
        from icil_metaworld.data.observation_filter import normalize_env_name

        tasks = tuple(normalize_env_name(t) for t in tasks)
        exclude_tasks = tuple(normalize_env_name(t) for t in exclude_tasks)
        store = MetaWorldEpisodeStore(
            cache_root=cache_root,
            keep_open_per_worker=bool(cfg.data.keep_open_per_worker),
            preload_to_memory=bool(getattr(cfg.data, 'preload_to_memory', False)),
        )
        if bool(getattr(cfg.data, 'preload_to_memory', False)):
            logging.info('Preloaded MetaWorld cache into RAM: %.2f MiB', float(store.preloaded_bytes) / (1024.0 * 1024.0))
        selected_tasks = tasks if tasks else tuple(store.list_task_names())
        state_dim, action_dim = store.infer_dims()
        num_points = 1
    # Metadata probing may have opened h5py handles on the main process store.
    # Clear them before the store is captured by spawned DataLoader workers.
    store.close()
    compute_dtype = resolve_dtype(cfg.train.amp_dtype if _as_bool(getattr(cfg.train, 'use_amp', False)) else 'float32')
    model_cfg = build_model_config_from_raw(model_cfg_raw, state_dim=state_dim, action_dim=action_dim, compute_dtype=compute_dtype)
    model = QueryMemoryDirectRegressionModel(cfg=model_cfg)

    init_rng = jax.random.PRNGKey(seed)
    dummy_query_xyz = jnp.zeros((1, int(dataset_cfg.T_obs), int(num_points), 3), dtype=compute_dtype)
    dummy_query_state = jnp.zeros((1, int(dataset_cfg.T_obs), int(state_dim)), dtype=compute_dtype)
    dummy_query_valid = jnp.ones((1, int(dataset_cfg.T_obs), int(num_points)), dtype=jnp.bool_)
    dummy_query_rgb = jnp.zeros((1, int(dataset_cfg.T_obs), int(num_points), 3), dtype=compute_dtype) if use_rgb else None
    dummy_query_mask_id = jnp.zeros((1, int(dataset_cfg.T_obs), int(num_points)), dtype=jnp.int32) if use_mask_id else None
    memory_init_mode = str(getattr(cfg.model.query_memory_direct_regression, 'memory_initialization_mode', 'base_only')).strip().lower()
    use_support_memory_init = memory_init_mode not in ('', 'none', 'learned', 'learned_base', 'base_only')
    dummy_support_count = max(1, int(getattr(memory_cfg, 'num_queries_per_step', 1)))
    dummy_support_state = (
        jnp.zeros((1, dummy_support_count, int(dataset_cfg.T_obs), int(state_dim)), dtype=compute_dtype)
        if use_support_memory_init
        else None
    )
    dummy_support_action = (
        jnp.zeros((1, dummy_support_count, int(dataset_cfg.H), int(action_dim)), dtype=compute_dtype)
        if use_support_memory_init
        else None
    )
    dummy_support_demo_id = jnp.zeros((1, dummy_support_count), dtype=jnp.int32) if use_support_memory_init else None
    dummy_support_chunk_start = jnp.zeros((1, dummy_support_count), dtype=compute_dtype) if use_support_memory_init else None
    params = model.init(
        init_rng,
        query_xyz=dummy_query_xyz,
        query_state=dummy_query_state,
        query_valid=dummy_query_valid,
        query_rgb=dummy_query_rgb,
        query_mask_id=dummy_query_mask_id,
        support_query_state=dummy_support_state,
        support_target_action=dummy_support_action,
        support_demo_id=dummy_support_demo_id,
        support_chunk_start=dummy_support_chunk_start,
        memory_tokens=None,
        train=False,
    )['params']

    state = create_train_state(
        params=params,
        outer_lr=float(memory_cfg.outer_lr),
        weight_decay=float(cfg.train.weight_decay),
        max_grad_norm=float(memory_cfg.max_grad_norm),
    )
    start_step = 0

    resume_path = str(getattr(cfg.train, 'resume_path', '')).strip()
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path).expanduser())
        state = state.replace(params=ckpt['params'], opt_state=ckpt.get('opt_state', state.opt_state), step=int(ckpt.get('step', 0)))
        start_step = int(ckpt.get('step', 0))

    replicated_state = jax_utils.replicate(state, devices=devices)
    p_train_step = create_train_step(
        model=model,
        inner_steps=int(memory_cfg.inner_steps),
        inner_lr=float(memory_cfg.inner_lr),
        max_grad_norm=float(memory_cfg.max_grad_norm),
        first_order=bool(memory_cfg.first_order),
        inner_loss_mode=str(memory_cfg.inner_loss_mode),
        memory_layer_norm_after_update=bool(memory_cfg.memory_layer_norm_after_update),
        use_read_improvement_margin=bool(memory_cfg.use_read_improvement_margin),
        read_improvement_margin=float(memory_cfg.read_improvement_margin),
        read_improvement_margin_weight=float(memory_cfg.read_improvement_margin_weight),
        log_output_delta=bool(memory_cfg.log_output_delta),
        training_mode_metrics_only=bool(memory_cfg.training_mode_metrics_only),
        use_wrong_support_margin=bool(memory_cfg.use_wrong_support_margin),
        wrong_support_margin=float(memory_cfg.wrong_support_margin),
        wrong_support_margin_weight=float(memory_cfg.wrong_support_margin_weight),
        use_memory_contrast=bool(memory_cfg.use_memory_contrast),
        memory_contrast_weight=float(memory_cfg.memory_contrast_weight),
        memory_contrast_temperature=float(memory_cfg.memory_contrast_temperature),
        memory_contrast_on_delta=bool(memory_cfg.memory_contrast_on_delta),
        query_goal_dropout_rate=float(memory_cfg.query_goal_dropout_rate),
        query_goal_dropout_state_start=int(memory_cfg.query_goal_dropout_state_start),
        log_attention_metrics=bool(memory_cfg.log_attention_metrics),
        goal_prediction_loss_weight=float(memory_cfg.goal_prediction_loss_weight),
        goal_prediction_loss_type=str(memory_cfg.goal_prediction_loss_type),
        rng_seed=int(seed),
    )
    adapt_fn = create_adapt_fn(
        model=model,
        inner_steps=int(memory_cfg.inner_steps),
        inner_lr=float(memory_cfg.inner_lr),
        max_grad_norm=float(memory_cfg.max_grad_norm),
        first_order=bool(memory_cfg.first_order),
        inner_loss_mode=str(memory_cfg.inner_loss_mode),
        memory_layer_norm_after_update=bool(memory_cfg.memory_layer_norm_after_update),
    )
    predict_fn = create_predict_fn(model=model)
    if source == 'rlbench':
        logging_task_builder = QueryMemoryTaskBuilder(
            store,
            cfg=dataset_cfg,
            seed=seed,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )
        task_dataset = PreparedQueryMemoryTaskBatchIterable(
            store,
            cfg=dataset_cfg,
            memory_cfg=memory_cfg,
            task_batch_size_B=int(global_task_batch),
            num_batches=int(cfg.train.num_steps) + 8,
            seed=seed,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
            use_mask_id=use_mask_id,
            load_rgb=use_rgb,
        )
    else:
        from icil_metaworld.data.metaworld_task_builder import PreparedMetaWorldQueryMemoryTaskBatchIterable

        logging_task_builder = None
        task_dataset = PreparedMetaWorldQueryMemoryTaskBatchIterable(
            store,
            cfg=dataset_cfg,
            memory_cfg=memory_cfg,
            task_batch_size_B=int(global_task_batch),
            num_batches=int(cfg.train.num_steps) + 8,
            seed=seed,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
            task_names=tasks,
            exclude_tasks=exclude_tasks,
        )
    num_workers = int(getattr(cfg.data, 'num_workers', 0))
    persistent_workers = bool(getattr(cfg.data, 'persistent_workers', False)) and num_workers > 0
    loader_kwargs: Dict[str, Any] = {
        'batch_size': None,
        'num_workers': num_workers,
        'pin_memory': bool(getattr(cfg.data, 'pin_memory', False)),
        'persistent_workers': persistent_workers,
    }
    if num_workers > 0:
        # JAX is multithreaded; avoid forking worker processes after the runtime
        # has initialized.
        loader_kwargs['multiprocessing_context'] = 'spawn'
        prefetch_factor = int(getattr(cfg.data, 'prefetch_factor', 2))
        if prefetch_factor < 1:
            raise ValueError(f'data.prefetch_factor must be >= 1 when num_workers > 0, got {prefetch_factor}.')
        loader_kwargs['prefetch_factor'] = prefetch_factor
    loader = DataLoader(
        task_dataset,
        **loader_kwargs,
    )

    run_root = Path(str(cfg.output_parent_dir)).expanduser()
    run_root.mkdir(parents=True, exist_ok=True)
    wandb_run = _maybe_init_wandb(cfg, run_root)
    run_id = _resolve_run_id(wandb_run)
    workdir = run_root / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(str(cfg.train.checkpoint_parent_dir)).expanduser() / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        'resolved': {
            'data_source': source,
            'state_dim': int(state_dim),
            'action_dim': int(action_dim),
            'num_points': int(num_points),
            'num_devices': int(num_devices),
            'per_device_batch_size': int(per_device_batch),
        },
        'model': cfg.model.to_dict(),
        'data': cfg.data.to_dict(),
        'dataset': cfg.dataset.to_dict(),
        'maml': cfg.maml.to_dict(),
        'train': cfg.train.to_dict(),
    }
    with (workdir / 'config.resolved.json').open('w') as f:
        json.dump(config_payload, f, indent=2)

    del selected_tasks

    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.ckpt_every)
    wandb_loss_every = int(getattr(cfg.wandb, 'n_loss_steps', 0)) if wandb_run is not None else 0
    wandb_sample_every = int(getattr(cfg.wandb, 'n_sample_steps', 0)) if wandb_run is not None else 0
    wandb_sample_batch = int(getattr(cfg.wandb, 'sample_batch_items', 4)) if wandb_run is not None else 0
    global_step = int(start_step)
    log_metric_sums: Dict[str, float] = {}
    log_timing_sums = {'data_wait_s': 0.0, 'train_step_s': 0.0}
    log_count = 0
    window_start = time.time()
    loader_iter = None

    try:
        loader_iter = iter(loader)
        while True:
            if global_step >= int(cfg.train.num_steps):
                break
            t_data_0 = time.time()
            try:
                prepared_tasks = next(loader_iter)
            except StopIteration:
                break
            data_wait_s = time.time() - t_data_0
            sharded_batch = prepared_tasks_to_sharded_batch(
                prepared_tasks,
                inner_steps=int(memory_cfg.inner_steps),
                num_devices=int(num_devices),
                per_device_batch=int(per_device_batch),
                devices=devices,
            )
            train_batch = {
                'inner': sharded_batch['inner'],
                'query': sharded_batch['query'],
                'meta': sharded_batch.get('meta', {}),
            }
            if bool(memory_cfg.use_wrong_support_margin):
                if 'wrong_inner' not in sharded_batch:
                    raise RuntimeError('maml.use_wrong_support_margin=True but prepared batch has no wrong_inner.')
                train_batch['wrong_inner'] = sharded_batch['wrong_inner']
            if bool(memory_cfg.use_memory_contrast):
                if 'contrast_inner' not in sharded_batch:
                    raise RuntimeError('maml.use_memory_contrast=True but prepared batch has no contrast_inner.')
                train_batch['contrast_inner'] = sharded_batch['contrast_inner']
            t_step_0 = time.time()
            replicated_state, metrics = p_train_step(replicated_state, train_batch)
            global_step = int(jax.device_get(jax_utils.unreplicate(replicated_state.step)))
            metrics_host = {key: float(jax.device_get(jax_utils.unreplicate(value))) for key, value in metrics.items()}
            train_step_s = time.time() - t_step_0
            for key, value in metrics_host.items():
                log_metric_sums.setdefault(key, 0.0)
                log_metric_sums[key] += float(value)
            log_timing_sums['data_wait_s'] += float(data_wait_s)
            log_timing_sums['train_step_s'] += float(train_step_s)
            log_count += 1

            if log_every > 0 and (global_step % log_every == 0 or global_step == start_step + 1):
                elapsed = max(1e-6, time.time() - window_start)
                steps_per_sec = log_count / elapsed
                avg_metrics = {key: value / max(1, log_count) for key, value in log_metric_sums.items()}
                avg_timings = {key: value / max(1, log_count) for key, value in log_timing_sums.items()}
                total_avg_step_s = max(1e-6, avg_timings['data_wait_s'] + avg_timings['train_step_s'])
                data_wait_frac = avg_timings['data_wait_s'] / total_avg_step_s
                logging.info(
                    'step %d/%d | meta_loss %.6f | read_before %.6f | read_after %.6f | inner_loss %.6f | inner_grad %.6f | mem_delta %.6f | rel_mem_delta %.6f | read_impr %.6f | out_delta %.6f | outer_lr %.3e | %.2f step/s | data_wait %.3fs | train_step %.3fs | data_wait_frac %.2f',
                    global_step,
                    int(cfg.train.num_steps),
                    avg_metrics.get('meta_loss', 0.0),
                    avg_metrics.get('read_loss_before', 0.0),
                    avg_metrics.get('read_loss_after', 0.0),
                    avg_metrics.get('inner_support_loss', 0.0),
                    avg_metrics.get('inner_memory_grad_norm', 0.0),
                    avg_metrics.get('memory_delta_norm', 0.0),
                    avg_metrics.get('memory_relative_delta_norm', 0.0),
                    avg_metrics.get('read_improvement', 0.0),
                    avg_metrics.get('action_output_delta', 0.0),
                    float(memory_cfg.outer_lr),
                    steps_per_sec,
                    avg_timings['data_wait_s'],
                    avg_timings['train_step_s'],
                    data_wait_frac,
                )
                log_metric_sums = {k: 0.0 for k in log_metric_sums}
                log_timing_sums = {k: 0.0 for k in log_timing_sums}
                log_count = 0
                window_start = time.time()

            if wandb_run is not None and wandb_loss_every > 0 and (global_step % wandb_loss_every == 0 or global_step == start_step + 1):
                step_total_s = max(1e-6, float(data_wait_s) + float(train_step_s))
                wandb_run.log(
                    {
                        **{f'train/{key}': float(value) for key, value in metrics_host.items()},
                        'train/outer_loss': metrics_host.get('meta_loss', 0.0),
                        'train/data_wait_s': float(data_wait_s),
                        'train/train_step_s': float(train_step_s),
                        'train/data_wait_frac': float(data_wait_s) / step_total_s,
                        'train/lr': float(memory_cfg.outer_lr),
                        'train/step': int(global_step),
                    },
                    step=int(global_step),
                )

            if (
                wandb_run is not None
                and wandb_sample_every > 0
                and logging_task_builder is not None
                and (global_step % wandb_sample_every == 0)
            ):
                fig = None
                try:
                    params_for_logging = jax_utils.unreplicate(replicated_state).params
                    sample_mse, fig = _sample_logging_artifacts(
                        params=params_for_logging,
                        task_spec=prepared_tasks[0]['task'],
                        task_builder=logging_task_builder,
                        memory_cfg=memory_cfg,
                        adapt_fn=adapt_fn,
                        predict_fn=predict_fn,
                        dataset_cfg=dataset_cfg,
                        use_mask_id=use_mask_id,
                        use_rgb=use_rgb,
                        sample_batch_items=max(1, int(wandb_sample_batch)),
                        rng=np.random.default_rng(seed + 1_000_003 + global_step),
                    )
                    log_dict: Dict[str, Any] = {
                        'train/step': int(global_step),
                    }
                    if sample_mse is not None:
                        log_dict['samples/action_chunk_mse'] = float(sample_mse)
                    if fig is not None:
                        import wandb

                        log_dict['samples/action_chunk_pred_vs_gt_3d'] = wandb.Image(fig)
                    wandb_run.log(log_dict, step=int(global_step))
                finally:
                    if fig is not None:
                        try:
                            import matplotlib.pyplot as plt

                            plt.close(fig)
                        except Exception:
                            pass

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                host_state = jax_utils.unreplicate(replicated_state)
                save_checkpoint(
                    checkpoint_dir / f'step_{global_step:07d}.pkl',
                    step=int(global_step),
                    params=host_state.params,
                    opt_state=host_state.opt_state,
                    config=config_payload,
                )
                logging.info('Saved checkpoint: %s', checkpoint_dir / f'step_{global_step:07d}.pkl')

        host_state = jax_utils.unreplicate(replicated_state)
        save_checkpoint(
            checkpoint_dir / 'last.pkl',
            step=int(global_step),
            params=host_state.params,
            opt_state=host_state.opt_state,
            config=config_payload,
        )
        logging.info('Training complete. Final checkpoint: %s', checkpoint_dir / 'last.pkl')
    finally:
        _shutdown_dataloader_workers(loader_iter, loader)
        if wandb_run is not None:
            wandb_run.finish()
        store.close()
