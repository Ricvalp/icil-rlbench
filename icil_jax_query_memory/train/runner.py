from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax import jax_utils
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
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
from icil_jax_query_memory.train.step import create_train_state, create_train_step
from icil_jax_query_memory.utils.checkpoints import load_checkpoint, save_checkpoint


def _as_bool(value: Any) -> bool:
    return bool(value)


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


def _resolve_memory_cfg(cfg: ConfigDict) -> QueryMemoryMetaConfig:
    grad_accum_steps = int(getattr(cfg.maml, 'grad_accum_steps', 1))
    if grad_accum_steps != 1:
        raise ValueError('JAX query-memory v1 only supports maml.grad_accum_steps=1.')
    if bool(getattr(cfg.maml, 'reuse_diffusion_noise', False)):
        raise ValueError('JAX query-memory v1 does not use diffusion noise; set maml.reuse_diffusion_noise=False.')
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

    dataset_cfg = _resolve_dataset_cfg(cfg)
    memory_cfg = _resolve_memory_cfg(cfg)
    model_cfg_raw = cfg.model
    use_mask_id = _resolve_use_mask_id(model_cfg_raw)
    use_rgb = _resolve_use_rgb(model_cfg_raw)

    if float(getattr(cfg.model.query_memory_direct_regression, 'dropout', 0.0)) != 0.0:
        raise ValueError('JAX query-memory v1 currently expects decoder dropout=0.0.')
    if float(getattr(cfg.model.query_memory_direct_regression, 'conditioner_dropout', 0.0)) != 0.0:
        raise ValueError('JAX query-memory v1 currently expects conditioner_dropout=0.0.')

    cache_root = Path(str(cfg.data.cache_root)).expanduser()
    tasks = _normalize_task_list(getattr(cfg.data, 'tasks', ()))
    exclude_tasks = _normalize_task_list(getattr(cfg.data, 'exclude_tasks', ()))
    store, selected_tasks = _build_store(
        cache_root=cache_root,
        tasks=tasks,
        exclude_tasks=exclude_tasks,
        keep_open_per_worker=bool(cfg.data.keep_open_per_worker),
    )

    state_dim, action_dim = _infer_dims(store)
    num_points = _infer_num_points(store, use_rgb=use_rgb, use_mask_id=use_mask_id)
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
    params = model.init(
        init_rng,
        query_xyz=dummy_query_xyz,
        query_state=dummy_query_state,
        query_valid=dummy_query_valid,
        query_rgb=dummy_query_rgb,
        query_mask_id=dummy_query_mask_id,
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
    global_step = int(start_step)
    log_metric_sums = {'meta_loss': 0.0, 'inner_support_loss': 0.0, 'inner_memory_grad_norm': 0.0}
    log_count = 0
    window_start = time.time()

    try:
        for prepared_tasks in loader:
            if global_step >= int(cfg.train.num_steps):
                break
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
            }
            replicated_state, metrics = p_train_step(replicated_state, train_batch)
            global_step = int(jax.device_get(jax_utils.unreplicate(replicated_state.step)))
            metrics_host = {key: float(jax.device_get(jax_utils.unreplicate(value))) for key, value in metrics.items()}
            for key, value in metrics_host.items():
                log_metric_sums[key] += float(value)
            log_count += 1

            if log_every > 0 and (global_step % log_every == 0 or global_step == start_step + 1):
                elapsed = max(1e-6, time.time() - window_start)
                steps_per_sec = log_count / elapsed
                avg_metrics = {key: value / max(1, log_count) for key, value in log_metric_sums.items()}
                logging.info(
                    'step %d/%d | meta_loss %.6f | inner_loss %.6f | inner_grad %.6f | outer_lr %.3e | %.2f step/s',
                    global_step,
                    int(cfg.train.num_steps),
                    avg_metrics['meta_loss'],
                    avg_metrics['inner_support_loss'],
                    avg_metrics['inner_memory_grad_norm'],
                    float(memory_cfg.outer_lr),
                    steps_per_sec,
                )
                log_metric_sums = {k: 0.0 for k in log_metric_sums}
                log_count = 0
                window_start = time.time()

            if wandb_run is not None and wandb_loss_every > 0 and (global_step % wandb_loss_every == 0 or global_step == start_step + 1):
                wandb_run.log(
                    {
                        'train/meta_loss': metrics_host['meta_loss'],
                        'train/outer_loss': metrics_host['meta_loss'],
                        'train/inner_support_loss': metrics_host['inner_support_loss'],
                        'train/inner_memory_grad_norm': metrics_host['inner_memory_grad_norm'],
                        'train/lr': float(memory_cfg.outer_lr),
                        'train/step': int(global_step),
                    },
                    step=int(global_step),
                )

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
        if wandb_run is not None:
            wandb_run.finish()
        store.close()
