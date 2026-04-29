from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from absl import logging
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

import icil.models.maml.memory_train as memory_train_lib
from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.models import (
    QueryMemoryDirectRegressionBuilderConfig,
    build_query_memory_builder_config_from_configdict,
    build_query_memory_direct_regression_policy,
)
from icil.models.maml.memory_core import memory_maml_step_with_stats
from icil.models.maml.query_memory_tasks import (
    QueryMemoryTaskBatchIterable,
    QueryMemoryTaskBuilder,
    prepare_outer_batch_for_query_memory_meta_step,
)
from icil.models.maml.train_utils import (
    build_optional_store as _build_optional_store,
    build_store as _build_store,
    count_parameters as _count_parameters,
    infer_dims as _infer_dims,
    maybe_init_wandb as _maybe_init_wandb,
    normalize_task_list as _normalize_task_list,
    resolve_run_id as _resolve_run_id,
)


def _init_distributed() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    return distributed, rank, world_size, local_rank



def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()



def _broadcast_object(obj: Any, *, src: int = 0) -> Any:
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]



def _distributed_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(dist.get_world_size())
    return float(tensor.item())



def _all_reduce_grads(params: Sequence[torch.nn.Parameter], device: torch.device) -> None:
    if not dist.is_initialized():
        return
    world_size = float(dist.get_world_size())
    for param in params:
        has_grad = torch.tensor(0 if param.grad is None else 1, device=device, dtype=torch.int32)
        grad = torch.zeros_like(param, memory_format=torch.preserve_format) if param.grad is None else param.grad
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            param.grad = None
        else:
            param.grad = grad / world_size



def _configdict_from_dict(data: Dict[str, Any]) -> ConfigDict:
    return ConfigDict(data)



def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint object type: {type(checkpoint).__name__}')
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' payload is not a state_dict dictionary.")
    if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
    return checkpoint, state_dict



def _resolve_model_cfg(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
) -> Tuple[QueryMemoryDirectRegressionBuilderConfig, str]:
    if isinstance(resume_config, dict) and isinstance(resume_config.get('model'), dict):
        return build_query_memory_builder_config_from_configdict(
            _configdict_from_dict(resume_config['model']),
            as_bool=memory_train_lib._as_bool,
        ), 'resume_checkpoint'
    return build_query_memory_builder_config_from_configdict(cfg.model, as_bool=memory_train_lib._as_bool), 'config'



def _resolve_dataset_cfg(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
) -> ICILConfig:
    dataset_ckpt = resume_config.get('dataset', {}) if isinstance(resume_config, dict) else {}

    def _resolve_int(name: str, default: int) -> int:
        local_value = int(getattr(cfg.dataset, name, default))
        if local_value > 0:
            return local_value
        if isinstance(dataset_ckpt, dict) and name in dataset_ckpt:
            return int(dataset_ckpt[name])
        return int(default)

    action_representation = str(getattr(cfg.dataset, 'action_representation', 'absolute'))
    if isinstance(dataset_ckpt, dict) and 'action_representation' in dataset_ckpt:
        action_representation = str(dataset_ckpt['action_representation'])

    return ICILConfig(
        K=_resolve_int('K', 1),
        L=_resolve_int('L', 16),
        T_obs=_resolve_int('T_obs', 2),
        H=_resolve_int('H', 16),
        stride=_resolve_int('stride', 2),
        action_representation=action_representation,
        task_sampling=str(getattr(cfg.data, 'task_sampling', 'variation_power')),
        task_sampling_alpha=float(getattr(cfg.data, 'task_sampling_alpha', 1.0)),
    )



def _resolve_task_filters(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    if isinstance(resume_config, dict):
        data_cfg = resume_config.get('data', {})
        if isinstance(data_cfg, dict):
            tasks = _normalize_task_list(data_cfg.get('tasks', getattr(cfg.data, 'tasks', ())))
            exclude = _normalize_task_list(data_cfg.get('exclude_tasks', getattr(cfg.data, 'exclude_tasks', ())))
            return tasks, exclude
    return _normalize_task_list(getattr(cfg.data, 'tasks', ())), _normalize_task_list(getattr(cfg.data, 'exclude_tasks', ()))



def _resolve_use_mask_id(model_cfg: QueryMemoryDirectRegressionBuilderConfig) -> bool:
    if str(model_cfg.query_encoder_name) == 'simple_query_point_encoder':
        return bool(model_cfg.simple_query_point_encoder.use_mask_id)
    if str(model_cfg.query_encoder_name) == 'dp3_query_frame_encoder':
        return bool(model_cfg.dp3_query_frame_encoder.use_mask_id)
    raise ValueError(f'Unknown query_encoder_name={model_cfg.query_encoder_name!r}.')



def _resolve_use_rgb(model_cfg: QueryMemoryDirectRegressionBuilderConfig) -> bool:
    if str(model_cfg.query_encoder_name) == 'simple_query_point_encoder':
        return bool(model_cfg.simple_query_point_encoder.use_rgb)
    if str(model_cfg.query_encoder_name) == 'dp3_query_frame_encoder':
        return bool(model_cfg.dp3_query_frame_encoder.use_rgb)
    raise ValueError(f'Unknown query_encoder_name={model_cfg.query_encoder_name!r}.')



def _build_model(
    cfg: QueryMemoryDirectRegressionBuilderConfig,
    *,
    state_dim: int,
    action_dim: int,
):
    return build_query_memory_direct_regression_policy(cfg, state_dim=state_dim, action_dim=action_dim)



def train(cfg: ConfigDict) -> None:
    distributed, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0
    seed = int(cfg.seed)
    first_order = memory_train_lib._as_bool(getattr(cfg.maml, 'first_order', False))
    if distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = memory_train_lib._resolve_device(str(cfg.device))
    memory_train_lib._set_seed(seed)
    data_seed = seed + rank * 1_000_003

    resume_path = Path(str(cfg.train.resume_path)).expanduser() if str(getattr(cfg.train, 'resume_path', '')) else None
    resume_checkpoint = None
    resume_state_dict = None
    resume_config = None
    if resume_path is not None:
        resume_checkpoint, resume_state_dict = _load_checkpoint(resume_path, device)
        config_obj = resume_checkpoint.get('config', None)
        if isinstance(config_obj, dict):
            resume_config = config_obj

    model_cfg, model_cfg_source = _resolve_model_cfg(cfg, resume_config=resume_config)
    dataset_cfg = _resolve_dataset_cfg(cfg, resume_config=resume_config)
    tasks, exclude_tasks = _resolve_task_filters(cfg, resume_config=resume_config)
    cache_root = Path(str(cfg.data.cache_root))
    store, tasks_used = _build_store(
        cache_root=cache_root,
        tasks=tasks,
        exclude_tasks=exclude_tasks,
        keep_open_per_worker=memory_train_lib._as_bool(cfg.data.keep_open_per_worker),
    )
    excluded_store, excluded_tasks_used = _build_optional_store(
        cache_root=cache_root,
        tasks=exclude_tasks,
        keep_open_per_worker=memory_train_lib._as_bool(cfg.data.keep_open_per_worker),
    )
    del excluded_store, excluded_tasks_used

    wandb_run = None
    try:
        output_parent = Path(
            str(
                getattr(
                    cfg,
                    'output_parent_dir',
                    getattr(cfg, 'workdir', 'output_data_playground_v3/.experiments'),
                )
            )
        )
        output_parent.mkdir(parents=True, exist_ok=True)
        if is_main:
            wandb_run = _maybe_init_wandb(cfg, output_parent)
            run_id = _resolve_run_id(wandb_run)
        else:
            run_id = ''
        run_id = _broadcast_object(run_id, src=0)
        if is_main and wandb_run is not None:
            wandb_run.name = run_id

        workdir = output_parent / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        ckpt_parent = Path(str(getattr(cfg.train, 'checkpoint_parent_dir', workdir.parent / 'checkpoints')))
        checkpoint_dir = ckpt_parent / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        state_dim, action_dim = _infer_dims(store)
        task_dataset = QueryMemoryTaskBatchIterable(
            store=store,
            cfg=dataset_cfg,
            task_batch_size_B=int(cfg.train.batch_size),
            num_batches=int(cfg.train.num_steps),
            seed=data_seed,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )
        num_workers = int(cfg.data.num_workers)
        pin_memory = memory_train_lib._as_bool(cfg.data.pin_memory) and device.type == 'cuda'
        persistent_workers = memory_train_lib._as_bool(cfg.data.persistent_workers) and num_workers > 0
        task_loader = DataLoader(
            task_dataset,
            batch_size=1,
            collate_fn=memory_train_lib._unwrap_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        task_builder = QueryMemoryTaskBuilder(
            store=store,
            cfg=dataset_cfg,
            seed=data_seed,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )

        policy = _build_model(model_cfg, state_dim=state_dim, action_dim=action_dim).to(device)
        if resume_state_dict is not None:
            policy.load_state_dict(resume_state_dict, strict=True)
            logging.info('Resumed model weights from %s', resume_path)

        use_mask_id = _resolve_use_mask_id(model_cfg)
        use_rgb = _resolve_use_rgb(model_cfg)
        memory_cfg = memory_train_lib._build_memory_cfg(cfg)
        resolved_inner_lr_mode = memory_train_lib._resolve_inner_lr_mode(
            cfg,
            resume_checkpoint=resume_checkpoint,
            pretrained_checkpoint=None,
            resume_config=resume_config,
            pretrained_config=None,
        )
        memory_cfg.inner_lr_mode = str(resolved_inner_lr_mode)
        inner_lr_schedule = memory_train_lib._build_inner_lr_schedule_wrapper(memory_cfg)
        if inner_lr_schedule is not None:
            inner_lr_schedule = inner_lr_schedule.to(device)
        memory_train_lib._load_inner_lr_schedule_state(
            inner_lr_schedule,
            checkpoint=resume_checkpoint,
            checkpoint_path=resume_path,
        )

        outer_params = list(policy.parameters())
        inner_lr_params = list(inner_lr_schedule.parameters()) if inner_lr_schedule is not None else []
        optimizer = torch.optim.AdamW(
            outer_params + inner_lr_params,
            lr=float(memory_cfg.outer_lr),
            weight_decay=float(memory_cfg.weight_decay),
        )

        global_step = 0
        if resume_checkpoint is not None:
            optimizer_state = resume_checkpoint.get('optimizer', None)
            if isinstance(optimizer_state, dict):
                try:
                    optimizer.load_state_dict(optimizer_state)
                    global_step = int(resume_checkpoint.get('step', 0))
                    logging.info('Resumed optimizer state from %s at step=%d', resume_path, global_step)
                except ValueError as exc:
                    logging.warning('Skipping optimizer state load from %s due to mismatch: %s', resume_path, exc)

        n_total, n_trainable = _count_parameters(policy)
        n_inner_lr = sum(int(param.numel()) for param in inner_lr_params)
        resolved_inner_lrs = memory_train_lib._inner_lr_values(schedule=inner_lr_schedule, cfg=memory_cfg)
        logging.info('Run id=%s', run_id)
        logging.info('Output dir=%s', workdir)
        logging.info('Checkpoint dir=%s', checkpoint_dir)
        logging.info('Using cache_root=%s', cache_root)
        logging.info('Tasks=%s | variations=%d', tasks_used, len(store))
        logging.info('Excluded tasks=%s', exclude_tasks)
        logging.info(
            'Model params: total=%s (%.3fM) | trainable=%s (%.3fM) | inner_lr=%s',
            f'{n_total:,}',
            n_total / 1e6,
            f'{n_trainable + n_inner_lr:,}',
            (n_trainable + n_inner_lr) / 1e6,
            f'{n_inner_lr:,}',
        )
        logging.info(
            'Resolved query-memory-%s DDP setup: model_source=%s | query_encoder=%s | data.K=%d | '
            'inner_steps=%d | inner_batch=%d | query_batch=%d | grad_accum=%d | distributed=%s | '
            'world_size=%d | effective_outer_batch=%d | inner_lr_mode=%s | inner_lrs=%s',
            'FOMAML' if first_order else 'MAML',
            model_cfg_source,
            str(model_cfg.query_encoder_name),
            int(dataset_cfg.K),
            int(memory_cfg.inner_steps),
            int(memory_cfg.num_queries_per_step),
            int(memory_cfg.num_query_loss_samples),
            int(memory_cfg.grad_accum_steps),
            str(bool(distributed)),
            int(world_size),
            int(cfg.train.batch_size) * int(world_size),
            str(memory_cfg.inner_lr_mode),
            resolved_inner_lrs,
        )

        config_payload = cfg.to_dict()
        config_payload['model'] = asdict(model_cfg)
        config_payload['dataset']['K'] = int(dataset_cfg.K)
        config_payload['dataset']['L'] = int(dataset_cfg.L)
        config_payload['dataset']['T_obs'] = int(dataset_cfg.T_obs)
        config_payload['dataset']['H'] = int(dataset_cfg.H)
        config_payload['dataset']['stride'] = int(dataset_cfg.stride)
        config_payload['data']['tasks'] = list(tasks)
        config_payload['data']['exclude_tasks'] = list(exclude_tasks)
        config_payload['maml']['first_order'] = bool(first_order)
        config_payload['maml']['inner_lr_mode'] = str(memory_cfg.inner_lr_mode)
        config_payload['algorithm'] = 'query_memory_fomaml_ddp' if bool(first_order) else 'query_memory_maml_ddp'
        config_payload['runtime'] = {
            'run_id': run_id,
            'output_dir': str(workdir),
            'checkpoint_dir': str(checkpoint_dir),
            'distributed': bool(distributed),
            'world_size': int(world_size),
            'rank': int(rank),
            'local_rank': int(local_rank),
            'local_batch_size_tasks': int(cfg.train.batch_size),
            'effective_batch_size_tasks': int(cfg.train.batch_size) * int(world_size),
        }
        config_payload['resolved'] = {
            'resume_path': str(resume_path) if resume_path is not None else '',
            'model_source': model_cfg_source,
            'initial_global_step': int(global_step),
            'fast_object': 'learned_memory_tokens',
            'first_order': bool(first_order),
            'inner_lr_mode': str(memory_cfg.inner_lr_mode),
            'initial_inner_lrs': resolved_inner_lrs,
        }
        config_path = workdir / 'config.json'
        if is_main:
            with config_path.open('w', encoding='utf-8') as f:
                json.dump(config_payload, f, indent=2)
        if is_main and wandb_run is not None:
            wandb_run.config.update(
                {
                    'runtime': config_payload['runtime'],
                    'resolved': config_payload['resolved'],
                    'model': config_payload['model'],
                    'data': config_payload['data'],
                    'dataset': config_payload['dataset'],
                    'maml': config_payload['maml'],
                },
                allow_val_change=True,
            )
            wandb_run.save(str(config_path), policy='now')

        log_every = int(cfg.train.log_every)
        ckpt_every = int(cfg.train.ckpt_every)
        wandb_loss_every = int(getattr(cfg.wandb, 'n_loss_steps', 0)) if wandb_run is not None else 0

        policy.train()
        log_loss = 0.0
        log_count = 0
        wb_loss_sum = 0.0
        wb_inner_grad_norm_sum = 0.0
        wb_inner_loss_sum = 0.0
        wb_count = 0
        window_start = time.time()

        for tasks_batch in task_loader:
            if global_step >= int(cfg.train.num_steps):
                break

            np_rng = np.random.default_rng(data_seed + 1_000_003 + global_step)
            prepared_tasks = prepare_outer_batch_for_query_memory_meta_step(
                tasks_batch,
                task_builder=task_builder,
                cfg=memory_cfg,
                device=device,
                use_mask_id=use_mask_id,
                rng=np_rng,
                load_rgb=use_rgb,
            )

            optimizer.zero_grad(set_to_none=True)
            meta_loss, avg_inner_memory_grad_norm_local, avg_inner_loss_local = memory_maml_step_with_stats(
                policy,
                prepared_tasks,
                cfg=memory_cfg,
                first_order=first_order,
                inner_lr_schedule=inner_lr_schedule,
            )
            meta_loss.backward()
            _all_reduce_grads(outer_params + inner_lr_params, device)
            loss_value = _distributed_mean(float(meta_loss.detach().cpu()), device)
            avg_inner_memory_grad_norm = _distributed_mean(float(avg_inner_memory_grad_norm_local), device)
            avg_inner_loss = _distributed_mean(float(avg_inner_loss_local), device)
            if float(memory_cfg.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(outer_params + inner_lr_params, float(memory_cfg.max_grad_norm))
            optimizer.step()
            global_step += 1

            log_loss += loss_value
            log_count += 1
            wb_loss_sum += loss_value
            wb_inner_grad_norm_sum += float(avg_inner_memory_grad_norm)
            wb_inner_loss_sum += float(avg_inner_loss)
            wb_count += 1

            if is_main and log_every > 0 and (global_step % log_every == 0 or global_step == 1):
                elapsed = max(1e-6, time.time() - window_start)
                steps_per_sec = log_count / elapsed
                avg_loss = log_loss / max(1, log_count)
                logging.info(
                    'step %d/%d | meta_loss %.6f | inner_loss %.6f | inner_grad %.6f | outer_lr %.3e | inner_lr_mean %.3e | %.2f step/s',
                    global_step,
                    int(cfg.train.num_steps),
                    avg_loss,
                    float(avg_inner_loss),
                    float(avg_inner_memory_grad_norm),
                    float(optimizer.param_groups[0]['lr']),
                    memory_train_lib._inner_lr_log_dict(schedule=inner_lr_schedule, cfg=memory_cfg).get('train/inner_lr_mean', float(memory_cfg.inner_lr)),
                    steps_per_sec,
                )
                log_loss = 0.0
                log_count = 0
                window_start = time.time()

            if is_main and wandb_run is not None and wandb_loss_every > 0 and (global_step % wandb_loss_every == 0 or global_step == 1):
                wandb_run.log(
                    {
                        'train/meta_loss': wb_loss_sum / max(1, wb_count),
                        'train/outer_loss': wb_loss_sum / max(1, wb_count),
                        'train/inner_memory_grad_norm': wb_inner_grad_norm_sum / max(1, wb_count),
                        'train/inner_support_loss': wb_inner_loss_sum / max(1, wb_count),
                        'train/lr': float(optimizer.param_groups[0]['lr']),
                        'train/step': global_step,
                        **memory_train_lib._inner_lr_log_dict(schedule=inner_lr_schedule, cfg=memory_cfg),
                    },
                    step=global_step,
                )
                wb_loss_sum = 0.0
                wb_inner_grad_norm_sum = 0.0
                wb_inner_loss_sum = 0.0
                wb_count = 0

            if is_main and ckpt_every > 0 and (global_step % ckpt_every == 0 or global_step == int(cfg.train.num_steps)):
                checkpoint = {
                    'step': int(global_step),
                    'model': policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config_payload,
                    'inner_lr_schedule': inner_lr_schedule.state_dict() if inner_lr_schedule is not None else None,
                }
                ckpt_path = checkpoint_dir / f'step_{global_step:07d}.pt'
                torch.save(checkpoint, ckpt_path)
                logging.info('Saved checkpoint to %s', ckpt_path)

        if is_main and wandb_run is not None:
            wandb_run.summary['train/final_step'] = int(global_step)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        store.close()
        _cleanup_distributed()
