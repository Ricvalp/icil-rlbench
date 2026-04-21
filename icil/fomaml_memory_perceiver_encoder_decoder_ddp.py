from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader

import icil.models.maml.memory_train as memory_train_lib
from icil.models import build_policy
from icil.models.maml.diagnostics import (
    memory_inner_loop_query_curves,
    plot_scalar_curve,
)
from icil.models.maml.memory_core import (
    memory_maml_step_with_stats,
    prepare_outer_batch_for_memory_meta_step,
)
from icil.models.maml.params import (
    count_params_by_name,
    get_outer_param_names,
    set_outer_trainable_params,
)
from icil.models.maml.tasks import ICILMAMLTaskBatchIterable, MAMLTaskBuilder

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/fomaml_memory_perceiver_encoder_decoder.py',
    help_string='Path to a ml_collections config file.',
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


def _all_reduce_outer_grads(params: Sequence[torch.nn.Parameter], device: torch.device) -> None:
    if not dist.is_initialized():
        return
    world_size = float(dist.get_world_size())
    for param in params:
        has_grad = torch.tensor(
            0 if param.grad is None else 1,
            device=device,
            dtype=torch.int32,
        )
        if param.grad is None:
            grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        else:
            grad = param.grad
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            param.grad = None
        else:
            param.grad = grad / world_size


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

    resume_path = Path(str(cfg.train.resume_path)).expanduser() if str(cfg.train.resume_path) else None
    pretrained_path = (
        Path(str(cfg.finetune.pretrained_checkpoint)).expanduser()
        if str(cfg.finetune.pretrained_checkpoint)
        else None
    )

    resume_checkpoint = None
    resume_state_dict = None
    resume_config = None
    if resume_path is not None:
        resume_checkpoint, resume_state_dict = memory_train_lib._load_checkpoint(resume_path, device)
        config_obj = resume_checkpoint.get('config', None)
        if isinstance(config_obj, dict):
            resume_config = config_obj

    pretrained_state_dict = None
    pretrained_config = None
    if pretrained_path is not None and resume_path is None:
        pretrained_checkpoint, pretrained_state_dict = memory_train_lib._load_checkpoint(pretrained_path, device)
        config_obj = pretrained_checkpoint.get('config', None)
        if not isinstance(config_obj, dict):
            raise ValueError(f'Pretrained checkpoint at {pretrained_path} does not contain a valid config dict.')
        pretrained_config = config_obj

    model_cfg, model_cfg_source = memory_train_lib._resolve_model_cfg(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    resolved_data_k = memory_train_lib._resolve_data_k(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    dataset_cfg = memory_train_lib._resolve_dataset_cfg(
        cfg,
        resolved_data_k=resolved_data_k,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    cache_root = Path(str(cfg.data.cache_root))
    tasks, exclude_tasks = memory_train_lib._resolve_task_filters(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    store, tasks_used = memory_train_lib._build_store(
        cache_root=cache_root,
        tasks=tasks,
        exclude_tasks=exclude_tasks,
        keep_open_per_worker=memory_train_lib._as_bool(cfg.data.keep_open_per_worker),
    )
    excluded_store, excluded_tasks_used = memory_train_lib._build_optional_store(
        cache_root=cache_root,
        tasks=exclude_tasks,
        keep_open_per_worker=memory_train_lib._as_bool(cfg.data.keep_open_per_worker),
    )

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
            wandb_run = memory_train_lib._maybe_init_wandb(cfg, output_parent)
            run_id = memory_train_lib._resolve_run_id(wandb_run)
        else:
            run_id = ''
        run_id = _broadcast_object(run_id, src=0)
        if is_main and wandb_run is not None:
            wandb_run.name = run_id

        workdir = output_parent / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        ckpt_parent = Path(
            str(getattr(cfg.train, 'checkpoint_parent_dir', workdir.parent / 'checkpoints'))
        )
        checkpoint_dir = ckpt_parent / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        state_dim, action_dim = memory_train_lib._infer_dims(store)
        task_dataset = ICILMAMLTaskBatchIterable(
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
        task_builder = MAMLTaskBuilder(
            store=store,
            cfg=dataset_cfg,
            seed=data_seed,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )
        excluded_task_builder = (
            MAMLTaskBuilder(
                store=excluded_store,
                cfg=dataset_cfg,
                seed=data_seed + 12345,
                num_tries_per_item=int(cfg.dataset.num_tries_per_item),
            )
            if excluded_store is not None
            else None
        )

        policy = build_policy(model_cfg, state_dim=state_dim, action_dim=action_dim).to(device)
        if resume_state_dict is not None:
            policy.load_state_dict(resume_state_dict, strict=True)
            logging.info('Resumed model weights from %s', resume_path)
        elif pretrained_state_dict is not None:
            load_result = policy.load_state_dict(
                pretrained_state_dict,
                strict=memory_train_lib._as_bool(cfg.finetune.strict_load),
            )
            if memory_train_lib._as_bool(cfg.finetune.strict_load):
                logging.info('Loaded pretrained checkpoint from %s', pretrained_path)
            else:
                logging.info(
                    'Loaded pretrained checkpoint from %s with missing_keys=%s unexpected_keys=%s',
                    pretrained_path,
                    load_result.missing_keys,
                    load_result.unexpected_keys,
                )

        use_mask_id = memory_train_lib._resolve_use_mask_id(model_cfg)
        resolved_inner_lr_mode = memory_train_lib._resolve_inner_lr_mode(
            cfg,
            resume_checkpoint=resume_checkpoint,
            pretrained_checkpoint=pretrained_checkpoint if pretrained_path is not None and resume_path is None else None,
            resume_config=resume_config,
            pretrained_config=pretrained_config,
        )
        memory_cfg = memory_train_lib._build_memory_cfg(cfg)
        memory_cfg.inner_lr_mode = str(resolved_inner_lr_mode)
        inner_lr_schedule = memory_train_lib._build_inner_lr_schedule_wrapper(memory_cfg)
        if inner_lr_schedule is not None:
            inner_lr_schedule = inner_lr_schedule.to(device)
        memory_train_lib._load_inner_lr_schedule_state(
            inner_lr_schedule,
            checkpoint=resume_checkpoint,
            checkpoint_path=resume_path,
        )
        outer_names = get_outer_param_names(
            policy,
            train_encoder=memory_train_lib._as_bool(cfg.outer.train_encoder),
            train_decoder=memory_train_lib._as_bool(cfg.outer.train_decoder),
            train_input_projections=memory_train_lib._as_bool(cfg.outer.train_input_projections),
            train_output_head=memory_train_lib._as_bool(cfg.outer.train_output_head),
            train_diffusion_conditioning=memory_train_lib._as_bool(cfg.outer.train_diffusion_conditioning),
        )
        set_outer_trainable_params(policy, outer_names)
        outer_params = [param for param in policy.parameters() if param.requires_grad]
        inner_lr_params = list(inner_lr_schedule.parameters()) if inner_lr_schedule is not None else []
        optimizer = torch.optim.AdamW(
            outer_params + inner_lr_params,
            lr=float(memory_cfg.outer_lr),
            weight_decay=float(memory_cfg.weight_decay),
        )

        global_step = 0
        if resume_checkpoint is not None:
            optimizer_state = resume_checkpoint.get('optimizer', None)
            optimizer_resumed = False
            if isinstance(optimizer_state, dict):
                try:
                    optimizer.load_state_dict(optimizer_state)
                    optimizer_resumed = True
                except ValueError as exc:
                    logging.warning('Skipping optimizer state load from %s due to mismatch: %s', resume_path, exc)
            global_step = int(resume_checkpoint.get('step', 0))
            if optimizer_resumed:
                logging.info('Resumed optimizer state from %s at step=%d', resume_path, global_step)

        n_total, n_trainable = memory_train_lib._count_parameters(policy)
        n_outer = count_params_by_name(policy, outer_names)
        n_inner_lr = sum(int(param.numel()) for param in inner_lr_params)
        resolved_inner_lrs = memory_train_lib._inner_lr_values(schedule=inner_lr_schedule, cfg=memory_cfg)
        logging.info('Run id=%s', run_id)
        logging.info('Output dir=%s', workdir)
        logging.info('Checkpoint dir=%s', checkpoint_dir)
        logging.info('Using cache_root=%s', cache_root)
        logging.info('Tasks=%s | variations=%d', tasks_used, len(store))
        logging.info('Excluded tasks=%s', exclude_tasks)
        if excluded_store is not None:
            logging.info('Excluded-task sampling store=%s | variations=%d', excluded_tasks_used, len(excluded_store))
        logging.info(
            'Model params: total=%s (%.3fM) | trainable=%s (%.3fM) | outer=%s (%.3fM) | inner_lr=%s',
            f'{n_total:,}',
            n_total / 1e6,
            f'{n_trainable + n_inner_lr:,}',
            (n_trainable + n_inner_lr) / 1e6,
            f'{n_outer + n_inner_lr:,}',
            (n_outer + n_inner_lr) / 1e6,
            f'{n_inner_lr:,}',
        )
        logging.info(
            'Resolved memory-%s DDP setup: model_source=%s | data.K=%d | memory_support=K-1=%d | '
            'outer_param_tensors=%d | inner_steps=%d | inner_batch=%d | query_batch=%d | grad_accum=%d | '
            'distributed=%s | world_size=%d | effective_outer_batch=%d | inner_lr_mode=%s | inner_lrs=%s',
            'FOMAML' if first_order else 'MAML',
            model_cfg_source,
            int(dataset_cfg.K),
            int(dataset_cfg.K) - 1,
            len(outer_names) + len(inner_lr_params),
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
        logging.info(
            'Resolved dataset: K=%d | L=%d | T_obs=%d | H=%d | stride=%d',
            int(dataset_cfg.K),
            int(dataset_cfg.L),
            int(dataset_cfg.T_obs),
            int(dataset_cfg.H),
            int(dataset_cfg.stride),
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
        config_payload['maml']['num_queries_per_step'] = int(memory_cfg.num_queries_per_step)
        config_payload['maml']['num_inner_batches'] = int(memory_cfg.num_inner_batches)
        config_payload['maml']['num_query_loss_samples'] = int(memory_cfg.num_query_loss_samples)
        config_payload['maml']['holdout_index'] = int(memory_cfg.holdout_index)
        config_payload['maml']['grad_accum_steps'] = int(memory_cfg.grad_accum_steps)
        config_payload['maml']['inner_lr_mode'] = str(memory_cfg.inner_lr_mode)
        config_payload['algorithm'] = 'memory_fomaml_ddp' if bool(first_order) else 'memory_maml_ddp'
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
            'pretrained_checkpoint': str(pretrained_path) if pretrained_path is not None else '',
            'model_source': model_cfg_source,
            'data_k': int(resolved_data_k),
            'memory_support_size': int(dataset_cfg.K) - 1,
            'initial_global_step': int(global_step),
            'outer_param_names': list(outer_names),
            'fast_object': 'context_encoder.support_tokens',
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
                    'outer': config_payload['outer'],
                },
                allow_val_change=True,
            )
            wandb_run.save(str(config_path), policy='now')
            wandb_run.log(
                {
                    'model/num_params_total': n_total,
                    'model/num_params_trainable': n_trainable + n_inner_lr,
                    'model/num_params_outer': n_outer + n_inner_lr,
                    'model/num_params_inner_lr': n_inner_lr,
                },
                step=global_step,
            )

        log_every = int(cfg.train.log_every)
        ckpt_every = int(cfg.train.ckpt_every)
        wandb_loss_every = int(getattr(cfg.wandb, 'n_loss_steps', 0)) if wandb_run is not None else 0
        wandb_sample_every = int(getattr(cfg.wandb, 'n_sample_steps', 0)) if wandb_run is not None else 0
        wandb_inner_loss_every = int(getattr(cfg.wandb, 'n_inner_loss_steps', 0)) if wandb_run is not None else 0
        wandb_sample_batch = int(getattr(cfg.wandb, 'sample_batch_items', 4)) if wandb_run is not None else 0
        wandb_sample_mse_items = int(getattr(cfg.wandb, 'sample_mse_items', wandb_sample_batch)) if wandb_run is not None else 0
        wandb_sample_inference_steps = int(getattr(cfg.wandb, 'sample_inference_steps', 0)) if wandb_run is not None else 0
        wandb_sample_eta = float(getattr(cfg.wandb, 'sample_eta', 0.0)) if wandb_run is not None else 0.0
        wandb_include_query_pc = (
            memory_train_lib._as_bool(getattr(cfg.wandb, 'include_query_pointcloud_in_x0_pred_vs_gt_3d', False))
            if wandb_run is not None
            else False
        )
        wandb_query_pc_max_points = int(getattr(cfg.wandb, 'query_pointcloud_max_points', 2048)) if wandb_run is not None else 2048

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
            torch_seed = data_seed + 2_000_003 + global_step
            torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
            torch_gen.manual_seed(torch_seed)
            prepared_tasks = prepare_outer_batch_for_memory_meta_step(
                tasks_batch,
                task_builder=task_builder,
                cfg=memory_cfg,
                device=device,
                num_train_timesteps=int(policy.noise_scheduler.config.num_train_timesteps),
                action_dim=int(action_dim),
                use_mask_id=use_mask_id,
                rng=np_rng,
                torch_generator=torch_gen,
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
            _all_reduce_outer_grads(outer_params + inner_lr_params, device)
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

            if (
                is_main
                and wandb_run is not None
                and wandb_inner_loss_every > 0
                and (global_step % wandb_inner_loss_every == 0 or global_step == 1)
            ):
                max_diag_tasks = max(1, min(len(prepared_tasks), max(1, wandb_sample_batch)))
                query_diffusion_curve, query_sample_mse_curve = memory_inner_loop_query_curves(
                    policy=policy,
                    prepared_tasks=prepared_tasks,
                    cfg=memory_cfg,
                    inference_steps=wandb_sample_inference_steps,
                    eta=wandb_sample_eta,
                    max_tasks=max_diag_tasks,
                    inner_lr_schedule=inner_lr_schedule,
                )
                fig_diff = plot_scalar_curve(
                    query_diffusion_curve,
                    ylabel='query diffusion loss',
                    title='Query diffusion loss vs inner step',
                    log_y=True,
                )
                fig_mse = plot_scalar_curve(
                    query_sample_mse_curve,
                    ylabel='query sampled action MSE',
                    title='Query sampled action MSE vs inner step',
                    log_y=True,
                )
                log_dict: Dict[str, Any] = {'train/step': global_step}
                import wandb

                if fig_diff is not None:
                    log_dict['inner_loop/query_diffusion_loss'] = wandb.Image(fig_diff)
                if fig_mse is not None:
                    log_dict['inner_loop/query_sample_mse'] = wandb.Image(fig_mse)
                wandb_run.log(log_dict, step=global_step)
                if fig_diff is not None or fig_mse is not None:
                    try:
                        import matplotlib.pyplot as plt

                        if fig_diff is not None:
                            plt.close(fig_diff)
                        if fig_mse is not None:
                            plt.close(fig_mse)
                    except Exception:
                        pass

            if is_main and wandb_run is not None and wandb_sample_every > 0 and (global_step % wandb_sample_every == 0):
                sample_tasks = list(tasks_batch[: max(0, min(len(tasks_batch), wandb_sample_batch))])
                pred_x0 = None
                gt_x0 = None
                query_xyz = None
                query_valid = None
                if sample_tasks:
                    pred_x0, gt_x0, query_xyz, query_valid = memory_train_lib._sample_adapted_queries_for_tasks(
                        policy=policy,
                        tasks=sample_tasks,
                        task_builder=task_builder,
                        memory_cfg=memory_cfg,
                        inner_lr_schedule=inner_lr_schedule,
                        device=device,
                        use_mask_id=use_mask_id,
                        inference_steps=wandb_sample_inference_steps,
                        eta=wandb_sample_eta,
                        seed=seed + 3_000_003 + global_step,
                    )
                sample_mse = memory_train_lib._estimate_adapted_x0_mse(
                    policy=policy,
                    store=store,
                    dataset_cfg=dataset_cfg,
                    task_builder=task_builder,
                    memory_cfg=memory_cfg,
                    inner_lr_schedule=inner_lr_schedule,
                    total_items=wandb_sample_mse_items,
                    per_batch_items=max(1, wandb_sample_batch),
                    seed=seed + 4_000_003 + global_step,
                    num_tries_per_item=int(cfg.dataset.num_tries_per_item),
                    device=device,
                    use_mask_id=use_mask_id,
                    inference_steps=wandb_sample_inference_steps,
                    eta=wandb_sample_eta,
                )
                sample_mse_excluded = memory_train_lib._estimate_adapted_x0_mse(
                    policy=policy,
                    store=excluded_store,
                    dataset_cfg=dataset_cfg,
                    task_builder=excluded_task_builder if excluded_task_builder is not None else task_builder,
                    memory_cfg=memory_cfg,
                    inner_lr_schedule=inner_lr_schedule,
                    total_items=wandb_sample_mse_items,
                    per_batch_items=max(1, wandb_sample_batch),
                    seed=seed + 5_000_003 + global_step,
                    num_tries_per_item=int(cfg.dataset.num_tries_per_item),
                    device=device,
                    use_mask_id=use_mask_id,
                    inference_steps=wandb_sample_inference_steps,
                    eta=wandb_sample_eta,
                )

                fig = None
                if pred_x0 is not None and gt_x0 is not None:
                    fig = memory_train_lib._plot_pred_vs_gt_3d(
                        pred_x0=pred_x0,
                        gt_x0=gt_x0,
                        max_items=max(1, wandb_sample_batch),
                        include_query_pointcloud=wandb_include_query_pc,
                        query_xyz=query_xyz,
                        query_valid=query_valid,
                        max_query_points=wandb_query_pc_max_points,
                    )
                log_dict: Dict[str, Any] = {'train/step': global_step}
                if sample_mse is not None:
                    log_dict['samples/x0_mse'] = float(sample_mse)
                if sample_mse_excluded is not None:
                    log_dict['samples_excluded/x0_mse'] = float(sample_mse_excluded)
                if fig is not None:
                    import wandb

                    log_dict['samples/x0_pred_vs_gt_3d'] = wandb.Image(fig)
                wandb_run.log(log_dict, step=global_step)
                if fig is not None:
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass

            if is_main and ckpt_every > 0 and global_step % ckpt_every == 0:
                ckpt_path = checkpoint_dir / f'step_{global_step:07d}.pt'
                memory_train_lib._save_checkpoint(
                    ckpt_path,
                    step=global_step,
                    model=policy,
                    optimizer=optimizer,
                    config_payload=config_payload,
                    extra_state=(
                        {'inner_lr_schedule': inner_lr_schedule.state_dict()}
                        if inner_lr_schedule is not None
                        else None
                    ),
                )
                logging.info('Saved checkpoint: %s', ckpt_path)

        if is_main:
            final_ckpt = checkpoint_dir / 'last.pt'
            memory_train_lib._save_checkpoint(
                final_ckpt,
                step=global_step,
                model=policy,
                optimizer=optimizer,
                config_payload=config_payload,
                extra_state=(
                    {'inner_lr_schedule': inner_lr_schedule.state_dict()}
                    if inner_lr_schedule is not None
                    else None
                ),
            )
            logging.info('Training complete. Final checkpoint: %s', final_ckpt)
        if distributed:
            dist.barrier()

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if excluded_store is not None:
            excluded_store.close()
        store.close()
        _cleanup_distributed()


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == '__main__':
    app.run(main)
