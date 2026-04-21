from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.models import build_policy
from icil.models.maml import (
    ICILMAMLTaskBatchIterable,
    MAMLConfig,
    MAMLTaskBuilder,
    MAMLTaskSpec,
    PolicyLossWrapper,
    adapt_fast_params_for_prepared_task,
    copy_fast_params_into_policy,
    count_params_by_name,
    get_fast_param_names,
    get_outer_param_names,
    maml_step_with_stats,
    prefix_param_names,
    prepare_outer_batch_for_meta_step,
    set_outer_trainable_params,
)
from icil.models.maml.diagnostics import (
    parameter_inner_loop_query_curves,
    plot_scalar_curve,
)
from icil.models.maml.inner_lr import (
    PositiveInnerLRSchedule,
    build_inner_lr_schedule,
    infer_inner_lr_mode,
    inner_lr_log_dict,
    resolved_inner_lr_values,
)
from icil.models.maml.train_utils import (
    build_model_cfg as _build_model_cfg,
    build_optional_store as _build_optional_store,
    build_store as _build_store,
    count_parameters as _count_parameters,
    infer_dims as _infer_dims,
    maybe_init_wandb as _maybe_init_wandb,
    normalize_task_list as _normalize_task_list,
    plot_denoising_trace_3d as _plot_denoising_trace_3d,
    plot_pred_vs_gt_3d as _plot_pred_vs_gt_3d,
    resolve_run_id as _resolve_run_id,
    resolve_use_mask_id as _resolve_use_mask_id,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/maml_perceiver_encoder_decoder.py',
    help_string='Path to a ml_collections config file.',
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _resolve_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and str(device_str).startswith('cuda'):
        return torch.device(str(device_str))
    return torch.device('cpu')


def _unwrap_batch(batch_list: List[Any]) -> Any:
    return batch_list[0]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
        return {key[len('module.'):]: value for key, value in state_dict.items()}
    return state_dict


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint object type: {type(checkpoint).__name__}')
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' payload is not a state_dict dictionary.")
    return checkpoint, _strip_module_prefix(state_dict)


def _configdict_from_dict(data: Dict[str, Any]) -> ConfigDict:
    return ConfigDict(data)


def _resolve_model_cfg(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> Tuple[Any, str]:
    if isinstance(resume_config, dict) and isinstance(resume_config.get('model'), dict):
        return _build_model_cfg(_configdict_from_dict(resume_config['model'])), 'resume_checkpoint'
    if isinstance(pretrained_config, dict) and isinstance(pretrained_config.get('model'), dict):
        return _build_model_cfg(_configdict_from_dict(pretrained_config['model'])), 'pretrained_checkpoint'
    return _build_model_cfg(cfg.model), 'config'


def _resolve_data_k(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> int:
    configured_k = int(cfg.dataset.K)
    if configured_k > 0:
        return configured_k
    if isinstance(resume_config, dict):
        dataset_cfg = resume_config.get('dataset', {})
        if isinstance(dataset_cfg, dict) and 'K' in dataset_cfg:
            resolved_k = int(dataset_cfg['K'])
            if resolved_k >= 2:
                return resolved_k
    if isinstance(pretrained_config, dict):
        dataset_cfg = pretrained_config.get('dataset', {})
        if isinstance(dataset_cfg, dict) and 'K' in dataset_cfg:
            pretrained_k = int(dataset_cfg['K'])
            if pretrained_k > 0:
                return pretrained_k + 1
    raise ValueError(
        'cfg.dataset.K=0 requires either cfg.train.resume_path or '
        'cfg.finetune.pretrained_checkpoint with a valid dataset.K.'
    )


def _resolve_outer_context_size(
    cfg: ConfigDict,
    *,
    data_k: int,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> int:
    configured_size = int(cfg.maml.outer_context_size)
    if configured_size > 0:
        resolved = configured_size
    else:
        resolved = 0
        if isinstance(resume_config, dict):
            maml_cfg = resume_config.get('maml', {})
            if isinstance(maml_cfg, dict) and int(maml_cfg.get('outer_context_size', 0)) > 0:
                resolved = int(maml_cfg['outer_context_size'])
            elif isinstance(resume_config.get('dataset', {}), dict) and int(resume_config['dataset'].get('K', 0)) > 0:
                resolved = int(resume_config['dataset']['K'])
        if resolved <= 0 and isinstance(pretrained_config, dict):
            dataset_cfg = pretrained_config.get('dataset', {})
            if isinstance(dataset_cfg, dict) and int(dataset_cfg.get('K', 0)) > 0:
                resolved = int(dataset_cfg['K'])
        if resolved <= 0:
            resolved = int(data_k)
    if resolved <= 0:
        raise ValueError('outer_context_size must be positive.')
    if resolved > data_k:
        raise ValueError(f'outer_context_size={resolved} exceeds data.K={data_k}.')
    return resolved


def _resolve_checkpoint_field(
    *,
    local_value: Any,
    section_name: str,
    field_name: str,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
    transform: Optional[Any] = None,
) -> Any:
    for source_config in (resume_config, pretrained_config):
        if not isinstance(source_config, dict):
            continue
        section = source_config.get(section_name, {})
        if isinstance(section, dict) and field_name in section:
            value = section[field_name]
            return transform(value) if transform is not None else value
    return transform(local_value) if transform is not None else local_value


def _resolve_dataset_cfg(
    cfg: ConfigDict,
    *,
    resolved_data_k: int,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> ICILConfig:
    return ICILConfig(
        K=int(resolved_data_k),
        L=_resolve_checkpoint_field(
            local_value=cfg.dataset.L,
            section_name='dataset',
            field_name='L',
            resume_config=resume_config,
            pretrained_config=pretrained_config,
            transform=int,
        ),
        T_obs=_resolve_checkpoint_field(
            local_value=cfg.dataset.T_obs,
            section_name='dataset',
            field_name='T_obs',
            resume_config=resume_config,
            pretrained_config=pretrained_config,
            transform=int,
        ),
        H=_resolve_checkpoint_field(
            local_value=cfg.dataset.H,
            section_name='dataset',
            field_name='H',
            resume_config=resume_config,
            pretrained_config=pretrained_config,
            transform=int,
        ),
        stride=_resolve_checkpoint_field(
            local_value=cfg.dataset.stride,
            section_name='dataset',
            field_name='stride',
            resume_config=resume_config,
            pretrained_config=pretrained_config,
            transform=int,
        ),
        task_sampling=str(getattr(cfg.data, 'task_sampling', 'variation_power')),
        task_sampling_alpha=float(getattr(cfg.data, 'task_sampling_alpha', 1.0)),
    )


def _resolve_task_filters(
    cfg: ConfigDict,
    *,
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    tasks = _resolve_checkpoint_field(
        local_value=getattr(cfg.data, 'tasks', ()),
        section_name='data',
        field_name='tasks',
        resume_config=resume_config,
        pretrained_config=pretrained_config,
        transform=_normalize_task_list,
    )
    exclude_tasks = _resolve_checkpoint_field(
        local_value=getattr(cfg.data, 'exclude_tasks', ()),
        section_name='data',
        field_name='exclude_tasks',
        resume_config=resume_config,
        pretrained_config=pretrained_config,
        transform=_normalize_task_list,
    )
    return tasks, exclude_tasks


def _resolve_inner_lr_mode(
    cfg: ConfigDict,
    *,
    resume_checkpoint: Optional[Dict[str, Any]],
    pretrained_checkpoint: Optional[Dict[str, Any]],
    resume_config: Optional[Dict[str, Any]],
    pretrained_config: Optional[Dict[str, Any]],
) -> str:
    if isinstance(resume_config, dict):
        return infer_inner_lr_mode(
            checkpoint=resume_checkpoint,
            checkpoint_config=resume_config,
            local_mode=None,
            legacy_learn_inner_lrs=getattr(cfg.maml, 'learn_inner_lrs', None),
        )
    if isinstance(pretrained_config, dict):
        inferred = infer_inner_lr_mode(
            checkpoint=pretrained_checkpoint,
            checkpoint_config=pretrained_config,
            local_mode=None,
            legacy_learn_inner_lrs=getattr(cfg.maml, 'learn_inner_lrs', None),
        )
        if inferred != 'fixed':
            return inferred
    return infer_inner_lr_mode(
        checkpoint=resume_checkpoint if resume_checkpoint is not None else pretrained_checkpoint,
        local_mode=getattr(cfg.maml, 'inner_lr_mode', 'fixed'),
        legacy_learn_inner_lrs=getattr(cfg.maml, 'learn_inner_lrs', None),
    )


def _inner_lr_values(
    *,
    schedule: Optional[PositiveInnerLRSchedule],
    cfg: MAMLConfig,
) -> List[float]:
    return resolved_inner_lr_values(
        mode=getattr(cfg, 'inner_lr_mode', 'fixed'),
        inner_steps=int(cfg.inner_steps),
        fixed_inner_lr=float(cfg.inner_lr),
        schedule=schedule,
    )


def _inner_lr_log_dict(
    *,
    schedule: Optional[PositiveInnerLRSchedule],
    cfg: MAMLConfig,
    prefix: str = 'train',
) -> Dict[str, float]:
    return inner_lr_log_dict(
        mode=getattr(cfg, 'inner_lr_mode', 'fixed'),
        inner_steps=int(cfg.inner_steps),
        fixed_inner_lr=float(cfg.inner_lr),
        schedule=schedule,
        prefix=prefix,
    )


def _build_logging_task_batch(
    *,
    store: Any,
    dataset_cfg: ICILConfig,
    batch_size: int,
    seed: int,
    num_tries_per_item: int,
) -> Optional[List[MAMLTaskSpec]]:
    if store is None or int(batch_size) <= 0:
        return None
    dataset = ICILMAMLTaskBatchIterable(
        store=store,
        cfg=dataset_cfg,
        task_batch_size_B=int(batch_size),
        num_batches=1,
        seed=int(seed),
        num_tries_per_item=int(num_tries_per_item),
    )
    try:
        return next(iter(dataset))
    except StopIteration:
        return None
    except RuntimeError as exc:
        logging.warning('Skipping MAML logging task batch (batch_size=%d): %s', int(batch_size), exc)
        return None


def _sample_adapted_queries_for_tasks(
    *,
    policy: torch.nn.Module,
    loss_wrapper: PolicyLossWrapper,
    tasks: Sequence[MAMLTaskSpec],
    task_builder: MAMLTaskBuilder,
    fast_names: Sequence[str],
    maml_cfg: MAMLConfig,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule],
    device: torch.device,
    use_mask_id: bool,
    inference_steps: int,
    eta: float,
    return_trace: bool,
    trace_steps: Optional[int],
    seed: int,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
]:
    if not tasks:
        return None, None, None, None, None

    base_params = dict(loss_wrapper.named_parameters())
    buffers = dict(loss_wrapper.named_buffers())
    adapted_policy = copy.deepcopy(policy).to(device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)

    pred_x0_items: List[torch.Tensor] = []
    gt_x0_items: List[torch.Tensor] = []
    query_xyz_items: List[torch.Tensor] = []
    query_valid_items: List[torch.Tensor] = []
    trace_items: List[torch.Tensor] = []
    trace_timesteps: Optional[torch.Tensor] = None

    for task_idx, task in enumerate(tasks):
        np_rng = np.random.default_rng(int(seed) + task_idx)
        torch_seed = int(seed) + 1_000_003 + task_idx
        torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
        torch_gen.manual_seed(torch_seed)
        prepared_task = prepare_outer_batch_for_meta_step(
            [task],
            task_builder=task_builder,
            cfg=maml_cfg,
            device=device,
            num_train_timesteps=int(policy.noise_scheduler.config.num_train_timesteps),
            action_dim=int(policy.action_dim),
            use_mask_id=use_mask_id,
            rng=np_rng,
            torch_generator=torch_gen,
        )[0]

        with torch.enable_grad():
            adapted_params = adapt_fast_params_for_prepared_task(
                loss_wrapper,
                prepared_task,
                fast_names=fast_names,
                cfg=maml_cfg,
                inner_lr_schedule=inner_lr_schedule,
                create_graph=False,
                base_params=base_params,
                buffers=buffers,
            )
        copy_fast_params_into_policy(
            adapted_policy,
            adapted_params=adapted_params,
            fast_names=fast_names,
        )

        query_batch = prepared_task['query_batch']
        with torch.no_grad():
            sample_out = adapted_policy.sample_actions(
                cond_xyz=query_batch.get('cond_xyz', None),
                cond_state=query_batch.get('cond_state', None),
                cond_traj=query_batch.get('cond_traj', None),
                cond_traj_mask=query_batch.get('cond_traj_mask', None),
                query_xyz=query_batch['query_xyz'],
                query_state=query_batch['query_state'],
                action_horizon=int(query_batch['target_action'].shape[1]),
                cond_rgb=query_batch.get('cond_rgb', None),
                query_rgb=query_batch.get('query_rgb', None),
                cond_mask_id=query_batch.get('cond_mask_id', None) if use_mask_id else None,
                query_mask_id=query_batch.get('query_mask_id', None) if use_mask_id else None,
                cond_valid=query_batch.get('cond_valid', None),
                query_valid=query_batch.get('query_valid', None),
                inference_steps=(int(inference_steps) if int(inference_steps) > 0 else None),
                eta=float(eta),
                return_trace=return_trace,
                trace_steps=trace_steps,
            )
        if isinstance(sample_out, tuple):
            pred_x0, denoise_trace = sample_out
        else:
            pred_x0, denoise_trace = sample_out, None
        pred_x0_items.append(pred_x0)
        gt_x0_items.append(query_batch['target_action'])
        query_xyz_items.append(query_batch['query_xyz'])
        query_valid_items.append(query_batch['query_valid'])
        if denoise_trace is not None:
            trace_items.append(denoise_trace['x0_hat'])
            if trace_timesteps is None:
                trace_timesteps = denoise_trace['timesteps']

    pred_x0_batch = torch.cat(pred_x0_items, dim=0)
    gt_x0_batch = torch.cat(gt_x0_items, dim=0)
    query_xyz_batch = torch.cat(query_xyz_items, dim=0)
    query_valid_batch = torch.cat(query_valid_items, dim=0)

    trace_out = None
    if trace_items and trace_timesteps is not None:
        trace_out = {
            'x0_hat': torch.cat(trace_items, dim=1),
            'timesteps': trace_timesteps,
        }
    return pred_x0_batch, gt_x0_batch, query_xyz_batch, query_valid_batch, trace_out


@torch.no_grad()
def _estimate_adapted_x0_mse(
    *,
    policy: torch.nn.Module,
    loss_wrapper: PolicyLossWrapper,
    store: Any,
    dataset_cfg: ICILConfig,
    task_builder: MAMLTaskBuilder,
    fast_names: Sequence[str],
    maml_cfg: MAMLConfig,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule],
    total_items: int,
    per_batch_items: int,
    seed: int,
    num_tries_per_item: int,
    device: torch.device,
    use_mask_id: bool,
    inference_steps: int,
    eta: float,
) -> Optional[float]:
    if store is None or int(total_items) <= 0:
        return None

    remaining = int(total_items)
    seed_cursor = int(seed)
    mse_sum = 0.0
    n_seen = 0
    chunk_size = max(1, int(per_batch_items))
    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        task_batch = _build_logging_task_batch(
            store=store,
            dataset_cfg=dataset_cfg,
            batch_size=batch_size,
            seed=seed_cursor,
            num_tries_per_item=num_tries_per_item,
        )
        seed_cursor += 1
        if not task_batch:
            break
        pred_x0, gt_x0, _, _, _ = _sample_adapted_queries_for_tasks(
            policy=policy,
            loss_wrapper=loss_wrapper,
            tasks=task_batch,
            task_builder=task_builder,
            fast_names=fast_names,
            maml_cfg=maml_cfg,
            inner_lr_schedule=inner_lr_schedule,
            device=device,
            use_mask_id=use_mask_id,
            inference_steps=inference_steps,
            eta=eta,
            return_trace=False,
            trace_steps=None,
            seed=seed_cursor + 10_000,
        )
        if pred_x0 is None or gt_x0 is None:
            break
        mse_sum += float(F.mse_loss(pred_x0, gt_x0, reduction='sum').detach().cpu())
        n_seen += int(gt_x0.numel())
        remaining -= int(gt_x0.shape[0])

    if n_seen == 0:
        return None
    return mse_sum / float(n_seen)


def _save_checkpoint(
    ckpt_path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_payload: Dict[str, Any],
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': int(step),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config_payload,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, ckpt_path)


def train(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    device = _resolve_device(str(cfg.device))

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
        resume_checkpoint, resume_state_dict = _load_checkpoint(resume_path, device)
        config_obj = resume_checkpoint.get('config', None)
        if isinstance(config_obj, dict):
            resume_config = config_obj

    pretrained_checkpoint = None
    pretrained_state_dict = None
    pretrained_config = None
    if pretrained_path is not None and resume_path is None:
        pretrained_checkpoint, pretrained_state_dict = _load_checkpoint(pretrained_path, device)
        config_obj = pretrained_checkpoint.get('config', None)
        if not isinstance(config_obj, dict):
            raise ValueError(
                f'Pretrained checkpoint at {pretrained_path} does not contain a valid config dict.'
            )
        pretrained_config = config_obj

    model_cfg, model_cfg_source = _resolve_model_cfg(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    resolved_data_k = _resolve_data_k(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    resolved_outer_context_size = _resolve_outer_context_size(
        cfg,
        data_k=resolved_data_k,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    dataset_cfg = _resolve_dataset_cfg(
        cfg,
        resolved_data_k=resolved_data_k,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )

    cache_root = Path(str(cfg.data.cache_root))
    tasks, exclude_tasks = _resolve_task_filters(
        cfg,
        resume_config=resume_config,
        pretrained_config=pretrained_config,
    )
    store, tasks_used = _build_store(
        cache_root=cache_root,
        tasks=tasks,
        exclude_tasks=exclude_tasks,
        keep_open_per_worker=_as_bool(cfg.data.keep_open_per_worker),
    )
    excluded_store, excluded_tasks_used = _build_optional_store(
        cache_root=cache_root,
        tasks=exclude_tasks,
        keep_open_per_worker=_as_bool(cfg.data.keep_open_per_worker),
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
        wandb_run = _maybe_init_wandb(cfg, output_parent)
        run_id = _resolve_run_id(wandb_run)
        if wandb_run is not None:
            wandb_run.name = run_id

        workdir = output_parent / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        ckpt_parent = Path(
            str(
                getattr(
                    cfg.train,
                    'checkpoint_parent_dir',
                    workdir.parent / 'checkpoints',
                )
            )
        )
        checkpoint_dir = ckpt_parent / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state_dim, action_dim = _infer_dims(store)
        task_dataset = ICILMAMLTaskBatchIterable(
            store=store,
            cfg=dataset_cfg,
            task_batch_size_B=int(cfg.train.batch_size),
            num_batches=int(cfg.train.num_steps),
            seed=seed,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )
        num_workers = int(cfg.data.num_workers)
        pin_memory = _as_bool(cfg.data.pin_memory) and device.type == 'cuda'
        persistent_workers = _as_bool(cfg.data.persistent_workers) and num_workers > 0
        task_loader = DataLoader(
            task_dataset,
            batch_size=1,
            collate_fn=_unwrap_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        task_builder = MAMLTaskBuilder(
            store=store,
            cfg=dataset_cfg,
            seed=seed,
            num_tries_per_item=int(cfg.dataset.num_tries_per_item),
        )
        excluded_task_builder = (
            MAMLTaskBuilder(
                store=excluded_store,
                cfg=dataset_cfg,
                seed=seed + 12345,
                num_tries_per_item=int(cfg.dataset.num_tries_per_item),
            )
            if excluded_store is not None
            else None
        )

        policy = build_policy(
            model_cfg,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)
        if resume_state_dict is not None:
            policy.load_state_dict(resume_state_dict, strict=True)
            logging.info('Resumed model weights from %s', resume_path)
        elif pretrained_state_dict is not None:
            load_result = policy.load_state_dict(
                pretrained_state_dict,
                strict=_as_bool(cfg.finetune.strict_load),
            )
            if _as_bool(cfg.finetune.strict_load):
                logging.info('Loaded pretrained checkpoint from %s', pretrained_path)
            else:
                logging.info(
                    'Loaded pretrained checkpoint from %s with missing_keys=%s unexpected_keys=%s',
                    pretrained_path,
                    load_result.missing_keys,
                    load_result.unexpected_keys,
                )

        use_mask_id = _resolve_use_mask_id(model_cfg)
        resolved_inner_lr_mode = _resolve_inner_lr_mode(
            cfg,
            resume_checkpoint=resume_checkpoint,
            pretrained_checkpoint=pretrained_checkpoint if pretrained_path is not None and resume_path is None else None,
            resume_config=resume_config,
            pretrained_config=pretrained_config,
        )
        maml_cfg = MAMLConfig(
            inner_steps=int(cfg.maml.inner_steps),
            inner_lr=float(cfg.maml.inner_lr),
            inner_lr_mode=str(resolved_inner_lr_mode),
            outer_lr=float(cfg.maml.outer_lr),
            weight_decay=float(cfg.train.weight_decay),
            max_grad_norm=float(cfg.maml.max_grad_norm),
            last_frac_fast=float(cfg.maml.last_frac_fast),
            include_decoder_mlp_fast=_as_bool(getattr(cfg.maml, 'include_decoder_mlp_fast', True)),
            include_ada_fast=_as_bool(cfg.maml.include_ada_fast),
            include_final_norm_fast=_as_bool(cfg.maml.include_final_norm_fast),
            include_input_projections_fast=_as_bool(getattr(cfg.maml, 'include_input_projections_fast', False)),
            include_output_head_fast=_as_bool(getattr(cfg.maml, 'include_output_head_fast', False)),
            include_diffusion_conditioning_fast=_as_bool(
                getattr(cfg.maml, 'include_diffusion_conditioning_fast', False)
            ),
            num_loo_per_task=int(cfg.maml.num_loo_per_task),
            outer_context_size=int(resolved_outer_context_size),
            reuse_diffusion_noise=_as_bool(cfg.maml.reuse_diffusion_noise),
        )
        inner_lr_schedule = build_inner_lr_schedule(
            mode=maml_cfg.inner_lr_mode,
            inner_steps=int(maml_cfg.inner_steps),
            init_lr=float(maml_cfg.inner_lr),
        )
        if inner_lr_schedule is not None:
            inner_lr_schedule = inner_lr_schedule.to(device)
            if isinstance(resume_checkpoint, dict) and isinstance(resume_checkpoint.get('inner_lr_schedule'), dict):
                inner_lr_schedule.load_state_dict(resume_checkpoint['inner_lr_schedule'], strict=True)
                logging.info('Resumed learned inner LR schedule from %s', resume_path)
        loss_wrapper = PolicyLossWrapper(policy)

        fast_names = get_fast_param_names(
            policy,
            last_frac=maml_cfg.last_frac_fast,
            include_decoder_mlp=maml_cfg.include_decoder_mlp_fast,
            include_ada=maml_cfg.include_ada_fast,
            include_final_norm=maml_cfg.include_final_norm_fast,
            include_input_projections=maml_cfg.include_input_projections_fast,
            include_output_head=maml_cfg.include_output_head_fast,
            include_diffusion_conditioning=maml_cfg.include_diffusion_conditioning_fast,
        )
        outer_names = get_outer_param_names(
            policy,
            train_encoder=_as_bool(cfg.outer.train_encoder),
            train_decoder=_as_bool(cfg.outer.train_decoder),
            train_input_projections=_as_bool(cfg.outer.train_input_projections),
            train_output_head=_as_bool(cfg.outer.train_output_head),
            train_diffusion_conditioning=_as_bool(cfg.outer.train_diffusion_conditioning),
        )
        missing_fast_outer = sorted(set(fast_names) - set(outer_names))
        if missing_fast_outer:
            raise ValueError(
                'Fast parameters must also be outer-trainable. '
                f'Examples: {missing_fast_outer[:5]}'
            )
        set_outer_trainable_params(policy, outer_names)
        fast_names_wrapped = prefix_param_names(fast_names)

        outer_params = [param for param in policy.parameters() if param.requires_grad]
        inner_lr_params = list(inner_lr_schedule.parameters()) if inner_lr_schedule is not None else []
        optimizer = torch.optim.AdamW(
            outer_params + inner_lr_params,
            lr=float(maml_cfg.outer_lr),
            weight_decay=float(maml_cfg.weight_decay),
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

        n_total, n_trainable = _count_parameters(policy)
        n_fast = count_params_by_name(policy, fast_names)
        n_outer = count_params_by_name(policy, outer_names)
        n_inner_lr = sum(int(param.numel()) for param in inner_lr_params)
        resolved_inner_lrs = _inner_lr_values(schedule=inner_lr_schedule, cfg=maml_cfg)
        logging.info('Run id=%s', run_id)
        logging.info('Output dir=%s', workdir)
        logging.info('Checkpoint dir=%s', checkpoint_dir)
        logging.info('Using cache_root=%s', cache_root)
        logging.info('Tasks=%s | variations=%d', tasks_used, len(store))
        logging.info('Excluded tasks=%s', exclude_tasks)
        if excluded_store is not None:
            logging.info(
                'Excluded-task sampling store=%s | variations=%d',
                excluded_tasks_used,
                len(excluded_store),
            )
        logging.info(
            'Model params: total=%s (%.3fM) | trainable=%s (%.3fM) | fast=%s (%.3fM) | inner_lr=%s',
            f'{n_total:,}',
            n_total / 1e6,
            f'{n_trainable + n_inner_lr:,}',
            (n_trainable + n_inner_lr) / 1e6,
            f'{n_fast:,}',
            n_fast / 1e6,
            f'{n_inner_lr:,}',
        )
        logging.info(
            'Resolved MAML setup: model_source=%s | data.K=%d | outer_context_size=%d | '
            'fast_param_tensors=%d | outer_param_tensors=%d | outer_param_count=%s | encoder_trainable=%s | '
            'inner_lr_mode=%s | inner_lrs=%s',
            model_cfg_source,
            resolved_data_k,
            resolved_outer_context_size,
            len(fast_names),
            len(outer_names) + len(inner_lr_params),
            f'{n_outer + n_inner_lr:,}',
            str(_as_bool(cfg.outer.train_encoder)),
            str(maml_cfg.inner_lr_mode),
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
        config_payload['maml']['outer_context_size'] = int(resolved_outer_context_size)
        config_payload['maml']['inner_lr_mode'] = str(maml_cfg.inner_lr_mode)
        config_payload['runtime'] = {
            'run_id': run_id,
            'output_dir': str(workdir),
            'checkpoint_dir': str(checkpoint_dir),
        }
        config_payload['resolved'] = {
            'resume_path': str(resume_path) if resume_path is not None else '',
            'pretrained_checkpoint': str(pretrained_path) if pretrained_path is not None else '',
            'model_source': model_cfg_source,
            'data_k': int(resolved_data_k),
            'outer_context_size': int(resolved_outer_context_size),
            'initial_global_step': int(global_step),
            'fast_param_names': list(fast_names),
            'outer_param_names': list(outer_names),
            'inner_lr_mode': str(maml_cfg.inner_lr_mode),
            'initial_inner_lrs': resolved_inner_lrs,
        }
        config_path = workdir / 'config.json'
        with config_path.open('w', encoding='utf-8') as f:
            json.dump(config_payload, f, indent=2)
        if wandb_run is not None:
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
            wandb_run.log(
                {
                    'model/num_params_total': n_total,
                    'model/num_params_trainable': n_trainable + n_inner_lr,
                    'model/num_params_fast': n_fast,
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
        wandb_sample_mse_items = (
            int(getattr(cfg.wandb, 'sample_mse_items', wandb_sample_batch)) if wandb_run is not None else 0
        )
        wandb_sample_inference_steps = (
            int(getattr(cfg.wandb, 'sample_inference_steps', 0)) if wandb_run is not None else 0
        )
        wandb_sample_eta = float(getattr(cfg.wandb, 'sample_eta', 0.0)) if wandb_run is not None else 0.0
        wandb_sample_trace_frames = int(getattr(cfg.wandb, 'sample_trace_frames', 8)) if wandb_run is not None else 0
        wandb_include_query_pc = (
            _as_bool(getattr(cfg.wandb, 'include_query_pointcloud_in_x0_pred_vs_gt_3d', False))
            if wandb_run is not None
            else False
        )
        wandb_query_pc_max_points = (
            int(getattr(cfg.wandb, 'query_pointcloud_max_points', 2048))
            if wandb_run is not None
            else 2048
        )

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        log_loss = 0.0
        log_count = 0
        wb_loss_sum = 0.0
        wb_inner_fast_grad_norm_sum = 0.0
        wb_count = 0
        window_start = time.time()

        for tasks_batch in task_loader:
            if global_step >= int(cfg.train.num_steps):
                break

            np_rng = np.random.default_rng(seed + 1_000_003 + global_step)
            torch_seed = seed + 2_000_003 + global_step
            torch_gen = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
            torch_gen.manual_seed(torch_seed)
            prepared_tasks = prepare_outer_batch_for_meta_step(
                tasks_batch,
                task_builder=task_builder,
                cfg=maml_cfg,
                device=device,
                num_train_timesteps=int(policy.noise_scheduler.config.num_train_timesteps),
                action_dim=int(action_dim),
                use_mask_id=use_mask_id,
                rng=np_rng,
                torch_generator=torch_gen,
            )

            optimizer.zero_grad(set_to_none=True)
            meta_loss, avg_inner_fast_grad_norm = maml_step_with_stats(
                loss_wrapper,
                prepared_tasks,
                fast_names=fast_names_wrapped,
                cfg=maml_cfg,
                inner_lr_schedule=inner_lr_schedule,
            )
            meta_loss.backward()
            all_outer_trainables = outer_params + inner_lr_params
            if float(maml_cfg.max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(all_outer_trainables, float(maml_cfg.max_grad_norm))
            optimizer.step()
            global_step += 1

            loss_value = float(meta_loss.detach().cpu())
            log_loss += loss_value
            log_count += 1
            wb_loss_sum += loss_value
            wb_inner_fast_grad_norm_sum += float(avg_inner_fast_grad_norm)
            wb_count += 1

            if log_every > 0 and (global_step % log_every == 0 or global_step == 1):
                elapsed = max(1e-6, time.time() - window_start)
                steps_per_sec = log_count / elapsed
                avg_loss = log_loss / max(1, log_count)
                logging.info(
                    'step %d/%d | meta_loss %.6f | outer_lr %.3e | inner_lr_mean %.3e | %.2f step/s',
                    global_step,
                    int(cfg.train.num_steps),
                    avg_loss,
                    float(optimizer.param_groups[0]['lr']),
                    _inner_lr_log_dict(schedule=inner_lr_schedule, cfg=maml_cfg).get('train/inner_lr_mean', float(maml_cfg.inner_lr)),
                    steps_per_sec,
                )
                log_loss = 0.0
                log_count = 0
                window_start = time.time()

            if wandb_run is not None and wandb_loss_every > 0 and (global_step % wandb_loss_every == 0 or global_step == 1):
                wandb_run.log(
                    {
                        'train/meta_loss': wb_loss_sum / max(1, wb_count),
                        'train/outer_loss': wb_loss_sum / max(1, wb_count),
                        'train/inner_fast_grad_norm': wb_inner_fast_grad_norm_sum / max(1, wb_count),
                        'train/lr': float(optimizer.param_groups[0]['lr']),
                        'train/step': global_step,
                        **_inner_lr_log_dict(schedule=inner_lr_schedule, cfg=maml_cfg),
                    },
                    step=global_step,
                )
                wb_loss_sum = 0.0
                wb_inner_fast_grad_norm_sum = 0.0
                wb_count = 0

            if (
                wandb_run is not None
                and wandb_inner_loss_every > 0
                and (global_step % wandb_inner_loss_every == 0 or global_step == 1)
            ):
                max_diag_tasks = max(1, min(len(prepared_tasks), max(1, wandb_sample_batch)))
                query_diffusion_curve, query_sample_mse_curve = parameter_inner_loop_query_curves(
                    policy=policy,
                    loss_wrapper=loss_wrapper,
                    prepared_tasks=prepared_tasks,
                    fast_names=fast_names_wrapped,
                    cfg=maml_cfg,
                    inference_steps=wandb_sample_inference_steps,
                    eta=wandb_sample_eta,
                    use_mask_id=use_mask_id,
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

            if wandb_run is not None and wandb_sample_every > 0 and (global_step % wandb_sample_every == 0):
                sample_tasks = list(tasks_batch[: max(0, min(len(tasks_batch), wandb_sample_batch))])
                pred_x0 = None
                gt_x0 = None
                query_xyz = None
                query_valid = None
                denoise_trace = None
                pred_x0_excluded = None
                gt_x0_excluded = None
                query_xyz_excluded = None
                query_valid_excluded = None
                denoise_trace_excluded = None
                sample_mse = None
                sample_mse_excluded = None

                if sample_tasks:
                    pred_x0, gt_x0, query_xyz, query_valid, denoise_trace = _sample_adapted_queries_for_tasks(
                        policy=policy,
                        loss_wrapper=loss_wrapper,
                        tasks=sample_tasks,
                        task_builder=task_builder,
                        fast_names=fast_names_wrapped,
                        maml_cfg=maml_cfg,
                        inner_lr_schedule=inner_lr_schedule,
                        device=device,
                        use_mask_id=use_mask_id,
                        inference_steps=wandb_sample_inference_steps,
                        eta=wandb_sample_eta,
                        return_trace=True,
                        trace_steps=(wandb_sample_trace_frames if wandb_sample_trace_frames > 0 else None),
                        seed=seed + 3_000_003 + global_step,
                    )
                sample_mse = _estimate_adapted_x0_mse(
                    policy=policy,
                    loss_wrapper=loss_wrapper,
                    store=store,
                    dataset_cfg=dataset_cfg,
                    task_builder=task_builder,
                    fast_names=fast_names_wrapped,
                    maml_cfg=maml_cfg,
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
                excluded_tasks = _build_logging_task_batch(
                    store=excluded_store,
                    dataset_cfg=dataset_cfg,
                    batch_size=wandb_sample_batch,
                    seed=seed + 5_000_003 + global_step,
                    num_tries_per_item=int(cfg.dataset.num_tries_per_item),
                )
                if excluded_tasks and excluded_task_builder is not None:
                    (
                        pred_x0_excluded,
                        gt_x0_excluded,
                        query_xyz_excluded,
                        query_valid_excluded,
                        denoise_trace_excluded,
                    ) = _sample_adapted_queries_for_tasks(
                        policy=policy,
                        loss_wrapper=loss_wrapper,
                        tasks=excluded_tasks,
                        task_builder=excluded_task_builder,
                        fast_names=fast_names_wrapped,
                        maml_cfg=maml_cfg,
                        inner_lr_schedule=inner_lr_schedule,
                        device=device,
                        use_mask_id=use_mask_id,
                        inference_steps=wandb_sample_inference_steps,
                        eta=wandb_sample_eta,
                        return_trace=True,
                        trace_steps=(wandb_sample_trace_frames if wandb_sample_trace_frames > 0 else None),
                        seed=seed + 6_000_003 + global_step,
                    )
                sample_mse_excluded = _estimate_adapted_x0_mse(
                    policy=policy,
                    loss_wrapper=loss_wrapper,
                    store=excluded_store,
                    dataset_cfg=dataset_cfg,
                    task_builder=excluded_task_builder if excluded_task_builder is not None else task_builder,
                    fast_names=fast_names_wrapped,
                    maml_cfg=maml_cfg,
                    inner_lr_schedule=inner_lr_schedule,
                    total_items=wandb_sample_mse_items,
                    per_batch_items=max(1, wandb_sample_batch),
                    seed=seed + 7_000_003 + global_step,
                    num_tries_per_item=int(cfg.dataset.num_tries_per_item),
                    device=device,
                    use_mask_id=use_mask_id,
                    inference_steps=wandb_sample_inference_steps,
                    eta=wandb_sample_eta,
                )

                fig = None
                fig_trace = None
                fig_excluded = None
                fig_trace_excluded = None
                if pred_x0 is not None and gt_x0 is not None:
                    fig = _plot_pred_vs_gt_3d(
                        pred_x0=pred_x0,
                        gt_x0=gt_x0,
                        max_items=max(1, wandb_sample_batch),
                        include_query_pointcloud=wandb_include_query_pc,
                        query_xyz=query_xyz,
                        query_valid=query_valid,
                        max_query_points=wandb_query_pc_max_points,
                    )
                if denoise_trace is not None:
                    fig_trace = _plot_denoising_trace_3d(
                        denoise_trace['x0_hat'],
                        denoise_trace['timesteps'],
                        max_items=max(1, min(2, wandb_sample_batch)),
                    )
                if pred_x0_excluded is not None and gt_x0_excluded is not None:
                    fig_excluded = _plot_pred_vs_gt_3d(
                        pred_x0=pred_x0_excluded,
                        gt_x0=gt_x0_excluded,
                        max_items=max(1, wandb_sample_batch),
                        include_query_pointcloud=wandb_include_query_pc,
                        query_xyz=query_xyz_excluded,
                        query_valid=query_valid_excluded,
                        max_query_points=wandb_query_pc_max_points,
                    )
                if denoise_trace_excluded is not None:
                    fig_trace_excluded = _plot_denoising_trace_3d(
                        denoise_trace_excluded['x0_hat'],
                        denoise_trace_excluded['timesteps'],
                        max_items=max(1, min(2, wandb_sample_batch)),
                    )

                log_dict: Dict[str, Any] = {
                    'train/step': global_step,
                }
                if sample_mse is not None:
                    log_dict['samples/x0_mse'] = float(sample_mse)
                if sample_mse_excluded is not None:
                    log_dict['samples_excluded/x0_mse'] = float(sample_mse_excluded)
                if fig is not None or fig_trace is not None or fig_excluded is not None or fig_trace_excluded is not None:
                    import wandb

                    if fig is not None:
                        log_dict['samples/x0_pred_vs_gt_3d'] = wandb.Image(fig)
                    if fig_trace is not None:
                        log_dict['samples/x0_denoising_trace_3d'] = wandb.Image(fig_trace)
                    if fig_excluded is not None:
                        log_dict['samples_excluded/x0_pred_vs_gt_3d'] = wandb.Image(fig_excluded)
                    if fig_trace_excluded is not None:
                        log_dict['samples_excluded/x0_denoising_trace_3d'] = wandb.Image(fig_trace_excluded)
                wandb_run.log(log_dict, step=global_step)
                if fig is not None or fig_trace is not None or fig_excluded is not None or fig_trace_excluded is not None:
                    try:
                        import matplotlib.pyplot as plt

                        if fig is not None:
                            plt.close(fig)
                        if fig_trace is not None:
                            plt.close(fig_trace)
                        if fig_excluded is not None:
                            plt.close(fig_excluded)
                        if fig_trace_excluded is not None:
                            plt.close(fig_trace_excluded)
                    except Exception:
                        pass

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                ckpt_path = checkpoint_dir / f'step_{global_step:07d}.pt'
                _save_checkpoint(
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

        final_ckpt = checkpoint_dir / 'last.pt'
        _save_checkpoint(
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

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if excluded_store is not None:
            excluded_store.close()
        store.close()


def main(argv=None):
    del argv
    cfg = _CONFIG.value
    train(cfg)


if __name__ == '__main__':
    app.run(main)
