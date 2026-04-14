from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from icil.models.maml.tasks import MAMLTaskBuilder, MAMLTaskSpec


@dataclass
class MAMLConfig:
    inner_steps: int = 1
    inner_lr: float = 1e-4
    outer_lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    last_frac_fast: float = 0.25
    include_decoder_mlp_fast: bool = True
    include_ada_fast: bool = True
    include_final_norm_fast: bool = True
    include_input_projections_fast: bool = False
    include_output_head_fast: bool = False
    include_diffusion_conditioning_fast: bool = False
    num_loo_per_task: int = 2
    outer_context_size: int = 0
    reuse_diffusion_noise: bool = True


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop("cond_mask_id", None)
    out.pop("query_mask_id", None)
    return out


def _clip_grads_in_list(grads: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    if max_norm <= 0.0:
        return grads
    valid_grads = [grad for grad in grads if grad is not None]
    if not valid_grads:
        return grads
    total_norm = torch.norm(torch.stack([grad.norm(2) for grad in valid_grads]), 2)
    if total_norm <= max_norm:
        return grads
    scale = max_norm / (total_norm + 1e-6)
    return [grad * scale if grad is not None else None for grad in grads]


def _grad_list_global_norm(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    valid_grads = [grad.detach() for grad in grads if grad is not None]
    if not valid_grads:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.norm(torch.stack([grad.norm(2) for grad in valid_grads]), 2)


def _sample_loo_indices(
    K: int,
    *,
    num_loo_per_task: int,
    rng: np.random.Generator,
) -> List[int]:
    if num_loo_per_task >= K:
        return list(range(K))
    perm = rng.permutation(K)
    return [int(idx) for idx in perm[:num_loo_per_task].tolist()]


def prepare_task_for_meta_step(
    task: MAMLTaskSpec,
    *,
    task_builder: MAMLTaskBuilder,
    cfg: MAMLConfig,
    device: torch.device,
    num_train_timesteps: int,
    action_dim: int,
    use_mask_id: bool,
    rng: np.random.Generator,
    torch_generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    support_count = len(task.support_episode_ids)
    loo_indices = _sample_loo_indices(
        support_count,
        num_loo_per_task=int(cfg.num_loo_per_task),
        rng=rng,
    )

    shared_noise = None
    shared_timesteps = None
    if bool(cfg.reuse_diffusion_noise):
        H = int(task_builder.cfg.H)
        shared_noise = torch.randn(
            (1, H, int(action_dim)),
            device=device,
            dtype=torch.float32,
            generator=torch_generator,
        )
        shared_timesteps = torch.randint(
            low=0,
            high=int(num_train_timesteps),
            size=(1,),
            device=device,
            dtype=torch.long,
            generator=torch_generator,
        )

    support_batches: List[Dict[str, Any]] = []
    for _ in range(int(cfg.inner_steps)):
        support_batch = task_builder.build_support_batch_loo_cached(
            task,
            holdout_indices=loo_indices,
            rng=rng,
            noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
            timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
            load_mask_id=use_mask_id,
        )
        support_batch = _to_device(support_batch, device)
        support_batches.append(_drop_mask_ids_if_disabled(support_batch, use_mask_id))

    num_context_episodes = int(cfg.outer_context_size) if int(cfg.outer_context_size) > 0 else None
    query_batch = task_builder.build_query_batch(
        task,
        rng=rng,
        num_context_episodes=num_context_episodes,
        noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
        timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
        load_mask_id=use_mask_id,
    )
    query_batch = _to_device(query_batch, device)
    query_batch = _drop_mask_ids_if_disabled(query_batch, use_mask_id)

    return {
        "task": task,
        "support_batches": support_batches,
        "query_batch": query_batch,
        "holdout_indices": loo_indices,
    }


def prepare_outer_batch_for_meta_step(
    tasks: Sequence[MAMLTaskSpec],
    *,
    task_builder: MAMLTaskBuilder,
    cfg: MAMLConfig,
    device: torch.device,
    num_train_timesteps: int,
    action_dim: int,
    use_mask_id: bool,
    rng: np.random.Generator,
    torch_generator: Optional[torch.Generator] = None,
) -> List[Dict[str, Any]]:
    prepared_tasks: List[Dict[str, Any]] = []
    for task in tasks:
        prepared_tasks.append(
            prepare_task_for_meta_step(
                task,
                task_builder=task_builder,
                cfg=cfg,
                device=device,
                num_train_timesteps=num_train_timesteps,
                action_dim=action_dim,
                use_mask_id=use_mask_id,
                rng=rng,
                torch_generator=torch_generator,
            )
        )
    return prepared_tasks


def adapt_fast_params_for_prepared_task(
    model: nn.Module,
    prepared_task: Dict[str, Any],
    *,
    fast_names: Sequence[str],
    cfg: MAMLConfig,
    create_graph: bool,
    base_params: Optional[Dict[str, torch.Tensor]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
    inner_grad_norms_out: Optional[List[torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    adapted_params = base_params if base_params is not None else dict(model.named_parameters())
    model_buffers = buffers if buffers is not None else dict(model.named_buffers())

    for support_batch in prepared_task["support_batches"]:
        support_loss = functional_call(model, (adapted_params, model_buffers), (support_batch,))
        fast_tensors = [adapted_params[name] for name in fast_names]
        grads = torch.autograd.grad(
            support_loss,
            fast_tensors,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        if inner_grad_norms_out is not None:
            inner_grad_norms_out.append(_grad_list_global_norm(grads))
        grads = _clip_grads_in_list(list(grads), float(cfg.max_grad_norm))

        new_params = dict(adapted_params)
        for name, param, grad in zip(fast_names, fast_tensors, grads):
            new_params[name] = param - float(cfg.inner_lr) * grad
        adapted_params = new_params

    return adapted_params


def maml_task_loss_second_order(
    model: nn.Module,
    prepared_task: Dict[str, Any],
    *,
    fast_names: Sequence[str],
    cfg: MAMLConfig,
    base_params: Optional[Dict[str, torch.Tensor]] = None,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    adapted_params = adapt_fast_params_for_prepared_task(
        model,
        prepared_task,
        fast_names=fast_names,
        cfg=cfg,
        create_graph=True,
        base_params=base_params,
        buffers=buffers,
    )
    model_buffers = buffers if buffers is not None else dict(model.named_buffers())
    return functional_call(model, (adapted_params, model_buffers), (prepared_task["query_batch"],))


def maml_step(
    model: nn.Module,
    prepared_tasks: Sequence[Dict[str, Any]],
    *,
    fast_names: Sequence[str],
    cfg: MAMLConfig,
) -> torch.Tensor:
    base_params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    losses: List[torch.Tensor] = []
    for prepared_task in prepared_tasks:
        losses.append(
            maml_task_loss_second_order(
                model,
                prepared_task,
                fast_names=fast_names,
                cfg=cfg,
                base_params=base_params,
                buffers=buffers,
            )
        )
    return torch.stack(losses).mean()


def maml_step_with_stats(
    model: nn.Module,
    prepared_tasks: Sequence[Dict[str, Any]],
    *,
    fast_names: Sequence[str],
    cfg: MAMLConfig,
) -> tuple[torch.Tensor, float]:
    base_params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    losses: List[torch.Tensor] = []
    inner_grad_norms: List[torch.Tensor] = []
    for prepared_task in prepared_tasks:
        adapted_params = adapt_fast_params_for_prepared_task(
            model,
            prepared_task,
            fast_names=fast_names,
            cfg=cfg,
            create_graph=True,
            base_params=base_params,
            buffers=buffers,
            inner_grad_norms_out=inner_grad_norms,
        )
        losses.append(functional_call(model, (adapted_params, buffers), (prepared_task["query_batch"],)))
    meta_loss = torch.stack(losses).mean()
    avg_inner_grad_norm = (
        float(torch.stack(inner_grad_norms).mean().item()) if inner_grad_norms else 0.0
    )
    return meta_loss, avg_inner_grad_norm


def copy_fast_params_into_policy(
    target_policy: nn.Module,
    *,
    adapted_params: Dict[str, torch.Tensor],
    fast_names: Sequence[str],
    wrapper_prefix: str = "policy.",
) -> None:
    target_params = dict(target_policy.named_parameters())
    with torch.no_grad():
        for fast_name in fast_names:
            target_name = fast_name
            if target_name.startswith(wrapper_prefix):
                target_name = target_name[len(wrapper_prefix) :]
            if target_name not in target_params:
                raise KeyError(f"Target policy is missing fast parameter '{target_name}'.")
            target_params[target_name].copy_(adapted_params[fast_name].detach())
