from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call

from icil.models.maml.core import copy_fast_params_into_policy
from icil.models.maml.inner_lr import (
    PositiveInnerLRSchedule,
    inner_lr_tensor_for_step,
    normalize_inner_lr_mode,
)
from icil.models.maml.memory_core import (
    _clip_grad,
    _iter_microbatches,
    init_memory_tokens_from_batch,
    memory_diffusion_loss,
    sample_actions_with_memory_tokens,
)
from icil.models.policies.policy import Policy


def plot_scalar_curve(
    values: Sequence[float],
    *,
    ylabel: str,
    title: str,
    xlabel: str = 'inner step',
    log_y: bool = True,
) -> Optional[Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    xs = list(range(len(values)))
    ys = [float(v) for v in values]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        positive = [v for v in ys if v > 0.0]
        if positive:
            ax.set_yscale('log')
    if len(xs) > 10:
        step = max(1, len(xs) // 8)
        ticks = xs[::step]
        if ticks[-1] != xs[-1]:
            ticks.append(xs[-1])
        ax.set_xticks(ticks)
    else:
        ax.set_xticks(xs)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _sample_policy_actions(
    policy: Policy,
    batch: Dict[str, Any],
    *,
    inference_steps: int,
    eta: float,
    use_mask_id: bool,
) -> torch.Tensor:
    return policy.sample_actions(
        cond_xyz=batch.get('cond_xyz', None),
        cond_state=batch.get('cond_state', None),
        cond_traj=batch.get('cond_traj', None),
        cond_traj_mask=batch.get('cond_traj_mask', None),
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        action_horizon=int(batch['target_action'].shape[1]),
        cond_rgb=batch.get('cond_rgb', None),
        query_rgb=batch.get('query_rgb', None),
        cond_mask_id=batch.get('cond_mask_id', None) if use_mask_id else None,
        query_mask_id=batch.get('query_mask_id', None) if use_mask_id else None,
        cond_valid=batch.get('cond_valid', None),
        query_valid=batch.get('query_valid', None),
        inference_steps=(int(inference_steps) if int(inference_steps) > 0 else None),
        eta=float(eta),
    )


def _clip_parameter_grads(grads: Sequence[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    out = list(grads)
    if float(max_norm) <= 0.0:
        return out
    valid = [grad for grad in out if grad is not None]
    if not valid:
        return out
    total_norm = torch.norm(torch.stack([grad.norm(2) for grad in valid]), 2)
    if total_norm <= float(max_norm):
        return out
    scale = float(max_norm) / (total_norm + 1e-6)
    return [grad * scale if grad is not None else None for grad in out]


def parameter_inner_loop_query_curves(
    *,
    policy: Policy,
    loss_wrapper: torch.nn.Module,
    prepared_tasks: Sequence[Dict[str, Any]],
    fast_names: Sequence[str],
    cfg: Any,
    inference_steps: int,
    eta: float,
    use_mask_id: bool,
    max_tasks: int,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule] = None,
) -> Tuple[List[float], List[float]]:
    selected_tasks = list(prepared_tasks[: max(1, int(max_tasks))])
    if not selected_tasks:
        return [], []

    num_steps = int(cfg.inner_steps)
    diffusion_sums = [0.0 for _ in range(num_steps + 1)]
    sample_mse_sums = [0.0 for _ in range(num_steps + 1)]
    count = 0

    was_training = policy.training
    policy.eval()
    loss_wrapper.eval()
    adapted_policy = copy.deepcopy(policy).to(next(policy.parameters()).device)
    adapted_policy.eval()
    for param in adapted_policy.parameters():
        param.requires_grad_(False)
    inner_lr_mode = normalize_inner_lr_mode(getattr(cfg, 'inner_lr_mode', 'fixed'))

    try:
        for prepared_task in selected_tasks:
            adapted_params = dict(loss_wrapper.named_parameters())
            buffers = dict(loss_wrapper.named_buffers())
            query_batch = prepared_task['query_batch']
            support_batches = list(prepared_task.get('support_batches', []))
            if num_steps > 0 and not support_batches:
                raise ValueError('Parameter inner-loop diagnostics require prepared support batches.')

            def eval_query(step_idx: int) -> None:
                with torch.no_grad():
                    query_loss = functional_call(loss_wrapper, (adapted_params, buffers), (query_batch,))
                copy_fast_params_into_policy(
                    adapted_policy,
                    adapted_params=adapted_params,
                    fast_names=fast_names,
                )
                with torch.no_grad():
                    pred = _sample_policy_actions(
                        adapted_policy,
                        query_batch,
                        inference_steps=int(inference_steps),
                        eta=float(eta),
                        use_mask_id=use_mask_id,
                    )
                    sample_mse = F.mse_loss(pred, query_batch['target_action'])
                diffusion_sums[step_idx] += float(query_loss.detach().cpu())
                sample_mse_sums[step_idx] += float(sample_mse.detach().cpu())

            eval_query(0)
            for step_idx in range(1, num_steps + 1):
                support_batch = support_batches[(step_idx - 1) % len(support_batches)]
                support_loss = functional_call(loss_wrapper, (adapted_params, buffers), (support_batch,))
                fast_tensors = [adapted_params[name] for name in fast_names]
                grads = torch.autograd.grad(
                    support_loss,
                    fast_tensors,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
                grads = _clip_parameter_grads(grads, float(cfg.max_grad_norm))
                new_params = dict(adapted_params)
                for name, param, grad in zip(fast_names, fast_tensors, grads):
                    step_lr = inner_lr_tensor_for_step(
                        step_idx=step_idx - 1,
                        mode=inner_lr_mode,
                        fixed_inner_lr=float(cfg.inner_lr),
                        schedule=inner_lr_schedule,
                        device=param.device,
                        dtype=param.dtype,
                    )
                    new_params[name] = (param - step_lr * grad).detach().requires_grad_(True)
                adapted_params = new_params
                eval_query(step_idx)
            count += 1
    finally:
        policy.train(was_training)
        loss_wrapper.train(was_training)

    denom = float(max(1, count))
    return [v / denom for v in diffusion_sums], [v / denom for v in sample_mse_sums]


def memory_inner_loop_query_curves(
    *,
    policy: Policy,
    prepared_tasks: Sequence[Dict[str, Any]],
    cfg: Any,
    inference_steps: int,
    eta: float,
    max_tasks: int,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule] = None,
) -> Tuple[List[float], List[float]]:
    selected_tasks = list(prepared_tasks[: max(1, int(max_tasks))])
    if not selected_tasks:
        return [], []

    num_steps = int(cfg.inner_steps)
    diffusion_sums = [0.0 for _ in range(num_steps + 1)]
    sample_mse_sums = [0.0 for _ in range(num_steps + 1)]
    count = 0

    was_training = policy.training
    policy.eval()
    inner_lr_mode = normalize_inner_lr_mode(getattr(cfg, 'inner_lr_mode', 'fixed'))
    try:
        for prepared_task in selected_tasks:
            memory_tokens, memory_token_mask = init_memory_tokens_from_batch(
                policy,
                prepared_task['memory_init_batch'],
            )
            inner_batches = list(prepared_task.get('inner_batches', []))
            query_batch = prepared_task['query_batch']
            if num_steps > 0 and not inner_batches:
                raise ValueError('Memory inner-loop diagnostics require prepared inner batches.')

            def eval_query(step_idx: int) -> None:
                with torch.no_grad():
                    query_loss = memory_diffusion_loss(
                        policy,
                        query_batch,
                        memory_tokens=memory_tokens,
                        memory_token_mask=memory_token_mask,
                    )
                    pred = sample_actions_with_memory_tokens(
                        policy,
                        query_batch,
                        memory_tokens=memory_tokens,
                        memory_token_mask=memory_token_mask,
                        inference_steps=(int(inference_steps) if int(inference_steps) > 0 else None),
                        eta=float(eta),
                    )
                    sample_mse = F.mse_loss(pred, query_batch['target_action'])
                diffusion_sums[step_idx] += float(query_loss.detach().cpu())
                sample_mse_sums[step_idx] += float(sample_mse.detach().cpu())

            eval_query(0)
            for step_idx in range(1, num_steps + 1):
                inner_batch = inner_batches[(step_idx - 1) % len(inner_batches)]
                grad_accum_steps = int(getattr(cfg, 'grad_accum_steps', 1))
                if grad_accum_steps == 1:
                    support_loss = memory_diffusion_loss(
                        policy,
                        inner_batch,
                        memory_tokens=memory_tokens,
                        memory_token_mask=memory_token_mask,
                    )
                    grad = torch.autograd.grad(
                        support_loss,
                        memory_tokens,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )[0]
                else:
                    grad = None
                    for micro_batch, weight in _iter_microbatches(inner_batch, grad_accum_steps):
                        micro_loss = memory_diffusion_loss(
                            policy,
                            micro_batch,
                            memory_tokens=memory_tokens,
                            memory_token_mask=memory_token_mask,
                        )
                        micro_grad = torch.autograd.grad(
                            micro_loss * float(weight),
                            memory_tokens,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=False,
                        )[0]
                        grad = micro_grad if grad is None else grad + micro_grad
                    if grad is None:
                        raise RuntimeError('No microbatches were produced for memory inner-loop diagnostics.')
                grad = _clip_grad(grad, float(cfg.max_grad_norm))
                step_lr = inner_lr_tensor_for_step(
                    step_idx=step_idx - 1,
                    mode=inner_lr_mode,
                    fixed_inner_lr=float(cfg.inner_lr),
                    schedule=inner_lr_schedule,
                    device=memory_tokens.device,
                    dtype=memory_tokens.dtype,
                )
                memory_tokens = (memory_tokens - step_lr * grad).detach().requires_grad_(True)
                eval_query(step_idx)
            count += 1
    finally:
        policy.train(was_training)

    denom = float(max(1, count))
    return [v / denom for v in diffusion_sums], [v / denom for v in sample_mse_sums]
