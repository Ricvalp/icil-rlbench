from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from icil.models.common import sinusoidal_position_embedding, sinusoidal_time_embedding
from icil.models.maml.inner_lr import (
    PositiveInnerLRSchedule,
    inner_lr_tensor_for_step,
    normalize_inner_lr_mode,
)
from icil.models.maml.tasks import MAMLTaskBuilder, MAMLTaskSpec


def _is_diffusion_policy(model: torch.nn.Module) -> bool:
    return hasattr(model, 'noise_scheduler') and hasattr(model, 'predict_model_output')


def _is_direct_regression_policy(model: torch.nn.Module) -> bool:
    return hasattr(model, 'decoder') and hasattr(model, 'forward_actions')


@dataclass
class MemoryMAMLConfig:
    inner_steps: int = 1
    inner_lr: float = 1e-4
    inner_lr_mode: str = 'fixed'
    outer_lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_queries_per_step: int = 1
    num_inner_batches: int = 0
    num_query_loss_samples: int = 1
    holdout_index: int = -1
    reuse_diffusion_noise: bool = False
    grad_accum_steps: int = 1

def _as_bool(value: Any) -> bool:
    return bool(value)


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop('cond_mask_id', None)
    out.pop('query_mask_id', None)
    return out


def _sample_balanced_indices(total: int, *, count: int, rng: np.random.Generator) -> List[int]:
    if total < 1:
        raise ValueError(f'total must be positive, got {total}.')
    if count < 1:
        raise ValueError(f'count must be positive, got {count}.')
    out: List[int] = []
    while len(out) < int(count):
        perm = rng.permutation(total)
        take = min(total, int(count) - len(out))
        out.extend(int(idx) for idx in perm[:take].tolist())
    return out


def _num_valid_query_t0s(task_builder: MAMLTaskBuilder, *, vidx: int, episode_id: int) -> int:
    T = int(task_builder.store.episode_length(int(vidx), int(episode_id)))
    required = 1 + ((int(task_builder.cfg.T_obs) - 1) * int(task_builder.cfg.stride))
    return max(0, T - required + 1)


def _sample_query_t0s_for_episode(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    episode_id: int,
    count: int,
    rng: np.random.Generator,
) -> List[int]:
    num_valid = _num_valid_query_t0s(task_builder, vidx=int(vidx), episode_id=int(episode_id))
    if num_valid < 1:
        raise RuntimeError(
            f'No valid query windows for vidx={vidx}, episode_id={episode_id}. '
            f'Need at least T_obs={int(task_builder.cfg.T_obs)} observed frames with stride={int(task_builder.cfg.stride)}.'
        )
    return _sample_balanced_indices(num_valid, count=int(count), rng=rng)


def _build_query_sample_at_t0(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    episode_id: int,
    t0: int,
    load_rgb: bool,
    load_mask_id: bool,
) -> Dict[str, Any]:
    episode_length = int(task_builder.store.episode_length(int(vidx), int(episode_id)))
    obs_idx, act_idx = task_builder._build_obs_act_indices(int(t0), episode_length=episode_length)
    q_obs = task_builder.store.load_episode_slices(
        int(vidx),
        int(episode_id),
        obs_idx,
        load_rgb=load_rgb,
        load_mask_id=load_mask_id,
        load_full_traj=False,
    )
    q_act = task_builder.store.load_episode_slices(
        int(vidx),
        int(episode_id),
        act_idx,
        load_rgb=False,
        load_mask_id=False,
        load_full_traj=False,
    )

    sample: Dict[str, Any] = {
        'query_xyz': q_obs['xyz'],
        'query_state': q_obs['state'],
        'query_valid': q_obs['valid'],
        'target_action': task_builder._encode_target_action(q_obs['state'], q_act['action']),
        'meta': {
            'vidx': int(vidx),
            'query_episode': int(episode_id),
            't0': int(t0),
        },
    }
    if load_mask_id and 'mask_id' in q_obs:
        sample['query_mask_id'] = q_obs['mask_id']
    if load_rgb and 'rgb' in q_obs:
        sample['query_rgb'] = q_obs['rgb']
    return sample


def _build_memory_query_batch(
    task_builder: MAMLTaskBuilder,
    *,
    vidx: int,
    support_cond: Dict[str, Any],
    support_ids: Sequence[int],
    query_episode_id: int,
    count: int,
    rng: np.random.Generator,
    noise: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    load_rgb: bool = True,
    load_mask_id: bool = True,
) -> Dict[str, Any]:
    if int(count) < 1:
        raise ValueError(f'count must be >= 1, got {count}.')
    sampled_t0s = _sample_query_t0s_for_episode(
        task_builder,
        vidx=int(vidx),
        episode_id=int(query_episode_id),
        count=int(count),
        rng=rng,
    )
    samples: List[Dict[str, Any]] = []
    for t0 in sampled_t0s:
        query = _build_query_sample_at_t0(
            task_builder,
            vidx=int(vidx),
            episode_id=int(query_episode_id),
            t0=int(t0),
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        query['meta'].update(
            {
                'memory_support_episodes': [int(ep_id) for ep_id in support_ids],
                'memory_query_episode': int(query_episode_id),
            }
        )
        samples.append({**support_cond, **query})
    batch = task_builder._stack_samples(samples)
    task_builder.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
    return batch


def _resolve_num_inner_batches(cfg: MemoryMAMLConfig) -> int:
    inner_steps = int(cfg.inner_steps)
    configured = int(cfg.num_inner_batches)
    if inner_steps <= 0:
        return 0
    if configured <= 0:
        return inner_steps
    return min(configured, inner_steps)


def _select_holdout_index(K: int, cfg: MemoryMAMLConfig, rng: np.random.Generator) -> int:
    configured = int(cfg.holdout_index)
    if configured >= 0:
        if configured >= int(K):
            raise ValueError(f'holdout_index={configured} out of range for K={K}.')
        return configured
    return int(rng.integers(low=0, high=int(K)))


def prepare_memory_task_for_meta_step(
    task: MAMLTaskSpec,
    *,
    task_builder: MAMLTaskBuilder,
    cfg: MemoryMAMLConfig,
    device: torch.device,
    num_train_timesteps: int,
    action_dim: int,
    use_mask_id: bool,
    rng: np.random.Generator,
    torch_generator: Optional[torch.Generator] = None,
    load_rgb: bool = True,
) -> Dict[str, Any]:
    support_ids = [int(ep_id) for ep_id in task.support_episode_ids]
    if len(support_ids) < 2:
        raise ValueError('Memory MAML requires at least two support episodes: one held-out target and one memory support.')

    holdout_index = _select_holdout_index(len(support_ids), cfg, rng)
    heldout_episode_id = int(support_ids[holdout_index])
    memory_support_ids = [int(ep_id) for idx, ep_id in enumerate(support_ids) if idx != holdout_index]
    if not memory_support_ids:
        raise RuntimeError('Memory MAML produced an empty memory support set.')

    memory_support = task_builder.build_conditioning_from_support_ids(
        rng,
        vidx=int(task.vidx),
        support_ids=memory_support_ids,
        load_rgb=load_rgb,
        load_mask_id=use_mask_id,
        load_full_traj=True,
    )
    if memory_support is None:
        raise RuntimeError('Failed to build memory-support conditioning.')

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

    memory_init_batch = _build_memory_query_batch(
        task_builder,
        vidx=int(task.vidx),
        support_cond=memory_support,
        support_ids=memory_support_ids,
        query_episode_id=heldout_episode_id,
        count=1,
        rng=rng,
        noise=None,
        timesteps=None,
        load_rgb=load_rgb,
        load_mask_id=use_mask_id,
    )
    memory_init_batch = _to_device(_drop_mask_ids_if_disabled(memory_init_batch, use_mask_id), device)

    inner_batches: List[Dict[str, Any]] = []
    num_inner_batches = _resolve_num_inner_batches(cfg)
    queries_per_step = int(cfg.num_queries_per_step)
    if queries_per_step < 1:
        raise ValueError(f'num_queries_per_step must be >= 1, got {queries_per_step}.')
    for _ in range(num_inner_batches):
        inner_batch = _build_memory_query_batch(
            task_builder,
            vidx=int(task.vidx),
            support_cond=memory_support,
            support_ids=memory_support_ids,
            query_episode_id=heldout_episode_id,
            count=queries_per_step,
            rng=rng,
            noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
            timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
            load_rgb=load_rgb,
            load_mask_id=use_mask_id,
        )
        inner_batches.append(_to_device(_drop_mask_ids_if_disabled(inner_batch, use_mask_id), device))

    query_count = max(1, int(cfg.num_query_loss_samples))
    query_batch = _build_memory_query_batch(
        task_builder,
        vidx=int(task.vidx),
        support_cond=memory_support,
        support_ids=memory_support_ids,
        query_episode_id=int(task.query_episode_id),
        count=query_count,
        rng=rng,
        noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
        timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
        load_rgb=load_rgb,
        load_mask_id=use_mask_id,
    )
    query_batch = _to_device(_drop_mask_ids_if_disabled(query_batch, use_mask_id), device)

    return {
        'task': task,
        'support_ids': support_ids,
        'memory_support_ids': memory_support_ids,
        'heldout_episode_id': heldout_episode_id,
        'holdout_index': int(holdout_index),
        'query_episode_id': int(task.query_episode_id),
        'memory_init_batch': memory_init_batch,
        'inner_batches': inner_batches,
        'query_batch': query_batch,
    }


def prepare_outer_batch_for_memory_meta_step(
    tasks: Sequence[MAMLTaskSpec],
    *,
    task_builder: MAMLTaskBuilder,
    cfg: MemoryMAMLConfig,
    device: torch.device,
    num_train_timesteps: int,
    action_dim: int,
    use_mask_id: bool,
    rng: np.random.Generator,
    torch_generator: Optional[torch.Generator] = None,
    load_rgb: bool = True,
) -> List[Dict[str, Any]]:
    return [
        prepare_memory_task_for_meta_step(
            task,
            task_builder=task_builder,
            cfg=cfg,
            device=device,
            num_train_timesteps=num_train_timesteps,
            action_dim=action_dim,
            use_mask_id=use_mask_id,
            rng=rng,
            torch_generator=torch_generator,
            load_rgb=load_rgb,
        )
        for task in tasks
    ]


def _encode_context(policy: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Any:
    ctx_out = policy.context_encoder(
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        cond_xyz=batch.get('cond_xyz', None),
        cond_state=batch.get('cond_state', None),
        cond_traj=batch.get('cond_traj', None),
        cond_traj_mask=batch.get('cond_traj_mask', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        query_valid=batch.get('query_valid', None),
        cond_rgb=batch.get('cond_rgb', None),
        cond_mask_id=batch.get('cond_mask_id', None),
        cond_valid=batch.get('cond_valid', None),
    )
    return policy._resolve_context_output(ctx_out)


def init_memory_tokens_from_batch(
    policy: torch.nn.Module,
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(policy, 'init_learned_memory_tokens'):
        tokens, mask = policy.init_learned_memory_tokens(
            batch_size=1,
            device=batch['query_xyz'].device,
            dtype=batch['query_xyz'].dtype,
            clone=True,
        )
        return tokens, mask
    ctx = _encode_context(policy, batch)
    if ctx.support_tokens is None:
        raise RuntimeError('Context encoder did not return support_tokens; memory-token MAML requires split support tokens.')
    support_tokens = ctx.support_tokens
    support_mask = ctx.support_token_mask.to(torch.bool) if torch.is_tensor(ctx.support_token_mask) else None
    if support_tokens.shape[0] != 1:
        support_tokens = support_tokens[:1].contiguous()
        if support_mask is not None:
            support_mask = support_mask[:1].contiguous()
    if not support_tokens.requires_grad:
        support_tokens = support_tokens.detach().requires_grad_(True)
    return support_tokens, support_mask


def query_tokens_from_batch(
    policy: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    ctx = _encode_context(policy, batch)
    if ctx.query_tokens is None:
        raise RuntimeError('Context encoder did not return query_tokens; memory-token MAML requires split query tokens.')
    query_mask = ctx.query_token_mask.to(torch.bool) if torch.is_tensor(ctx.query_token_mask) else None
    return ctx.query_tokens, query_mask


def _resolve_batch_timesteps(policy: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not _is_diffusion_policy(policy):
        raise TypeError('_resolve_batch_timesteps is only valid for diffusion policies.')
    x0 = batch['target_action']
    B = int(x0.shape[0])
    provided = batch.get('timesteps', None)
    if provided is None:
        return torch.randint(
            low=0,
            high=int(policy.noise_scheduler.config.num_train_timesteps),
            size=(B,),
            device=x0.device,
            dtype=torch.long,
        )
    t = provided.to(device=x0.device, dtype=torch.long)
    if t.ndim == 0:
        t = t.view(1)
    if t.shape == (1,) and B > 1:
        t = t.expand(B)
    if t.shape != (B,):
        raise ValueError(f'timesteps must have shape ({B},), got {tuple(t.shape)}.')
    return t


def _resolve_batch_noise(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    x0 = batch['target_action']
    B = int(x0.shape[0])
    provided = batch.get('noise', None)
    if provided is None:
        return torch.randn_like(x0)
    noise = provided.to(device=x0.device, dtype=x0.dtype)
    if noise.ndim == 2:
        noise = noise.unsqueeze(0)
    if noise.shape[0] == 1 and B > 1:
        noise = noise.expand(B, -1, -1)
    if noise.shape != x0.shape:
        raise ValueError(f'noise must have shape {tuple(x0.shape)}, got {tuple(noise.shape)}.')
    return noise


def diffusion_training_target(
    policy: torch.nn.Module,
    *,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if not _is_diffusion_policy(policy):
        raise TypeError('diffusion_training_target is only valid for diffusion policies.')
    pred_type = str(policy.noise_scheduler.config.prediction_type)
    if pred_type == 'epsilon':
        return noise
    if pred_type == 'sample':
        return x0
    if pred_type == 'v_prediction':
        if hasattr(policy.noise_scheduler, 'get_velocity'):
            return policy.noise_scheduler.get_velocity(x0, noise, t)
        alpha_t = policy.noise_scheduler.alphas_cumprod[t].sqrt().to(x0.device)
        sigma_t = (1.0 - policy.noise_scheduler.alphas_cumprod[t]).sqrt().to(x0.device)
        alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
        return alpha_t * noise - sigma_t * x0
    raise ValueError(f'Unsupported prediction type {pred_type!r}.')


def _expand_tokens_for_batch(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tokens.shape[0] == batch_size:
        return tokens
    if tokens.shape[0] == 1:
        return tokens.expand(batch_size, -1, -1)
    raise ValueError(f'Cannot expand token batch from B={tokens.shape[0]} to B={batch_size}.')


def _expand_mask_for_batch(mask: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.expand(batch_size, -1)
    raise ValueError(f'Cannot expand mask batch from B={mask.shape[0]} to B={batch_size}.')


def _infer_num_support_demos_from_batch(batch: Dict[str, Any]) -> Optional[int]:
    for key in ('cond_xyz', 'cond_state', 'cond_traj'):
        value = batch.get(key, None)
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[1])
    return None


def _traj_support_token_count(
    policy: torch.nn.Module,
    support_tokens: Optional[torch.Tensor],
    *,
    num_support_demos: Optional[int],
) -> int:
    if support_tokens is None or num_support_demos is None or int(num_support_demos) <= 0:
        return 0
    context_encoder = getattr(policy.context_encoder, 'base_encoder', policy.context_encoder)
    cfg = getattr(context_encoder, 'cfg', None)
    if cfg is None or not hasattr(cfg, 'm_traj_tokens'):
        return 0
    if not _as_bool(getattr(cfg, 'include_traj_tokens', True)):
        return 0
    count = int(num_support_demos) * int(getattr(cfg, 'm_traj_tokens'))
    if count <= 0 or count > int(support_tokens.shape[1]):
        return 0
    return count


def _single_context_from_memory_tokens(
    policy: torch.nn.Module,
    *,
    support_tokens: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    query_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    num_support_demos: Optional[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    traj_count = _traj_support_token_count(policy, support_tokens, num_support_demos=num_support_demos)
    if traj_count <= 0 or support_tokens is None:
        return policy._concat_token_groups(support_tokens, support_mask, query_tokens, query_mask)

    demo_count = int(support_tokens.shape[1]) - int(traj_count)
    demo_tokens = support_tokens[:, :demo_count] if demo_count > 0 else None
    traj_tokens = support_tokens[:, demo_count:]
    demo_mask = None
    traj_mask = None
    if support_mask is not None:
        demo_mask = support_mask[:, :demo_count] if demo_count > 0 else None
        traj_mask = support_mask[:, demo_count:]

    ctx_tokens, ctx_mask = policy._concat_token_groups(demo_tokens, demo_mask, query_tokens, query_mask)
    return policy._concat_token_groups(ctx_tokens, ctx_mask, traj_tokens, traj_mask)


def predict_model_output_from_tokens(
    policy: torch.nn.Module,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    query_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    support_tokens: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    num_support_demos: Optional[int] = None,
) -> torch.Tensor:
    if not _is_diffusion_policy(policy):
        raise TypeError('predict_model_output_from_tokens is only valid for diffusion policies.')
    B, H, _ = x_t.shape
    d = int(policy.cfg.d_model)

    query_tokens = _expand_tokens_for_batch(query_tokens.to(device=x_t.device, dtype=x_t.dtype), B)
    query_mask = _expand_mask_for_batch(
        query_mask.to(device=x_t.device, dtype=torch.bool) if query_mask is not None else None,
        B,
    )
    if support_tokens is not None:
        support_tokens = _expand_tokens_for_batch(support_tokens.to(device=x_t.device, dtype=x_t.dtype), B)
    support_mask = _expand_mask_for_batch(
        support_mask.to(device=x_t.device, dtype=torch.bool) if support_mask is not None else None,
        B,
    )

    t_emb = sinusoidal_time_embedding(t, d)
    t_cond = policy.t_mlp(t_emb)
    h = policy.action_in(x_t)
    h = h + sinusoidal_position_embedding(H, d, device=x_t.device).to(dtype=h.dtype).unsqueeze(0)
    use_dit_ckpt = bool(policy.training and policy.cfg.grad_checkpoint_dit and torch.is_grad_enabled())

    if str(policy.context_attention_mode) == 'single':
        ctx_tokens, ctx_mask = _single_context_from_memory_tokens(
            policy,
            support_tokens=support_tokens,
            support_mask=support_mask,
            query_tokens=query_tokens,
            query_mask=query_mask,
            num_support_demos=num_support_demos,
        )
        if ctx_tokens is None:
            raise RuntimeError('Memory-token decoder received no context tokens.')
        for blk in policy.denoiser:
            h = policy._apply_single_context_block(
                blk,
                h,
                t_cond,
                ctx_tokens,
                ctx_mask,
                use_checkpoint=use_dit_ckpt,
            )
    else:
        for blk in policy.denoiser:
            h = policy._apply_two_context_block(
                blk,
                h,
                t_cond,
                query_tokens,
                query_mask,
                support_tokens,
                support_mask,
                use_checkpoint=use_dit_ckpt,
            )
    return policy.action_out(h)


def predict_direct_actions_from_tokens(
    policy: torch.nn.Module,
    *,
    query_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor],
    support_tokens: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    action_horizon: int,
    num_support_demos: Optional[int] = None,
) -> torch.Tensor:
    if not _is_direct_regression_policy(policy):
        raise TypeError('predict_direct_actions_from_tokens is only valid for direct-regression policies.')
    policy._check_action_horizon(int(action_horizon))
    B = int(query_tokens.shape[0])
    d = int(policy.cfg.d_model)

    query_tokens = _expand_tokens_for_batch(query_tokens, B)
    query_mask = _expand_mask_for_batch(
        query_mask.to(device=query_tokens.device, dtype=torch.bool) if query_mask is not None else None,
        B,
    )
    if support_tokens is not None:
        support_tokens = _expand_tokens_for_batch(
            support_tokens.to(device=query_tokens.device, dtype=query_tokens.dtype),
            B,
        )
    support_mask = _expand_mask_for_batch(
        support_mask.to(device=query_tokens.device, dtype=torch.bool) if support_mask is not None else None,
        B,
    )

    tokens = None
    token_mask = None
    if str(policy.context_attention_mode) == 'single':
        tokens, token_mask = _single_context_from_memory_tokens(
            policy,
            support_tokens=support_tokens,
            support_mask=support_mask,
            query_tokens=query_tokens,
            query_mask=query_mask,
            num_support_demos=num_support_demos,
        )
        if tokens is None:
            raise RuntimeError('Memory-token direct decoder received no context tokens.')

    cond_vec = policy.context_conditioner(
        tokens=tokens,
        token_mask=token_mask,
        support_tokens=support_tokens,
        support_token_mask=support_mask,
        query_tokens=query_tokens,
        query_token_mask=query_mask,
    )
    h = policy.action_queries.unsqueeze(0).expand(B, -1, -1)
    h = h + policy.action_slot_embed.unsqueeze(0)
    use_decoder_ckpt = bool(
        policy.training and policy.cfg.grad_checkpoint_decoder and torch.is_grad_enabled()
    )

    if str(policy.context_attention_mode) == 'single':
        for blk in policy.decoder:
            h = policy._apply_single_context_block(
                blk,
                h,
                cond_vec,
                tokens,
                token_mask,
                use_checkpoint=use_decoder_ckpt,
            )
    else:
        for blk in policy.decoder:
            h = policy._apply_two_context_block(
                blk,
                h,
                cond_vec,
                query_tokens,
                query_mask,
                support_tokens,
                support_mask,
                use_checkpoint=use_decoder_ckpt,
            )
    return policy.action_out(h)


def memory_diffusion_loss(
    policy: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    memory_tokens: torch.Tensor,
    memory_token_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    query_tokens, query_mask = query_tokens_from_batch(policy, batch)
    if _is_diffusion_policy(policy):
        x0 = batch['target_action']
        t = _resolve_batch_timesteps(policy, batch)
        noise = _resolve_batch_noise(batch)
        x_t = policy.noise_scheduler.add_noise(x0, noise, t)
        model_out = predict_model_output_from_tokens(
            policy,
            x_t=x_t,
            t=t,
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=memory_tokens,
            support_mask=memory_token_mask,
            num_support_demos=_infer_num_support_demos_from_batch(batch),
        )
        target = diffusion_training_target(policy, x0=x0, noise=noise, t=t)
        return F.mse_loss(model_out, target)

    if _is_direct_regression_policy(policy):
        pred = predict_direct_actions_from_tokens(
            policy,
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=memory_tokens,
            support_mask=memory_token_mask,
            action_horizon=int(batch['target_action'].shape[1]),
            num_support_demos=_infer_num_support_demos_from_batch(batch),
        )
        target = batch['target_action']
        loss_type = str(policy.cfg.loss_type).lower()
        if loss_type == 'mse':
            return F.mse_loss(pred, target)
        if loss_type == 'l1':
            return F.l1_loss(pred, target)
        raise ValueError(f'Unsupported loss_type={policy.cfg.loss_type!r}.')

    raise TypeError('Unsupported policy type for memory-token MAML loss.')


def _grad_global_norm(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    valid = [grad.detach() for grad in grads if grad is not None]
    if not valid:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.norm(torch.stack([grad.norm(2) for grad in valid]), 2)


def _clip_grad(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    if float(max_norm) <= 0.0:
        return grad
    norm = grad.norm(2)
    if norm <= float(max_norm):
        return grad
    return grad * (float(max_norm) / (norm + 1e-6))


def _batch_size_from_target(batch: Dict[str, Any]) -> int:
    target = batch.get('target_action', None)
    if not torch.is_tensor(target):
        raise KeyError("Batch is missing tensor key 'target_action'.")
    return int(target.shape[0])


def _slice_batch_dim0(batch: Dict[str, Any], start: int, end: int, batch_size: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            out[key] = value[start:end]
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def _iter_microbatches(batch: Dict[str, Any], grad_accum_steps: int):
    batch_size = _batch_size_from_target(batch)
    steps = max(1, min(int(grad_accum_steps), batch_size))
    if steps == 1:
        yield batch, 1.0
        return
    base = batch_size // steps
    remainder = batch_size % steps
    start = 0
    for idx in range(steps):
        size = base + (1 if idx < remainder else 0)
        if size <= 0:
            continue
        end = start + size
        yield _slice_batch_dim0(batch, start, end, batch_size), float(size) / float(batch_size)
        start = end


def adapt_memory_tokens_for_prepared_task(
    policy: torch.nn.Module,
    prepared_task: Dict[str, Any],
    *,
    cfg: MemoryMAMLConfig,
    create_graph: bool,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule] = None,
    inner_grad_norms_out: Optional[List[torch.Tensor]] = None,
    inner_losses_out: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    memory_tokens, memory_token_mask = init_memory_tokens_from_batch(policy, prepared_task['memory_init_batch'])
    inner_batches = list(prepared_task.get('inner_batches', []))
    if int(cfg.inner_steps) > 0 and not inner_batches:
        raise ValueError('Memory MAML inner_steps > 0 requires at least one prepared inner batch.')
    grad_accum_steps = int(cfg.grad_accum_steps)
    if grad_accum_steps < 1:
        raise ValueError(f'grad_accum_steps must be >= 1, got {grad_accum_steps}.')
    inner_lr_mode = normalize_inner_lr_mode(getattr(cfg, 'inner_lr_mode', 'fixed'))

    for step_idx in range(int(cfg.inner_steps)):
        inner_batch = inner_batches[step_idx % len(inner_batches)]
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
                create_graph=create_graph,
                retain_graph=create_graph,
                allow_unused=False,
            )[0]
            loss_for_stats = support_loss.detach()
        else:
            grad = None
            weighted_loss = None
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
                    create_graph=create_graph,
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                grad = micro_grad if grad is None else grad + micro_grad
                weighted_loss = micro_loss.detach() * float(weight) if weighted_loss is None else weighted_loss + micro_loss.detach() * float(weight)
            if grad is None or weighted_loss is None:
                raise RuntimeError('No microbatches were produced for memory-token adaptation.')
            loss_for_stats = weighted_loss

        if inner_grad_norms_out is not None:
            inner_grad_norms_out.append(_grad_global_norm([grad]))
        if inner_losses_out is not None:
            inner_losses_out.append(loss_for_stats)
        grad = _clip_grad(grad, float(cfg.max_grad_norm))
        step_lr = inner_lr_tensor_for_step(
            step_idx=step_idx,
            mode=inner_lr_mode,
            fixed_inner_lr=float(cfg.inner_lr),
            schedule=inner_lr_schedule,
            device=memory_tokens.device,
            dtype=memory_tokens.dtype,
        )
        memory_tokens = memory_tokens - step_lr * grad

    return memory_tokens, memory_token_mask


def memory_maml_step_with_stats(
    policy: torch.nn.Module,
    prepared_tasks: Sequence[Dict[str, Any]],
    *,
    cfg: MemoryMAMLConfig,
    first_order: bool,
    inner_lr_schedule: Optional[PositiveInnerLRSchedule] = None,
) -> Tuple[torch.Tensor, float, float]:
    if not prepared_tasks:
        raise ValueError('prepared_tasks must contain at least one task.')
    losses: List[torch.Tensor] = []
    inner_grad_norms: List[torch.Tensor] = []
    inner_losses: List[torch.Tensor] = []
    for prepared_task in prepared_tasks:
        adapted_tokens, token_mask = adapt_memory_tokens_for_prepared_task(
            policy,
            prepared_task,
            cfg=cfg,
            create_graph=not bool(first_order),
            inner_lr_schedule=inner_lr_schedule,
            inner_grad_norms_out=inner_grad_norms,
            inner_losses_out=inner_losses,
        )
        losses.append(
            memory_diffusion_loss(
                policy,
                prepared_task['query_batch'],
                memory_tokens=adapted_tokens,
                memory_token_mask=token_mask,
            )
        )
    meta_loss = torch.stack(losses).mean()
    avg_inner_grad_norm = float(torch.stack(inner_grad_norms).mean().item()) if inner_grad_norms else 0.0
    avg_inner_loss = float(torch.stack(inner_losses).mean().item()) if inner_losses else 0.0
    return meta_loss, avg_inner_grad_norm, avg_inner_loss


@torch.no_grad()
def sample_actions_with_memory_tokens(
    policy: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    memory_tokens: torch.Tensor,
    memory_token_mask: Optional[torch.Tensor],
    inference_steps: Optional[int] = None,
    eta: float = 0.0,
) -> torch.Tensor:
    if eta < 0.0:
        raise ValueError('eta must be >= 0.')
    x0 = batch['target_action']
    B, H, A = x0.shape
    query_tokens, query_mask = query_tokens_from_batch(policy, batch)
    support_tokens = memory_tokens.to(device=x0.device, dtype=query_tokens.dtype)
    support_mask = memory_token_mask.to(device=x0.device, dtype=torch.bool) if memory_token_mask is not None else None

    if _is_direct_regression_policy(policy):
        del B, A  # kept only for parity with the diffusion branch.
        return predict_direct_actions_from_tokens(
            policy,
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
            action_horizon=int(H),
            num_support_demos=_infer_num_support_demos_from_batch(batch),
        )

    if not _is_diffusion_policy(policy):
        raise TypeError('Unsupported policy type for memory-token action sampling.')

    scheduler = policy.noise_scheduler
    total_T = int(scheduler.config.num_train_timesteps)
    steps = policy.num_inference_steps if inference_steps is None else int(inference_steps)
    steps = max(1, min(steps, total_T))
    try:
        scheduler.set_timesteps(steps, device=x0.device)
    except TypeError:
        scheduler.set_timesteps(steps)

    x_t = torch.randn(int(B), int(H), int(A), device=x0.device, dtype=x0.dtype)
    step_sig = inspect.signature(scheduler.step).parameters
    for t_now in scheduler.timesteps:
        t_int = int(t_now.item() if torch.is_tensor(t_now) else t_now)
        t_batch = torch.full((int(B),), t_int, device=x0.device, dtype=torch.long)
        model_out = predict_model_output_from_tokens(
            policy,
            x_t=x_t,
            t=t_batch,
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
            num_support_demos=_infer_num_support_demos_from_batch(batch),
        )
        step_kwargs: Dict[str, Any] = {}
        if 'eta' in step_sig:
            step_kwargs['eta'] = float(eta)
        step_out = scheduler.step(model_out, t_now, x_t, **step_kwargs)
        x_t = step_out[0] if isinstance(step_out, tuple) else step_out.prev_sample
    return x_t
