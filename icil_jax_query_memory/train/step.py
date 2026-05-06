from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict
from flax.training import train_state

from icil_jax_query_memory.models.query_memory_direct_regression import QueryMemoryDirectRegressionConfig, QueryMemoryDirectRegressionModel


class TrainState(train_state.TrainState):
    pass


def _tree_global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum([jnp.sum(jnp.square(jnp.asarray(x, dtype=jnp.float32))) for x in leaves]))


def _clip_tree_by_global_norm(tree: Any, max_norm: float) -> Any:
    if float(max_norm) <= 0.0:
        return tree
    norm = _tree_global_norm(tree)
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda x: x * scale.astype(x.dtype), tree)


def _memory_layer_norm(mem: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(mem, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(mem - mean), axis=-1, keepdims=True)
    return (mem - mean) / jnp.sqrt(var + jnp.asarray(eps, dtype=mem.dtype))


def _loss_type_from_model(model: QueryMemoryDirectRegressionModel) -> str:
    action_loss_type = str(getattr(model.cfg.decoder, 'action_loss_type', '')).strip().lower()
    if action_loss_type:
        return action_loss_type
    return str(model.cfg.decoder.loss_type).strip().lower()


def _action_chunk_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    *,
    loss_type: str,
    position_weight: float,
    rotation_weight: float,
    gripper_weight: float,
    chunk_decay: float,
) -> jnp.ndarray:
    pred = jnp.asarray(pred, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    if str(loss_type).lower() == 'mse':
        elem = jnp.square(pred - target)
    elif str(loss_type).lower() == 'l1':
        elem = jnp.abs(pred - target)
    elif str(loss_type).lower() == 'huber':
        diff = pred - target
        abs_diff = jnp.abs(diff)
        delta = jnp.asarray(1.0, dtype=pred.dtype)
        elem = jnp.where(abs_diff <= delta, 0.5 * jnp.square(diff), delta * (abs_diff - 0.5 * delta))
    else:
        raise ValueError(f'Unsupported action loss type={loss_type!r}.')

    A = int(target.shape[-1])
    H = int(target.shape[-2])
    dim_weights = jnp.ones((A,), dtype=pred.dtype)
    if A >= 3:
        dim_weights = dim_weights.at[:3].set(jnp.asarray(position_weight, dtype=pred.dtype))
        if A >= 8:
            dim_weights = dim_weights.at[3 : A - 1].set(jnp.asarray(rotation_weight, dtype=pred.dtype))
            dim_weights = dim_weights.at[A - 1].set(jnp.asarray(gripper_weight, dtype=pred.dtype))
        elif A > 3:
            dim_weights = dim_weights.at[3:].set(jnp.asarray(rotation_weight, dtype=pred.dtype))

    if float(chunk_decay) > 0.0:
        h = jnp.arange(H, dtype=pred.dtype)
        chunk_weights = jnp.exp(-jnp.asarray(chunk_decay, dtype=pred.dtype) * h)
        chunk_weights = chunk_weights / jnp.maximum(jnp.mean(chunk_weights), jnp.asarray(1e-6, dtype=pred.dtype))
    else:
        chunk_weights = jnp.ones((H,), dtype=pred.dtype)

    weights = chunk_weights[None, :, None] * dim_weights[None, None, :]
    denom = jnp.maximum(
        jnp.asarray(target.shape[0], dtype=pred.dtype) * jnp.sum(weights),
        jnp.asarray(1e-6, dtype=pred.dtype),
    )
    return jnp.sum(elem * weights) / denom


def _read_predict(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    return model.apply(
        {'params': params},
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        query_valid=batch.get('query_valid', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        memory_tokens=memory_tokens,
        mode='read',
        train=train,
    )


def _attention_metrics_from_intermediates(intermediates: Any) -> Dict[str, jnp.ndarray]:
    zero = jnp.array(0.0, dtype=jnp.float32)
    query_weights = []
    memory_weights = []
    flat = flatten_dict(intermediates, sep='/') if intermediates else {}
    for path, value in flat.items():
        if 'attention_weights' not in str(path):
            continue
        x = value[-1] if isinstance(value, tuple) and value else value
        if not hasattr(x, 'shape') or len(x.shape) < 4:
            continue
        if 'cross_attn_q' in str(path):
            query_weights.append(x)
        elif 'cross_attn_s' in str(path):
            memory_weights.append(x)

    def _stats(values):
        if not values:
            return zero, zero
        entropies = []
        maxes = []
        for weights in values:
            w = jnp.asarray(weights, dtype=jnp.float32)
            p = jnp.clip(w, jnp.asarray(1e-8, dtype=jnp.float32), jnp.asarray(1.0, dtype=jnp.float32))
            entropy = -jnp.sum(p * jnp.log(p), axis=-1)
            denom = jnp.log(jnp.asarray(max(2, int(w.shape[-1])), dtype=jnp.float32))
            entropies.append(jnp.mean(entropy / denom))
            maxes.append(jnp.mean(jnp.max(w, axis=-1)))
        return jnp.mean(jnp.stack(entropies)), jnp.mean(jnp.stack(maxes))

    query_entropy, query_max = _stats(query_weights)
    memory_entropy, memory_max = _stats(memory_weights)
    return {
        'attn_query_entropy': query_entropy,
        'attn_query_max': query_max,
        'attn_memory_entropy': memory_entropy,
        'attn_memory_max': memory_max,
    }


def _read_predict_with_attention_metrics(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    pred, mutable = model.apply(
        {'params': params},
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        query_valid=batch.get('query_valid', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        memory_tokens=memory_tokens,
        mode='read',
        train=train,
        mutable=['intermediates'],
    )
    return pred, _attention_metrics_from_intermediates(mutable.get('intermediates', {}))


def _read_loss(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    pred = _read_predict(
        params,
        model=model,
        batch=batch,
        memory_tokens=memory_tokens,
        train=train,
    )
    return _action_chunk_loss(
        pred,
        batch['target_action'],
        loss_type=_loss_type_from_model(model),
        position_weight=float(model.cfg.decoder.position_loss_weight),
        rotation_weight=float(model.cfg.decoder.rotation_loss_weight),
        gripper_weight=float(model.cfg.decoder.gripper_loss_weight),
        chunk_decay=float(model.cfg.decoder.chunk_decay),
    )


def _goal_predict(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    return model.apply(
        {'params': params},
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        query_valid=batch.get('query_valid', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        memory_tokens=memory_tokens,
        mode='goal',
        train=train,
    )


def _goal_prediction_loss(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
    loss_type: str,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    zero = jnp.array(0.0, dtype=jnp.float32)
    if 'target_goal' not in batch:
        return zero, {
            'goal_prediction_loss': zero,
            'goal_prediction_l1': zero,
            'goal_prediction_l2': zero,
        }
    pred = _goal_predict(params, model=model, batch=batch, memory_tokens=memory_tokens, train=train)
    target = jnp.asarray(batch['target_goal'], dtype=jnp.float32)
    pred = jnp.asarray(pred, dtype=jnp.float32)
    diff = pred - target
    loss_name = str(loss_type).strip().lower()
    if loss_name == 'mse':
        loss = jnp.mean(jnp.square(diff))
    elif loss_name == 'l1':
        loss = jnp.mean(jnp.abs(diff))
    elif loss_name == 'huber':
        abs_diff = jnp.abs(diff)
        loss = jnp.mean(jnp.where(abs_diff <= 1.0, 0.5 * jnp.square(diff), abs_diff - 0.5))
    else:
        raise ValueError(f"goal_prediction_loss_type must be 'mse', 'l1', or 'huber', got {loss_type!r}.")
    return loss, {
        'goal_prediction_loss': loss,
        'goal_prediction_l1': jnp.mean(jnp.abs(diff)),
        'goal_prediction_l2': jnp.sqrt(jnp.mean(jnp.square(diff)) + jnp.asarray(1e-8, dtype=jnp.float32)),
    }


def _metadata_or_zeros(batch: Dict[str, jnp.ndarray], primary: str, fallback: str) -> jnp.ndarray:
    if primary in batch:
        return batch[primary]
    if fallback in batch:
        return batch[fallback]
    return jnp.zeros((batch['target_action'].shape[0],), dtype=jnp.float32)


def _write_predict(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    return model.apply(
        {'params': params},
        query_xyz=batch.get('query_xyz', None),
        query_state=batch.get('query_state', None),
        query_valid=batch.get('query_valid', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        memory_tokens=memory_tokens,
        mode='write',
        write_demo_id=_metadata_or_zeros(batch, 'demo_id', 'support_demo_id'),
        write_chunk_start=_metadata_or_zeros(batch, 'chunk_start', 'support_chunk_start'),
        train=train,
    )


def _write_loss(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    pred = _write_predict(
        params,
        model=model,
        batch=batch,
        memory_tokens=memory_tokens,
        train=train,
    )
    return _action_chunk_loss(
        pred,
        batch['target_action'],
        loss_type=_loss_type_from_model(model),
        position_weight=float(model.cfg.decoder.position_loss_weight),
        rotation_weight=float(model.cfg.decoder.rotation_loss_weight),
        gripper_weight=float(model.cfg.decoder.gripper_loss_weight),
        chunk_decay=float(model.cfg.decoder.chunk_decay),
    )


def _inner_loss(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
    inner_loss_mode: str,
) -> jnp.ndarray:
    mode = str(inner_loss_mode).lower()
    if mode == 'read':
        return _read_loss(params, model=model, batch=batch, memory_tokens=memory_tokens, train=train)
    if mode == 'write':
        return _write_loss(params, model=model, batch=batch, memory_tokens=memory_tokens, train=train)
    raise ValueError(f"inner_loss_mode must be 'read' or 'write', got {inner_loss_mode!r}.")


def _apply_query_goal_dropout(
    batch: Dict[str, jnp.ndarray],
    *,
    rng: jnp.ndarray,
    rate: float,
    state_start: int,
) -> Dict[str, jnp.ndarray]:
    rate = float(rate)
    if rate <= 0.0 or 'query_state' not in batch:
        return batch
    q = batch['query_state']
    start = int(state_start)
    if start < 0 or q.shape[-1] < start + 3:
        return batch
    keep_prob = max(0.0, min(1.0, 1.0 - rate))
    mask_shape = tuple(q.shape[:-2]) + (1, 1)
    keep = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape).astype(q.dtype)
    q = q.at[..., start : start + 3].set(q[..., start : start + 3] * keep)
    return {**batch, 'query_state': q}


def _adapt_one_task(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    initial_memory_tokens: Optional[jnp.ndarray] = None,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str,
    memory_layer_norm_after_update: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    memory_tokens = params['memory_token_init'] if initial_memory_tokens is None else initial_memory_tokens
    initial_memory = memory_tokens
    effective_clip = float(model.cfg.decoder.memory_update_clip_norm)
    if effective_clip <= 0.0:
        effective_clip = float(max_grad_norm)

    def inner_step(carry: jnp.ndarray, step_batch: Dict[str, jnp.ndarray]):
        def support_loss_fn(mem: jnp.ndarray) -> jnp.ndarray:
            return _inner_loss(
                params,
                model=model,
                batch=step_batch,
                memory_tokens=mem,
                train=True,
                inner_loss_mode=inner_loss_mode,
            )

        support_loss, grad = jax.value_and_grad(support_loss_fn)(carry)
        if bool(first_order):
            grad = jax.lax.stop_gradient(grad)
        grad_norm = _tree_global_norm(grad)
        grad = _clip_tree_by_global_norm(grad, effective_clip)
        next_mem = carry - jnp.asarray(inner_lr, dtype=carry.dtype) * grad
        if bool(memory_layer_norm_after_update):
            next_mem = _memory_layer_norm(next_mem)
        return next_mem, (support_loss, grad_norm)

    if int(inner_steps) <= 0:
        zero = jnp.array(0.0, dtype=jnp.float32)
        stats = {
            'inner_loss_mean': zero,
            'inner_memory_grad_norm': zero,
            'inner_loss_before_first_step': zero,
            'inner_loss_after_last_step': zero,
            'memory_delta_norm': zero,
            'memory_relative_delta_norm': zero,
        }
        return memory_tokens, stats

    adapted_memory, (inner_losses, inner_grad_norms) = jax.lax.scan(inner_step, memory_tokens, task_inner)
    memory_delta = adapted_memory - initial_memory
    memory_delta_norm = _tree_global_norm(memory_delta)
    memory_relative_delta_norm = memory_delta_norm / (_tree_global_norm(initial_memory) + jnp.asarray(1e-6, dtype=jnp.float32))
    stats = {
        'inner_loss_mean': jnp.mean(inner_losses),
        'inner_memory_grad_norm': jnp.mean(inner_grad_norms),
        'inner_loss_before_first_step': inner_losses[0],
        'inner_loss_after_last_step': jnp.mean(inner_losses),
        'memory_delta_norm': memory_delta_norm,
        'memory_relative_delta_norm': memory_relative_delta_norm,
    }
    return adapted_memory, stats


def _flatten_support_for_memory(task_inner: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    if 'query_state' not in task_inner or 'target_action' not in task_inner:
        raise ValueError('Support-encoder memory requires task_inner query_state and target_action.')
    state = task_inner['query_state']
    action = task_inner['target_action']
    if state.ndim < 4 or action.ndim < 4:
        raise ValueError(
            'Support-encoder memory expects per-task inner tensors with leading [inner_steps, support_chunks]. '
            f'Got query_state={tuple(state.shape)}, target_action={tuple(action.shape)}.'
        )
    out = {
        'support_query_state': state.reshape((-1,) + tuple(state.shape[2:])),
        'support_target_action': action.reshape((-1,) + tuple(action.shape[2:])),
    }
    for src_key, dst_key in (
        ('support_demo_id', 'support_demo_id'),
        ('demo_id', 'support_demo_id'),
        ('support_chunk_start', 'support_chunk_start'),
        ('chunk_start', 'support_chunk_start'),
    ):
        if src_key in task_inner and dst_key not in out:
            x = task_inner[src_key]
            out[dst_key] = x.reshape((-1,) + tuple(x.shape[2:])) if x.ndim > 2 else x.reshape((-1,))
    return out


def _initial_memory_for_task(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    train: bool,
) -> jnp.ndarray:
    mode = str(getattr(model.cfg, 'memory_initialization_mode', 'base_only')).strip().lower()
    if mode in ('', 'none', 'learned', 'learned_base', 'base_only'):
        return params['memory_token_init']
    support = _flatten_support_for_memory(task_inner)
    return model.apply(
        {'params': params},
        support_query_state=support['support_query_state'],
        support_target_action=support['support_target_action'],
        support_demo_id=support.get('support_demo_id'),
        support_chunk_start=support.get('support_chunk_start'),
        train=train,
        method=QueryMemoryDirectRegressionModel.initial_memory_from_support,
    )


def _memory_embedding(memory_tokens: jnp.ndarray, initial_memory: jnp.ndarray, *, on_delta: bool) -> jnp.ndarray:
    x = memory_tokens - initial_memory if bool(on_delta) else memory_tokens
    z = jnp.mean(jnp.asarray(x, dtype=jnp.float32), axis=0)
    return z / (jnp.linalg.norm(z) + jnp.asarray(1e-6, dtype=jnp.float32))


def _masked_info_nce(
    z_a: jnp.ndarray,
    z_b: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    temperature: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    labels = jnp.asarray(labels)
    B = int(z_a.shape[0])
    temp = jnp.maximum(jnp.asarray(float(temperature), dtype=jnp.float32), jnp.asarray(1e-6, dtype=jnp.float32))
    logits = (z_a @ z_b.T) / temp
    same = labels[:, None] == labels[None, :]
    eye = jnp.eye(B, dtype=jnp.bool_)
    valid = jnp.logical_or(jnp.logical_not(same), eye)
    logits = jnp.where(valid, logits, jnp.asarray(-1e9, dtype=logits.dtype))
    log_probs = jax.nn.log_softmax(logits, axis=1)
    targets = jnp.arange(B, dtype=jnp.int32)
    per_row = -log_probs[targets, targets]
    has_negative = jnp.any(jnp.logical_and(jnp.logical_not(same), jnp.logical_not(eye)), axis=1)
    denom = jnp.maximum(jnp.sum(has_negative.astype(jnp.float32)), jnp.asarray(1.0, dtype=jnp.float32))
    loss = jnp.sum(jnp.where(has_negative, per_row, 0.0)) / denom
    pred = jnp.argmax(logits, axis=1)
    acc = jnp.sum(jnp.where(has_negative, (pred == targets).astype(jnp.float32), 0.0)) / denom

    sim = z_a @ z_b.T
    within = jnp.mean(jnp.diag(sim))
    between_mask = jnp.logical_and(jnp.logical_not(same), jnp.logical_not(eye))
    between_denom = jnp.maximum(jnp.sum(between_mask.astype(jnp.float32)), jnp.asarray(1.0, dtype=jnp.float32))
    between = jnp.sum(jnp.where(between_mask, sim, 0.0)) / between_denom
    return loss, {
        'memory_contrast_accuracy': acc,
        'within_task_memory_similarity': within,
        'between_task_memory_similarity': between,
    }


def _symmetric_memory_contrast_loss(
    z_a: jnp.ndarray,
    z_b: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    temperature: float,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    loss_ab, metrics_ab = _masked_info_nce(z_a, z_b, labels, temperature=temperature)
    loss_ba, metrics_ba = _masked_info_nce(z_b, z_a, labels, temperature=temperature)
    return 0.5 * (loss_ab + loss_ba), {
        key: 0.5 * (metrics_ab[key] + metrics_ba[key])
        for key in metrics_ab
    }


def _task_meta_objective(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    task_query: Dict[str, jnp.ndarray],
    task_wrong_inner: Dict[str, jnp.ndarray],
    task_contrast_inner: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str,
    memory_layer_norm_after_update: bool,
    use_read_improvement_margin: bool,
    read_improvement_margin: float,
    read_improvement_margin_weight: float,
    log_output_delta: bool,
    training_mode_metrics_only: bool,
    use_wrong_support_margin: bool,
    wrong_support_margin: float,
    wrong_support_margin_weight: float,
    use_memory_contrast: bool,
    memory_contrast_on_delta: bool,
    query_goal_dropout_rate: float,
    query_goal_dropout_state_start: int,
    log_attention_metrics: bool,
    goal_prediction_loss_weight: float,
    goal_prediction_loss_type: str,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    initial_memory = _initial_memory_for_task(params, model=model, task_inner=task_inner, train=True)
    rng_query_dropout, _rng_aux = jax.random.split(rng)
    task_query_for_read = _apply_query_goal_dropout(
        task_query,
        rng=rng_query_dropout,
        rate=float(query_goal_dropout_rate),
        state_start=int(query_goal_dropout_state_start),
    )
    adapted_memory, inner_stats = _adapt_one_task(
        params,
        model=model,
        task_inner=task_inner,
        initial_memory_tokens=initial_memory,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        max_grad_norm=max_grad_norm,
        first_order=first_order,
        inner_loss_mode=inner_loss_mode,
        memory_layer_norm_after_update=memory_layer_norm_after_update,
    )
    read_loss_after = _read_loss(params, model=model, batch=task_query_for_read, memory_tokens=adapted_memory, train=True)
    meta_loss = read_loss_after
    need_read_before = (
        bool(use_read_improvement_margin)
        or bool(log_output_delta)
        or not bool(training_mode_metrics_only)
    )
    if need_read_before:
        read_loss_before = _read_loss(params, model=model, batch=task_query_for_read, memory_tokens=initial_memory, train=True)
    else:
        read_loss_before = jax.lax.stop_gradient(read_loss_after)

    if bool(use_read_improvement_margin):
        margin = jnp.asarray(read_improvement_margin, dtype=read_loss_after.dtype)
        weight = jnp.asarray(read_improvement_margin_weight, dtype=read_loss_after.dtype)
        # Do not let the margin objective satisfy itself by worsening the
        # pre-adaptation READ loss. The main READ-after loss still trains the
        # post-adaptation policy; read_loss_before remains available for logs.
        read_loss_before_ref = jax.lax.stop_gradient(read_loss_before)
        meta_loss = meta_loss + weight * jnp.maximum(0.0, margin + read_loss_after - read_loss_before_ref)

    read_loss_wrong = jnp.array(0.0, dtype=jnp.float32)
    wrong_rank_loss = jnp.array(0.0, dtype=jnp.float32)
    wrong_minus_correct = jnp.array(0.0, dtype=jnp.float32)
    wrong_ranking_accuracy = jnp.array(0.0, dtype=jnp.float32)
    wrong_memory_delta_norm = jnp.array(0.0, dtype=jnp.float32)
    wrong_memory_relative_delta_norm = jnp.array(0.0, dtype=jnp.float32)
    if bool(use_wrong_support_margin):
        wrong_initial_memory = _initial_memory_for_task(params, model=model, task_inner=task_wrong_inner, train=True)
        wrong_memory, wrong_stats = _adapt_one_task(
            params,
            model=model,
            task_inner=task_wrong_inner,
            initial_memory_tokens=wrong_initial_memory,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            max_grad_norm=max_grad_norm,
            first_order=first_order,
            inner_loss_mode=inner_loss_mode,
            memory_layer_norm_after_update=memory_layer_norm_after_update,
        )
        read_loss_wrong = _read_loss(params, model=model, batch=task_query_for_read, memory_tokens=wrong_memory, train=True)
        wrong_minus_correct = read_loss_wrong - read_loss_after
        margin = jnp.asarray(wrong_support_margin, dtype=read_loss_after.dtype)
        weight = jnp.asarray(wrong_support_margin_weight, dtype=read_loss_after.dtype)
        wrong_rank_loss = jnp.maximum(0.0, margin + read_loss_after - read_loss_wrong)
        meta_loss = meta_loss + weight * wrong_rank_loss
        wrong_ranking_accuracy = (read_loss_wrong > read_loss_after).astype(jnp.float32)
        wrong_memory_delta_norm = wrong_stats['memory_delta_norm']
        wrong_memory_relative_delta_norm = wrong_stats['memory_relative_delta_norm']

    if bool(use_memory_contrast):
        contrast_initial_memory = _initial_memory_for_task(params, model=model, task_inner=task_contrast_inner, train=True)
        contrast_memory, _ = _adapt_one_task(
            params,
            model=model,
            task_inner=task_contrast_inner,
            initial_memory_tokens=contrast_initial_memory,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            max_grad_norm=max_grad_norm,
            first_order=first_order,
            inner_loss_mode=inner_loss_mode,
            memory_layer_norm_after_update=memory_layer_norm_after_update,
        )

    if bool(log_output_delta):
        pred_before = _read_predict(params, model=model, batch=task_query_for_read, memory_tokens=initial_memory, train=True)
        pred_after = _read_predict(params, model=model, batch=task_query_for_read, memory_tokens=adapted_memory, train=True)
        action_output_delta = jnp.mean(jnp.abs(pred_after - pred_before))
    else:
        action_output_delta = jnp.array(0.0, dtype=jnp.float32)

    goal_loss = jnp.array(0.0, dtype=jnp.float32)
    goal_metrics = {
        'goal_prediction_loss': goal_loss,
        'goal_prediction_l1': goal_loss,
        'goal_prediction_l2': goal_loss,
    }
    if float(goal_prediction_loss_weight) > 0.0:
        goal_loss, goal_metrics = _goal_prediction_loss(
            params,
            model=model,
            batch=task_query_for_read,
            memory_tokens=adapted_memory,
            train=True,
            loss_type=str(goal_prediction_loss_type),
        )
        meta_loss = meta_loss + jnp.asarray(float(goal_prediction_loss_weight), dtype=meta_loss.dtype) * goal_loss

    attention_metrics = {
        'attn_query_entropy': jnp.array(0.0, dtype=jnp.float32),
        'attn_query_max': jnp.array(0.0, dtype=jnp.float32),
        'attn_memory_entropy': jnp.array(0.0, dtype=jnp.float32),
        'attn_memory_max': jnp.array(0.0, dtype=jnp.float32),
    }
    if bool(log_attention_metrics):
        _, attention_metrics = _read_predict_with_attention_metrics(
            params,
            model=model,
            batch=task_query_for_read,
            memory_tokens=adapted_memory,
            train=True,
        )

    metrics = {
        'meta_loss': meta_loss,
        'read_loss_after': read_loss_after,
        'read_loss_before': read_loss_before,
        'read_improvement': read_loss_before - read_loss_after,
        'inner_support_loss': inner_stats['inner_loss_mean'],
        'inner_write_loss_mean': inner_stats['inner_loss_mean'],
        'inner_memory_grad_norm': inner_stats['inner_memory_grad_norm'],
        'write_loss_before_first_step': inner_stats['inner_loss_before_first_step'],
        'write_loss_after_last_step': (
            jax.lax.stop_gradient(inner_stats['inner_loss_mean'])
            if bool(training_mode_metrics_only)
            else inner_stats['inner_loss_after_last_step']
        ),
        'memory_delta_norm': inner_stats['memory_delta_norm'],
        'memory_relative_delta_norm': inner_stats['memory_relative_delta_norm'],
        'action_output_delta': action_output_delta,
        'read_loss_wrong_support': read_loss_wrong,
        'wrong_support_rank_loss': wrong_rank_loss,
        'wrong_minus_correct_loss': wrong_minus_correct,
        'wrong_support_ranking_accuracy': wrong_ranking_accuracy,
        'wrong_memory_delta_norm': wrong_memory_delta_norm,
        'wrong_memory_relative_delta_norm': wrong_memory_relative_delta_norm,
    }
    metrics.update(goal_metrics)
    metrics.update(attention_metrics)
    if bool(use_memory_contrast):
        metrics['memory_contrast_z_a'] = _memory_embedding(
            adapted_memory,
            initial_memory,
            on_delta=bool(memory_contrast_on_delta),
        )
        metrics['memory_contrast_z_b'] = _memory_embedding(
            contrast_memory,
            contrast_initial_memory,
            on_delta=bool(memory_contrast_on_delta),
        )
    return meta_loss, metrics


def create_train_step(
    *,
    model: QueryMemoryDirectRegressionModel,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str = 'read',
    memory_layer_norm_after_update: bool = False,
    use_read_improvement_margin: bool = False,
    read_improvement_margin: float = 0.0,
    read_improvement_margin_weight: float = 0.0,
    log_output_delta: bool = False,
    training_mode_metrics_only: bool = False,
    use_wrong_support_margin: bool = False,
    wrong_support_margin: float = 0.0,
    wrong_support_margin_weight: float = 0.0,
    use_memory_contrast: bool = False,
    memory_contrast_weight: float = 0.0,
    memory_contrast_temperature: float = 0.1,
    memory_contrast_on_delta: bool = True,
    query_goal_dropout_rate: float = 0.0,
    query_goal_dropout_state_start: int = 36,
    log_attention_metrics: bool = False,
    goal_prediction_loss_weight: float = 0.0,
    goal_prediction_loss_type: str = 'mse',
    rng_seed: int = 0,
):
    @partial(jax.pmap, axis_name='device')
    def train_step(state: TrainState, batch: Dict[str, Any]):
        def loss_fn(params: Any):
            task_wrong_inner = batch['wrong_inner'] if bool(use_wrong_support_margin) else batch['inner']
            task_contrast_inner = batch['contrast_inner'] if bool(use_memory_contrast) else batch['inner']
            base_rng = jax.random.fold_in(jax.random.PRNGKey(int(rng_seed)), state.step)
            base_rng = jax.random.fold_in(base_rng, jax.lax.axis_index('device'))
            # Query-only baselines intentionally have inner_steps=0, so the
            # sharded inner batch is empty. The query batch is always present
            # and carries the task axis for every meta-objective variant.
            task_rngs = jax.random.split(base_rng, batch['query']['target_action'].shape[0])
            task_losses, task_aux = jax.vmap(
                lambda task_inner, task_query, wrong_inner, contrast_inner, task_rng: _task_meta_objective(
                    params,
                    model=model,
                    task_inner=task_inner,
                    task_query=task_query,
                    task_wrong_inner=wrong_inner,
                    task_contrast_inner=contrast_inner,
                    rng=task_rng,
                    inner_steps=int(inner_steps),
                    inner_lr=float(inner_lr),
                    max_grad_norm=float(max_grad_norm),
                    first_order=bool(first_order),
                    inner_loss_mode=str(inner_loss_mode),
                    memory_layer_norm_after_update=bool(memory_layer_norm_after_update),
                    use_read_improvement_margin=bool(use_read_improvement_margin),
                    read_improvement_margin=float(read_improvement_margin),
                    read_improvement_margin_weight=float(read_improvement_margin_weight),
                    log_output_delta=bool(log_output_delta),
                    training_mode_metrics_only=bool(training_mode_metrics_only),
                    use_wrong_support_margin=bool(use_wrong_support_margin),
                    wrong_support_margin=float(wrong_support_margin),
                    wrong_support_margin_weight=float(wrong_support_margin_weight),
                    use_memory_contrast=bool(use_memory_contrast),
                    memory_contrast_on_delta=bool(memory_contrast_on_delta),
                    query_goal_dropout_rate=float(query_goal_dropout_rate),
                    query_goal_dropout_state_start=int(query_goal_dropout_state_start),
                    log_attention_metrics=bool(log_attention_metrics),
                    goal_prediction_loss_weight=float(goal_prediction_loss_weight),
                    goal_prediction_loss_type=str(goal_prediction_loss_type),
                ),
                in_axes=(0, 0, 0, 0, 0),
            )(batch['inner'], batch['query'], task_wrong_inner, task_contrast_inner, task_rngs)
            meta_loss = jnp.mean(task_losses)
            contrast_metrics = {
                'memory_contrast_loss': jnp.array(0.0, dtype=jnp.float32),
                'memory_contrast_accuracy': jnp.array(0.0, dtype=jnp.float32),
                'within_task_memory_similarity': jnp.array(0.0, dtype=jnp.float32),
                'between_task_memory_similarity': jnp.array(0.0, dtype=jnp.float32),
            }
            if bool(use_memory_contrast):
                z_a = task_aux['memory_contrast_z_a']
                z_b = task_aux['memory_contrast_z_b']
                labels = batch['meta']['vidx'] if 'meta' in batch and 'vidx' in batch['meta'] else jnp.arange(z_a.shape[0])
                contrast_loss, contrast_metrics = _symmetric_memory_contrast_loss(
                    z_a,
                    z_b,
                    labels,
                    temperature=float(memory_contrast_temperature),
                )
                meta_loss = meta_loss + jnp.asarray(float(memory_contrast_weight), dtype=meta_loss.dtype) * contrast_loss
            metrics = {
                key: jnp.mean(value)
                for key, value in task_aux.items()
                if key not in ('memory_contrast_z_a', 'memory_contrast_z_b')
            }
            metrics.update(contrast_metrics)
            metrics['meta_loss'] = meta_loss
            return metrics['meta_loss'], metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='device')
        metrics = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, axis_name='device'), metrics)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step


def create_adapt_fn(
    *,
    model: QueryMemoryDirectRegressionModel,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str = 'read',
    memory_layer_norm_after_update: bool = False,
):
    @jax.jit
    def adapt_fn(params: Any, inner_batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        initial_memory = _initial_memory_for_task(params, model=model, task_inner=inner_batch, train=False)
        adapted_memory, _ = _adapt_one_task(
            params,
            model=model,
            task_inner=inner_batch,
            initial_memory_tokens=initial_memory,
            inner_steps=int(inner_steps),
            inner_lr=float(inner_lr),
            max_grad_norm=float(max_grad_norm),
            first_order=bool(first_order),
            inner_loss_mode=str(inner_loss_mode),
            memory_layer_norm_after_update=bool(memory_layer_norm_after_update),
        )
        return adapted_memory

    return adapt_fn


def create_adapt_with_stats_fn(
    *,
    model: QueryMemoryDirectRegressionModel,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str = 'read',
    memory_layer_norm_after_update: bool = False,
):
    @jax.jit
    def adapt_with_stats_fn(params: Any, inner_batch: Dict[str, jnp.ndarray]):
        initial_memory = _initial_memory_for_task(params, model=model, task_inner=inner_batch, train=False)
        effective_clip = float(model.cfg.decoder.memory_update_clip_norm)
        if effective_clip <= 0.0:
            effective_clip = float(max_grad_norm)

        def inner_step(carry: jnp.ndarray, step_batch: Dict[str, jnp.ndarray]):
            def support_loss_fn(mem: jnp.ndarray) -> jnp.ndarray:
                return _inner_loss(
                    params,
                    model=model,
                    batch=step_batch,
                    memory_tokens=mem,
                    train=False,
                    inner_loss_mode=inner_loss_mode,
                )

            support_loss, grad = jax.value_and_grad(support_loss_fn)(carry)
            if bool(first_order):
                grad = jax.lax.stop_gradient(grad)
            grad_norm = _tree_global_norm(grad)
            grad = _clip_tree_by_global_norm(grad, effective_clip)
            next_mem = carry - jnp.asarray(inner_lr, dtype=carry.dtype) * grad
            if bool(memory_layer_norm_after_update):
                next_mem = _memory_layer_norm(next_mem)
            return next_mem, (support_loss, grad_norm)

        if int(inner_steps) <= 0:
            return (
                initial_memory,
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.float32),
            )

        adapted_memory, (inner_losses, inner_grad_norms) = jax.lax.scan(inner_step, initial_memory, inner_batch)
        return adapted_memory, inner_losses, inner_grad_norms

    return adapt_with_stats_fn


def create_predict_fn(*, model: QueryMemoryDirectRegressionModel):
    @jax.jit
    def predict_fn(params: Any, query_batch: Dict[str, jnp.ndarray], memory_tokens: jnp.ndarray) -> jnp.ndarray:
        return model.apply(
            {'params': params},
            query_xyz=query_batch['query_xyz'],
            query_state=query_batch['query_state'],
            query_valid=query_batch.get('query_valid', None),
            query_rgb=query_batch.get('query_rgb', None),
            query_mask_id=query_batch.get('query_mask_id', None),
            memory_tokens=memory_tokens,
            mode='read',
            train=False,
        )

    return predict_fn


def create_write_predict_fn(*, model: QueryMemoryDirectRegressionModel):
    @jax.jit
    def write_predict_fn(params: Any, write_batch: Dict[str, jnp.ndarray], memory_tokens: jnp.ndarray) -> jnp.ndarray:
        return _write_predict(
            params,
            model=model,
            batch=write_batch,
            memory_tokens=memory_tokens,
            train=False,
        )

    return write_predict_fn


def create_train_state(*, params: Any, outer_lr: float, weight_decay: float, max_grad_norm: float) -> TrainState:
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adamw(learning_rate=float(outer_lr), weight_decay=float(weight_decay)),
    )
    return TrainState.create(apply_fn=None, params=params, tx=tx)
