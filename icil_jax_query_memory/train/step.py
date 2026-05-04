from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
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


def _adapt_one_task(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
    inner_loss_mode: str,
    memory_layer_norm_after_update: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    memory_tokens = params['memory_token_init']
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


def _task_meta_objective(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    task_query: Dict[str, jnp.ndarray],
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
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    initial_memory = params['memory_token_init']
    adapted_memory, inner_stats = _adapt_one_task(
        params,
        model=model,
        task_inner=task_inner,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        max_grad_norm=max_grad_norm,
        first_order=first_order,
        inner_loss_mode=inner_loss_mode,
        memory_layer_norm_after_update=memory_layer_norm_after_update,
    )
    read_loss_after = _read_loss(params, model=model, batch=task_query, memory_tokens=adapted_memory, train=True)
    meta_loss = read_loss_after
    need_read_before = (
        bool(use_read_improvement_margin)
        or bool(log_output_delta)
        or not bool(training_mode_metrics_only)
    )
    if need_read_before:
        read_loss_before = _read_loss(params, model=model, batch=task_query, memory_tokens=initial_memory, train=True)
    else:
        read_loss_before = jax.lax.stop_gradient(read_loss_after)

    if bool(use_read_improvement_margin):
        margin = jnp.asarray(read_improvement_margin, dtype=read_loss_after.dtype)
        weight = jnp.asarray(read_improvement_margin_weight, dtype=read_loss_after.dtype)
        meta_loss = meta_loss + weight * jnp.maximum(0.0, margin + read_loss_after - read_loss_before)

    if bool(log_output_delta):
        pred_before = _read_predict(params, model=model, batch=task_query, memory_tokens=initial_memory, train=True)
        pred_after = _read_predict(params, model=model, batch=task_query, memory_tokens=adapted_memory, train=True)
        action_output_delta = jnp.mean(jnp.abs(pred_after - pred_before))
    else:
        action_output_delta = jnp.array(0.0, dtype=jnp.float32)

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
    }
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
):
    @partial(jax.pmap, axis_name='device')
    def train_step(state: TrainState, batch: Dict[str, Any]):
        def loss_fn(params: Any):
            task_losses, task_aux = jax.vmap(
                lambda task_inner, task_query: _task_meta_objective(
                    params,
                    model=model,
                    task_inner=task_inner,
                    task_query=task_query,
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
                ),
                in_axes=(0, 0),
            )(batch['inner'], batch['query'])
            metrics = {key: jnp.mean(value) for key, value in task_aux.items()}
            metrics['meta_loss'] = jnp.mean(task_losses)
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
        adapted_memory, _ = _adapt_one_task(
            params,
            model=model,
            task_inner=inner_batch,
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
        memory_tokens = params['memory_token_init']
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
                memory_tokens,
                jnp.zeros((0,), dtype=jnp.float32),
                jnp.zeros((0,), dtype=jnp.float32),
            )

        adapted_memory, (inner_losses, inner_grad_norms) = jax.lax.scan(inner_step, memory_tokens, inner_batch)
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
