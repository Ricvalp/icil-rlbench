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


def _batch_loss(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    batch: Dict[str, jnp.ndarray],
    memory_tokens: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    pred = model.apply(
        {'params': params},
        query_xyz=batch['query_xyz'],
        query_state=batch['query_state'],
        query_valid=batch.get('query_valid', None),
        query_rgb=batch.get('query_rgb', None),
        query_mask_id=batch.get('query_mask_id', None),
        memory_tokens=memory_tokens,
        train=train,
    )
    target = batch['target_action']
    loss_type = str(model.cfg.decoder.loss_type).lower()
    if loss_type == 'mse':
        return jnp.mean(jnp.square(pred - target))
    if loss_type == 'l1':
        return jnp.mean(jnp.abs(pred - target))
    raise ValueError(f'Unsupported loss_type={model.cfg.decoder.loss_type!r}.')


def _adapt_one_task(
    params: Any,
    *,
    model: QueryMemoryDirectRegressionModel,
    task_inner: Dict[str, jnp.ndarray],
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    memory_tokens = params['memory_token_init']

    def inner_step(carry: jnp.ndarray, step_batch: Dict[str, jnp.ndarray]):
        def support_loss_fn(mem: jnp.ndarray) -> jnp.ndarray:
            return _batch_loss(params, model=model, batch=step_batch, memory_tokens=mem, train=True)

        support_loss, grad = jax.value_and_grad(support_loss_fn)(carry)
        if bool(first_order):
            grad = jax.lax.stop_gradient(grad)
        grad_norm = _tree_global_norm(grad)
        grad = _clip_tree_by_global_norm(grad, max_grad_norm)
        next_mem = carry - jnp.asarray(inner_lr, dtype=carry.dtype) * grad
        return next_mem, (support_loss, grad_norm)

    if int(inner_steps) <= 0:
        return memory_tokens, jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)

    adapted_memory, (inner_losses, inner_grad_norms) = jax.lax.scan(inner_step, memory_tokens, task_inner)
    return adapted_memory, jnp.mean(inner_losses), jnp.mean(inner_grad_norms)


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
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    adapted_memory, inner_loss_mean, inner_grad_mean = _adapt_one_task(
        params,
        model=model,
        task_inner=task_inner,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        max_grad_norm=max_grad_norm,
        first_order=first_order,
    )
    meta_loss = _batch_loss(params, model=model, batch=task_query, memory_tokens=adapted_memory, train=True)
    return meta_loss, (inner_loss_mean, inner_grad_mean)


def create_train_step(
    *,
    model: QueryMemoryDirectRegressionModel,
    inner_steps: int,
    inner_lr: float,
    max_grad_norm: float,
    first_order: bool,
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
                ),
                in_axes=(0, 0),
            )(batch['inner'], batch['query'])
            inner_loss_means, inner_grad_means = task_aux
            metrics = {
                'meta_loss': jnp.mean(task_losses),
                'inner_support_loss': jnp.mean(inner_loss_means),
                'inner_memory_grad_norm': jnp.mean(inner_grad_means),
            }
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
):
    @jax.jit
    def adapt_fn(params: Any, inner_batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        adapted_memory, _, _ = _adapt_one_task(
            params,
            model=model,
            task_inner=inner_batch,
            inner_steps=int(inner_steps),
            inner_lr=float(inner_lr),
            max_grad_norm=float(max_grad_norm),
            first_order=bool(first_order),
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
):
    @jax.jit
    def adapt_with_stats_fn(params: Any, inner_batch: Dict[str, jnp.ndarray]):
        memory_tokens = params['memory_token_init']

        def inner_step(carry: jnp.ndarray, step_batch: Dict[str, jnp.ndarray]):
            def support_loss_fn(mem: jnp.ndarray) -> jnp.ndarray:
                return _batch_loss(params, model=model, batch=step_batch, memory_tokens=mem, train=False)

            support_loss, grad = jax.value_and_grad(support_loss_fn)(carry)
            if bool(first_order):
                grad = jax.lax.stop_gradient(grad)
            grad_norm = _tree_global_norm(grad)
            grad = _clip_tree_by_global_norm(grad, max_grad_norm)
            next_mem = carry - jnp.asarray(inner_lr, dtype=carry.dtype) * grad
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
            train=False,
        )

    return predict_fn


def create_train_state(*, params: Any, outer_lr: float, weight_decay: float, max_grad_norm: float) -> TrainState:
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adamw(learning_rate=float(outer_lr), weight_decay=float(weight_decay)),
    )
    return TrainState.create(apply_fn=None, params=params, tx=tx)
