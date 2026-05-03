from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import jax_utils

from icil_jax_query_memory.models.direct_decoder import DirectDecoderConfig
from icil_jax_query_memory.models.query_memory_direct_regression import (
    QueryMemoryDirectRegressionConfig,
    QueryMemoryDirectRegressionModel,
)
from icil_jax_query_memory.models.simple_query_point_encoder import SimpleQueryPointEncoderConfig
from icil_jax_query_memory.train.step import create_train_state, create_train_step


def main() -> None:
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError('No JAX devices available.')
    num_devices = len(devices)

    state_dim = 8
    action_dim = 8
    t_obs = 2
    num_points = 16
    horizon = 4
    d_model = 32
    memory_tokens = 8
    inner_steps = 1
    inner_batch = 3
    query_batch = 2
    per_device_tasks = 1

    cfg = QueryMemoryDirectRegressionConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        query_encoder=SimpleQueryPointEncoderConfig(
            d_model=d_model,
            use_rgb=True,
            use_mask_id=False,
            max_T_obs=4,
            add_state_token=True,
        ),
        decoder=DirectDecoderConfig(
            d_model=d_model,
            n_heads=4,
            decoder_layers=1,
            decoder_mlp_mult=2,
            horizon=horizon,
            conditioner_mlp_mult=1,
            memory_num_tokens=memory_tokens,
            separate_write_read_heads=True,
            use_decoder_mode_embed=True,
            write_num_query_tokens=4,
            memory_layer_norm_after_update=True,
            action_loss_type='l1',
        ),
    )
    model = QueryMemoryDirectRegressionModel(cfg=cfg)
    query = {
        'query_xyz': jnp.zeros((query_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'query_state': jnp.zeros((query_batch, t_obs, state_dim), dtype=jnp.float32),
        'query_valid': jnp.ones((query_batch, t_obs, num_points), dtype=jnp.bool_),
        'query_rgb': jnp.zeros((query_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'target_action': jnp.zeros((query_batch, horizon, action_dim), dtype=jnp.float32),
    }
    params = model.init(
        jax.random.PRNGKey(0),
        query_xyz=query['query_xyz'],
        query_state=query['query_state'],
        query_valid=query['query_valid'],
        query_rgb=query['query_rgb'],
        memory_tokens=None,
        mode='read',
        train=False,
    )['params']

    read_pred = model.apply(
        {'params': params},
        query_xyz=query['query_xyz'],
        query_state=query['query_state'],
        query_valid=query['query_valid'],
        query_rgb=query['query_rgb'],
        memory_tokens=None,
        mode='read',
        train=False,
    )
    if read_pred.shape != (query_batch, horizon, action_dim):
        raise AssertionError(f'bad READ shape: {read_pred.shape}')

    write_memory = jnp.broadcast_to(params['memory_token_init'][None, :, :], (inner_batch, memory_tokens, d_model))
    write_pred = model.apply(
        {'params': params},
        memory_tokens=write_memory,
        mode='write',
        write_demo_id=jnp.arange(inner_batch, dtype=jnp.int32),
        write_chunk_start=jnp.arange(inner_batch, dtype=jnp.float32),
        train=False,
    )
    if write_pred.shape != (inner_batch, horizon, action_dim):
        raise AssertionError(f'bad WRITE shape: {write_pred.shape}')

    write_missing_meta = model.apply(
        {'params': params},
        memory_tokens=write_memory,
        mode='write',
        train=False,
    )
    if write_missing_meta.shape != (inner_batch, horizon, action_dim):
        raise AssertionError(f'bad WRITE missing-meta shape: {write_missing_meta.shape}')

    state = create_train_state(params=params, outer_lr=1e-4, weight_decay=0.0, max_grad_norm=1.0)
    p_state = jax_utils.replicate(state, devices=devices)
    p_train_step = create_train_step(
        model=model,
        inner_steps=inner_steps,
        inner_lr=3e-3,
        max_grad_norm=1.0,
        first_order=True,
        inner_loss_mode='write',
        memory_layer_norm_after_update=True,
        log_output_delta=True,
    )
    inner = {
        'query_xyz': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'query_state': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch, t_obs, state_dim), dtype=jnp.float32),
        'query_valid': jnp.ones((num_devices, per_device_tasks, inner_steps, inner_batch, t_obs, num_points), dtype=jnp.bool_),
        'query_rgb': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'target_action': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch, horizon, action_dim), dtype=jnp.float32),
        'demo_id': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch), dtype=jnp.int32),
        'chunk_start': jnp.zeros((num_devices, per_device_tasks, inner_steps, inner_batch), dtype=jnp.float32),
    }
    query_p = {
        'query_xyz': jnp.zeros((num_devices, per_device_tasks, query_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'query_state': jnp.zeros((num_devices, per_device_tasks, query_batch, t_obs, state_dim), dtype=jnp.float32),
        'query_valid': jnp.ones((num_devices, per_device_tasks, query_batch, t_obs, num_points), dtype=jnp.bool_),
        'query_rgb': jnp.zeros((num_devices, per_device_tasks, query_batch, t_obs, num_points, 3), dtype=jnp.float32),
        'target_action': jnp.zeros((num_devices, per_device_tasks, query_batch, horizon, action_dim), dtype=jnp.float32),
    }
    p_state, metrics = p_train_step(p_state, {'inner': inner, 'query': query_p})
    del p_state
    metrics_host = {k: float(jax.device_get(jax_utils.unreplicate(v))) for k, v in metrics.items()}
    required = (
        'meta_loss',
        'read_loss_before',
        'read_loss_after',
        'inner_write_loss_mean',
        'inner_memory_grad_norm',
        'memory_delta_norm',
        'memory_relative_delta_norm',
        'action_output_delta',
    )
    missing = [key for key in required if key not in metrics_host]
    if missing:
        raise AssertionError(f'missing metrics: {missing}')
    print('JAX WRITE/READ GradMem smoke check passed.')
    for key in required:
        print(f'{key}: {metrics_host[key]:.6f}')


if __name__ == '__main__':
    main()
