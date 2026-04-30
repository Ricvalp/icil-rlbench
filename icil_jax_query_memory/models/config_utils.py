from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .direct_decoder import DirectDecoderConfig
from .query_memory_direct_regression import QueryMemoryDirectRegressionConfig
from .simple_query_point_encoder import SimpleQueryPointEncoderConfig


def resolve_dtype(name: str) -> jnp.dtype:
    value = str(name).strip().lower()
    if value in ('float32', 'fp32'):
        return jnp.float32
    if value in ('bf16', 'bfloat16'):
        return jnp.bfloat16
    raise ValueError(f'Unsupported dtype name {name!r}. Expected float32 or bf16.')


def build_model_config_from_raw(raw_model_cfg: Any, *, state_dim: int, action_dim: int, compute_dtype: jnp.dtype) -> QueryMemoryDirectRegressionConfig:
    query_encoder_name = str(getattr(raw_model_cfg, 'query_encoder_name', 'simple_query_point_encoder'))
    if query_encoder_name != 'simple_query_point_encoder':
        raise ValueError(
            'JAX query-memory v1 only supports query_encoder_name="simple_query_point_encoder". '
            f'Got {query_encoder_name!r}.'
        )
    decoder_raw = raw_model_cfg.query_memory_direct_regression
    encoder_raw = raw_model_cfg.simple_query_point_encoder
    encoder_cfg = SimpleQueryPointEncoderConfig(
        d_model=int(getattr(encoder_raw, 'd_model', 512)),
        use_rgb=bool(getattr(encoder_raw, 'use_rgb', True)),
        use_mask_id=bool(getattr(encoder_raw, 'use_mask_id', False)),
        mask_hash_buckets=int(getattr(encoder_raw, 'mask_hash_buckets', 2048)),
        use_gripper_point_features=bool(getattr(encoder_raw, 'use_gripper_point_features', False)),
        gripper_xyz_state_start=int(getattr(encoder_raw, 'gripper_xyz_state_start', 0)),
        max_T_obs=int(getattr(encoder_raw, 'max_T_obs', 16)),
        add_state_token=bool(getattr(encoder_raw, 'add_state_token', True)),
        dtype=compute_dtype,
        param_dtype=jnp.float32,
    )
    decoder_cfg = DirectDecoderConfig(
        d_model=int(getattr(decoder_raw, 'd_model', 512)),
        n_heads=int(getattr(decoder_raw, 'n_heads', 8)),
        decoder_layers=int(getattr(decoder_raw, 'decoder_layers', 8)),
        decoder_mlp_mult=int(getattr(decoder_raw, 'decoder_mlp_mult', 4)),
        dropout=float(getattr(decoder_raw, 'dropout', 0.0)),
        loss_type=str(getattr(decoder_raw, 'loss_type', 'l1')),
        horizon=int(getattr(decoder_raw, 'horizon', 16)),
        conditioner_mlp_mult=int(getattr(decoder_raw, 'conditioner_mlp_mult', 2)),
        conditioner_dropout=float(getattr(decoder_raw, 'conditioner_dropout', 0.0)),
        context_attention_mode=str(getattr(decoder_raw, 'context_attention_mode', 'two_ctx')),
        memory_num_tokens=int(getattr(decoder_raw, 'memory_num_tokens', 128)),
        dtype=compute_dtype,
        param_dtype=jnp.float32,
    )
    if decoder_cfg.context_attention_mode != 'two_ctx':
        raise ValueError(
            'JAX query-memory v1 only supports context_attention_mode="two_ctx". '
            f'Got {decoder_cfg.context_attention_mode!r}.'
        )
    if encoder_cfg.d_model != decoder_cfg.d_model:
        raise ValueError(
            f'd_model mismatch between encoder ({encoder_cfg.d_model}) and decoder ({decoder_cfg.d_model}).'
        )
    if decoder_cfg.horizon < 1:
        raise ValueError('decoder horizon must be >= 1.')
    return QueryMemoryDirectRegressionConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        query_encoder=encoder_cfg,
        decoder=decoder_cfg,
    )
