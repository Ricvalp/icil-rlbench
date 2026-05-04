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
        separate_write_read_heads=bool(getattr(decoder_raw, 'separate_write_read_heads', False)),
        shared_write_read_head=bool(getattr(decoder_raw, 'shared_write_read_head', False)),
        write_num_query_tokens=int(getattr(decoder_raw, 'write_num_query_tokens', 4)),
        write_use_demo_id_embed=bool(getattr(decoder_raw, 'write_use_demo_id_embed', True)),
        write_use_time_embed=bool(getattr(decoder_raw, 'write_use_time_embed', True)),
        write_max_demo_id=int(getattr(decoder_raw, 'write_max_demo_id', 16)),
        write_max_time_bins=int(getattr(decoder_raw, 'write_max_time_bins', 512)),
        write_time_embed_type=str(getattr(decoder_raw, 'write_time_embed_type', 'continuous_sinusoidal')),
        write_query_mlp_mult=int(getattr(decoder_raw, 'write_query_mlp_mult', 2)),
        write_use_support_obs=bool(getattr(decoder_raw, 'write_use_support_obs', False)),
        use_decoder_mode_embed=bool(getattr(decoder_raw, 'use_decoder_mode_embed', False)),
        memory_layer_norm_after_update=bool(getattr(decoder_raw, 'memory_layer_norm_after_update', False)),
        memory_update_clip_norm=float(getattr(decoder_raw, 'memory_update_clip_norm', 0.0)),
        action_loss_type=str(getattr(decoder_raw, 'action_loss_type', '')),
        position_loss_weight=float(getattr(decoder_raw, 'position_loss_weight', 1.0)),
        rotation_loss_weight=float(getattr(decoder_raw, 'rotation_loss_weight', 1.0)),
        gripper_loss_weight=float(getattr(decoder_raw, 'gripper_loss_weight', 1.0)),
        chunk_decay=float(getattr(decoder_raw, 'chunk_decay', 0.0)),
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
    if decoder_cfg.write_num_query_tokens < 1:
        raise ValueError('write_num_query_tokens must be >= 1.')
    if decoder_cfg.write_max_demo_id < 1:
        raise ValueError('write_max_demo_id must be >= 1.')
    if decoder_cfg.write_max_time_bins < 1:
        raise ValueError('write_max_time_bins must be >= 1.')
    return QueryMemoryDirectRegressionConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        query_encoder=encoder_cfg,
        decoder=decoder_cfg,
    )
