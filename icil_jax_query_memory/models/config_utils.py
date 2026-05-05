from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .direct_decoder import DirectDecoderConfig
from .object_centric_state import ObjectCentricStateTokenizerConfig, parse_slice
from .query_memory_direct_regression import QueryMemoryDirectRegressionConfig
from .simple_query_point_encoder import SimpleQueryPointEncoderConfig
from .support_encoder_memory import SupportEncoderConfig


def resolve_dtype(name: str) -> jnp.dtype:
    value = str(name).strip().lower()
    if value in ('float32', 'fp32'):
        return jnp.float32
    if value in ('bf16', 'bfloat16'):
        return jnp.bfloat16
    raise ValueError(f'Unsupported dtype name {name!r}. Expected float32 or bf16.')


def build_model_config_from_raw(raw_model_cfg: Any, *, state_dim: int, action_dim: int, compute_dtype: jnp.dtype) -> QueryMemoryDirectRegressionConfig:
    query_encoder_name = str(getattr(raw_model_cfg, 'query_encoder_name', 'simple_query_point_encoder'))
    if query_encoder_name not in ('simple_query_point_encoder', 'object_centric_state'):
        raise ValueError(
            'JAX query-memory supports query_encoder_name="simple_query_point_encoder" or "object_centric_state". '
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
    object_raw = getattr(raw_model_cfg, 'object_centric_state', None)
    object_cfg = None
    if object_raw is not None:
        object_cfg = ObjectCentricStateTokenizerConfig(
            d_model=int(getattr(object_raw, 'd_model', getattr(encoder_raw, 'd_model', 512))),
            max_T_obs=int(getattr(object_raw, 'max_T_obs', getattr(encoder_raw, 'max_T_obs', 16))),
            hand_pos_slice=parse_slice(getattr(object_raw, 'hand_pos_slice', (0, 3)), (0, 3)),
            gripper_slice=parse_slice(getattr(object_raw, 'gripper_slice', (3, 4)), (3, 4)),
            obj1_pos_slice=parse_slice(getattr(object_raw, 'obj1_pos_slice', (4, 7)), (4, 7)),
            obj2_pos_slice=parse_slice(getattr(object_raw, 'obj2_pos_slice', (11, 14)), (11, 14)),
            goal_pos_slice=parse_slice(getattr(object_raw, 'goal_pos_slice', (36, 39)), (36, 39)),
            has_obj2=bool(getattr(object_raw, 'has_obj2', True)),
            goal_available=bool(getattr(object_raw, 'goal_available', int(state_dim) >= 39)),
            goal_visible=bool(getattr(object_raw, 'goal_visible', True)),
            hidden_goal_token_policy=str(getattr(object_raw, 'hidden_goal_token_policy', 'mask')),
            mlp_mult=int(getattr(object_raw, 'mlp_mult', 2)),
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
        memory_conditioning_mode=str(getattr(decoder_raw, 'memory_conditioning_mode', 'none')),
        memory_conditioning_strength=float(getattr(decoder_raw, 'memory_conditioning_strength', 1.0)),
        log_attention_weights=bool(getattr(decoder_raw, 'log_attention_weights', False)),
        use_goal_prediction_head=bool(getattr(decoder_raw, 'use_goal_prediction_head', False)),
        goal_prediction_mlp_mult=int(getattr(decoder_raw, 'goal_prediction_mlp_mult', 2)),
        dtype=compute_dtype,
        param_dtype=jnp.float32,
    )
    if decoder_cfg.context_attention_mode != 'two_ctx':
        raise ValueError(
            'JAX query-memory v1 only supports context_attention_mode="two_ctx". '
            f'Got {decoder_cfg.context_attention_mode!r}.'
        )
    support_raw = getattr(raw_model_cfg, 'support_encoder_memory', None)
    support_cfg = None
    if support_raw is not None:
        support_cfg = SupportEncoderConfig(
            d_model=int(getattr(support_raw, 'd_model', decoder_cfg.d_model)),
            n_heads=int(getattr(support_raw, 'n_heads', decoder_cfg.n_heads)),
            memory_num_tokens=int(getattr(support_raw, 'memory_num_tokens', decoder_cfg.memory_num_tokens)),
            support_encoder_layers=int(getattr(support_raw, 'support_encoder_layers', 2)),
            memory_self_attn_layers=int(getattr(support_raw, 'memory_self_attn_layers', 1)),
            mlp_mult=int(getattr(support_raw, 'mlp_mult', 2)),
            max_support_chunks=int(getattr(support_raw, 'max_support_chunks', 256)),
            max_demo_id=int(getattr(support_raw, 'max_demo_id', decoder_cfg.write_max_demo_id)),
            max_time_bins=int(getattr(support_raw, 'max_time_bins', decoder_cfg.write_max_time_bins)),
            dropout=float(getattr(support_raw, 'dropout', 0.0)),
            goal_visible=bool(getattr(support_raw, 'goal_visible', True)),
            dtype=compute_dtype,
            param_dtype=jnp.float32,
        )

    tokenizer_d_model = object_cfg.d_model if query_encoder_name == 'object_centric_state' and object_cfg is not None else encoder_cfg.d_model
    if tokenizer_d_model != decoder_cfg.d_model:
        raise ValueError(
            f'd_model mismatch between encoder/tokenizer ({tokenizer_d_model}) and decoder ({decoder_cfg.d_model}).'
        )
    if support_cfg is not None and int(support_cfg.d_model) != int(decoder_cfg.d_model):
        raise ValueError(f'd_model mismatch between support encoder ({support_cfg.d_model}) and decoder ({decoder_cfg.d_model}).')
    if support_cfg is not None and int(support_cfg.memory_num_tokens) != int(decoder_cfg.memory_num_tokens):
        raise ValueError(
            f'memory_num_tokens mismatch between support encoder ({support_cfg.memory_num_tokens}) '
            f'and decoder ({decoder_cfg.memory_num_tokens}).'
        )
    if decoder_cfg.horizon < 1:
        raise ValueError('decoder horizon must be >= 1.')
    if decoder_cfg.write_num_query_tokens < 1:
        raise ValueError('write_num_query_tokens must be >= 1.')
    if decoder_cfg.write_max_demo_id < 1:
        raise ValueError('write_max_demo_id must be >= 1.')
    if decoder_cfg.write_max_time_bins < 1:
        raise ValueError('write_max_time_bins must be >= 1.')
    if str(decoder_cfg.memory_conditioning_mode).strip().lower() not in (
        '',
        'none',
        'off',
        'false',
        'cross_attn',
        'film',
        'adaln',
        'cross_attn_plus_film',
        'cross_attn_plus_adaln',
    ):
        raise ValueError(
            "memory_conditioning_mode must be one of: 'none', 'cross_attn', 'film', 'adaln', "
            "'cross_attn_plus_film', 'cross_attn_plus_adaln'. "
            f"Got {decoder_cfg.memory_conditioning_mode!r}."
        )
    if decoder_cfg.goal_prediction_mlp_mult < 1:
        raise ValueError('goal_prediction_mlp_mult must be >= 1.')
    return QueryMemoryDirectRegressionConfig(
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        query_encoder=encoder_cfg,
        decoder=decoder_cfg,
        query_tokenizer_name=query_encoder_name,
        object_tokenizer=object_cfg,
        support_encoder=support_cfg,
        memory_initialization_mode=str(getattr(decoder_raw, 'memory_initialization_mode', 'base_only')),
        query_goal_visible=not bool(getattr(raw_model_cfg, 'query_goal_hidden', False)),
        support_goal_visible=not bool(getattr(raw_model_cfg, 'support_goal_hidden', False)),
    )
