from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .attention import CrossAttention, SelfAttention
from .object_centric_state import ObjectCentricStateTokenizer, ObjectCentricStateTokenizerConfig


@dataclass(frozen=True)
class SupportEncoderConfig:
    d_model: int = 256
    n_heads: int = 4
    memory_num_tokens: int = 64
    support_encoder_layers: int = 2
    memory_self_attn_layers: int = 1
    mlp_mult: int = 2
    max_support_chunks: int = 256
    max_demo_id: int = 16
    max_time_bins: int = 512
    dropout: float = 0.0
    goal_visible: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class _Mlp(nn.Module):
    d_model: int
    mlp_mult: int
    dropout: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        hidden = max(int(self.d_model), int(self.mlp_mult) * int(self.d_model))
        x = nn.Dense(hidden, dtype=self.dtype, param_dtype=self.param_dtype, name='in')(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=float(self.dropout))(x, deterministic=not train)
        return nn.Dense(int(self.d_model), dtype=self.dtype, param_dtype=self.param_dtype, name='out')(x)


def _masked_mean(tokens: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    keep = mask.astype(tokens.dtype)[..., None]
    denom = jnp.maximum(jnp.sum(keep, axis=1), 1.0)
    return jnp.sum(tokens * keep, axis=1) / denom


def _sinusoidal_scalar_embedding(x: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = max(1, int(dim) // 2)
    freqs = jnp.exp(
        -jnp.log(jnp.asarray(10000.0, dtype=x.dtype))
        * jnp.arange(half, dtype=x.dtype)
        / jnp.asarray(max(1, half - 1), dtype=x.dtype)
    )
    angles = x[..., None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < int(dim):
        emb = jnp.pad(emb, ((0, 0), (0, int(dim) - emb.shape[-1])))
    return emb[..., : int(dim)]


class SupportEncoderMemory(nn.Module):
    cfg: SupportEncoderConfig
    state_tokenizer_cfg: ObjectCentricStateTokenizerConfig
    state_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        *,
        support_state: jnp.ndarray,
        support_target_action: jnp.ndarray,
        support_demo_id: Optional[jnp.ndarray] = None,
        support_chunk_start: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        if support_state.ndim not in (3, 4):
            raise ValueError(f'support_state must be [S,T,D] or [B,S,T,D], got {tuple(support_state.shape)}')
        unbatched = support_state.ndim == 3
        if unbatched:
            support_state = support_state[None, ...]
            support_target_action = support_target_action[None, ...]
            if support_demo_id is not None:
                support_demo_id = support_demo_id[None, ...]
            if support_chunk_start is not None:
                support_chunk_start = support_chunk_start[None, ...]
        if support_target_action.ndim != 4:
            raise ValueError(
                f'support_target_action must be [B,S,H,A] after batching, got {tuple(support_target_action.shape)}'
            )
        B, S, T, _ = support_state.shape
        d = int(self.cfg.d_model)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype

        flat_state = support_state.reshape(B * S, T, int(self.state_dim))
        state_tokens, state_mask = ObjectCentricStateTokenizer(
            cfg=self.state_tokenizer_cfg,
            state_dim=int(self.state_dim),
            name='state_tokenizer',
        )(
            query_state=flat_state,
            goal_visible=bool(self.cfg.goal_visible),
        )
        state_pooled = _masked_mean(state_tokens, state_mask).reshape(B, S, d)

        action_flat = support_target_action.astype(dtype).reshape(B, S, -1)
        action_tok = nn.Dense(d, dtype=dtype, param_dtype=param_dtype, name='action_proj_in')(action_flat)
        action_tok = nn.silu(action_tok)
        action_tok = nn.Dense(d, dtype=dtype, param_dtype=param_dtype, name='action_proj_out')(action_tok)

        support_tok = jnp.concatenate([state_pooled, action_tok], axis=-1)
        support_tok = nn.Dense(d, dtype=dtype, param_dtype=param_dtype, name='support_fuse')(support_tok)

        if support_demo_id is None:
            demo_id = jnp.zeros((B, S), dtype=jnp.int32)
        else:
            demo_id = jnp.asarray(support_demo_id)
            if demo_id.ndim == 3:
                demo_id = demo_id[..., 0]
            demo_id = jnp.clip(jnp.rint(demo_id).astype(jnp.int32), 0, max(1, int(self.cfg.max_demo_id)) - 1)
        demo_embed = nn.Embed(
            num_embeddings=max(1, int(self.cfg.max_demo_id)),
            features=d,
            dtype=dtype,
            param_dtype=param_dtype,
            name='demo_embed',
        )(demo_id)
        support_tok = support_tok + demo_embed

        if support_chunk_start is not None:
            t = jnp.asarray(support_chunk_start, dtype=jnp.float32)
            if t.ndim == 3:
                t = t[..., 0]
            max_time = jnp.asarray(max(1, int(self.cfg.max_time_bins) - 1), dtype=jnp.float32)
            time_emb = _sinusoidal_scalar_embedding(jnp.clip(t, 0.0, max_time).reshape(-1) / max_time, d)
            time_emb = time_emb.reshape(B, S, d).astype(dtype)
            time_emb = nn.Dense(d, dtype=dtype, param_dtype=param_dtype, name='time_proj')(time_emb)
            support_tok = support_tok + time_emb

        support_mask = jnp.ones((B, S), dtype=jnp.bool_)
        for layer_idx in range(int(self.cfg.support_encoder_layers)):
            h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name=f'support_ln_attn_{layer_idx}')(support_tok)
            support_tok = support_tok + SelfAttention(
                d,
                int(self.cfg.n_heads),
                float(self.cfg.dropout),
                dtype=dtype,
                param_dtype=param_dtype,
                name=f'support_self_attn_{layer_idx}',
            )(h, x_mask=support_mask, train=train)
            h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name=f'support_ln_mlp_{layer_idx}')(support_tok)
            support_tok = support_tok + _Mlp(
                d,
                int(self.cfg.mlp_mult),
                float(self.cfg.dropout),
                dtype,
                param_dtype,
                name=f'support_mlp_{layer_idx}',
            )(h, train=train)

        mem_queries = self.param(
            'memory_queries',
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.memory_num_tokens), d),
            param_dtype,
        )
        memory = jnp.broadcast_to(mem_queries.astype(dtype)[None, :, :], (B, int(self.cfg.memory_num_tokens), d))
        h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name='memory_cross_ln')(memory)
        memory = memory + CrossAttention(
            d,
            int(self.cfg.n_heads),
            float(self.cfg.dropout),
            dtype=dtype,
            param_dtype=param_dtype,
            name='memory_cross_attn',
        )(h, support_tok, kv_mask=support_mask, train=train)

        memory_mask = jnp.ones(memory.shape[:2], dtype=jnp.bool_)
        for layer_idx in range(int(self.cfg.memory_self_attn_layers)):
            h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name=f'memory_ln_attn_{layer_idx}')(memory)
            memory = memory + SelfAttention(
                d,
                int(self.cfg.n_heads),
                float(self.cfg.dropout),
                dtype=dtype,
                param_dtype=param_dtype,
                name=f'memory_self_attn_{layer_idx}',
            )(h, x_mask=memory_mask, train=train)
            h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name=f'memory_ln_mlp_{layer_idx}')(memory)
            memory = memory + _Mlp(
                d,
                int(self.cfg.mlp_mult),
                float(self.cfg.dropout),
                dtype,
                param_dtype,
                name=f'memory_mlp_{layer_idx}',
            )(h, train=train)

        return memory[0] if unbatched else memory
