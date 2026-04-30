from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .attention import CrossAttention, IdentityContextAdaLN, SelfAttention


@dataclass(frozen=True)
class DirectDecoderConfig:
    d_model: int = 512
    n_heads: int = 8
    decoder_layers: int = 8
    decoder_mlp_mult: int = 4
    dropout: float = 0.0
    loss_type: str = 'l1'
    horizon: int = 16
    conditioner_mlp_mult: int = 2
    conditioner_dropout: float = 0.0
    context_attention_mode: str = 'two_ctx'
    memory_num_tokens: int = 128
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class PooledContextConditioner(nn.Module):
    cfg: DirectDecoderConfig

    def _masked_mean(self, tokens: Optional[jnp.ndarray], mask: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
        if tokens is None:
            return None
        if mask is None:
            return jnp.mean(tokens, axis=1)
        keep = mask.astype(tokens.dtype)[..., None]
        denom = jnp.maximum(jnp.sum(keep, axis=1), 1.0)
        pooled = jnp.sum(tokens * keep, axis=1) / denom
        empty = jnp.sum(mask.astype(jnp.int32), axis=1) == 0
        return jnp.where(empty[:, None], jnp.zeros_like(pooled), pooled)

    @nn.compact
    def __call__(
        self,
        *,
        query_tokens: jnp.ndarray,
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        pooled_query = self._masked_mean(query_tokens, query_mask)
        pooled_support = self._masked_mean(support_tokens, support_mask)
        if pooled_support is None:
            pooled_support = jnp.zeros_like(pooled_query)
        x = jnp.concatenate([pooled_query, pooled_support], axis=-1)
        hidden = max(int(self.cfg.d_model), int(self.cfg.conditioner_mlp_mult) * int(self.cfg.d_model))
        x = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype)(x)
        x = nn.Dense(hidden, dtype=dtype, param_dtype=param_dtype)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=float(self.cfg.conditioner_dropout))(x, deterministic=True)
        x = nn.Dense(int(self.cfg.d_model), dtype=dtype, param_dtype=param_dtype)(x)
        return x


class MlpBlock(nn.Module):
    d_model: int
    mlp_mult: int
    dropout: float
    dtype: jnp.dtype
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        hidden = int(self.mlp_mult) * int(self.d_model)
        x = nn.Dense(hidden, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=float(self.dropout))(x, deterministic=not train)
        x = nn.Dense(int(self.d_model), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x


class DirectChunkBlockTwoCtx(nn.Module):
    cfg: DirectDecoderConfig

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cond: jnp.ndarray,
        *,
        query_tokens: jnp.ndarray,
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
        train: bool,
    ) -> jnp.ndarray:
        d = int(self.cfg.d_model)
        n_heads = int(self.cfg.n_heads)
        dropout = float(self.cfg.dropout)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        drop = lambda y: nn.Dropout(rate=dropout)(y, deterministic=not train)

        h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln1')(x, cond)
        x = x + drop(SelfAttention(d, n_heads, dropout, dtype=dtype, param_dtype=param_dtype, name='self_attn')(h, train=train))

        h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln_q')(x, cond)
        x = x + drop(
            CrossAttention(d, n_heads, dropout, dtype=dtype, param_dtype=param_dtype, name='cross_attn_q')(
                h,
                query_tokens,
                kv_mask=query_mask,
                train=train,
            )
        )
        if support_tokens is not None:
            h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln_s')(x, cond)
            x = x + drop(
                CrossAttention(d, n_heads, dropout, dtype=dtype, param_dtype=param_dtype, name='cross_attn_s')(
                    h,
                    support_tokens,
                    kv_mask=support_mask,
                    train=train,
                )
            )
        h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln3')(x, cond)
        x = x + drop(MlpBlock(d, int(self.cfg.decoder_mlp_mult), dropout, dtype, param_dtype, name='mlp')(h, train=train))
        return x


class DirectDecoderCore(nn.Module):
    cfg: DirectDecoderConfig
    action_dim: int

    @nn.compact
    def __call__(
        self,
        *,
        query_tokens: jnp.ndarray,
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
        train: bool,
    ) -> jnp.ndarray:
        B = int(query_tokens.shape[0])
        d = int(self.cfg.d_model)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        cond_vec = PooledContextConditioner(self.cfg, name='context_conditioner')(
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
        )
        action_queries = self.param('action_queries', nn.initializers.normal(stddev=0.02), (int(self.cfg.horizon), d), param_dtype)
        action_slot_embed = self.param('action_slot_embed', nn.initializers.normal(stddev=0.02), (int(self.cfg.horizon), d), param_dtype)
        x = jnp.broadcast_to((action_queries + action_slot_embed).astype(dtype)[None, :, :], (B, int(self.cfg.horizon), d))
        for layer_idx in range(int(self.cfg.decoder_layers)):
            x = DirectChunkBlockTwoCtx(self.cfg, name=f'decoder_{layer_idx}')(
                x,
                cond_vec,
                query_tokens=query_tokens,
                query_mask=query_mask,
                support_tokens=support_tokens,
                support_mask=support_mask,
                train=train,
            )
        return nn.Dense(int(self.action_dim), dtype=dtype, param_dtype=param_dtype, name='action_out')(x)
