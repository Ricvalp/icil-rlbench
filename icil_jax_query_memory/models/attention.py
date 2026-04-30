from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class ModuleDTypeConfig:
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class IdentityContextAdaLN(nn.Module):
    d_model: int
    cond_dim: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        ss = nn.Dense(
            2 * int(self.d_model),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(cond)
        scale, shift = jnp.split(ss[:, None, :], 2, axis=-1)
        return h * (1.0 + scale) + shift


class CrossAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        kv: jnp.ndarray,
        *,
        kv_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        mask = None
        if kv_mask is not None:
            keep = kv_mask.astype(jnp.bool_)
            mask = jnp.broadcast_to(keep[:, None, None, :], (q.shape[0], 1, q.shape[1], kv.shape[1]))
        return nn.MultiHeadDotProductAttention(
            num_heads=int(self.n_heads),
            qkv_features=int(self.d_model),
            out_features=int(self.d_model),
            use_bias=False,
            dropout_rate=float(self.dropout),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(q, kv, mask=mask, deterministic=not train)


class SelfAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        x_mask: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        mask = None
        if x_mask is not None:
            keep = x_mask.astype(jnp.bool_)
            mask = jnp.logical_and(keep[:, None, :, None], keep[:, None, None, :])
        return nn.MultiHeadDotProductAttention(
            num_heads=int(self.n_heads),
            qkv_features=int(self.d_model),
            out_features=int(self.d_model),
            use_bias=False,
            dropout_rate=float(self.dropout),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x, x, mask=mask, deterministic=not train)
