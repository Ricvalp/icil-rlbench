from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .direct_decoder import DirectDecoderConfig, DirectDecoderCore
from .simple_query_point_encoder import SimpleQueryPointEncoder, SimpleQueryPointEncoderConfig


@dataclass(frozen=True)
class QueryMemoryDirectRegressionConfig:
    state_dim: int
    action_dim: int
    query_encoder: SimpleQueryPointEncoderConfig
    decoder: DirectDecoderConfig


class QueryMemoryDirectRegressionModel(nn.Module):
    cfg: QueryMemoryDirectRegressionConfig

    @nn.compact
    def __call__(
        self,
        *,
        query_xyz: jnp.ndarray,
        query_state: jnp.ndarray,
        query_valid: Optional[jnp.ndarray] = None,
        query_rgb: Optional[jnp.ndarray] = None,
        query_mask_id: Optional[jnp.ndarray] = None,
        memory_tokens: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        encoder = SimpleQueryPointEncoder(cfg=self.cfg.query_encoder, state_dim=int(self.cfg.state_dim), name='context_encoder')
        query_tokens, query_mask = encoder(
            query_xyz=query_xyz,
            query_state=query_state,
            query_valid=query_valid,
            query_rgb=query_rgb,
            query_mask_id=query_mask_id,
        )
        mem_init = self.param(
            'memory_token_init',
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.decoder.memory_num_tokens), int(self.cfg.decoder.d_model)),
            self.cfg.decoder.param_dtype,
        )
        if memory_tokens is None:
            support_tokens = jnp.broadcast_to(mem_init.astype(self.cfg.decoder.dtype)[None, :, :], (query_xyz.shape[0],) + mem_init.shape)
        else:
            if memory_tokens.ndim == 2:
                support_tokens = jnp.broadcast_to(memory_tokens.astype(self.cfg.decoder.dtype)[None, :, :], (query_xyz.shape[0],) + tuple(memory_tokens.shape))
            elif memory_tokens.ndim == 3:
                support_tokens = memory_tokens.astype(self.cfg.decoder.dtype)
            else:
                raise ValueError(f'memory_tokens must have ndim 2 or 3, got {memory_tokens.ndim}.')
        support_mask = jnp.ones(support_tokens.shape[:2], dtype=jnp.bool_)
        decoder = DirectDecoderCore(cfg=self.cfg.decoder, action_dim=int(self.cfg.action_dim), name='decoder')
        return decoder(
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
            train=train,
        )
