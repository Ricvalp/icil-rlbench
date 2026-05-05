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
    separate_write_read_heads: bool = False
    shared_write_read_head: bool = False
    write_num_query_tokens: int = 4
    write_use_demo_id_embed: bool = True
    write_use_time_embed: bool = True
    write_max_demo_id: int = 16
    write_max_time_bins: int = 512
    write_time_embed_type: str = 'continuous_sinusoidal'
    write_query_mlp_mult: int = 2
    write_use_support_obs: bool = False
    use_decoder_mode_embed: bool = False
    memory_layer_norm_after_update: bool = False
    memory_update_clip_norm: float = 0.0
    action_loss_type: str = ''
    position_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    gripper_loss_weight: float = 1.0
    chunk_decay: float = 0.0
    memory_conditioning_mode: str = 'none'
    memory_conditioning_strength: float = 1.0
    log_attention_weights: bool = False
    use_goal_prediction_head: bool = False
    goal_prediction_mlp_mult: int = 2
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


def _as_token_metadata(
    value: Optional[jnp.ndarray],
    *,
    batch_size: int,
    num_tokens: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if value is None:
        return jnp.zeros((int(batch_size),), dtype=dtype)
    x = jnp.asarray(value, dtype=dtype)
    if x.ndim == 0:
        return jnp.broadcast_to(x[None], (int(batch_size),))
    if x.ndim == 1:
        return x
    # If callers pass per-write-token metadata, average it into a stable
    # per-example descriptor. The token-specific learned base already breaks
    # symmetry across WRITE query tokens.
    return jnp.mean(x[:, : int(num_tokens)], axis=1)


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


class WriteQueryTokenBuilder(nn.Module):
    cfg: DirectDecoderConfig

    @nn.compact
    def __call__(
        self,
        *,
        batch_size: int,
        demo_id: Optional[jnp.ndarray],
        chunk_start: Optional[jnp.ndarray],
        train: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del train  # deterministic token construction for now.
        B = int(batch_size)
        Q = max(1, int(self.cfg.write_num_query_tokens))
        d = int(self.cfg.d_model)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype

        base = self.param(
            'write_query_base',
            nn.initializers.normal(stddev=0.02),
            (Q, d),
            param_dtype,
        )
        tokens = jnp.broadcast_to(base.astype(dtype)[None, :, :], (B, Q, d))

        if bool(self.cfg.write_use_demo_id_embed):
            demo = _as_token_metadata(demo_id, batch_size=B, num_tokens=Q, dtype=jnp.float32)
            demo = jnp.clip(jnp.rint(demo).astype(jnp.int32), 0, max(1, int(self.cfg.write_max_demo_id)) - 1)
            demo_embed = self.param(
                'write_demo_id_embed',
                nn.initializers.normal(stddev=0.02),
                (max(1, int(self.cfg.write_max_demo_id)), d),
                param_dtype,
            )
            tokens = tokens + demo_embed[demo].astype(dtype)[:, None, :]

        if bool(self.cfg.write_use_time_embed):
            time_value = _as_token_metadata(chunk_start, batch_size=B, num_tokens=Q, dtype=jnp.float32)
            time_type = str(self.cfg.write_time_embed_type).lower()
            if time_type == 'learned':
                bucket = jnp.clip(
                    jnp.rint(time_value).astype(jnp.int32),
                    0,
                    max(1, int(self.cfg.write_max_time_bins)) - 1,
                )
                time_embed = self.param(
                    'write_time_embed',
                    nn.initializers.normal(stddev=0.02),
                    (max(1, int(self.cfg.write_max_time_bins)), d),
                    param_dtype,
                )
                t = time_embed[bucket].astype(dtype)
            elif time_type == 'continuous_sinusoidal':
                max_time = jnp.asarray(max(1, int(self.cfg.write_max_time_bins) - 1), dtype=jnp.float32)
                tau = jnp.clip(time_value, 0.0, max_time) / max_time
                t = _sinusoidal_scalar_embedding(tau, d).astype(dtype)
                hidden = max(d, int(self.cfg.write_query_mlp_mult) * d)
                t = nn.Dense(hidden, dtype=dtype, param_dtype=param_dtype, name='write_time_mlp_in')(t)
                t = nn.silu(t)
                t = nn.Dense(d, dtype=dtype, param_dtype=param_dtype, name='write_time_mlp_out')(t)
            else:
                raise ValueError(
                    "write_time_embed_type must be one of: 'continuous_sinusoidal', 'learned'. "
                    f"Got {self.cfg.write_time_embed_type!r}."
                )
            tokens = tokens + t[:, None, :]

        mask = jnp.ones((B, Q), dtype=jnp.bool_)
        return tokens, mask


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
        query_tokens: Optional[jnp.ndarray],
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        pooled_query = self._masked_mean(query_tokens, query_mask)
        pooled_support = self._masked_mean(support_tokens, support_mask)
        if pooled_query is None and pooled_support is None:
            raise ValueError('At least one of query_tokens or support_tokens must be provided.')
        if pooled_query is None:
            pooled_query = jnp.zeros_like(pooled_support)
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


class MemoryAdaLNFiLM(nn.Module):
    cfg: DirectDecoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, memory_cond: Optional[jnp.ndarray]) -> jnp.ndarray:
        mode = str(self.cfg.memory_conditioning_mode).strip().lower()
        if mode in ('', 'none', 'off', 'false'):
            return x
        if memory_cond is None:
            return x
        if mode not in ('film', 'adaln'):
            raise ValueError(
                "memory_conditioning_mode must be one of: 'none', 'film', 'adaln'. "
                f"Got {self.cfg.memory_conditioning_mode!r}."
            )
        d = int(self.cfg.d_model)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        cond = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name='memory_cond_ln')(memory_cond.astype(dtype))
        ss = nn.Dense(
            2 * d,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='memory_cond_out',
        )(cond)
        scale, shift = jnp.split(ss[:, None, :], 2, axis=-1)
        strength = jnp.asarray(float(self.cfg.memory_conditioning_strength), dtype=x.dtype)
        if mode == 'film':
            return x * (1.0 + strength * scale) + strength * shift
        h = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name='memory_adaln_x_ln')(x)
        modulated = h * (1.0 + scale) + shift
        # Residual form preserves the identity at initialization while still
        # giving memory a direct AdaLN/FiLM path into every decoder block.
        return x + strength * (modulated - h)


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
        memory_cond: Optional[jnp.ndarray],
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
            CrossAttention(
                d,
                n_heads,
                dropout,
                dtype=dtype,
                param_dtype=param_dtype,
                log_attention_weights=bool(self.cfg.log_attention_weights),
                name='cross_attn_q',
            )(
                h,
                query_tokens,
                kv_mask=query_mask,
                train=train,
            )
        )
        if support_tokens is not None:
            h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln_s')(x, cond)
            x = x + drop(
                CrossAttention(
                    d,
                    n_heads,
                    dropout,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    log_attention_weights=bool(self.cfg.log_attention_weights),
                    name='cross_attn_s',
                )(
                    h,
                    support_tokens,
                    kv_mask=support_mask,
                    train=train,
                )
            )
        x = MemoryAdaLNFiLM(self.cfg, name='memory_adaln_film')(x, memory_cond)
        h = IdentityContextAdaLN(d, d, dtype=dtype, param_dtype=param_dtype, name='adaln3')(x, cond)
        x = x + drop(MlpBlock(d, int(self.cfg.decoder_mlp_mult), dropout, dtype, param_dtype, name='mlp')(h, train=train))
        return x


class GoalPredictionHead(nn.Module):
    cfg: DirectDecoderConfig

    @nn.compact
    def __call__(
        self,
        *,
        query_tokens: Optional[jnp.ndarray],
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        x = PooledContextConditioner(self.cfg, name='goal_context_conditioner')(
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
        )
        hidden = max(int(self.cfg.d_model), int(self.cfg.goal_prediction_mlp_mult) * int(self.cfg.d_model))
        x = nn.LayerNorm(dtype=dtype, param_dtype=param_dtype, name='goal_ln')(x)
        x = nn.Dense(hidden, dtype=dtype, param_dtype=param_dtype, name='goal_mlp_in')(x)
        x = nn.silu(x)
        return nn.Dense(3, dtype=dtype, param_dtype=param_dtype, name='goal_xyz_out')(x)


class DirectDecoderCore(nn.Module):
    cfg: DirectDecoderConfig
    action_dim: int

    @nn.compact
    def __call__(
        self,
        *,
        query_tokens: Optional[jnp.ndarray],
        query_mask: Optional[jnp.ndarray],
        support_tokens: Optional[jnp.ndarray],
        support_mask: Optional[jnp.ndarray],
        train: bool,
        mode: str = 'read',
        write_demo_id: Optional[jnp.ndarray] = None,
        write_chunk_start: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        mode = str(mode).lower()
        if mode not in ('read', 'write', 'goal'):
            raise ValueError(f"mode must be one of: 'read', 'write', 'goal'. Got {mode!r}.")
        if mode in ('read', 'goal'):
            if query_tokens is None:
                raise ValueError(f'{mode.upper()} mode requires query_tokens.')
            B = int(query_tokens.shape[0])
            # Touch WRITE-only parameters when the new WRITE/READ mode is
            # enabled so model.init from a READ call still initializes the full
            # checkpoint parameter tree.
            if mode == 'read' and (bool(self.cfg.separate_write_read_heads) or bool(self.cfg.use_decoder_mode_embed)):
                _unused_write_tokens, _unused_write_mask = WriteQueryTokenBuilder(
                    self.cfg,
                    name='write_query_builder',
                )(
                    batch_size=B,
                    demo_id=None,
                    chunk_start=None,
                    train=train,
                )
                del _unused_write_tokens, _unused_write_mask
            if mode == 'goal':
                return GoalPredictionHead(self.cfg, name='goal_prediction_head')(
                    query_tokens=query_tokens,
                    query_mask=query_mask,
                    support_tokens=support_tokens,
                    support_mask=support_mask,
                )
            if bool(self.cfg.use_goal_prediction_head):
                _unused_goal = GoalPredictionHead(self.cfg, name='goal_prediction_head')(
                    query_tokens=query_tokens,
                    query_mask=query_mask,
                    support_tokens=support_tokens,
                    support_mask=support_mask,
                )
                del _unused_goal
        else:
            if support_tokens is None:
                raise ValueError('WRITE mode requires support_tokens/memory_tokens.')
            B = int(support_tokens.shape[0])
            write_tokens, write_mask = WriteQueryTokenBuilder(self.cfg, name='write_query_builder')(
                batch_size=B,
                demo_id=write_demo_id,
                chunk_start=write_chunk_start,
                train=train,
            )
            if query_tokens is None:
                query_tokens, query_mask = write_tokens, write_mask
            else:
                query_tokens = jnp.concatenate([query_tokens, write_tokens], axis=1)
                if query_mask is None:
                    query_mask = jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_)
                else:
                    query_mask = jnp.concatenate([query_mask, write_mask], axis=1)

        d = int(self.cfg.d_model)
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        cond_vec = PooledContextConditioner(self.cfg, name='context_conditioner')(
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
        )
        memory_cond = PooledContextConditioner(self.cfg, name='memory_context_conditioner')(
            query_tokens=None,
            query_mask=None,
            support_tokens=support_tokens,
            support_mask=support_mask,
        ) if str(self.cfg.memory_conditioning_mode).strip().lower() not in ('', 'none', 'off', 'false') else None
        if bool(self.cfg.use_decoder_mode_embed):
            mode_embed = self.param(
                'mode_embed',
                nn.initializers.normal(stddev=0.02),
                (2, d),
                param_dtype,
            )
            mode_idx = 1 if mode == 'write' else 0
            cond_vec = cond_vec + mode_embed[mode_idx].astype(dtype)[None, :]
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
                memory_cond=memory_cond,
                train=train,
            )
        if bool(self.cfg.shared_write_read_head) or not bool(self.cfg.separate_write_read_heads):
            return nn.Dense(int(self.action_dim), dtype=dtype, param_dtype=param_dtype, name='action_out')(x)

        read_out = nn.Dense(int(self.action_dim), dtype=dtype, param_dtype=param_dtype, name='read_action_out')(x)
        write_out = nn.Dense(int(self.action_dim), dtype=dtype, param_dtype=param_dtype, name='write_action_out')(x)
        return write_out if mode == 'write' else read_out
