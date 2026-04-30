from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class SimpleQueryPointEncoderConfig:
    d_model: int = 512
    use_rgb: bool = True
    use_mask_id: bool = False
    mask_hash_buckets: int = 2048
    use_gripper_point_features: bool = False
    gripper_xyz_state_start: int = 0
    max_T_obs: int = 16
    add_state_token: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


class SimpleQueryPointEncoder(nn.Module):
    cfg: SimpleQueryPointEncoderConfig
    state_dim: int

    def _hash_mask_ids(self, mask_id: jnp.ndarray) -> jnp.ndarray:
        return jnp.mod(mask_id.astype(jnp.int32), int(self.cfg.mask_hash_buckets))

    def _gripper_xyz_from_state(self, state: jnp.ndarray) -> jnp.ndarray:
        start = int(self.cfg.gripper_xyz_state_start)
        end = start + 3
        if end > int(self.state_dim):
            raise ValueError(
                f'gripper_xyz_state_start={start} requires state_dim >= {end}, got {int(self.state_dim)}.'
            )
        return state[..., start:end]

    @nn.compact
    def __call__(
        self,
        *,
        query_xyz: jnp.ndarray,
        query_state: jnp.ndarray,
        query_valid: Optional[jnp.ndarray] = None,
        query_rgb: Optional[jnp.ndarray] = None,
        query_mask_id: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if query_xyz.ndim != 4:
            raise ValueError(f'query_xyz must be [B,T,N,3], got {tuple(query_xyz.shape)}')
        if query_state.ndim != 3:
            raise ValueError(f'query_state must be [B,T,S], got {tuple(query_state.shape)}')
        B, T, N, Dxyz = query_xyz.shape
        if int(Dxyz) != 3:
            raise ValueError(f'query_xyz last dim must be 3, got {Dxyz}')
        if int(T) > int(self.cfg.max_T_obs):
            raise ValueError(
                f'query T_obs={int(T)} exceeds max_T_obs={int(self.cfg.max_T_obs)} for SimpleQueryPointEncoder.'
            )

        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        xyz = query_xyz.astype(dtype)
        h = nn.Dense(int(self.cfg.d_model), use_bias=False, dtype=dtype, param_dtype=param_dtype, name='xyz_proj')(xyz)
        if bool(self.cfg.use_rgb):
            if query_rgb is None:
                raise ValueError('SimpleQueryPointEncoder requires query_rgb when use_rgb=True.')
            h = h + nn.Dense(
                int(self.cfg.d_model), use_bias=False, dtype=dtype, param_dtype=param_dtype, name='rgb_proj'
            )(query_rgb.astype(dtype))
        if bool(self.cfg.use_mask_id):
            if query_mask_id is None:
                raise ValueError('SimpleQueryPointEncoder requires query_mask_id when use_mask_id=True.')
            mask_feat = nn.Embed(
                num_embeddings=int(self.cfg.mask_hash_buckets),
                features=int(self.cfg.d_model),
                dtype=dtype,
                param_dtype=param_dtype,
                name='mask_embed',
            )(self._hash_mask_ids(query_mask_id))
            h = h + mask_feat
        if bool(self.cfg.use_gripper_point_features):
            gripper_xyz = self._gripper_xyz_from_state(query_state).astype(dtype)[:, :, None, :]
            rel = xyz - gripper_xyz
            dist = jnp.linalg.norm(rel, axis=-1, keepdims=True)
            h = h + nn.Dense(
                int(self.cfg.d_model), use_bias=False, dtype=dtype, param_dtype=param_dtype, name='gripper_proj'
            )(jnp.concatenate([rel, dist], axis=-1))

        state_tok = nn.Dense(int(self.cfg.d_model), dtype=dtype, param_dtype=param_dtype, name='state_proj_0')(
            query_state.astype(dtype)
        )
        state_tok = nn.silu(state_tok)
        state_tok = nn.Dense(int(self.cfg.d_model), dtype=dtype, param_dtype=param_dtype, name='state_proj_1')(state_tok)
        frame_ids = jnp.arange(T, dtype=jnp.int32)
        frame_embed = nn.Embed(
            num_embeddings=int(self.cfg.max_T_obs),
            features=int(self.cfg.d_model),
            dtype=dtype,
            param_dtype=param_dtype,
            name='frame_embed',
        )(frame_ids)[None, :, None, :]
        point_role = self.param('point_role_embed', nn.initializers.normal(stddev=0.02), (int(self.cfg.d_model),), param_dtype)
        state_role = self.param('state_role_embed', nn.initializers.normal(stddev=0.02), (int(self.cfg.d_model),), param_dtype)
        h = h + state_tok[:, :, None, :] + frame_embed + point_role.astype(dtype)[None, None, None, :]

        if query_valid is None:
            point_mask = jnp.ones((B, T, N), dtype=jnp.bool_)
        else:
            point_mask = query_valid.astype(jnp.bool_)

        if bool(self.cfg.add_state_token):
            state_tokens = state_tok + frame_embed[:, :, 0, :] + state_role.astype(dtype)[None, None, :]
            state_tokens = state_tokens[:, :, None, :]
            state_mask = jnp.ones((B, T, 1), dtype=jnp.bool_)
            per_frame_tokens = jnp.concatenate([state_tokens, h], axis=2)
            per_frame_mask = jnp.concatenate([state_mask, point_mask], axis=2)
            tokens = per_frame_tokens.reshape(B, T * (N + 1), int(self.cfg.d_model))
            token_mask = per_frame_mask.reshape(B, T * (N + 1))
        else:
            tokens = h.reshape(B, T * N, int(self.cfg.d_model))
            token_mask = point_mask.reshape(B, T * N)
        return tokens, token_mask
