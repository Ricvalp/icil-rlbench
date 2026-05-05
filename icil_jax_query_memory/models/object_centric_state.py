from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class ObjectCentricStateTokenizerConfig:
    d_model: int = 256
    max_T_obs: int = 16
    hand_pos_slice: Tuple[int, int] = (0, 3)
    gripper_slice: Tuple[int, int] = (3, 4)
    obj1_pos_slice: Tuple[int, int] = (4, 7)
    obj2_pos_slice: Tuple[int, int] = (11, 14)
    goal_pos_slice: Tuple[int, int] = (36, 39)
    has_obj2: bool = True
    goal_available: bool = True
    goal_visible: bool = True
    hidden_goal_token_policy: str = 'mask'
    mlp_mult: int = 2
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32


def _slice_to_tuple(value: object, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if ':' in text:
            a, b = text.split(':', 1)
            return int(a), int(b)
        raise ValueError(f'Expected slice string "start:end", got {value!r}.')
    seq = tuple(int(v) for v in value)  # type: ignore[arg-type]
    if len(seq) != 2:
        raise ValueError(f'Expected 2 slice values, got {seq!r}.')
    return seq[0], seq[1]


def parse_slice(value: object, default: Tuple[int, int]) -> Tuple[int, int]:
    return _slice_to_tuple(value, default)


class _TokenMLP(nn.Module):
    d_model: int
    mlp_mult: int
    dtype: jnp.dtype
    param_dtype: jnp.dtype

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hidden = max(int(self.d_model), int(self.mlp_mult) * int(self.d_model))
        x = nn.Dense(hidden, dtype=self.dtype, param_dtype=self.param_dtype, name='in')(x)
        x = nn.silu(x)
        return nn.Dense(int(self.d_model), dtype=self.dtype, param_dtype=self.param_dtype, name='out')(x)


class ObjectCentricStateTokenizer(nn.Module):
    cfg: ObjectCentricStateTokenizerConfig
    state_dim: int

    def _slice(self, state: jnp.ndarray, sl: Sequence[int], width: int) -> jnp.ndarray:
        start, end = int(sl[0]), int(sl[1])
        if start < 0 or end < start:
            raise ValueError(f'Invalid state slice {(start, end)!r}.')
        if end <= int(self.state_dim):
            x = state[..., start:end]
        elif start < int(self.state_dim):
            x = state[..., start:int(self.state_dim)]
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, max(0, end - int(self.state_dim)))])
        else:
            x = jnp.zeros(state.shape[:-1] + (max(0, end - start),), dtype=state.dtype)
        if x.shape[-1] < int(width):
            x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, int(width) - x.shape[-1])])
        return x[..., : int(width)]

    @nn.compact
    def __call__(
        self,
        *,
        query_state: jnp.ndarray,
        goal_visible: Optional[bool] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if query_state.ndim != 3:
            raise ValueError(f'query_state must be [B,T,S], got {tuple(query_state.shape)}')
        B, T, _ = query_state.shape
        if int(T) > int(self.cfg.max_T_obs):
            raise ValueError(
                f'T_obs={int(T)} exceeds max_T_obs={int(self.cfg.max_T_obs)} for ObjectCentricStateTokenizer.'
            )
        dtype = self.cfg.dtype
        param_dtype = self.cfg.param_dtype
        d = int(self.cfg.d_model)
        mlp_mult = int(self.cfg.mlp_mult)
        goal_is_visible = bool(self.cfg.goal_visible if goal_visible is None else goal_visible)
        goal_is_visible = bool(goal_is_visible and bool(self.cfg.goal_available))

        state = query_state.astype(dtype)
        hand = self._slice(state, self.cfg.hand_pos_slice, 3)
        grip = self._slice(state, self.cfg.gripper_slice, 1)
        obj1 = self._slice(state, self.cfg.obj1_pos_slice, 3)
        obj2 = self._slice(state, self.cfg.obj2_pos_slice, 3)
        goal = self._slice(state, self.cfg.goal_pos_slice, 3)

        hand_tok = _TokenMLP(d, mlp_mult, dtype, param_dtype, name='hand_mlp')(jnp.concatenate([hand, grip], axis=-1))
        obj_mlp = _TokenMLP(d, mlp_mult, dtype, param_dtype, name='obj_mlp')
        obj1_tok = obj_mlp(obj1)
        obj2_tok = obj_mlp(obj2)
        goal_tok = _TokenMLP(d, mlp_mult, dtype, param_dtype, name='goal_mlp')(goal)
        rel_mlp = _TokenMLP(d, mlp_mult, dtype, param_dtype, name='rel_mlp')
        obj1_hand_tok = rel_mlp(obj1 - hand)
        obj2_hand_tok = rel_mlp(obj2 - hand)
        goal_hand_tok = rel_mlp(goal - hand)
        goal_obj1_tok = rel_mlp(goal - obj1)
        goal_obj2_tok = rel_mlp(goal - obj2)

        per_frame = jnp.stack(
            [
                hand_tok,
                obj1_tok,
                obj2_tok,
                goal_tok,
                obj1_hand_tok,
                obj2_hand_tok,
                goal_hand_tok,
                goal_obj1_tok,
                goal_obj2_tok,
            ],
            axis=2,
        )
        num_roles = int(per_frame.shape[2])
        role_embed = self.param('role_embed', nn.initializers.normal(stddev=0.02), (num_roles, d), param_dtype)
        frame_embed = nn.Embed(
            num_embeddings=int(self.cfg.max_T_obs),
            features=d,
            dtype=dtype,
            param_dtype=param_dtype,
            name='frame_embed',
        )(jnp.arange(T, dtype=jnp.int32))
        per_frame = per_frame + role_embed.astype(dtype)[None, None, :, :] + frame_embed[None, :, None, :]

        role_mask = jnp.asarray(
            [
                True,
                True,
                bool(self.cfg.has_obj2),
                goal_is_visible,
                True,
                bool(self.cfg.has_obj2),
                goal_is_visible,
                goal_is_visible,
                bool(goal_is_visible and bool(self.cfg.has_obj2)),
            ],
            dtype=jnp.bool_,
        )
        mask = jnp.broadcast_to(role_mask[None, None, :], (B, T, num_roles))
        tokens = per_frame.reshape(B, T * num_roles, d)
        token_mask = mask.reshape(B, T * num_roles)
        if str(self.cfg.hidden_goal_token_policy).lower() not in ('mask', 'zero'):
            raise ValueError("hidden_goal_token_policy must be 'mask' or 'zero'.")
        if str(self.cfg.hidden_goal_token_policy).lower() == 'zero':
            tokens = jnp.where(token_mask[..., None], tokens, jnp.zeros_like(tokens))
            token_mask = jnp.ones_like(token_mask, dtype=jnp.bool_)
        return tokens, token_mask
