from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np

ACTION_REPRESENTATION_ABSOLUTE: Final[str] = 'absolute'
ACTION_REPRESENTATION_DELTA_XYZ: Final[str] = 'delta_xyz'
_SUPPORTED: Final[tuple[str, ...]] = (
    ACTION_REPRESENTATION_ABSOLUTE,
    ACTION_REPRESENTATION_DELTA_XYZ,
)


def normalize_action_representation(value: str) -> str:
    out = str(value).strip().lower()
    if out not in _SUPPORTED:
        raise ValueError(f'action_representation must be one of {_SUPPORTED}, got {value!r}.')
    return out


def _validate_action_np(action: np.ndarray) -> None:
    if action.ndim < 2:
        raise ValueError(f'Expected action shape [..., H, A], got {tuple(action.shape)}.')
    if int(action.shape[-1]) < 3:
        raise ValueError(f'Expected at least 3 xyz dims, got {tuple(action.shape)}.')


def _validate_query_state_np(query_state: np.ndarray, batch_shape: tuple[int, ...]) -> None:
    if query_state.ndim < 2:
        raise ValueError(f'Expected query_state shape [..., T_obs, S], got {tuple(query_state.shape)}.')
    if int(query_state.shape[-1]) < 3:
        raise ValueError(f'Expected query_state last dim >= 3, got {tuple(query_state.shape)}.')
    if tuple(query_state.shape[:-2]) != tuple(batch_shape):
        raise ValueError(
            'query_state/action leading shape mismatch: '
            f'query_state={tuple(query_state.shape)}, action_batch_shape={tuple(batch_shape)}'
        )


def encode_action_chunk_np(action_chunk: np.ndarray, *, query_state: np.ndarray, representation: str) -> np.ndarray:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return np.asarray(action_chunk)
    action = np.asarray(action_chunk)
    q_state = np.asarray(query_state)
    _validate_action_np(action)
    _validate_query_state_np(q_state, tuple(action.shape[:-2]))
    out = np.array(action, copy=True)
    anchor_xyz = q_state[..., -1, :3]
    out[..., 0, :3] = action[..., 0, :3] - anchor_xyz
    if int(action.shape[-2]) > 1:
        out[..., 1:, :3] = action[..., 1:, :3] - action[..., :-1, :3]
    return out


def decode_action_chunk_np(action_chunk: np.ndarray, *, query_state: np.ndarray, representation: str) -> np.ndarray:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return np.asarray(action_chunk)
    action = np.asarray(action_chunk)
    q_state = np.asarray(query_state)
    _validate_action_np(action)
    _validate_query_state_np(q_state, tuple(action.shape[:-2]))
    out = np.array(action, copy=True)
    anchor_xyz = q_state[..., -1, :3][..., None, :]
    out[..., :, :3] = np.cumsum(action[..., :, :3], axis=-2) + anchor_xyz
    return out


def decode_action_chunk_jnp(action_chunk: jnp.ndarray, *, query_state: jnp.ndarray, representation: str) -> jnp.ndarray:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return action_chunk
    anchor_xyz = query_state[..., -1, :3][..., None, :]
    pos = jnp.cumsum(action_chunk[..., :, :3], axis=-2) + anchor_xyz
    return action_chunk.at[..., :, :3].set(pos)
