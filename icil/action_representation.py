from __future__ import annotations

from typing import Final

import torch


ACTION_REPRESENTATION_ABSOLUTE: Final[str] = 'absolute'
ACTION_REPRESENTATION_DELTA_XYZ: Final[str] = 'delta_xyz'
_SUPPORTED_ACTION_REPRESENTATIONS: Final[tuple[str, ...]] = (
    ACTION_REPRESENTATION_ABSOLUTE,
    ACTION_REPRESENTATION_DELTA_XYZ,
)


def normalize_action_representation(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _SUPPORTED_ACTION_REPRESENTATIONS:
        raise ValueError(
            f'action_representation must be one of {_SUPPORTED_ACTION_REPRESENTATIONS}, got {value!r}.'
        )
    return normalized


def _validate_action_tensor(action: torch.Tensor) -> None:
    if action.ndim < 2:
        raise ValueError(f'Expected action tensor with shape [..., H, A], got {tuple(action.shape)}.')
    if int(action.shape[-1]) < 3:
        raise ValueError(
            f'Expected action tensor with at least 3 position dims in the last axis, got {tuple(action.shape)}.'
        )


def _validate_query_state_tensor(query_state: torch.Tensor, *, batch_shape: torch.Size) -> None:
    if query_state.ndim < 2:
        raise ValueError(
            f'Expected query_state tensor with shape [..., T_obs, S], got {tuple(query_state.shape)}.'
        )
    if int(query_state.shape[-1]) < 3:
        raise ValueError(
            f'Expected query_state tensor with at least 3 position dims in the last axis, got {tuple(query_state.shape)}.'
        )
    if query_state.shape[:-2] != batch_shape:
        raise ValueError(
            'query_state/action leading shape mismatch: '
            f'query_state={tuple(query_state.shape)}, action_batch_shape={tuple(batch_shape)}'
        )


def encode_action_chunk(
    action_chunk: torch.Tensor,
    *,
    query_state: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return action_chunk

    _validate_action_tensor(action_chunk)
    _validate_query_state_tensor(query_state, batch_shape=action_chunk.shape[:-2])

    out = action_chunk.clone()
    anchor_xyz = query_state[..., -1, :3]
    out[..., 0, :3] = action_chunk[..., 0, :3] - anchor_xyz
    if int(action_chunk.shape[-2]) > 1:
        out[..., 1:, :3] = action_chunk[..., 1:, :3] - action_chunk[..., :-1, :3]
    return out


def encode_support_traj(
    traj: torch.Tensor,
    *,
    representation: str,
) -> torch.Tensor:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return traj

    _validate_action_tensor(traj)
    out = traj.clone()
    out[..., 0, :3] = 0.0
    if int(traj.shape[-2]) > 1:
        out[..., 1:, :3] = traj[..., 1:, :3] - traj[..., :-1, :3]
    return out


def decode_action_chunk(
    action_chunk: torch.Tensor,
    *,
    query_state: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return action_chunk

    _validate_action_tensor(action_chunk)
    _validate_query_state_tensor(query_state, batch_shape=action_chunk.shape[:-2])

    out = action_chunk.clone()
    anchor_xyz = query_state[..., -1, :3].unsqueeze(-2)
    out[..., :, :3] = torch.cumsum(action_chunk[..., :, :3], dim=-2) + anchor_xyz
    return out


def decode_action_trace(
    action_trace: torch.Tensor,
    *,
    query_state: torch.Tensor,
    representation: str,
) -> torch.Tensor:
    representation = normalize_action_representation(representation)
    if representation == ACTION_REPRESENTATION_ABSOLUTE:
        return action_trace
    if action_trace.ndim < 4:
        raise ValueError(
            f'Expected action trace tensor with shape [S, ..., H, A], got {tuple(action_trace.shape)}.'
        )
    if query_state.ndim < 2:
        raise ValueError(
            f'Expected query_state tensor with shape [..., T_obs, S], got {tuple(query_state.shape)}.'
        )
    if action_trace.shape[1:-2] != query_state.shape[:-2]:
        raise ValueError(
            'query_state/action_trace leading shape mismatch: '
            f'query_state={tuple(query_state.shape)}, action_trace={tuple(action_trace.shape)}'
        )

    num_steps = int(action_trace.shape[0])
    flat_size = 1
    for dim in action_trace.shape[1:-2]:
        flat_size *= int(dim)
    flat_trace = action_trace.reshape(num_steps * flat_size, *action_trace.shape[-2:])
    anchor_state = query_state.unsqueeze(0).expand(num_steps, *query_state.shape)
    flat_query_state = anchor_state.reshape(num_steps * flat_size, *query_state.shape[-2:])
    decoded = decode_action_chunk(
        flat_trace,
        query_state=flat_query_state,
        representation=representation,
    )
    return decoded.reshape(action_trace.shape)
