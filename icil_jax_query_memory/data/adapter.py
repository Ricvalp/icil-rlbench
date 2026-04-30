from __future__ import annotations

from typing import Any, Dict, List, Sequence

import jax
import numpy as np
import torch


_MANDATORY_QUERY_KEYS = ('query_xyz', 'query_state', 'query_valid', 'target_action')
_OPTIONAL_QUERY_KEYS = ('query_rgb', 'query_mask_id')


def _to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_numpy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy(v) for v in value)
    return value


def _stack_batches(batches: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not batches:
        raise ValueError('batches must be non-empty.')
    out: Dict[str, np.ndarray] = {}
    for key in _MANDATORY_QUERY_KEYS:
        out[key] = np.stack([_to_numpy(batch[key]) for batch in batches], axis=0)
    for key in _OPTIONAL_QUERY_KEYS:
        if all(key in batch for batch in batches):
            out[key] = np.stack([_to_numpy(batch[key]) for batch in batches], axis=0)
    return out


def prepared_tasks_to_host_batch(
    prepared_tasks: Sequence[Dict[str, Any]],
    *,
    inner_steps: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    if not prepared_tasks:
        raise ValueError('prepared_tasks must be non-empty.')
    task_count = len(prepared_tasks)
    if int(inner_steps) < 0:
        raise ValueError(f'inner_steps must be >= 0, got {inner_steps}.')

    expanded_inner: List[List[Dict[str, Any]]] = []
    for task in prepared_tasks:
        base_inner = list(task.get('inner_batches', []))
        if int(inner_steps) > 0 and not base_inner:
            raise ValueError('prepared task is missing inner_batches while inner_steps > 0.')
        steps = [base_inner[idx % len(base_inner)] for idx in range(int(inner_steps))] if inner_steps > 0 else []
        expanded_inner.append(steps)

    inner: Dict[str, np.ndarray] = {}
    if inner_steps > 0:
        for key in _MANDATORY_QUERY_KEYS:
            inner[key] = np.stack(
                [
                    np.stack([_to_numpy(step_batch[key]) for step_batch in task_steps], axis=0)
                    for task_steps in expanded_inner
                ],
                axis=0,
            )
        for key in _OPTIONAL_QUERY_KEYS:
            if all(all(key in step_batch for step_batch in task_steps) for task_steps in expanded_inner):
                inner[key] = np.stack(
                    [
                        np.stack([_to_numpy(step_batch[key]) for step_batch in task_steps], axis=0)
                        for task_steps in expanded_inner
                    ],
                    axis=0,
                )

    query = _stack_batches([task['query_batch'] for task in prepared_tasks])
    meta = {
        'support_ids': np.asarray([task['support_ids'] for task in prepared_tasks], dtype=np.int32),
        'query_episode_id': np.asarray([task['query_episode_id'] for task in prepared_tasks], dtype=np.int32),
        'vidx': np.asarray([task['task'].vidx for task in prepared_tasks], dtype=np.int32),
    }
    return {
        'inner': inner,
        'query': query,
        'meta': meta,
        'task_count': np.asarray(task_count, dtype=np.int32),
    }


def _reshape_for_devices(tree: Any, num_devices: int, per_device_batch: int) -> Any:
    def _reshape_leaf(x: Any) -> Any:
        if not hasattr(x, 'shape'):
            return x
        if len(x.shape) == 0:
            return x
        if x.shape[0] != num_devices * per_device_batch:
            return x
        return x.reshape((num_devices, per_device_batch) + x.shape[1:])

    return jax.tree_util.tree_map(_reshape_leaf, tree)


def prepared_tasks_to_sharded_batch(
    prepared_tasks: Sequence[Dict[str, Any]],
    *,
    inner_steps: int,
    num_devices: int,
    per_device_batch: int,
    devices: Sequence[jax.Device],
) -> Any:
    host_batch = prepared_tasks_to_host_batch(prepared_tasks, inner_steps=inner_steps)
    host_batch = _reshape_for_devices(host_batch, int(num_devices), int(per_device_batch))

    def _maybe_shard(x: Any) -> Any:
        if not hasattr(x, 'shape'):
            return x
        if len(x.shape) < 2:
            return x
        if x.shape[:2] != (int(num_devices), int(per_device_batch)):
            return x
        return jax.device_put_sharded([x[i] for i in range(int(num_devices))], devices)

    return jax.tree_util.tree_map(
        _maybe_shard,
        host_batch,
    )
