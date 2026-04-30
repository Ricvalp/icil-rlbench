from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x if isinstance(x, (int, float, bool, str, type(None))) else jax.device_get(x), tree)


def _tree_to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if hasattr(x, 'shape') else x, tree)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    params: Any,
    opt_state: Any,
    config: Dict[str, Any],
    extra_state: Dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': int(step),
        'params': _tree_to_numpy(params),
        'opt_state': _tree_to_numpy(opt_state),
        'config': config,
    }
    if extra_state:
        payload.update(_tree_to_numpy(extra_state))
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with path.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f'Unsupported checkpoint payload type: {type(payload).__name__}')
    if 'params' in payload:
        payload['params'] = _tree_to_jax(payload['params'])
    if 'opt_state' in payload and payload['opt_state'] is not None:
        payload['opt_state'] = _tree_to_jax(payload['opt_state'])
    return payload
