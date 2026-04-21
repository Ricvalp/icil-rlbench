from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

VALID_INNER_LR_MODES = (
    'fixed',
    'shared_learned',
    'per_step_learned',
)


def normalize_inner_lr_mode(mode: Any) -> str:
    if mode is None:
        return 'fixed'
    value = str(mode).strip().lower()
    if value in ('', 'none'):
        return 'fixed'
    aliases = {
        'fixed': 'fixed',
        'shared': 'shared_learned',
        'shared_learned': 'shared_learned',
        'per_step': 'per_step_learned',
        'per_step_learned': 'per_step_learned',
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(
        f'Unsupported inner_lr_mode={mode!r}. '
        f'Expected one of {VALID_INNER_LR_MODES}.'
    )


def infer_inner_lr_mode(
    *,
    checkpoint: Optional[Dict[str, Any]] = None,
    checkpoint_config: Optional[Dict[str, Any]] = None,
    local_mode: Any = None,
    legacy_learn_inner_lrs: Any = None,
) -> str:
    config_dict = checkpoint_config
    if config_dict is None and isinstance(checkpoint, dict):
        config_obj = checkpoint.get('config', None)
        config_dict = config_obj if isinstance(config_obj, dict) else None

    if isinstance(config_dict, dict):
        maml_cfg = config_dict.get('maml', {}) or {}
        if isinstance(maml_cfg, dict):
            if 'inner_lr_mode' in maml_cfg:
                return normalize_inner_lr_mode(maml_cfg['inner_lr_mode'])
            if bool(maml_cfg.get('learn_inner_lrs', False)):
                return 'per_step_learned'

    if isinstance(checkpoint, dict):
        schedule_state = checkpoint.get('inner_lr_schedule', None)
        if isinstance(schedule_state, dict):
            raw_lrs = schedule_state.get('raw_lrs', None)
            if torch.is_tensor(raw_lrs):
                return 'shared_learned' if int(raw_lrs.numel()) == 1 else 'per_step_learned'

    if legacy_learn_inner_lrs is not None and bool(legacy_learn_inner_lrs):
        return 'per_step_learned'
    if local_mode is not None:
        return normalize_inner_lr_mode(local_mode)
    return 'fixed'


class PositiveInnerLRSchedule(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        inner_steps: int,
        init_lr: float,
        min_lr: float = 1e-8,
    ) -> None:
        super().__init__()
        resolved_mode = normalize_inner_lr_mode(mode)
        if resolved_mode == 'fixed':
            raise ValueError('PositiveInnerLRSchedule does not support mode="fixed".')
        if int(inner_steps) < 1:
            raise ValueError(f'inner_steps must be >= 1, got {inner_steps}.')
        if float(init_lr) <= 0.0:
            raise ValueError(f'init_lr must be > 0, got {init_lr}.')
        if float(min_lr) < 0.0:
            raise ValueError(f'min_lr must be >= 0, got {min_lr}.')
        self.mode = resolved_mode
        self.min_lr = float(min_lr)
        num_values = 1 if resolved_mode == 'shared_learned' else int(inner_steps)
        init = torch.full((num_values,), float(init_lr) - self.min_lr, dtype=torch.float32)
        init = init.clamp_min(1e-8)
        self.raw_lrs = nn.Parameter(torch.log(torch.expm1(init)))

    def values(self) -> torch.Tensor:
        return F.softplus(self.raw_lrs) + float(self.min_lr)

    def lr_at(self, step_idx: int) -> torch.Tensor:
        values = self.values()
        if self.mode == 'shared_learned':
            return values[0]
        if not 0 <= int(step_idx) < int(values.shape[0]):
            raise IndexError(
                f'step_idx={step_idx} out of range for {int(values.shape[0])} '
                f'inner LR values in mode={self.mode!r}.'
            )
        return values[int(step_idx)]


def build_inner_lr_schedule(
    *,
    mode: str,
    inner_steps: int,
    init_lr: float,
) -> Optional[PositiveInnerLRSchedule]:
    resolved_mode = normalize_inner_lr_mode(mode)
    if resolved_mode == 'fixed' or int(inner_steps) < 1:
        return None
    return PositiveInnerLRSchedule(
        mode=resolved_mode,
        inner_steps=int(inner_steps),
        init_lr=float(init_lr),
    )


def resolved_inner_lr_values(
    *,
    mode: str,
    inner_steps: int,
    fixed_inner_lr: float,
    schedule: Optional[PositiveInnerLRSchedule],
) -> List[float]:
    resolved_mode = normalize_inner_lr_mode(mode)
    if int(inner_steps) <= 0:
        return []
    if resolved_mode == 'fixed' or schedule is None:
        return [float(fixed_inner_lr) for _ in range(int(inner_steps))]
    learned = [float(v) for v in schedule.values().detach().cpu().tolist()]
    if resolved_mode == 'shared_learned':
        return [float(learned[0]) for _ in range(int(inner_steps))]
    if len(learned) < int(inner_steps):
        raise ValueError(
            f'Per-step inner LR schedule has {len(learned)} values but inner_steps={inner_steps}.'
        )
    return learned[: int(inner_steps)]


def inner_lr_log_dict(
    *,
    mode: str,
    inner_steps: int,
    fixed_inner_lr: float,
    schedule: Optional[PositiveInnerLRSchedule],
    prefix: str = 'train',
) -> Dict[str, float]:
    values = resolved_inner_lr_values(
        mode=mode,
        inner_steps=int(inner_steps),
        fixed_inner_lr=float(fixed_inner_lr),
        schedule=schedule,
    )
    if not values:
        return {}
    return {
        f'{prefix}/inner_lr_min': float(min(values)),
        f'{prefix}/inner_lr_max': float(max(values)),
        f'{prefix}/inner_lr_mean': float(sum(values) / len(values)),
    }


def inner_lr_tensor_for_step(
    *,
    step_idx: int,
    mode: str,
    fixed_inner_lr: float,
    schedule: Optional[PositiveInnerLRSchedule],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    resolved_mode = normalize_inner_lr_mode(mode)
    if resolved_mode == 'fixed' or schedule is None:
        return torch.tensor(float(fixed_inner_lr), device=device, dtype=dtype)
    return schedule.lr_at(int(step_idx)).to(device=device, dtype=dtype)
