from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    d_model: int

    def forward(
        self,
        *,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        cond_xyz: Optional[torch.Tensor] = None,
        cond_state: Optional[torch.Tensor] = None,
        cond_traj: Optional[torch.Tensor] = None,
        cond_traj_mask: Optional[torch.Tensor] = None,
        query_rgb: Optional[torch.Tensor] = None,
        query_mask_id: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        cond_rgb: Optional[torch.Tensor] = None,
        cond_mask_id: Optional[torch.Tensor] = None,
        cond_valid: Optional[torch.Tensor] = None,
    ) -> ContextEncoderOutput:
        raise NotImplementedError


@dataclass
class ContextEncoderOutput:
    tokens: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    support_tokens: Optional[torch.Tensor] = None
    support_token_mask: Optional[torch.Tensor] = None
    query_tokens: Optional[torch.Tensor] = None
    query_token_mask: Optional[torch.Tensor] = None
