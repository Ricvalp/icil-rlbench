from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from icil.models.encoders.base import ContextEncoder
from icil.models.policies.direct_regression_policy import (
    DirectRegressionPolicy,
    DirectRegressionPolicyConfig,
)


@dataclass
class QueryMemoryDirectRegressionPolicyConfig(DirectRegressionPolicyConfig):
    memory_num_tokens: int = 128


class QueryMemoryDirectRegressionPolicy(DirectRegressionPolicy):
    def __init__(
        self,
        *,
        cfg: QueryMemoryDirectRegressionPolicyConfig,
        context_encoder: ContextEncoder,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__(
            cfg=cfg,
            context_encoder=context_encoder,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.cfg = cfg
        self.memory_num_tokens = int(cfg.memory_num_tokens)
        if self.memory_num_tokens < 1:
            raise ValueError('QueryMemoryDirectRegressionPolicyConfig.memory_num_tokens must be >= 1.')
        d_model = int(cfg.d_model)
        self.memory_token_init = torch.nn.Parameter(torch.randn(self.memory_num_tokens, d_model) * 0.02)

    def init_learned_memory_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        clone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.memory_token_init.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1, -1)
        if clone:
            tokens = tokens.clone()
        mask = torch.ones((int(batch_size), self.memory_num_tokens), device=device, dtype=torch.bool)
        return tokens, mask

    def forward_actions(
        self,
        *,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        cond_xyz: Optional[torch.Tensor] = None,
        cond_state: Optional[torch.Tensor] = None,
        cond_traj: Optional[torch.Tensor] = None,
        cond_traj_mask: Optional[torch.Tensor] = None,
        cond_rgb: Optional[torch.Tensor] = None,
        query_rgb: Optional[torch.Tensor] = None,
        cond_mask_id: Optional[torch.Tensor] = None,
        query_mask_id: Optional[torch.Tensor] = None,
        cond_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        del cond_xyz, cond_state, cond_traj, cond_traj_mask, cond_rgb, cond_mask_id, cond_valid
        if action_horizon is not None:
            self._check_action_horizon(int(action_horizon))
        batch_size = int(query_xyz.shape[0])
        ctx = self._encode_context(
            query_xyz=query_xyz,
            query_state=query_state,
            query_rgb=query_rgb,
            query_mask_id=query_mask_id,
            query_valid=query_valid,
        )
        support_tokens, support_mask = self.init_learned_memory_tokens(
            batch_size=batch_size,
            device=query_xyz.device,
            dtype=query_xyz.dtype,
            clone=False,
        )
        ctx.support_tokens = support_tokens
        ctx.support_token_mask = support_mask
        if self.context_attention_mode == 'single':
            ctx.tokens, ctx.token_mask = self._concat_token_groups(
                ctx.support_tokens,
                ctx.support_token_mask,
                ctx.query_tokens,
                ctx.query_token_mask,
            )
        elif ctx.tokens is None:
            ctx.tokens = ctx.query_tokens
            ctx.token_mask = ctx.query_token_mask
        return self._predict_actions_from_context(ctx, batch_size=batch_size)
