from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from icil.models.common import TimeLatentPerceiver, continuous_sinusoidal_embedding
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput
from icil.models.encoders.perceiver_demo_query_v2 import (
    PerceiverDemoQueryEncoderV2,
    PerceiverDemoQueryEncoderV2Config,
)


@dataclass
class TrajPerceiverV2Config(PerceiverDemoQueryEncoderV2Config):
    m_traj_tokens: int = 16
    traj_perceiver_layers: int = 2
    traj_dim: int = 8
    use_demo_id_embed: bool = True
    include_traj_tokens: bool = True
    use_cond_state_as_traj_fallback: bool = True


class TrajectoryPerceiverEncoderV2(ContextEncoder):
    def __init__(self, *, cfg: TrajPerceiverV2Config, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)

        self.demo_query_encoder = PerceiverDemoQueryEncoderV2(
            cfg=cfg,
            state_dim=int(state_dim),
            action_dim=0,
        )

        self.step_mlp = nn.Sequential(
            nn.Linear(int(cfg.traj_dim), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.demo_id_embed = (
            nn.Embedding(int(cfg.role_embed_max_K), self.d_model)
            if bool(cfg.use_demo_id_embed)
            else None
        )
        self.traj_perceiver = TimeLatentPerceiver(
            d=self.d_model,
            m=int(cfg.m_traj_tokens),
            n_heads=int(cfg.n_heads),
            n_layers=int(cfg.traj_perceiver_layers),
            dropout=float(cfg.dropout),
        )

    def _resolve_traj_source(
        self,
        *,
        cond_traj: Optional[torch.Tensor],
        cond_traj_mask: Optional[torch.Tensor],
        cond_state: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if cond_traj is not None:
            mask = cond_traj_mask
            if mask is None:
                mask = torch.ones(cond_traj.shape[:3], device=cond_traj.device, dtype=torch.bool)
            return cond_traj, mask

        if bool(self.cfg.use_cond_state_as_traj_fallback) and cond_state is not None:
            traj = cond_state
            if int(traj.shape[-1]) != int(self.cfg.traj_dim):
                raise ValueError(
                    f"cond_state last dim={int(traj.shape[-1])} cannot be used as traj_dim={int(self.cfg.traj_dim)}. "
                    "Set cfg.model.traj_perceiver_v2.traj_dim to match state_dim or provide cond_traj explicitly."
                )
            mask = torch.ones(traj.shape[:3], device=traj.device, dtype=torch.bool)
            return traj, mask

        return None, None

    def _build_traj_tokens(
        self,
        *,
        cond_traj: torch.Tensor,
        cond_traj_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T, Dt = cond_traj.shape
        if Dt != int(self.cfg.traj_dim):
            raise ValueError(f"cond_traj last dim={Dt} != cfg.traj_dim={int(self.cfg.traj_dim)}")
        if K > int(self.cfg.role_embed_max_K):
            raise ValueError(
                f"K={K} exceeds role_embed_max_K={int(self.cfg.role_embed_max_K)} for trajectory encoder."
            )

        traj_f = cond_traj.reshape(B * K, T, Dt)
        mask_f = cond_traj_mask.reshape(B * K, T).to(torch.bool)
        h = self.step_mlp(traj_f)

        if T <= 1:
            tau = torch.zeros((B * K, T), device=cond_traj.device, dtype=torch.float32)
        else:
            tau_1d = torch.linspace(0.0, 1.0, T, device=cond_traj.device, dtype=torch.float32)
            tau = tau_1d.view(1, T).expand(B * K, T)
        t_emb = continuous_sinusoidal_embedding(tau, self.d_model).to(dtype=h.dtype)
        h = h + self.time_mlp(t_emb)

        if self.demo_id_embed is not None:
            demo_ids = torch.arange(K, device=cond_traj.device).clamp_max(int(self.cfg.role_embed_max_K) - 1)
            demo_e = self.demo_id_embed(demo_ids)
            demo_e = demo_e.view(1, K, 1, self.d_model).expand(B, K, T, self.d_model)
            h = h + demo_e.reshape(B * K, T, self.d_model)

        z = self.traj_perceiver(h, token_mask=mask_f)
        return z.reshape(B, K * int(self.cfg.m_traj_tokens), self.d_model)

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
        dq_out = self.demo_query_encoder(
            query_xyz=query_xyz,
            query_state=query_state,
            cond_xyz=cond_xyz,
            cond_state=cond_state,
            cond_traj=None,
            cond_traj_mask=None,
            query_rgb=query_rgb,
            query_mask_id=query_mask_id,
            query_valid=query_valid,
            cond_rgb=cond_rgb,
            cond_mask_id=cond_mask_id,
            cond_valid=cond_valid,
        )
        ctx = dq_out.tokens
        ctx_mask = dq_out.token_mask
        support_tokens = dq_out.support_tokens
        support_token_mask = dq_out.support_token_mask
        query_tokens = dq_out.query_tokens
        query_token_mask = dq_out.query_token_mask

        if bool(self.cfg.include_traj_tokens):
            traj_src, traj_mask = self._resolve_traj_source(
                cond_traj=cond_traj,
                cond_traj_mask=cond_traj_mask,
                cond_state=cond_state,
            )
            if traj_src is not None and traj_mask is not None:
                z_traj = self._build_traj_tokens(cond_traj=traj_src, cond_traj_mask=traj_mask)
                ctx = z_traj if ctx is None else torch.cat([ctx, z_traj], dim=1)
                support_tokens = z_traj if support_tokens is None else torch.cat([support_tokens, z_traj], dim=1)
                if ctx_mask is not None:
                    traj_keep = torch.ones(z_traj.shape[:2], device=z_traj.device, dtype=torch.bool)
                    ctx_mask = torch.cat([ctx_mask.to(torch.bool), traj_keep], dim=1)
                if support_token_mask is not None:
                    traj_keep = torch.ones(z_traj.shape[:2], device=z_traj.device, dtype=torch.bool)
                    support_token_mask = torch.cat([support_token_mask.to(torch.bool), traj_keep], dim=1)

        return ContextEncoderOutput(
            tokens=ctx,
            token_mask=ctx_mask,
            support_tokens=support_tokens,
            support_token_mask=support_token_mask,
            query_tokens=query_tokens,
            query_token_mask=query_token_mask,
        )
