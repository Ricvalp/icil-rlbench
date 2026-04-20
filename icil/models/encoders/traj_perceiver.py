from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from icil.models.common import TimeLatentPerceiver, continuous_sinusoidal_embedding
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput
from icil.models.encoders.perceiver_demo_query import (
    PerceiverDemoQueryEncoder,
    PerceiverDemoQueryEncoderConfig,
)


@dataclass
class TrajPerceiverConfig:
    # Shared token space.
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.0

    # Demo/query Perceiver branch (same knobs as PerceiverDemoQueryEncoder).
    m_frame_tokens: int = 64
    frame_tokenizer_layers: int = 2
    M_demo_latents: int = 256
    demo_perceiver_layers: int = 3
    mask_hash_buckets: int = 2048
    use_mask_id: bool = True
    role_embed_max_K: int = 32
    role_embed_max_L: int = 64
    role_embed_max_Tobs: int = 16
    rgb_alpha_init: float = 1.0
    attention_backend: str = "manual"
    ignore_demos: bool = False
    compress_demo_latents: bool = True
    checkpoint_demo_memory: bool = False
    checkpoint_build_demo_memory: bool = False
    checkpoint_frame_tokenizer: bool = False
    tokenize_frames_chunked: bool = False
    chunk_frames: int = 32

    # Trajectory branch.
    m_traj_tokens: int = 16
    traj_perceiver_layers: int = 2
    traj_dim: int = 8  # [x,y,z,qx,qy,qz,qw,grip]
    use_demo_id_embed: bool = True
    include_traj_tokens: bool = True
    use_cond_state_as_traj_fallback: bool = True


class TrajectoryPerceiverEncoder(ContextEncoder):
    """
    Full context encoder: demo/query point-cloud encoder + support-trajectory encoder.

    Context output is:
      concat( demo_query_tokens, trajectory_tokens )
    where trajectory tokens are optional depending on config and available inputs.

    Expected trajectory inputs:
      - cond_traj: [B, K, T, traj_dim] (preferred)
      - cond_traj_mask: [B, K, T] bool (optional)

    If cond_traj is missing and use_cond_state_as_traj_fallback=True, cond_state is used as
    trajectory source with shape [B, K, L, state_dim], requiring state_dim == traj_dim.
    """

    def __init__(self, *, cfg: TrajPerceiverConfig, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)

        # Reuse the existing demo/query encoder implementation.
        self.demo_query_encoder = PerceiverDemoQueryEncoder(
            cfg=PerceiverDemoQueryEncoderConfig(
                d_model=int(cfg.d_model),
                n_heads=int(cfg.n_heads),
                m_frame_tokens=int(cfg.m_frame_tokens),
                frame_tokenizer_layers=int(cfg.frame_tokenizer_layers),
                M_demo_latents=int(cfg.M_demo_latents),
                demo_perceiver_layers=int(cfg.demo_perceiver_layers),
                mask_hash_buckets=int(cfg.mask_hash_buckets),
                use_mask_id=bool(cfg.use_mask_id),
                role_embed_max_K=int(cfg.role_embed_max_K),
                role_embed_max_L=int(cfg.role_embed_max_L),
                role_embed_max_Tobs=int(cfg.role_embed_max_Tobs),
                rgb_alpha_init=float(cfg.rgb_alpha_init),
                dropout=float(cfg.dropout),
                attention_backend=str(cfg.attention_backend),
                ignore_demos=bool(cfg.ignore_demos),
                compress_demo_latents=bool(cfg.compress_demo_latents),
                checkpoint_demo_memory=bool(cfg.checkpoint_demo_memory),
                checkpoint_build_demo_memory=bool(cfg.checkpoint_build_demo_memory),
                checkpoint_frame_tokenizer=bool(cfg.checkpoint_frame_tokenizer),
                tokenize_frames_chunked=bool(cfg.tokenize_frames_chunked),
                chunk_frames=int(cfg.chunk_frames),
            ),
            state_dim=int(state_dim),
            action_dim=0,
        )

        # Trajectory branch.
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
            attention_backend=str(cfg.attention_backend),
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
            # cond_state is [B, K, L, S], use L as trajectory-length proxy.
            
            print("Warning: using cond_state as trajectory!")
            
            traj = cond_state
            if int(traj.shape[-1]) != int(self.cfg.traj_dim):
                raise ValueError(
                    f"cond_state last dim={int(traj.shape[-1])} cannot be used as traj_dim={int(self.cfg.traj_dim)}. "
                    "Set cfg.model.traj_perceiver.traj_dim to match state_dim or provide cond_traj explicitly."
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
        # cond_traj: [B,K,T,Dt]
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
            demo_e = self.demo_id_embed(demo_ids)  # [K,d]
            demo_e = demo_e.view(1, K, 1, self.d_model).expand(B, K, T, self.d_model)
            demo_e = demo_e.reshape(B * K, T, self.d_model)
            h = h + demo_e

        z = self.traj_perceiver(h, token_mask=mask_f)  # [BK,m,d]
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
        # Demo/query context (same as legacy perceiver encoder).
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

        # Optional trajectory tokens.
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


class TrajectoryOnlyPerceiverEncoder(ContextEncoder):
    """
    Reference implementation kept from the old file:
    trajectory-only context encoder (no point-cloud demo/query branch).
    """

    def __init__(self, *, cfg: TrajPerceiverConfig, state_dim: int, action_dim: int):
        super().__init__()
        del state_dim, action_dim
        self.cfg = cfg
        self.d_model = int(cfg.d_model)

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
            attention_backend=str(cfg.attention_backend),
        )

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
        del query_xyz, query_state, cond_xyz, cond_state, query_rgb, query_mask_id, query_valid, cond_rgb, cond_mask_id, cond_valid
        if cond_traj is None:
            raise ValueError("TrajectoryOnlyPerceiverEncoder requires cond_traj")

        if cond_traj_mask is None:
            cond_traj_mask = torch.ones(cond_traj.shape[:3], device=cond_traj.device, dtype=torch.bool)

        B, K, T, Dt = cond_traj.shape
        if Dt != int(self.cfg.traj_dim):
            raise ValueError(f"cond_traj last dim={Dt} != cfg.traj_dim={int(self.cfg.traj_dim)}")

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
            demo_e = demo_e.reshape(B * K, T, self.d_model)
            h = h + demo_e

        z = self.traj_perceiver(h, token_mask=mask_f)
        z = z.reshape(B, K * int(self.cfg.m_traj_tokens), self.d_model)
        return ContextEncoderOutput(
            tokens=z,
            token_mask=None,
            support_tokens=z,
            support_token_mask=None,
            query_tokens=None,
            query_token_mask=None,
        )
