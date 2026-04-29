from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


@dataclass
class SimpleQueryPointEncoderConfig:
    d_model: int = 512
    use_rgb: bool = True
    use_mask_id: bool = False
    mask_hash_buckets: int = 2048
    use_gripper_point_features: bool = False
    gripper_xyz_state_start: int = 0
    max_T_obs: int = 16
    add_state_token: bool = True


class SimpleQueryPointEncoder(ContextEncoder):
    def __init__(self, *, cfg: SimpleQueryPointEncoderConfig, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)
        self.use_rgb = bool(cfg.use_rgb)
        self.use_mask_id = bool(cfg.use_mask_id)
        self.use_gripper_point_features = bool(cfg.use_gripper_point_features)
        self.gripper_xyz_state_start = int(cfg.gripper_xyz_state_start)
        self.add_state_token = bool(cfg.add_state_token)

        self.xyz_proj = nn.Linear(3, self.d_model, bias=False)
        self.rgb_proj = nn.Linear(3, self.d_model, bias=False) if self.use_rgb else None
        self.mask_embed = (
            nn.Embedding(int(cfg.mask_hash_buckets), self.d_model) if self.use_mask_id else None
        )
        self.gripper_proj = (
            nn.Linear(4, self.d_model, bias=False) if self.use_gripper_point_features else None
        )
        self.state_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.frame_embed = nn.Embedding(int(cfg.max_T_obs), self.d_model)
        self.point_role_embed = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.state_role_embed = nn.Parameter(torch.randn(self.d_model) * 0.02)

    def _hash_mask_ids(self, mask_id: torch.Tensor) -> torch.Tensor:
        if self.mask_embed is None:
            raise RuntimeError('mask_embed is not initialized.')
        return torch.remainder(mask_id, self.mask_embed.num_embeddings)

    def _gripper_xyz_from_state(self, state: torch.Tensor) -> torch.Tensor:
        start = int(self.gripper_xyz_state_start)
        end = start + 3
        if end > int(state.shape[-1]):
            raise ValueError(
                f'gripper_xyz_state_start={start} requires state_dim >= {end}, got {int(state.shape[-1])}.'
            )
        return state[..., start:end]

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
        del cond_xyz, cond_state, cond_traj, cond_traj_mask, cond_rgb, cond_mask_id, cond_valid
        if query_xyz.dim() != 4:
            raise ValueError(f'query_xyz must be [B,T,N,3], got {tuple(query_xyz.shape)}')
        if query_state.dim() != 3:
            raise ValueError(f'query_state must be [B,T,S], got {tuple(query_state.shape)}')
        B, T, N, Dxyz = query_xyz.shape
        if int(Dxyz) != 3:
            raise ValueError(f'query_xyz last dim must be 3, got {Dxyz}')
        if T > int(self.cfg.max_T_obs):
            raise ValueError(
                f'query T_obs={T} exceeds max_T_obs={int(self.cfg.max_T_obs)} for SimpleQueryPointEncoder.'
            )

        h = self.xyz_proj(query_xyz)
        if self.use_rgb:
            if query_rgb is None:
                raise ValueError('SimpleQueryPointEncoder requires query_rgb when use_rgb=True.')
            h = h + self.rgb_proj(query_rgb.to(dtype=query_xyz.dtype))
        if self.use_mask_id:
            if query_mask_id is None:
                raise ValueError('SimpleQueryPointEncoder requires query_mask_id when use_mask_id=True.')
            h = h + self.mask_embed(self._hash_mask_ids(query_mask_id))
        if self.use_gripper_point_features:
            gripper_xyz = self._gripper_xyz_from_state(query_state).unsqueeze(2)
            rel = query_xyz - gripper_xyz
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            h = h + self.gripper_proj(torch.cat([rel, dist], dim=-1))

        state_tok = self.state_proj(query_state)
        frame_ids = torch.arange(T, device=query_xyz.device)
        frame_emb = self.frame_embed(frame_ids).to(dtype=h.dtype).view(1, T, 1, self.d_model)
        h = h + state_tok.unsqueeze(2) + frame_emb + self.point_role_embed.view(1, 1, 1, self.d_model)

        point_mask = (
            query_valid.to(torch.bool)
            if query_valid is not None
            else torch.ones((B, T, N), device=query_xyz.device, dtype=torch.bool)
        )
        tokens = h.reshape(B, T * N, self.d_model)
        token_mask = point_mask.reshape(B, T * N)

        if self.add_state_token:
            state_tokens = state_tok + self.frame_embed(frame_ids).to(dtype=state_tok.dtype).view(1, T, self.d_model)
            state_tokens = state_tokens + self.state_role_embed.view(1, 1, self.d_model)
            state_mask = torch.ones((B, T), device=query_xyz.device, dtype=torch.bool)
            per_frame_tokens = torch.cat([state_tokens.unsqueeze(2), h], dim=2)
            per_frame_mask = torch.cat([state_mask.unsqueeze(-1), point_mask], dim=2)
            tokens = per_frame_tokens.reshape(B, T * (N + 1), self.d_model)
            token_mask = per_frame_mask.reshape(B, T * (N + 1))

        return ContextEncoderOutput(
            tokens=tokens,
            token_mask=token_mask,
            query_tokens=tokens,
            query_token_mask=token_mask,
        )
