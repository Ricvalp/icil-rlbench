from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


def _create_mlp(input_dim: int, output_dim: int, net_arch: Sequence[int]) -> nn.Sequential:
    dims = [int(input_dim), *[int(v) for v in net_arch], int(output_dim)]
    layers = []
    for idx in range(len(dims) - 2):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class _PointNetPoolEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        use_layernorm: bool,
        final_norm: str,
    ):
        super().__init__()
        widths = (64, 128, 256, 512)
        layers = []
        prev = int(in_channels)
        for width in widths:
            layers.append(nn.Linear(prev, int(width)))
            if bool(use_layernorm):
                layers.append(nn.LayerNorm(int(width)))
            layers.append(nn.ReLU())
            prev = int(width)
        self.mlp = nn.Sequential(*layers)
        if str(final_norm) == 'layernorm':
            self.out = nn.Sequential(
                nn.Linear(prev, int(out_channels)),
                nn.LayerNorm(int(out_channels)),
            )
        elif str(final_norm) == 'none':
            self.out = nn.Linear(prev, int(out_channels))
        else:
            raise ValueError(f'Unsupported final_norm={final_norm!r}.')

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.mlp(x)
        if valid_mask is None:
            pooled = h.max(dim=1).values
            return self.out(pooled)

        mask = valid_mask.to(torch.bool).unsqueeze(-1)
        any_valid = mask.any(dim=1)
        h = h.masked_fill(~mask, torch.finfo(h.dtype).min)
        pooled = h.max(dim=1).values
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return self.out(pooled)


@dataclass
class DP3QueryFrameEncoderConfig:
    d_model: int = 512
    pointcloud_out_channels: int = 256
    pointcloud_use_layernorm: bool = True
    pointcloud_final_norm: str = 'layernorm'
    use_rgb: bool = True
    use_mask_id: bool = False
    mask_hash_buckets: int = 2048
    mask_embed_dim: int = 8
    use_gripper_point_features: bool = False
    gripper_xyz_state_start: int = 0
    state_mlp_hidden_dims: Tuple[int, ...] = (64,)
    state_feat_dim: int = 64
    max_T_obs: int = 16


class DP3QueryFrameEncoder(ContextEncoder):
    def __init__(self, *, cfg: DP3QueryFrameEncoderConfig, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)
        self.use_rgb = bool(cfg.use_rgb)
        self.use_mask_id = bool(cfg.use_mask_id)
        self.use_gripper_point_features = bool(cfg.use_gripper_point_features)
        self.gripper_xyz_state_start = int(cfg.gripper_xyz_state_start)

        point_in_dim = 3
        if self.use_rgb:
            point_in_dim += 3
        if self.use_mask_id:
            point_in_dim += int(cfg.mask_embed_dim)
            self.mask_embed = nn.Embedding(int(cfg.mask_hash_buckets), int(cfg.mask_embed_dim))
        else:
            self.mask_embed = None
        if self.use_gripper_point_features:
            point_in_dim += 4

        self.point_encoder = _PointNetPoolEncoder(
            in_channels=int(point_in_dim),
            out_channels=int(cfg.pointcloud_out_channels),
            use_layernorm=bool(cfg.pointcloud_use_layernorm),
            final_norm=str(cfg.pointcloud_final_norm),
        )
        self.state_mlp = _create_mlp(
            self.state_dim,
            int(cfg.state_feat_dim),
            tuple(int(v) for v in cfg.state_mlp_hidden_dims),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(int(cfg.pointcloud_out_channels) + int(cfg.state_feat_dim), self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.frame_embed = nn.Embedding(int(cfg.max_T_obs), self.d_model)
        self.role_embed = nn.Parameter(torch.randn(self.d_model) * 0.02)

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
                f'query T_obs={T} exceeds max_T_obs={int(self.cfg.max_T_obs)} for DP3QueryFrameEncoder.'
            )

        feats = [query_xyz]
        if self.use_rgb:
            if query_rgb is None:
                raise ValueError('DP3QueryFrameEncoder requires query_rgb when use_rgb=True.')
            feats.append(query_rgb.to(dtype=query_xyz.dtype))
        if self.use_mask_id:
            if query_mask_id is None:
                raise ValueError('DP3QueryFrameEncoder requires query_mask_id when use_mask_id=True.')
            mask_feat = self.mask_embed(self._hash_mask_ids(query_mask_id))
            feats.append(mask_feat.to(dtype=query_xyz.dtype))
        if self.use_gripper_point_features:
            gripper_xyz = self._gripper_xyz_from_state(query_state).unsqueeze(2)
            rel = query_xyz - gripper_xyz
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            feats.append(torch.cat([rel, dist], dim=-1))

        point_feat = torch.cat(feats, dim=-1)
        point_mask = (
            query_valid.to(torch.bool)
            if query_valid is not None
            else torch.ones((B, T, N), device=query_xyz.device, dtype=torch.bool)
        )
        point_feat = point_feat.reshape(B * T, N, point_feat.shape[-1])
        point_mask_flat = point_mask.reshape(B * T, N)
        state_flat = query_state.reshape(B * T, self.state_dim)

        pc_feat = self.point_encoder(point_feat, point_mask_flat)
        state_feat = self.state_mlp(state_flat)
        tokens = self.out_proj(torch.cat([pc_feat, state_feat], dim=-1)).reshape(B, T, self.d_model)
        frame_ids = torch.arange(T, device=query_xyz.device)
        tokens = tokens + self.frame_embed(frame_ids).to(dtype=tokens.dtype).view(1, T, self.d_model)
        tokens = tokens + self.role_embed.view(1, 1, self.d_model)
        token_mask = point_mask.any(dim=-1)

        return ContextEncoderOutput(
            tokens=tokens,
            token_mask=token_mask,
            query_tokens=tokens,
            query_token_mask=token_mask,
        )
