from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B] int64 or float32
    returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def sinusoidal_position_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    length: number of positions
    returns: [length, dim]
    """
    pos = torch.arange(length, device=device, dtype=torch.float32)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / max(1, half - 1)
    )
    args = pos.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def two_pi_continuous_sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [B, T] float in [0,1] (or any continuous)
    returns: [B, T, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=x.device, dtype=torch.float32)
        / max(1, half - 1)
    )  # [half]
    args = 2.0 * math.pi * x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # [B,T,half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B,T,2*half]
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def continuous_sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [B, T] float in [0,1] (or any continuous)
    returns: [B, T, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=x.device, dtype=torch.float32)
        / max(1, half - 1)
    )  # [half]
    args = x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # [B,T,half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B,T,2*half]
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)
