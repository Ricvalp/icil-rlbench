
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from icil.models.common.attention import CrossAttention


class FramePerceiverTokenizer(nn.Module):
    """
    Tokenize N point features into m learned latents via cross-attn.
    This avoids O(N^2) attention over points.
    """
    def __init__(self, d: int, m: int, n_heads: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.m = m
        self.latents = nn.Parameter(torch.randn(m, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "xattn": CrossAttention(d, n_heads, dropout),
                "mlp": nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 4 * d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d, d),
                ),
                "ln": nn.LayerNorm(d),
            })
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, point_tokens: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        point_tokens: [Bf, N, d]
        point_mask:   [Bf, N] bool True=keep (optional)
        returns:      [Bf, m, d]
        """
        Bf, _, d = point_tokens.shape
        z = self.latents.unsqueeze(0).expand(Bf, -1, -1)  # [Bf,m,d]
        for layer in self.layers:
            z = z + self.dropout(layer["xattn"](layer["ln"](z), point_tokens, kv_mask=point_mask))
            z = z + self.dropout(layer["mlp"](z))
        return z


class DemoMemoryPerceiver(nn.Module):
    """
    Compress many tokens (demo frames) into M memory latents.
    """
    def __init__(self, d: int, M: int, n_heads: int, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.M = M
        self.latents = nn.Parameter(torch.randn(M, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "xattn": CrossAttention(d, n_heads, dropout),
                "mlp": nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 4 * d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d, d),
                ),
                "ln": nn.LayerNorm(d),
            })
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens: [B, S, d]
        token_mask: [B, S] bool True=keep
        returns: [B, M, d]
        """
        B, _, d = tokens.shape
        z = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            z = z + self.dropout(layer["xattn"](layer["ln"](z), tokens, kv_mask=token_mask))
            z = z + self.dropout(layer["mlp"](z))
        return z


class TimeLatentPerceiver(nn.Module):
    """
    Perceiver-style compressor over time:
      z (m latents) cross-attends to tokens (T steps)
    """
    def __init__(self, d: int, m: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.m = int(m)
        self.latents = nn.Parameter(torch.randn(self.m, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln": nn.LayerNorm(d),
                "xattn": CrossAttention(d, n_heads, dropout),
                "mlp": nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 4 * d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d, d),
                ),
            })
            for _ in range(int(n_layers))
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens: [B, T, d]
        token_mask: [B, T] bool True=keep
        returns: [B, m, d]
        """
        B, _, d = tokens.shape
        z = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B,m,d]
        for layer in self.layers:
            z = z + self.drop(layer["xattn"](layer["ln"](z), tokens, kv_mask=token_mask))
            z = z + self.drop(layer["mlp"](z))
        return z