from __future__ import annotations

import torch
import torch.nn as nn

from icil.models.common.attention import CrossAttention, SelfAttention


class IdentityContextAdaLN(nn.Module):
    def __init__(self, d: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class DirectChunkBlock(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        cond_dim: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.adaln1 = IdentityContextAdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln2 = IdentityContextAdaLN(d, cond_dim)
        self.cross_attn = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln3 = IdentityContextAdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, cond)))
        x = x + self.drop(self.cross_attn(self.adaln2(x, cond), ctx, kv_mask=ctx_mask))
        x = x + self.drop(self.mlp(self.adaln3(x, cond)))
        return x


class DirectChunkBlock2Ctx(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        cond_dim: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.adaln1 = IdentityContextAdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)

        self.adaln_q = IdentityContextAdaLN(d, cond_dim)
        self.cross_attn_q = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)

        self.adaln_s = IdentityContextAdaLN(d, cond_dim)
        self.cross_attn_s = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)

        self.adaln3 = IdentityContextAdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        ctx_query: torch.Tensor,
        ctx_support: torch.Tensor | None = None,
        ctx_query_mask: torch.Tensor | None = None,
        ctx_support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, cond)))
        x = x + self.drop(self.cross_attn_q(self.adaln_q(x, cond), ctx_query, kv_mask=ctx_query_mask))
        if ctx_support is not None:
            x = x + self.drop(
                self.cross_attn_s(self.adaln_s(x, cond), ctx_support, kv_mask=ctx_support_mask)
            )
        x = x + self.drop(self.mlp(self.adaln3(x, cond)))
        return x
