from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_attention_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value in {"manual", "old", "legacy"}:
        return "manual"
    if value in {"sdpa", "flash", "flash_attention", "flashattention"}:
        return "sdpa"
    raise ValueError(
        "attention_backend must be one of: 'manual', 'sdpa'. "
        f"Got {backend!r}."
    )


class CrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0, attention_backend: str = "manual"):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.dh = d // n_heads
        self.attention_backend = normalize_attention_backend(attention_backend)
        self.dropout_p = float(dropout)

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def _forward_manual(
        self,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        kv_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, _, Lq, _ = qh.shape
        Lk = kh.shape[2]
        att = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.dh)   # [B,h,Lq,Lk]
        if kv_mask is not None:
            # kv_mask: True=keep. Use NaN-safe masked softmax.
            keep = kv_mask.to(torch.bool).view(B, 1, 1, Lk)
            neg_large = -torch.finfo(att.dtype).max
            att = att.masked_fill(~keep, neg_large)
            w = torch.softmax(att, dim=-1)
            w = w * keep.to(dtype=w.dtype)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            w = torch.softmax(att, dim=-1)
        w = self.drop(w)
        return torch.matmul(w, vh)  # [B,h,Lq,dh]

    def _forward_sdpa(
        self,
        qh: torch.Tensor,
        kh: torch.Tensor,
        vh: torch.Tensor,
        kv_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_mask = None
        empty_rows = None
        if kv_mask is not None:
            keep = kv_mask.to(torch.bool)
            empty_rows = ~keep.any(dim=1)
            if bool(empty_rows.any().item()):
                keep = keep.clone()
                keep[empty_rows, 0] = True
            attn_mask = keep[:, None, None, :]

        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        if empty_rows is not None and bool(empty_rows.any().item()):
            out = out.masked_fill(empty_rows.view(-1, 1, 1, 1), 0.0)
        return out

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q:  [B, Lq, d]
        kv: [B, Lk, d]
        kv_mask: [B, Lk] bool, True=keep (optional)
        """
        B, Lq, d = q.shape
        _, Lk, _ = kv.shape

        qh = self.q(q).view(B, Lq, self.n_heads, self.dh).transpose(1, 2)  # [B,h,Lq,dh]
        kh = self.k(kv).view(B, Lk, self.n_heads, self.dh).transpose(1, 2) # [B,h,Lk,dh]
        vh = self.v(kv).view(B, Lk, self.n_heads, self.dh).transpose(1, 2) # [B,h,Lk,dh]

        if self.attention_backend == "sdpa":
            out = self._forward_sdpa(qh, kh, vh, kv_mask)
        else:
            out = self._forward_manual(qh, kh, vh, kv_mask)
        out = out.transpose(1, 2).contiguous().view(B, Lq, d)
        return self.proj(out)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0, attention_backend: str = "manual"):
        super().__init__()
        self.attn = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attn(x, x, kv_mask=mask)


class AdaLN(nn.Module):
    """
    Adaptive LayerNorm conditioning: scale/shift from time embedding.
    """
    def __init__(self, d: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * d)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B,L,d]
        cond: [B,cond_dim]
        """
        h = self.norm(x)
        ss = self.to_scale_shift(cond).unsqueeze(1)  # [B,1,2d]
        scale, shift = ss.chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """
    Transformer block for action tokens with:
      - AdaLN (time-conditioned)
      - self-attn
      - cross-attn to context
      - MLP
    """
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
        self.adaln1 = AdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln2 = AdaLN(d, cond_dim)
        self.cross_attn = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln3 = AdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, t_cond)))
        x = x + self.drop(self.cross_attn(self.adaln2(x, t_cond), ctx, kv_mask=ctx_mask))
        x = x + self.drop(self.mlp(self.adaln3(x, t_cond)))
        return x


class DiTBlock2Ctx(nn.Module):
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
        self.adaln1 = AdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)

        # cross-attn #1: query
        self.adaln_q = AdaLN(d, cond_dim)
        self.cross_attn_q = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)

        # cross-attn #2: support
        self.adaln_s = AdaLN(d, cond_dim)
        self.cross_attn_s = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)

        self.adaln3 = AdaLN(d, cond_dim)
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
        t_cond: torch.Tensor,
        ctx_query: torch.Tensor,
        ctx_support: Optional[torch.Tensor] = None,
        ctx_query_mask: Optional[torch.Tensor] = None,
        ctx_support_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, t_cond)))

        # query-first cross-attn (forces grounding in current obs)
        x = x + self.drop(self.cross_attn_q(self.adaln_q(x, t_cond), ctx_query, kv_mask=ctx_query_mask))

        # then support cross-attn (demos / traj)
        if ctx_support is not None:
            x = x + self.drop(
                self.cross_attn_s(self.adaln_s(x, t_cond), ctx_support, kv_mask=ctx_support_mask)
            )

        x = x + self.drop(self.mlp(self.adaln3(x, t_cond)))
        return x
