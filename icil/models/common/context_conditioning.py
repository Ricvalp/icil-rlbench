from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ContextConditionerConfig:
    d_model: int = 512
    context_attention_mode: str = "single"  # "single" | "two_ctx"
    mlp_mult: int = 2
    dropout: float = 0.0


class PooledContextConditioner(nn.Module):
    def __init__(self, cfg: ContextConditionerConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        hidden = max(d, int(cfg.mlp_mult) * d)
        mode = str(cfg.context_attention_mode)
        if mode not in {"single", "two_ctx"}:
            raise ValueError(
                f"Unsupported context_attention_mode={mode!r}. Expected 'single' or 'two_ctx'."
            )
        self.context_attention_mode = mode
        in_dim = d if mode == "single" else 2 * d
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, d),
        )

    @staticmethod
    def _masked_mean_pool(
        tokens: Optional[torch.Tensor],
        token_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        if token_mask is None:
            return tokens.mean(dim=1)
        mask = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * mask).sum(dim=1) / denom
        empty = token_mask.to(torch.bool).sum(dim=1) == 0
        if bool(empty.any().item()):
            pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)
        return pooled

    def forward(
        self,
        *,
        tokens: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        support_tokens: Optional[torch.Tensor] = None,
        support_token_mask: Optional[torch.Tensor] = None,
        query_tokens: Optional[torch.Tensor] = None,
        query_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.context_attention_mode == "single":
            pooled = self._masked_mean_pool(tokens, token_mask)
            if pooled is None:
                raise ValueError("Single-context conditioner requires tokens.")
            return self.proj(pooled)

        pooled_query = self._masked_mean_pool(query_tokens, query_token_mask)
        if pooled_query is None:
            raise ValueError("Two-context conditioner requires query_tokens.")
        pooled_support = self._masked_mean_pool(support_tokens, support_token_mask)
        if pooled_support is None:
            pooled_support = torch.zeros_like(pooled_query)
        return self.proj(torch.cat([pooled_query, pooled_support], dim=-1))
