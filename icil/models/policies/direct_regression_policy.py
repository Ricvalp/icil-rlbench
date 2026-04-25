from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from icil.models.common import (
    ContextConditionerConfig,
    DirectChunkBlock,
    DirectChunkBlock2Ctx,
    PooledContextConditioner,
)
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


@dataclass
class DirectRegressionPolicyConfig:
    d_model: int = 512
    n_heads: int = 8
    decoder_layers: int = 8
    decoder_mlp_mult: int = 4
    dropout: float = 0.0
    grad_checkpoint_decoder: bool = False
    context_attention_mode: str = "single"  # "single" | "two_ctx"
    attention_backend: str = "manual"  # "manual" | "sdpa"
    loss_type: str = "l1"  # "l1" | "mse"
    horizon: int = 16
    conditioner_mlp_mult: int = 2
    conditioner_dropout: float = 0.0


@dataclass
class ResolvedContext:
    tokens: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    support_tokens: Optional[torch.Tensor] = None
    support_token_mask: Optional[torch.Tensor] = None
    query_tokens: Optional[torch.Tensor] = None
    query_token_mask: Optional[torch.Tensor] = None


class DirectRegressionPolicy(nn.Module):
    def __init__(
        self,
        *,
        cfg: DirectRegressionPolicyConfig,
        context_encoder: ContextEncoder,
        state_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(cfg.horizon)
        self.context_attention_mode = str(cfg.context_attention_mode)
        if self.horizon < 1:
            raise ValueError("DirectRegressionPolicyConfig.horizon must be >= 1.")
        if self.context_attention_mode not in {"single", "two_ctx"}:
            raise ValueError(
                f"Unsupported context_attention_mode={self.context_attention_mode!r}. "
                "Expected 'single' or 'two_ctx'."
            )
        d = int(cfg.d_model)
        if d % int(cfg.n_heads) != 0:
            raise ValueError(f"d_model={d} must be divisible by n_heads={cfg.n_heads}.")
        if getattr(context_encoder, "d_model", d) != d:
            raise ValueError(
                f"context_encoder.d_model={getattr(context_encoder, 'd_model', None)} "
                f"must match direct head d_model={d}."
            )
        self.context_encoder = context_encoder
        self.context_conditioner = PooledContextConditioner(
            ContextConditionerConfig(
                d_model=d,
                context_attention_mode=self.context_attention_mode,
                mlp_mult=int(cfg.conditioner_mlp_mult),
                dropout=float(cfg.conditioner_dropout),
            )
        )
        self.action_queries = nn.Parameter(torch.randn(self.horizon, d) * 0.02)
        self.action_slot_embed = nn.Parameter(torch.randn(self.horizon, d) * 0.02)
        block_cls = DirectChunkBlock if self.context_attention_mode == "single" else DirectChunkBlock2Ctx
        self.decoder = nn.ModuleList(
            [
                block_cls(
                    d=d,
                    n_heads=int(cfg.n_heads),
                    cond_dim=d,
                    mlp_mult=int(cfg.decoder_mlp_mult),
                    dropout=float(cfg.dropout),
                    attention_backend=str(cfg.attention_backend),
                )
                for _ in range(int(cfg.decoder_layers))
            ]
        )
        self.action_out = nn.Linear(d, self.action_dim)

    @staticmethod
    def _concat_token_groups(
        tokens_a: Optional[torch.Tensor],
        mask_a: Optional[torch.Tensor],
        tokens_b: Optional[torch.Tensor],
        mask_b: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tokens_a is None:
            return tokens_b, mask_b
        if tokens_b is None:
            return tokens_a, mask_a

        tokens = torch.cat([tokens_a, tokens_b], dim=1)
        if mask_a is None and mask_b is None:
            return tokens, None
        if mask_a is None:
            mask_a = torch.ones(tokens_a.shape[:2], device=tokens_a.device, dtype=torch.bool)
        else:
            mask_a = mask_a.to(torch.bool)
        if mask_b is None:
            mask_b = torch.ones(tokens_b.shape[:2], device=tokens_b.device, dtype=torch.bool)
        else:
            mask_b = mask_b.to(torch.bool)
        return tokens, torch.cat([mask_a, mask_b], dim=1)

    def _resolve_context_output(self, ctx_out: Any) -> ResolvedContext:
        if isinstance(ctx_out, ContextEncoderOutput):
            ctx = ResolvedContext(
                tokens=ctx_out.tokens,
                token_mask=ctx_out.token_mask,
                support_tokens=ctx_out.support_tokens,
                support_token_mask=ctx_out.support_token_mask,
                query_tokens=ctx_out.query_tokens,
                query_token_mask=ctx_out.query_token_mask,
            )
        elif isinstance(ctx_out, tuple):
            ctx = ResolvedContext(
                tokens=ctx_out[0],
                token_mask=ctx_out[1] if len(ctx_out) > 1 else None,
            )
        else:
            ctx = ResolvedContext(tokens=ctx_out)

        if ctx.tokens is None:
            ctx.tokens, ctx.token_mask = self._concat_token_groups(
                ctx.support_tokens,
                ctx.support_token_mask,
                ctx.query_tokens,
                ctx.query_token_mask,
            )
        return ctx

    @staticmethod
    def _require_single_context(ctx: ResolvedContext) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if ctx.tokens is None:
            raise ValueError(
                "Single-context direct decoder requires ContextEncoderOutput.tokens "
                "or a split context that can be combined."
            )
        return ctx.tokens, ctx.token_mask

    @staticmethod
    def _require_two_context(
        ctx: ResolvedContext,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ctx.query_tokens is None:
            raise ValueError(
                "Two-context direct decoder requires ContextEncoderOutput.query_tokens."
            )
        return (
            ctx.query_tokens,
            ctx.query_token_mask,
            ctx.support_tokens,
            ctx.support_token_mask,
        )

    @staticmethod
    def _apply_single_context_block(
        blk: DirectChunkBlock,
        h: torch.Tensor,
        cond: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_mask: Optional[torch.Tensor],
        *,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if use_checkpoint:
            if ctx_mask is None:
                return ckpt.checkpoint(
                    lambda h_, cond_, ctx_, blk_=blk: blk_(h_, cond_, ctx_, ctx_mask=None),
                    h,
                    cond,
                    ctx_tokens,
                    use_reentrant=False,
                )
            return ckpt.checkpoint(
                lambda h_, cond_, ctx_, ctx_mask_, blk_=blk: blk_(h_, cond_, ctx_, ctx_mask_),
                h,
                cond,
                ctx_tokens,
                ctx_mask,
                use_reentrant=False,
            )
        return blk(h, cond, ctx_tokens, ctx_mask)

    @staticmethod
    def _apply_two_context_block(
        blk: DirectChunkBlock2Ctx,
        h: torch.Tensor,
        cond: torch.Tensor,
        query_tokens: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        support_tokens: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor],
        *,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if use_checkpoint:
            if support_tokens is None:
                return ckpt.checkpoint(
                    lambda h_, cond_, query_tokens_, blk_=blk: blk_(
                        h_,
                        cond_,
                        ctx_query=query_tokens_,
                        ctx_support=None,
                        ctx_query_mask=query_mask,
                        ctx_support_mask=None,
                    ),
                    h,
                    cond,
                    query_tokens,
                    use_reentrant=False,
                )
            return ckpt.checkpoint(
                lambda h_, cond_, query_tokens_, support_tokens_, blk_=blk: blk_(
                    h_,
                    cond_,
                    ctx_query=query_tokens_,
                    ctx_support=support_tokens_,
                    ctx_query_mask=query_mask,
                    ctx_support_mask=support_mask,
                ),
                h,
                cond,
                query_tokens,
                support_tokens,
                use_reentrant=False,
            )
        return blk(
            h,
            cond,
            ctx_query=query_tokens,
            ctx_support=support_tokens,
            ctx_query_mask=query_mask,
            ctx_support_mask=support_mask,
        )

    def _check_action_horizon(self, action_horizon: int) -> None:
        if int(action_horizon) != self.horizon:
            raise ValueError(
                f"DirectRegressionPolicy was initialized with horizon={self.horizon}, "
                f"but received action_horizon={int(action_horizon)}."
            )

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
        if action_horizon is not None:
            self._check_action_horizon(int(action_horizon))
        B = int(query_xyz.shape[0])
        ctx_out = self.context_encoder(
            query_xyz=query_xyz,
            query_state=query_state,
            cond_xyz=cond_xyz,
            cond_state=cond_state,
            cond_traj=cond_traj,
            cond_traj_mask=cond_traj_mask,
            query_rgb=query_rgb,
            query_mask_id=query_mask_id,
            query_valid=query_valid,
            cond_rgb=cond_rgb,
            cond_mask_id=cond_mask_id,
            cond_valid=cond_valid,
        )
        ctx = self._resolve_context_output(ctx_out)
        cond_vec = self.context_conditioner(
            tokens=ctx.tokens,
            token_mask=ctx.token_mask,
            support_tokens=ctx.support_tokens,
            support_token_mask=ctx.support_token_mask,
            query_tokens=ctx.query_tokens,
            query_token_mask=ctx.query_token_mask,
        )
        h = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        h = h + self.action_slot_embed.unsqueeze(0)
        use_decoder_ckpt = bool(
            self.training and self.cfg.grad_checkpoint_decoder and torch.is_grad_enabled()
        )
        if self.context_attention_mode == "single":
            ctx_tokens, ctx_mask = self._require_single_context(ctx)
            for blk in self.decoder:
                h = self._apply_single_context_block(
                    blk,
                    h,
                    cond_vec,
                    ctx_tokens,
                    ctx_mask,
                    use_checkpoint=use_decoder_ckpt,
                )
        else:
            query_tokens, query_mask, support_tokens, support_mask = self._require_two_context(ctx)
            for blk in self.decoder:
                h = self._apply_two_context_block(
                    blk,
                    h,
                    cond_vec,
                    query_tokens,
                    query_mask,
                    support_tokens,
                    support_mask,
                    use_checkpoint=use_decoder_ckpt,
                )
        return self.action_out(h)

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = batch["target_action"]
        self._check_action_horizon(int(target.shape[1]))
        pred = self.forward_actions(
            cond_xyz=batch.get("cond_xyz", None),
            cond_state=batch.get("cond_state", None),
            cond_traj=batch.get("cond_traj", None),
            cond_traj_mask=batch.get("cond_traj_mask", None),
            query_xyz=batch["query_xyz"],
            query_state=batch["query_state"],
            cond_rgb=batch.get("cond_rgb", None),
            query_rgb=batch.get("query_rgb", None),
            cond_mask_id=batch.get("cond_mask_id", None),
            query_mask_id=batch.get("query_mask_id", None),
            cond_valid=batch.get("cond_valid", None),
            query_valid=batch.get("query_valid", None),
        )
        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        loss_type = str(self.cfg.loss_type).lower()
        if loss_type == "mse":
            loss = mse
        elif loss_type == "l1":
            loss = l1
        else:
            raise ValueError(f"Unsupported loss_type={self.cfg.loss_type!r}. Expected 'l1' or 'mse'.")
        return {
            "loss": loss,
            "l1": l1.detach(),
            "mse": mse.detach(),
            "pred_action": pred,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        cond_xyz: Optional[torch.Tensor] = None,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        action_horizon: int,
        cond_state: Optional[torch.Tensor] = None,
        cond_traj: Optional[torch.Tensor] = None,
        cond_traj_mask: Optional[torch.Tensor] = None,
        cond_rgb: Optional[torch.Tensor] = None,
        query_rgb: Optional[torch.Tensor] = None,
        cond_mask_id: Optional[torch.Tensor] = None,
        query_mask_id: Optional[torch.Tensor] = None,
        cond_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        inference_steps: Optional[int] = None,
        eta: float = 0.0,
        return_trace: bool = False,
        trace_steps: Optional[int] = None,
    ) -> Any:
        del inference_steps, eta, trace_steps
        pred = self.forward_actions(
            cond_xyz=cond_xyz,
            cond_state=cond_state,
            cond_traj=cond_traj,
            cond_traj_mask=cond_traj_mask,
            query_xyz=query_xyz,
            query_state=query_state,
            cond_rgb=cond_rgb,
            query_rgb=query_rgb,
            cond_mask_id=cond_mask_id,
            query_mask_id=query_mask_id,
            cond_valid=cond_valid,
            query_valid=query_valid,
            action_horizon=action_horizon,
        )
        if not return_trace:
            return pred
        trace = {
            "x0_hat": pred.unsqueeze(0),
            "timesteps": torch.zeros((1,), device=pred.device, dtype=torch.long),
        }
        return pred, trace


DirectRegressionModelConfig = DirectRegressionPolicyConfig
