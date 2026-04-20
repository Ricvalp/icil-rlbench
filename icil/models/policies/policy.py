from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


from icil.models.common import (
    DiTBlock,
    DiTBlock2Ctx,
    TimeMLP,
    sinusoidal_position_embedding,
    sinusoidal_time_embedding,
)
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


@dataclass
class PolicyConfig:
    # Core model dims
    d_model: int = 512
    n_heads: int = 8

    # diffusion transformer
    denoiser_layers: int = 10
    denoiser_mlp_mult: int = 4
    dropout: float = 0.0
    grad_checkpoint_dit: bool = False
    context_attention_mode: str = "single"  # "single" | "two_ctx"
    attention_backend: str = "manual"  # "manual" | "sdpa" ("flash" alias)

    # diffusion (DDIM via diffusers)
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"  # "epsilon" | "sample" | "v_prediction"
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    num_inference_steps: Optional[int] = None


@dataclass
class ResolvedContext:
    tokens: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    support_tokens: Optional[torch.Tensor] = None
    support_token_mask: Optional[torch.Tensor] = None
    query_tokens: Optional[torch.Tensor] = None
    query_token_mask: Optional[torch.Tensor] = None


class Policy(nn.Module):
    def __init__(
        self,
        *,
        cfg: PolicyConfig,
        context_encoder: ContextEncoder,
        state_dim: int,
        action_dim: int
        ):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_attention_mode = str(cfg.context_attention_mode)
        if d % int(cfg.n_heads) != 0:
            raise ValueError(f"d_model={d} must be divisible by n_heads={cfg.n_heads}.")
        if self.context_attention_mode not in {"single", "two_ctx"}:
            raise ValueError(
                f"Unsupported context_attention_mode={self.context_attention_mode!r}. "
                "Expected 'single' or 'two_ctx'."
            )
        if getattr(context_encoder, "d_model", d) != d:
            raise ValueError(
                f"context_encoder.d_model={getattr(context_encoder, 'd_model', None)} "
                f"must match policy d_model={d}."
            )
        self.context_encoder = context_encoder

        # diffusion time embedding (for denoiser blocks)
        t_emb_dim = d  # base sinusoid dim
        self.t_mlp = TimeMLP(emb_dim=t_emb_dim, out_dim=d)

        # action embedding/projection
        self.action_in = nn.Linear(action_dim, d)
        self.action_out = nn.Linear(d, action_dim)

        block_cls = DiTBlock if self.context_attention_mode == "single" else DiTBlock2Ctx
        self.denoiser = nn.ModuleList(
            [
                block_cls(
                    d=d,
                    n_heads=cfg.n_heads,
                    cond_dim=d,
                    mlp_mult=cfg.denoiser_mlp_mult,
                    dropout=cfg.dropout,
                    attention_backend=str(cfg.attention_backend),
                )
                for _ in range(cfg.denoiser_layers)
            ]
        )

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=int(cfg.num_train_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            beta_schedule=str(cfg.beta_schedule),
            clip_sample=False,
            set_alpha_to_one=bool(cfg.set_alpha_to_one),
            steps_offset=int(cfg.steps_offset),
            prediction_type=str(cfg.prediction_type),
        )
        self.num_inference_steps = (
            int(cfg.num_inference_steps)
            if cfg.num_inference_steps is not None
            else int(self.noise_scheduler.config.num_train_timesteps)
        )

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
    def _require_single_context(
        ctx: ResolvedContext,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if ctx.tokens is None:
            raise ValueError(
                "Single-context denoiser requires ContextEncoderOutput.tokens "
                "or a split context that can be combined."
            )
        return ctx.tokens, ctx.token_mask

    @staticmethod
    def _require_two_context(
        ctx: ResolvedContext,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if ctx.query_tokens is None:
            raise ValueError(
                "Two-context denoiser requires ContextEncoderOutput.query_tokens. "
                "Update the context encoder to return split query/support tokens."
            )
        return (
            ctx.query_tokens,
            ctx.query_token_mask,
            ctx.support_tokens,
            ctx.support_token_mask,
        )

    @staticmethod
    def _apply_single_context_block(
        blk: DiTBlock,
        h: torch.Tensor,
        t_cond: torch.Tensor,
        ctx_tokens: torch.Tensor,
        ctx_mask: Optional[torch.Tensor],
        *,
        use_checkpoint: bool,
    ) -> torch.Tensor:
        if use_checkpoint:
            if ctx_mask is None:
                return ckpt.checkpoint(
                    lambda h_, t_cond_, ctx_, blk_=blk: blk_(
                        h_, t_cond=t_cond_, ctx=ctx_, ctx_mask=None
                    ),
                    h,
                    t_cond,
                    ctx_tokens,
                    use_reentrant=False,
                )
            return ckpt.checkpoint(
                lambda h_, t_cond_, ctx_, ctx_mask_, blk_=blk: blk_(
                    h_, t_cond=t_cond_, ctx=ctx_, ctx_mask=ctx_mask_
                ),
                h,
                t_cond,
                ctx_tokens,
                ctx_mask,
                use_reentrant=False,
            )
        return blk(h, t_cond=t_cond, ctx=ctx_tokens, ctx_mask=ctx_mask)

    @staticmethod
    def _apply_two_context_block(
        blk: DiTBlock2Ctx,
        h: torch.Tensor,
        t_cond: torch.Tensor,
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
                    lambda h_, t_cond_, query_tokens_, blk_=blk: blk_(
                        h_,
                        t_cond=t_cond_,
                        ctx_query=query_tokens_,
                        ctx_support=None,
                        ctx_query_mask=query_mask,
                        ctx_support_mask=None,
                    ),
                    h,
                    t_cond,
                    query_tokens,
                    use_reentrant=False,
                )
            return ckpt.checkpoint(
                lambda h_, t_cond_, query_tokens_, support_tokens_, blk_=blk: blk_(
                    h_,
                    t_cond=t_cond_,
                    ctx_query=query_tokens_,
                    ctx_support=support_tokens_,
                    ctx_query_mask=query_mask,
                    ctx_support_mask=support_mask,
                ),
                h,
                t_cond,
                query_tokens,
                support_tokens,
                use_reentrant=False,
            )
        return blk(
            h,
            t_cond=t_cond,
            ctx_query=query_tokens,
            ctx_support=support_tokens,
            ctx_query_mask=query_mask,
            ctx_support_mask=support_mask,
        )


    # --------------------
    # Denoiser forward
    # --------------------

    def predict_model_output(
        self,
        x_t: torch.Tensor,                # [B,H,A]
        t: torch.Tensor,                  # [B]
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
    ) -> torch.Tensor:
        """
        Returns model output for the configured diffusion prediction type: [B,H,A]
        """
        B, H, A = x_t.shape
        d = self.cfg.d_model

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

        # diffusion timestep embedding
        t_emb = sinusoidal_time_embedding(t, d)  # [B,d]
        t_cond = self.t_mlp(t_emb)               # [B,d]

        # action tokens
        h = self.action_in(x_t)  # [B,H,d]
        # Add action-position signal so chunk order is identifiable.
        h = h + sinusoidal_position_embedding(H, d, device=x_t.device).to(dtype=h.dtype).unsqueeze(0)
        use_dit_ckpt = bool(self.training and self.cfg.grad_checkpoint_dit and torch.is_grad_enabled())
        if self.context_attention_mode == "single":
            ctx_tokens, ctx_mask = self._require_single_context(ctx)
            for blk in self.denoiser:
                h = self._apply_single_context_block(
                    blk,
                    h,
                    t_cond,
                    ctx_tokens,
                    ctx_mask,
                    use_checkpoint=use_dit_ckpt,
                )
        else:
            query_tokens, query_mask, support_tokens, support_mask = self._require_two_context(ctx)
            for blk in self.denoiser:
                h = self._apply_two_context_block(
                    blk,
                    h,
                    t_cond,
                    query_tokens,
                    query_mask,
                    support_tokens,
                    support_mask,
                    use_checkpoint=use_dit_ckpt,
                )
        model_out = self.action_out(h)  # [B,H,A]
        return model_out

    # --------------------
    # Training loss
    # --------------------

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Expects your batch keys:
          cond_xyz, cond_state, query_xyz, query_state, target_action
          optional: cond_mask_id, query_mask_id

        Returns dict with loss and diagnostic scalars.
        """
        device = batch["target_action"].device
        x0 = batch["target_action"]  # [B,H,A]
        B = x0.shape[0]

        t = torch.randint(
            low=0,
            high=int(self.noise_scheduler.config.num_train_timesteps),
            size=(B,),
            device=device,
        ).long()  # [B]
        noise = torch.randn_like(x0)
        x_t = self.noise_scheduler.add_noise(x0, noise, t)

        model_out = self.predict_model_output(
            x_t=x_t,
            t=t,
            cond_xyz=batch.get("cond_xyz", None),
            query_xyz=batch["query_xyz"],
            query_state=batch["query_state"],
            cond_state=batch.get("cond_state", None),
            cond_traj=batch.get("cond_traj", None),
            cond_traj_mask=batch.get("cond_traj_mask", None),
            cond_rgb=batch.get("cond_rgb", None),
            query_rgb=batch.get("query_rgb", None),
            cond_mask_id=batch.get("cond_mask_id", None),
            query_mask_id=batch.get("query_mask_id", None),
            cond_valid=batch.get("cond_valid", None),
            query_valid=batch.get("query_valid", None),
        )

        pred_type = str(self.noise_scheduler.config.prediction_type)
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = x0
        elif pred_type == "v_prediction":
            if hasattr(self.noise_scheduler, "get_velocity"):
                target = self.noise_scheduler.get_velocity(x0, noise, t)
            else:
                alpha_t = self.noise_scheduler.alphas_cumprod[t].sqrt().to(x0.device)
                sigma_t = (1.0 - self.noise_scheduler.alphas_cumprod[t]).sqrt().to(x0.device)
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
                target = alpha_t * noise - sigma_t * x0
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(model_out, target)

        return {
            "loss": loss,
            "mse": loss.detach(),
            "t_mean": t.float().mean().detach(),
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
        """
        DDIM sampling for action chunk prediction.
        Returns sampled actions: [B, H, A]
        If return_trace=True, returns a tuple:
          (sampled_actions, {"x0_hat": [S,B,H,A], "timesteps": [S]})
        where S is the number of captured denoising snapshots.
        """
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")
        if eta < 0.0:
            raise ValueError("eta must be >= 0.")

        device = query_xyz.device
        B = query_xyz.shape[0]
        H = int(action_horizon)
        A = self.action_dim

        scheduler = self.noise_scheduler
        total_T = int(scheduler.config.num_train_timesteps)
        steps = self.num_inference_steps if inference_steps is None else int(inference_steps)
        steps = max(1, min(steps, total_T))

        try:
            scheduler.set_timesteps(steps, device=device)
        except TypeError:
            scheduler.set_timesteps(steps)

        x_t = torch.randn(B, H, A, device=device)
        trace_x0: List[torch.Tensor] = []
        trace_t: List[int] = []
        capture_idx = None
        if return_trace:
            if trace_steps is None or int(trace_steps) <= 0 or int(trace_steps) >= steps:
                capture_idx = set(range(steps))
            else:
                n = int(trace_steps)
                if n == 1:
                    capture_idx = {steps - 1}
                else:
                    capture_idx = {
                        int(round(i * (steps - 1) / float(n - 1)))
                        for i in range(n)
                    }

        step_sig = inspect.signature(scheduler.step).parameters
        for i, t_now in enumerate(scheduler.timesteps):
            t_int = int(t_now.item() if torch.is_tensor(t_now) else t_now)
            t_batch = torch.full((B,), t_int, device=device, dtype=torch.long)

            model_out = self.predict_model_output(
                x_t=x_t,
                t=t_batch,
                cond_xyz=cond_xyz,
                cond_state=cond_state,
                query_xyz=query_xyz,
                query_state=query_state,
                cond_traj=cond_traj,
                cond_traj_mask=cond_traj_mask,
                cond_rgb=cond_rgb,
                query_rgb=query_rgb,
                cond_mask_id=cond_mask_id,
                query_mask_id=query_mask_id,
                cond_valid=cond_valid,
                query_valid=query_valid,
            )

            step_kwargs: Dict[str, Any] = {}
            if "eta" in step_sig:
                step_kwargs["eta"] = float(eta)

            step_out = scheduler.step(model_out, t_now, x_t, **step_kwargs)
            x0_hat = getattr(step_out, "pred_original_sample", None)
            if isinstance(step_out, tuple):
                x_t = step_out[0]
            else:
                x_t = step_out.prev_sample

            if return_trace and capture_idx is not None and i in capture_idx:
                if x0_hat is None:
                    x0_hat = x_t
                trace_x0.append(x0_hat.detach())
                trace_t.append(t_int)

        if return_trace:
            if len(trace_x0) == 0:
                trace_x0 = [x_t.detach()]
                trace_t = [0]
            trace = {
                "x0_hat": torch.stack(trace_x0, dim=0),
                "timesteps": torch.tensor(trace_t, device=x_t.device, dtype=torch.long),
            }
            return x_t, trace
        return x_t


ModelConfig = PolicyConfig
