from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# =========================
# Utilities: embeddings
# =========================

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


# =========================
# Attention blocks
# =========================

class CrossAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.dh = d // n_heads

        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

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
        out = torch.matmul(w, vh)  # [B,h,Lq,dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, d)
        return self.proj(out)


class SelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CrossAttention(d, n_heads, dropout)

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
    def __init__(self, d: int, n_heads: int, cond_dim: int, mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.adaln1 = AdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout)
        self.adaln2 = AdaLN(d, cond_dim)
        self.cross_attn = CrossAttention(d, n_heads, dropout)
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


# =========================
# Frame tokenizer (Perceiver)
# =========================

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


# =========================
# Full ICIL Diffusion Policy
# =========================

@dataclass
class ModelConfig:
    # geometry/token dims
    d_model: int = 512
    n_heads: int = 8

    # frame tokenizer
    m_frame_tokens: int = 64
    frame_tokenizer_layers: int = 2

    # demo memory
    M_demo_latents: int = 256
    demo_perceiver_layers: int = 3
    ignore_demos: bool = False

    # diffusion transformer
    denoiser_layers: int = 10
    denoiser_mlp_mult: int = 4
    dropout: float = 0.0

    # embeddings
    mask_hash_buckets: int = 2048  # to avoid huge embedding tables for raw RLBench mask ids
    use_mask_id: bool = True
    role_embed_max_K: int = 32
    role_embed_max_L: int = 64
    role_embed_max_Tobs: int = 16

    # RGB fusion
    rgb_alpha_init: float = 1.0

    # diffusion (DDIM via diffusers)
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"  # "epsilon" | "sample" | "v_prediction"
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    num_inference_steps: Optional[int] = None


class ICILPerceiverDiffusionPolicy(nn.Module):
    def __init__(self, *, cfg: ModelConfig, state_dim: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.state_dim = state_dim
        self.action_dim = action_dim

        # --- point feature stem ---
        self.xyz_proj = nn.Linear(3, d, bias=False)
        self.rgb_proj = nn.Linear(3, d, bias=False)
        # Learnable scalar for RGB contribution in point tokens.
        self.rgb_alpha = nn.Parameter(torch.tensor(float(cfg.rgb_alpha_init), dtype=torch.float32))

        # mask embedding: hashed buckets -> d
        self.mask_embed = nn.Embedding(cfg.mask_hash_buckets, d)

        # per-frame role embeddings (added at token level)
        self.demo_id_embed = nn.Embedding(cfg.role_embed_max_K, d)
        self.keyframe_embed = nn.Embedding(cfg.role_embed_max_L, d)
        self.query_time_embed = nn.Embedding(cfg.role_embed_max_Tobs, d)

        # state token projection
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # frame tokenizer (Perceiver)
        self.frame_tokenizer = FramePerceiverTokenizer(
            d=d, m=cfg.m_frame_tokens, n_heads=cfg.n_heads,
            n_layers=cfg.frame_tokenizer_layers, dropout=cfg.dropout
        )

        # demo memory perceiver
        self.demo_memory = DemoMemoryPerceiver(
            d=d, M=cfg.M_demo_latents, n_heads=cfg.n_heads,
            n_layers=cfg.demo_perceiver_layers, dropout=cfg.dropout
        )

        # diffusion time embedding (for denoiser blocks)
        t_emb_dim = d  # base sinusoid dim
        self.t_mlp = TimeMLP(emb_dim=t_emb_dim, out_dim=d)

        # action embedding/projection
        self.action_in = nn.Linear(action_dim, d)
        self.action_out = nn.Linear(d, action_dim)

        self.denoiser = nn.ModuleList([
            DiTBlock(d=d, n_heads=cfg.n_heads, cond_dim=d, mlp_mult=cfg.denoiser_mlp_mult, dropout=cfg.dropout)
            for _ in range(cfg.denoiser_layers)
        ])

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

    # --------------------
    # Encoding helpers
    # --------------------

    def _hash_mask_ids(self, mask_id: torch.Tensor) -> torch.Tensor:
        # mask_id: [...], int64
        return torch.remainder(mask_id, self.cfg.mask_hash_buckets)

    def _encode_points(
        self,
        xyz: torch.Tensor,              # [Bf, N, 3]
        mask_id: Optional[torch.Tensor], # [Bf, N] int64
        rgb: Optional[torch.Tensor] = None, # [Bf, N, 3] in [0,1]
    ) -> torch.Tensor:
        """
        returns point tokens: [Bf, N, d]
        """
        h = self.xyz_proj(xyz)  # [Bf,N,d]
        if rgb is not None:
            rgb_f = rgb.to(dtype=xyz.dtype)
            h = h + self.rgb_alpha.to(dtype=h.dtype) * self.rgb_proj(rgb_f)
        if bool(getattr(self.cfg, "use_mask_id", True)) and mask_id is not None:
            mid = self._hash_mask_ids(mask_id)
            h = h + self.mask_embed(mid)
        return h

    def _tokenize_frames(
        self,
        xyz: torch.Tensor,               # [Bf, N, 3]
        state: torch.Tensor,             # [Bf, S]
        mask_id: Optional[torch.Tensor], # [Bf, N]
        rgb: Optional[torch.Tensor] = None, # [Bf, N, 3]
        point_valid: Optional[torch.Tensor] = None,  # [Bf, N] bool
    ) -> torch.Tensor:
        """
        returns per-frame tokens including state token:
          [Bf, m+1, d]  (m point-derived + 1 state token)
        """
        pt = self._encode_points(xyz, mask_id, rgb=rgb)  # [Bf,N,d]
        z = self.frame_tokenizer(pt, point_mask=point_valid)  # [Bf,m,d]
        s_tok = self.state_proj(state).unsqueeze(1)  # [Bf,1,d]
        return torch.cat([z, s_tok], dim=1)          # [Bf,m+1,d]

    def _build_demo_memory(
        self,
        cond_xyz: torch.Tensor,         # [B,K,L,N,3]
        cond_state: torch.Tensor,       # [B,K,L,S]
        cond_rgb: Optional[torch.Tensor] = None, # [B,K,L,N,3]
        cond_mask_id: Optional[torch.Tensor] = None, # [B,K,L,N]
        cond_valid: Optional[torch.Tensor] = None,   # [B,K,L,N] bool
    ) -> torch.Tensor:
        """
        returns Z_demo: [B, M, d]
        """
        B, K, L, N, _ = cond_xyz.shape
        d = self.cfg.d_model

        if K > self.cfg.role_embed_max_K:
            raise ValueError(
                f"K={K} exceeds role_embed_max_K={self.cfg.role_embed_max_K}. "
                "Increase cfg.role_embed_max_K."
            )
        if L > self.cfg.role_embed_max_L:
            raise ValueError(
                f"L={L} exceeds role_embed_max_L={self.cfg.role_embed_max_L}. "
                "Increase cfg.role_embed_max_L."
            )

        # flatten frames
        xyz_f = cond_xyz.reshape(B * K * L, N, 3)
        state_f = cond_state.reshape(B * K * L, -1)
        rgb_f = cond_rgb.reshape(B * K * L, N, 3) if cond_rgb is not None else None
        mask_f = cond_mask_id.reshape(B * K * L, N) if cond_mask_id is not None else None
        valid_f = cond_valid.reshape(B * K * L, N).to(torch.bool) if cond_valid is not None else None

        frame_tokens = self._tokenize_frames(xyz_f, state_f, mask_f, rgb=rgb_f, point_valid=valid_f)  # [B*K*L, m+1, d]
        m1 = frame_tokens.shape[1]
        frame_tokens = frame_tokens.reshape(B, K, L, m1, d)

        # add role embeddings at token level (broadcast across tokens within the frame)
        # demo id
        demo_ids = torch.arange(K, device=cond_xyz.device).clamp_max(self.cfg.role_embed_max_K - 1)
        demo_e = self.demo_id_embed(demo_ids)  # [K,d]
        # keyframe id
        kf_ids = torch.arange(L, device=cond_xyz.device).clamp_max(self.cfg.role_embed_max_L - 1)
        kf_e = self.keyframe_embed(kf_ids)     # [L,d]

        frame_tokens = frame_tokens + demo_e.view(1, K, 1, 1, d) + kf_e.view(1, 1, L, 1, d)

        # flatten all conditioning tokens into one set
        tokens = frame_tokens.reshape(B, K * L * m1, d)  # [B,S,d]

        # compress to demo memory latents
        Z_demo = self.demo_memory(tokens)  # [B,M,d]
        return Z_demo

    def _build_query_tokens(
        self,
        query_xyz: torch.Tensor,           # [B,T_obs,N,3]
        query_state: torch.Tensor,         # [B,T_obs,S]
        query_rgb: Optional[torch.Tensor] = None, # [B,T_obs,N,3]
        query_mask_id: Optional[torch.Tensor] = None, # [B,T_obs,N]
        query_valid: Optional[torch.Tensor] = None,   # [B,T_obs,N] bool
    ) -> torch.Tensor:
        """
        returns Z_query: [B, T_obs*(m+1), d]
        """
        B, Tobs, N, _ = query_xyz.shape
        d = self.cfg.d_model

        if Tobs > self.cfg.role_embed_max_Tobs:
            raise ValueError(
                f"T_obs={Tobs} exceeds role_embed_max_Tobs={self.cfg.role_embed_max_Tobs}. "
                "Increase cfg.role_embed_max_Tobs."
            )

        xyz_f = query_xyz.reshape(B * Tobs, N, 3)
        state_f = query_state.reshape(B * Tobs, -1)
        rgb_f = query_rgb.reshape(B * Tobs, N, 3) if query_rgb is not None else None
        mask_f = query_mask_id.reshape(B * Tobs, N) if query_mask_id is not None else None
        valid_f = query_valid.reshape(B * Tobs, N).to(torch.bool) if query_valid is not None else None

        frame_tokens = self._tokenize_frames(xyz_f, state_f, mask_f, rgb=rgb_f, point_valid=valid_f)  # [B*Tobs, m+1, d]
        m1 = frame_tokens.shape[1]
        frame_tokens = frame_tokens.reshape(B, Tobs, m1, d)

        # time embedding for query frames (0..T_obs-1)
        t_ids = torch.arange(Tobs, device=query_xyz.device).clamp_max(self.cfg.role_embed_max_Tobs - 1)
        t_e = self.query_time_embed(t_ids)  # [Tobs,d]
        frame_tokens = frame_tokens + t_e.view(1, Tobs, 1, d)

        return frame_tokens.reshape(B, Tobs * m1, d)

    # --------------------
    # Denoiser forward
    # --------------------

    def predict_model_output(
        self,
        x_t: torch.Tensor,                # [B,H,A]
        t: torch.Tensor,                  # [B]
        cond_xyz: torch.Tensor,
        cond_state: torch.Tensor,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
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

        Z_query = self._build_query_tokens(
            query_xyz, query_state, query_rgb=query_rgb, query_mask_id=query_mask_id, query_valid=query_valid
        )  # [B,Sq,d]
        # Optional ablation: ignore support demos and condition only on query tokens.
        if bool(getattr(self.cfg, "ignore_demos", False)):
            ctx = Z_query
        else:
            Z_demo = self._build_demo_memory(
                cond_xyz, cond_state, cond_rgb=cond_rgb, cond_mask_id=cond_mask_id, cond_valid=cond_valid
            )  # [B,M,d]
            ctx = torch.cat([Z_demo, Z_query], dim=1)  # [B, M+Sq, d]

        # diffusion timestep embedding
        t_emb = sinusoidal_time_embedding(t, d)  # [B,d]
        t_cond = self.t_mlp(t_emb)               # [B,d]

        # action tokens
        h = self.action_in(x_t)  # [B,H,d]
        # Add action-position signal so chunk order is identifiable.
        h = h + sinusoidal_position_embedding(H, d, device=x_t.device).to(dtype=h.dtype).unsqueeze(0)
        for blk in self.denoiser:
            h = blk(h, t_cond=t_cond, ctx=ctx, ctx_mask=None)
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
            cond_xyz=batch["cond_xyz"],
            cond_state=batch["cond_state"],
            query_xyz=batch["query_xyz"],
            query_state=batch["query_state"],
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
        cond_xyz: torch.Tensor,
        cond_state: torch.Tensor,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        action_horizon: int,
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

        device = cond_xyz.device
        B = cond_xyz.shape[0]
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


# =========================
# Sanity check helper
# =========================

def check_batch_shapes(batch: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    Quick runtime checks; returns (state_dim, action_dim, N) inferred.
    """
    assert batch["cond_xyz"].dim() == 5, batch["cond_xyz"].shape  # [B,K,L,N,3]
    assert batch["query_xyz"].dim() == 4, batch["query_xyz"].shape  # [B,T_obs,N,3]
    assert batch["target_action"].dim() == 3, batch["target_action"].shape  # [B,H,A]
    assert batch["cond_state"].dim() == 4, batch["cond_state"].shape  # [B,K,L,S]
    assert batch["query_state"].dim() == 3, batch["query_state"].shape  # [B,T_obs,S]
    if "cond_valid" in batch:
        assert batch["cond_valid"].dim() == 4, batch["cond_valid"].shape  # [B,K,L,N]
    if "query_valid" in batch:
        assert batch["query_valid"].dim() == 3, batch["query_valid"].shape  # [B,T_obs,N]
    if "cond_rgb" in batch:
        assert batch["cond_rgb"].dim() == 5, batch["cond_rgb"].shape  # [B,K,L,N,3]
    if "query_rgb" in batch:
        assert batch["query_rgb"].dim() == 4, batch["query_rgb"].shape  # [B,T_obs,N,3]
    N = batch["cond_xyz"].shape[3]
    state_dim = batch["cond_state"].shape[-1]
    action_dim = batch["target_action"].shape[-1]
    return state_dim, action_dim, N
