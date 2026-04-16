from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from icil.models.common import DemoMemoryPerceiver, FramePerceiverTokenizer
from icil.models.common.attention import SelfAttention
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


@dataclass
class PerceiverDemoQueryEncoderV2Config:
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.0

    # Demo/query frame tokenizers have separate parameters and can use different hyperparams.
    demo_m_frame_tokens: int = 128
    demo_frame_tokenizer_layers: int = 2
    demo_n_heads: int = 8
    query_m_frame_tokens: int = 64
    query_frame_tokenizer_layers: int = 2
    query_n_heads: int = 8

    # Demo memory compressor.
    M_demo_latents: int = 256
    demo_perceiver_layers: int = 3

    # Shared role/mask settings.
    mask_hash_buckets: int = 2048
    use_mask_id: bool = True
    role_embed_max_K: int = 32
    role_embed_max_L: int = 64
    role_embed_max_Tobs: int = 16
    ignore_demos: bool = False
    compress_demo_latents: bool = True

    # Input feature knobs.
    demo_rgb_alpha_init: float = 1.0
    query_rgb_alpha_init: float = 1.0
    use_gripper_point_features: bool = False
    gripper_xyz_state_start: int = 0
    gripper_alpha_init: float = 1.0

    # Optional token refiners after the perceiver stages.
    demo_post_self_attn_layers: int = 0
    query_post_self_attn_layers: int = 0
    post_self_attn_mlp_mult: int = 4

    # Memory/runtime knobs.
    checkpoint_demo_memory: bool = False
    checkpoint_build_demo_memory: bool = False
    checkpoint_frame_tokenizer: bool = False
    tokenize_frames_chunked: bool = False
    chunk_frames: int = 32


class _FrameTokenizationStack(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        m_frame_tokens: int,
        frame_tokenizer_layers: int,
        state_dim: int,
        mask_hash_buckets: int,
        use_mask_id: bool,
        rgb_alpha_init: float,
        dropout: float,
        use_gripper_point_features: bool,
        gripper_xyz_state_start: int,
        gripper_alpha_init: float,
    ):
        super().__init__()
        self.d = int(d)
        self.use_mask_id = bool(use_mask_id)
        self.use_gripper_point_features = bool(use_gripper_point_features)
        self.gripper_xyz_state_start = int(gripper_xyz_state_start)

        self.xyz_proj = nn.Linear(3, d, bias=False)
        self.rgb_proj = nn.Linear(3, d, bias=False)
        self.rgb_alpha = nn.Parameter(torch.tensor(float(rgb_alpha_init), dtype=torch.float32))

        self.mask_embed = nn.Embedding(int(mask_hash_buckets), d)
        self.gripper_proj = (
            nn.Linear(4, d, bias=False) if self.use_gripper_point_features else None
        )
        self.gripper_alpha = (
            nn.Parameter(torch.tensor(float(gripper_alpha_init), dtype=torch.float32))
            if self.use_gripper_point_features
            else None
        )

        self.state_proj = nn.Sequential(
            nn.Linear(int(state_dim), d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.frame_tokenizer = FramePerceiverTokenizer(
            d=d,
            m=int(m_frame_tokens),
            n_heads=int(n_heads),
            n_layers=int(frame_tokenizer_layers),
            dropout=float(dropout),
        )

    def _hash_mask_ids(self, mask_id: torch.Tensor) -> torch.Tensor:
        return torch.remainder(mask_id, self.mask_embed.num_embeddings)

    def _gripper_xyz_from_state(self, state: torch.Tensor) -> torch.Tensor:
        start = int(self.gripper_xyz_state_start)
        end = start + 3
        if int(state.shape[-1]) < end:
            raise ValueError(
                f"state_dim={int(state.shape[-1])} is too small for gripper_xyz_state_start={start}."
            )
        return state[..., start:end]

    def _encode_points(
        self,
        xyz: torch.Tensor,
        *,
        state: torch.Tensor,
        mask_id: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.xyz_proj(xyz)
        if rgb is not None:
            h = h + self.rgb_alpha.to(dtype=h.dtype) * self.rgb_proj(rgb.to(dtype=xyz.dtype))
        if self.use_gripper_point_features:
            if self.gripper_proj is None or self.gripper_alpha is None:
                raise RuntimeError("gripper point feature modules were not initialized.")
            gripper_xyz = self._gripper_xyz_from_state(state).to(device=xyz.device, dtype=xyz.dtype)
            rel = xyz - gripper_xyz.unsqueeze(1)
            dist = torch.norm(rel, dim=-1, keepdim=True)
            grip_feat = torch.cat([rel, dist], dim=-1)
            h = h + self.gripper_alpha.to(dtype=h.dtype) * self.gripper_proj(grip_feat)
        if self.use_mask_id and mask_id is not None:
            h = h + self.mask_embed(self._hash_mask_ids(mask_id))
        return h

    def tokenize_frames(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        mask_id: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor] = None,
        point_valid: Optional[torch.Tensor] = None,
        checkpoint_frame_tokenizer: bool = False,
    ) -> torch.Tensor:
        pt = self._encode_points(xyz, state=state, mask_id=mask_id, rgb=rgb)
        if checkpoint_frame_tokenizer and self.training and torch.is_grad_enabled():
            if point_valid is None:
                z = checkpoint(
                    lambda pt_: self.frame_tokenizer(pt_, point_mask=None),
                    pt,
                    use_reentrant=False,
                )
            else:
                z = checkpoint(
                    lambda pt_, point_valid_: self.frame_tokenizer(pt_, point_mask=point_valid_),
                    pt,
                    point_valid,
                    use_reentrant=False,
                )
        else:
            z = self.frame_tokenizer(pt, point_mask=point_valid)
        s_tok = self.state_proj(state).unsqueeze(1)
        return torch.cat([z, s_tok], dim=1)

    def tokenize_frames_chunked(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        mask_id: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor] = None,
        point_valid: Optional[torch.Tensor] = None,
        chunk_frames: int,
        checkpoint_frame_tokenizer: bool = False,
    ) -> torch.Tensor:
        if int(chunk_frames) < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}.")
        outs = []
        Bf = int(xyz.shape[0])
        for s in range(0, Bf, int(chunk_frames)):
            e = min(Bf, s + int(chunk_frames))
            outs.append(
                self.tokenize_frames(
                    xyz=xyz[s:e],
                    state=state[s:e],
                    mask_id=None if mask_id is None else mask_id[s:e],
                    rgb=None if rgb is None else rgb[s:e],
                    point_valid=None if point_valid is None else point_valid[s:e],
                    checkpoint_frame_tokenizer=checkpoint_frame_tokenizer,
                )
            )
        return torch.cat(outs, dim=0)


class _TokenSelfAttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = SelfAttention(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(mlp_mult) * d),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_mult) * d, d),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.ln1(x), mask=mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class _TokenSelfAttentionRefiner(nn.Module):
    def __init__(self, *, d: int, n_heads: int, n_layers: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TokenSelfAttentionBlock(
                    d=d,
                    n_heads=n_heads,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class PerceiverDemoQueryEncoderV2(ContextEncoder):
    def __init__(self, *, cfg: PerceiverDemoQueryEncoderV2Config, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)
        d = self.d_model

        self.demo_frame_stack = _FrameTokenizationStack(
            d=d,
            n_heads=int(cfg.demo_n_heads),
            m_frame_tokens=int(cfg.demo_m_frame_tokens),
            frame_tokenizer_layers=int(cfg.demo_frame_tokenizer_layers),
            state_dim=int(state_dim),
            mask_hash_buckets=int(cfg.mask_hash_buckets),
            use_mask_id=bool(cfg.use_mask_id),
            rgb_alpha_init=float(cfg.demo_rgb_alpha_init),
            dropout=float(cfg.dropout),
            use_gripper_point_features=bool(cfg.use_gripper_point_features),
            gripper_xyz_state_start=int(cfg.gripper_xyz_state_start),
            gripper_alpha_init=float(cfg.gripper_alpha_init),
        )
        self.query_frame_stack = _FrameTokenizationStack(
            d=d,
            n_heads=int(cfg.query_n_heads),
            m_frame_tokens=int(cfg.query_m_frame_tokens),
            frame_tokenizer_layers=int(cfg.query_frame_tokenizer_layers),
            state_dim=int(state_dim),
            mask_hash_buckets=int(cfg.mask_hash_buckets),
            use_mask_id=bool(cfg.use_mask_id),
            rgb_alpha_init=float(cfg.query_rgb_alpha_init),
            dropout=float(cfg.dropout),
            use_gripper_point_features=bool(cfg.use_gripper_point_features),
            gripper_xyz_state_start=int(cfg.gripper_xyz_state_start),
            gripper_alpha_init=float(cfg.gripper_alpha_init),
        )

        self.demo_id_embed = nn.Embedding(int(cfg.role_embed_max_K), d)
        self.keyframe_embed = nn.Embedding(int(cfg.role_embed_max_L), d)
        self.query_time_embed = nn.Embedding(int(cfg.role_embed_max_Tobs), d)

        self.demo_memory = None
        if bool(cfg.compress_demo_latents):
            self.demo_memory = DemoMemoryPerceiver(
                d=d,
                M=int(cfg.M_demo_latents),
                n_heads=int(cfg.n_heads),
                n_layers=int(cfg.demo_perceiver_layers),
                dropout=float(cfg.dropout),
            )

        self.demo_post_refiner = (
            _TokenSelfAttentionRefiner(
                d=d,
                n_heads=int(cfg.n_heads),
                n_layers=int(cfg.demo_post_self_attn_layers),
                mlp_mult=int(cfg.post_self_attn_mlp_mult),
                dropout=float(cfg.dropout),
            )
            if int(cfg.demo_post_self_attn_layers) > 0
            else None
        )
        self.query_post_refiner = (
            _TokenSelfAttentionRefiner(
                d=d,
                n_heads=int(cfg.n_heads),
                n_layers=int(cfg.query_post_self_attn_layers),
                mlp_mult=int(cfg.post_self_attn_mlp_mult),
                dropout=float(cfg.dropout),
            )
            if int(cfg.query_post_self_attn_layers) > 0
            else None
        )

    def _tokenize_with_stack(
        self,
        stack: _FrameTokenizationStack,
        *,
        xyz_f: torch.Tensor,
        state_f: torch.Tensor,
        mask_f: Optional[torch.Tensor],
        rgb_f: Optional[torch.Tensor],
        valid_f: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if bool(self.cfg.tokenize_frames_chunked):
            return stack.tokenize_frames_chunked(
                xyz=xyz_f,
                state=state_f,
                mask_id=mask_f,
                rgb=rgb_f,
                point_valid=valid_f,
                chunk_frames=int(self.cfg.chunk_frames),
                checkpoint_frame_tokenizer=bool(self.cfg.checkpoint_frame_tokenizer),
            )
        return stack.tokenize_frames(
            xyz=xyz_f,
            state=state_f,
            mask_id=mask_f,
            rgb=rgb_f,
            point_valid=valid_f,
            checkpoint_frame_tokenizer=bool(self.cfg.checkpoint_frame_tokenizer),
        )

    def _build_demo_memory(
        self,
        cond_xyz: torch.Tensor,
        cond_state: torch.Tensor,
        cond_rgb: Optional[torch.Tensor] = None,
        cond_mask_id: Optional[torch.Tensor] = None,
        cond_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, K, L, N, _ = cond_xyz.shape
        d = self.d_model
        if K > int(self.cfg.role_embed_max_K):
            raise ValueError(
                f"K={K} exceeds role_embed_max_K={int(self.cfg.role_embed_max_K)}. Increase cfg.role_embed_max_K."
            )
        if L > int(self.cfg.role_embed_max_L):
            raise ValueError(
                f"L={L} exceeds role_embed_max_L={int(self.cfg.role_embed_max_L)}. Increase cfg.role_embed_max_L."
            )

        xyz_f = cond_xyz.reshape(B * K * L, N, 3)
        state_f = cond_state.reshape(B * K * L, -1)
        rgb_f = cond_rgb.reshape(B * K * L, N, 3) if cond_rgb is not None else None
        mask_f = cond_mask_id.reshape(B * K * L, N) if cond_mask_id is not None else None
        valid_f = cond_valid.reshape(B * K * L, N).to(torch.bool) if cond_valid is not None else None

        frame_tokens = self._tokenize_with_stack(
            self.demo_frame_stack,
            xyz_f=xyz_f,
            state_f=state_f,
            mask_f=mask_f,
            rgb_f=rgb_f,
            valid_f=valid_f,
        )
        m1 = int(frame_tokens.shape[1])
        frame_tokens = frame_tokens.reshape(B, K, L, m1, d)

        demo_ids = torch.arange(K, device=cond_xyz.device).clamp_max(int(self.cfg.role_embed_max_K) - 1)
        keyframe_ids = torch.arange(L, device=cond_xyz.device).clamp_max(int(self.cfg.role_embed_max_L) - 1)
        frame_tokens = (
            frame_tokens
            + self.demo_id_embed(demo_ids).view(1, K, 1, 1, d)
            + self.keyframe_embed(keyframe_ids).view(1, 1, L, 1, d)
        )
        tokens = frame_tokens.reshape(B, K * L * m1, d)

        if bool(self.cfg.compress_demo_latents):
            if self.demo_memory is None:
                raise RuntimeError("compress_demo_latents=True but demo_memory is not initialized.")
            if bool(self.cfg.checkpoint_demo_memory) and self.training and torch.is_grad_enabled():
                z_demo = checkpoint(self.demo_memory, tokens, use_reentrant=False)
            else:
                z_demo = self.demo_memory(tokens)
        else:
            z_demo = tokens

        if self.demo_post_refiner is not None:
            z_demo = self.demo_post_refiner(z_demo)
        return z_demo

    def _build_query_tokens(
        self,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        query_rgb: Optional[torch.Tensor] = None,
        query_mask_id: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Tobs, N, _ = query_xyz.shape
        d = self.d_model
        if Tobs > int(self.cfg.role_embed_max_Tobs):
            raise ValueError(
                f"T_obs={Tobs} exceeds role_embed_max_Tobs={int(self.cfg.role_embed_max_Tobs)}. "
                "Increase cfg.role_embed_max_Tobs."
            )

        xyz_f = query_xyz.reshape(B * Tobs, N, 3)
        state_f = query_state.reshape(B * Tobs, -1)
        rgb_f = query_rgb.reshape(B * Tobs, N, 3) if query_rgb is not None else None
        mask_f = query_mask_id.reshape(B * Tobs, N) if query_mask_id is not None else None
        valid_f = query_valid.reshape(B * Tobs, N).to(torch.bool) if query_valid is not None else None

        frame_tokens = self._tokenize_with_stack(
            self.query_frame_stack,
            xyz_f=xyz_f,
            state_f=state_f,
            mask_f=mask_f,
            rgb_f=rgb_f,
            valid_f=valid_f,
        )
        m1 = int(frame_tokens.shape[1])
        frame_tokens = frame_tokens.reshape(B, Tobs, m1, d)
        t_ids = torch.arange(Tobs, device=query_xyz.device).clamp_max(int(self.cfg.role_embed_max_Tobs) - 1)
        frame_tokens = frame_tokens + self.query_time_embed(t_ids).view(1, Tobs, 1, d)
        z_query = frame_tokens.reshape(B, Tobs * m1, d)
        if self.query_post_refiner is not None:
            z_query = self.query_post_refiner(z_query)
        return z_query

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
        del cond_traj, cond_traj_mask

        z_query = self._build_query_tokens(
            query_xyz,
            query_state,
            query_rgb=query_rgb,
            query_mask_id=query_mask_id,
            query_valid=query_valid,
        )
        support_tokens: Optional[torch.Tensor] = None
        if bool(self.cfg.ignore_demos):
            ctx = z_query
        else:
            if cond_xyz is None or cond_state is None:
                raise ValueError("PerceiverDemoQueryEncoderV2 requires cond_xyz and cond_state when ignore_demos=False.")
            use_demo_ckpt = bool(
                self.cfg.checkpoint_build_demo_memory and self.training and torch.is_grad_enabled()
            )
            if use_demo_ckpt:
                z_demo = checkpoint(
                    lambda cond_xyz_, cond_state_: self._build_demo_memory(
                        cond_xyz_,
                        cond_state_,
                        cond_rgb=cond_rgb,
                        cond_mask_id=cond_mask_id,
                        cond_valid=cond_valid,
                    ),
                    cond_xyz,
                    cond_state,
                    use_reentrant=False,
                )
            else:
                z_demo = self._build_demo_memory(
                    cond_xyz,
                    cond_state,
                    cond_rgb=cond_rgb,
                    cond_mask_id=cond_mask_id,
                    cond_valid=cond_valid,
                )
            support_tokens = z_demo
            ctx = torch.cat([z_demo, z_query], dim=1)

        return ContextEncoderOutput(
            tokens=ctx,
            token_mask=None,
            support_tokens=support_tokens,
            support_token_mask=None,
            query_tokens=z_query,
            query_token_mask=None,
        )
