from __future__ import annotations


from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from icil.models.common import SparseVoxelConvTokenizer, DemoMemoryPerceiver
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput


@dataclass
class Conv3dDemoQueryEncoderConfig:
    d_model: int = 512
    n_heads: int = 8
    m_frame_tokens: int = 64
    max_voxels: int = 4096
    voxel_size: float = 0.01
    use_learned_topk: bool = True
    n_mix_layers: int = 2
    M_demo_latents: int = 256
    demo_perceiver_layers: int = 3
    mask_hash_buckets: int = 2048
    use_mask_id: bool = True
    role_embed_max_K: int = 32
    role_embed_max_L: int = 64
    role_embed_max_Tobs: int = 16
    rgb_alpha_init: float = 1.0
    dropout: float = 0.0
    attention_backend: str = "manual"
    ignore_demos: bool = False

class Conv3dDemoQueryEncoder(ContextEncoder):

    def __init__(self, *, cfg: Conv3dDemoQueryEncoderConfig, state_dim: int, action_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.state_dim = state_dim
        self.d_model = d
        
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

        # frame tokenizer (Sparse 3D convolution)
        self.frame_tokenizer = SparseVoxelConvTokenizer(
            d=d, m=cfg.m_frame_tokens, voxel_size=cfg.voxel_size,
            n_mix_layers=cfg.n_mix_layers, dropout=cfg.dropout, max_voxels=cfg.max_voxels,
            use_learned_topk=cfg.use_learned_topk
        )

        # demo memory perceiver
        self.demo_memory = DemoMemoryPerceiver(
            d=d, M=cfg.M_demo_latents, n_heads=cfg.n_heads,
            n_layers=cfg.demo_perceiver_layers, dropout=cfg.dropout,
            attention_backend=str(cfg.attention_backend),
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
        z = self.frame_tokenizer(pt, xyz, point_mask=point_valid)  # [Bf,m,d]
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
        del cond_traj, cond_traj_mask  # Not used in this encoder.

        Z_query = self._build_query_tokens(
            query_xyz, query_state, query_rgb=query_rgb, query_mask_id=query_mask_id, query_valid=query_valid
        )  # [B,Sq,d]
        support_tokens: Optional[torch.Tensor] = None
        # Optional ablation: ignore support demos and condition only on query tokens.
        if bool(getattr(self.cfg, "ignore_demos", False)):
            ctx = Z_query
        else:
            if cond_xyz is None or cond_state is None:
                raise ValueError("Conv3DDemoQueryEncoder requires cond_xyz and cond_state when ignore_demos=False.")
            Z_demo = self._build_demo_memory(
                cond_xyz, cond_state, cond_rgb=cond_rgb, cond_mask_id=cond_mask_id, cond_valid=cond_valid
            )  # [B,M,d]
            support_tokens = Z_demo
            ctx = torch.cat([Z_demo, Z_query], dim=1)  # [B, M+Sq, d]

        return ContextEncoderOutput(
            tokens=ctx,
            token_mask=None,
            support_tokens=support_tokens,
            support_token_mask=None,
            query_tokens=Z_query,
            query_token_mask=None,
        )
