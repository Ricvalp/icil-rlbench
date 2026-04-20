from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from icil.models.common import DemoMemoryPerceiver, SupernodeFrameTokenizer, SupernodeFrameTokenizerConfig
from icil.models.common.attention import SelfAttention
from icil.models.encoders.base import ContextEncoder, ContextEncoderOutput
from icil.models.encoders.perceiver_demo_query_v2 import PerceiverDemoQueryEncoderV2Config


@dataclass
class PerceiverDemoQuerySupernodeEncoderV2Config(PerceiverDemoQueryEncoderV2Config):
    # Request mask ids from the dataset for quota sampling. Mask embeddings remain separate.
    use_mask_id: bool = True
    use_mask_embedding: bool = False
    use_mask_instance_quota: bool = True
    supernode_sampling_mode: str = "fps"

    demo_supernodes: int = 128
    query_supernodes: int = 128
    demo_frame_tokens_out: int = 64
    query_frame_tokens_out: int = 128
    neighbors_per_supernode: int = 32
    demo_supernode_refine_layers: int = 1
    query_supernode_refine_layers: int = 2
    compress_supernodes_demo: bool = True
    compress_supernodes_query: bool = True
    supernode_pool_layers: int = 1
    min_gripper_supernodes: int = 2
    min_mask_supernodes: int = 4
    gripper_sampling_radius: float = 0.10
    use_gripper_point_features: bool = True
    gripper_xyz_state_start: int = 0
    gripper_alpha_init: float = 1.0
    chunk_frames: int = 32
    tokenize_frames_chunked: bool = True


class _TokenSelfAttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_mult: int, dropout: float, attention_backend: str = "manual"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)
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
    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        dropout: float,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TokenSelfAttentionBlock(
                    d=d,
                    n_heads=n_heads,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                    attention_backend=attention_backend,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class _SupernodeFrameTokenizationStack(nn.Module):
    def __init__(
        self,
        *,
        cfg: PerceiverDemoQuerySupernodeEncoderV2Config,
        state_dim: int,
        branch: str,
    ):
        super().__init__()
        if branch not in {"demo", "query"}:
            raise ValueError(f"Unsupported branch={branch!r}.")
        if branch == "demo":
            num_supernodes = int(cfg.demo_supernodes)
            frame_tokens_out = int(cfg.demo_frame_tokens_out)
            n_heads = int(cfg.demo_n_heads)
            refine_layers = int(cfg.demo_supernode_refine_layers)
            compress = bool(cfg.compress_supernodes_demo)
            rgb_alpha_init = float(cfg.demo_rgb_alpha_init)
        else:
            num_supernodes = int(cfg.query_supernodes)
            frame_tokens_out = int(cfg.query_frame_tokens_out)
            n_heads = int(cfg.query_n_heads)
            refine_layers = int(cfg.query_supernode_refine_layers)
            compress = bool(cfg.compress_supernodes_query)
            rgb_alpha_init = float(cfg.query_rgb_alpha_init)

        tokenizer_cfg = SupernodeFrameTokenizerConfig(
            d_model=int(cfg.d_model),
            n_heads=n_heads,
            dropout=float(cfg.dropout),
            num_supernodes=num_supernodes,
            frame_tokens_out=frame_tokens_out,
            neighbors_per_supernode=int(cfg.neighbors_per_supernode),
            supernode_refine_layers=refine_layers,
            compress_supernodes=compress,
            supernode_pool_layers=int(cfg.supernode_pool_layers),
            use_mask_id=bool(cfg.use_mask_id),
            use_mask_embedding=bool(cfg.use_mask_embedding),
            mask_hash_buckets=int(cfg.mask_hash_buckets),
            supernode_sampling_mode=str(cfg.supernode_sampling_mode),
            attention_backend=str(cfg.attention_backend),
            use_mask_instance_quota=bool(cfg.use_mask_instance_quota),
            min_mask_supernodes=int(cfg.min_mask_supernodes),
            use_gripper_point_features=bool(cfg.use_gripper_point_features),
            gripper_xyz_state_start=int(cfg.gripper_xyz_state_start),
            gripper_alpha_init=float(cfg.gripper_alpha_init),
            min_gripper_supernodes=int(cfg.min_gripper_supernodes),
            gripper_sampling_radius=float(cfg.gripper_sampling_radius),
            rgb_alpha_init=rgb_alpha_init,
        )
        self.tokenizer = SupernodeFrameTokenizer(cfg=tokenizer_cfg, state_dim=int(state_dim))
        self.checkpoint_frame_tokenizer = bool(cfg.checkpoint_frame_tokenizer)

    def tokenize_frames(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        mask_id: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor] = None,
        point_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if point_valid is None:
            point_valid = torch.ones(xyz.shape[:2], device=xyz.device, dtype=torch.bool)
        if self.checkpoint_frame_tokenizer and self.training and torch.is_grad_enabled():
            has_rgb = rgb is not None
            has_mask = mask_id is not None
            rgb_arg = rgb if has_rgb else xyz.new_empty((0,))
            mask_arg = mask_id if has_mask else torch.empty((0,), device=xyz.device, dtype=torch.long)

            def _forward(
                xyz_: torch.Tensor,
                state_: torch.Tensor,
                valid_: torch.Tensor,
                rgb_: torch.Tensor,
                mask_: torch.Tensor,
            ) -> torch.Tensor:
                return self.tokenizer(
                    xyz=xyz_,
                    valid=valid_,
                    state=state_,
                    rgb=rgb_ if has_rgb else None,
                    mask_id=mask_ if has_mask else None,
                )

            return checkpoint(_forward, xyz, state, point_valid, rgb_arg, mask_arg, use_reentrant=False)

        return self.tokenizer(xyz=xyz, valid=point_valid, state=state, rgb=rgb, mask_id=mask_id)

    def tokenize_frames_chunked(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        mask_id: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor] = None,
        point_valid: Optional[torch.Tensor] = None,
        chunk_frames: int,
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
                )
            )
        return torch.cat(outs, dim=0)


class PerceiverDemoQuerySupernodeEncoderV2(ContextEncoder):
    def __init__(self, *, cfg: PerceiverDemoQuerySupernodeEncoderV2Config, state_dim: int, action_dim: int):
        super().__init__()
        del action_dim
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.d_model = int(cfg.d_model)
        d = self.d_model

        self.demo_frame_stack = _SupernodeFrameTokenizationStack(cfg=cfg, state_dim=state_dim, branch="demo")
        self.query_frame_stack = _SupernodeFrameTokenizationStack(cfg=cfg, state_dim=state_dim, branch="query")

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
                attention_backend=str(cfg.attention_backend),
            )

        self.demo_post_refiner = (
            _TokenSelfAttentionRefiner(
                d=d,
                n_heads=int(cfg.n_heads),
                n_layers=int(cfg.demo_post_self_attn_layers),
                mlp_mult=int(cfg.post_self_attn_mlp_mult),
                dropout=float(cfg.dropout),
                attention_backend=str(cfg.attention_backend),
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
                attention_backend=str(cfg.attention_backend),
            )
            if int(cfg.query_post_self_attn_layers) > 0
            else None
        )

    def _tokenize_with_stack(
        self,
        stack: _SupernodeFrameTokenizationStack,
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
            )
        return stack.tokenize_frames(
            xyz=xyz_f,
            state=state_f,
            mask_id=mask_f,
            rgb=rgb_f,
            point_valid=valid_f,
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
                raise ValueError("PerceiverDemoQuerySupernodeEncoderV2 requires cond_xyz and cond_state when ignore_demos=False.")
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
