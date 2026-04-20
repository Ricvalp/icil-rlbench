from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from icil.models.common.attention import SelfAttention
from icil.models.common.perceiver import FramePerceiverTokenizer


@dataclass
class SupernodeFrameTokenizerConfig:
    d_model: int = 512
    n_heads: int = 4
    dropout: float = 0.0
    num_supernodes: int = 128
    frame_tokens_out: int = 128
    neighbors_per_supernode: int = 32
    supernode_refine_layers: int = 1
    compress_supernodes: bool = True
    supernode_pool_layers: int = 1
    use_mask_id: bool = True
    use_mask_embedding: bool = False
    mask_hash_buckets: int = 1
    supernode_sampling_mode: str = "fps"
    attention_backend: str = "manual"
    use_mask_instance_quota: bool = True
    min_mask_supernodes: int = 4
    use_gripper_point_features: bool = False
    gripper_xyz_state_start: int = 0
    gripper_alpha_init: float = 1.0
    min_gripper_supernodes: int = 2
    gripper_sampling_radius: float = 0.10
    rgb_alpha_init: float = 1.0


class SupernodeSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
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


class SupernodeRefiner(nn.Module):
    def __init__(self, *, d: int, n_heads: int, n_layers: int, dropout: float, attention_backend: str = "manual"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SupernodeSelfAttentionBlock(
                    d=d,
                    n_heads=n_heads,
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


def _first_valid_index(valid: torch.Tensor) -> int:
    idx = torch.nonzero(valid, as_tuple=False).flatten()
    return int(idx[0].item()) if idx.numel() > 0 else 0


def _append_unique(
    selected: list[int],
    buckets: list[int],
    candidate_indices: torch.Tensor,
    bucket_id: int,
    max_count: int,
) -> None:
    selected_set = set(selected)
    for idx_t in candidate_indices.flatten():
        if len(selected) >= int(max_count):
            break
        idx = int(idx_t.item())
        if idx in selected_set:
            continue
        selected.append(idx)
        buckets.append(int(bucket_id))
        selected_set.add(idx)


def _fps_fill_single(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    selected: list[int],
    buckets: list[int],
    *,
    target_count: int,
    bucket_id: int = 0,
) -> None:
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        while len(selected) < int(target_count):
            selected.append(0)
            buckets.append(-1)
        return

    selected_set = set(selected)
    if not selected:
        pts_valid = xyz[valid_idx]
        centroid = pts_valid.mean(dim=0, keepdim=True)
        dist = torch.norm(pts_valid - centroid, dim=-1)
        first = valid_idx[int(torch.argmax(dist).item())]
        selected.append(int(first.item()))
        buckets.append(int(bucket_id))
        selected_set.add(int(first.item()))

    while len(selected) < int(target_count):
        remaining = [int(idx.item()) for idx in valid_idx if int(idx.item()) not in selected_set]
        if not remaining:
            selected.append(selected[0])
            buckets.append(-1)
            continue
        rem_t = torch.tensor(remaining, device=xyz.device, dtype=torch.long)
        sel_t = torch.tensor(selected, device=xyz.device, dtype=torch.long)
        rem_xyz = xyz[rem_t]
        sel_xyz = xyz[sel_t]
        dist = torch.cdist(rem_xyz.unsqueeze(0), sel_xyz.unsqueeze(0)).squeeze(0)
        min_dist = dist.min(dim=1).values
        chosen = int(rem_t[int(torch.argmax(min_dist).item())].item())
        selected.append(chosen)
        buckets.append(int(bucket_id))
        selected_set.add(chosen)


def quota_based_supernode_sampling(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_supernodes: int,
    state: Optional[torch.Tensor] = None,
    mask_id: Optional[torch.Tensor] = None,
    use_mask_instance_quota: bool = True,
    min_mask_supernodes: int = 0,
    min_gripper_supernodes: int = 0,
    gripper_xyz_state_start: int = 0,
    gripper_sampling_radius: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sampled point indices, supernode validity, and bucket labels.

    Bucket labels:
      0 = global FPS fill
      1 = gripper quota
      2 = mask-instance quota
     -1 = duplicate/padded fallback
    """
    if xyz.ndim != 3 or valid.ndim != 2:
        raise ValueError(f"Expected xyz [Bf,N,3] and valid [Bf,N], got {tuple(xyz.shape)}, {tuple(valid.shape)}")
    Bf, N, _ = xyz.shape
    M = int(num_supernodes)
    out_idx = torch.zeros((Bf, M), device=xyz.device, dtype=torch.long)
    out_valid = torch.zeros((Bf, M), device=xyz.device, dtype=torch.bool)
    out_bucket = torch.full((Bf, M), -1, device=xyz.device, dtype=torch.long)

    valid = valid.to(torch.bool)
    for b in range(Bf):
        vb = valid[b]
        selected: list[int] = []
        buckets: list[int] = []
        if not bool(vb.any().item()):
            continue

        if int(min_gripper_supernodes) > 0 and state is not None:
            start = int(gripper_xyz_state_start)
            if int(state.shape[-1]) >= start + 3:
                gripper_xyz = state[b, start:start + 3].to(device=xyz.device, dtype=xyz.dtype)
                dist = torch.norm(xyz[b] - gripper_xyz.view(1, 3), dim=-1)
                eligible = vb
                if float(gripper_sampling_radius) > 0.0:
                    in_radius = dist <= float(gripper_sampling_radius)
                    if bool((vb & in_radius).any().item()):
                        eligible = vb & in_radius
                dist_rank = dist.masked_fill(~eligible, torch.inf)
                take = min(int(min_gripper_supernodes), M)
                nearest = torch.argsort(dist_rank)[:take]
                _append_unique(selected, buckets, nearest, bucket_id=1, max_count=M)

        if bool(use_mask_instance_quota) and mask_id is not None and len(selected) < M:
            mask_b = mask_id[b]
            unique_ids = torch.unique(mask_b[vb])
            unique_ids = unique_ids[torch.argsort(unique_ids)]
            if unique_ids.numel() > 0:
                selected_set = set(selected)
                mask_selected = 0

                # First pass: cover every distinct mask id if there is enough room.
                for mid in unique_ids:
                    if len(selected) >= M:
                        break
                    candidates = torch.nonzero(vb & (mask_b == mid), as_tuple=False).flatten()
                    candidates = candidates[torch.argsort(candidates)]
                    for idx_t in candidates:
                        idx = int(idx_t.item())
                        if idx in selected_set:
                            continue
                        selected.append(idx)
                        buckets.append(2)
                        selected_set.add(idx)
                        mask_selected += 1
                        break

                # Optional extra mask quota, spread round-robin over visible ids.
                quota = min(
                    max(mask_selected, int(min_mask_supernodes)),
                    M - (len(selected) - mask_selected),
                )
                cursor = 0
                while mask_selected < quota and len(selected) < M:
                    mid = unique_ids[cursor % int(unique_ids.numel())]
                    candidates = torch.nonzero(vb & (mask_b == mid), as_tuple=False).flatten()
                    candidates = candidates[torch.argsort(candidates)]
                    for idx_t in candidates:
                        idx = int(idx_t.item())
                        if idx in selected_set:
                            continue
                        selected.append(idx)
                        buckets.append(2)
                        selected_set.add(idx)
                        mask_selected += 1
                        break
                    cursor += 1
                    if cursor > int(unique_ids.numel()) * max(2, quota + 1):
                        break

        _fps_fill_single(xyz[b], vb, selected, buckets, target_count=M, bucket_id=0)
        idx_t = torch.tensor(selected[:M], device=xyz.device, dtype=torch.long)
        bucket_t = torch.tensor(buckets[:M], device=xyz.device, dtype=torch.long)
        out_idx[b] = idx_t
        out_bucket[b] = bucket_t
        out_valid[b] = vb[idx_t]

    return out_idx, out_valid, out_bucket


def _topk_with_pad(
    scores: torch.Tensor,
    *,
    k: int,
    largest: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N = scores.shape
    if int(k) <= 0:
        idx = torch.empty((B, 0), device=scores.device, dtype=torch.long)
        vals = torch.empty((B, 0), device=scores.device, dtype=scores.dtype)
        return vals, idx
    k_eff = min(int(k), N)
    vals, idx = torch.topk(scores, k=k_eff, dim=1, largest=largest)
    if k_eff < int(k):
        pad = int(k) - k_eff
        pad_val = torch.inf if not largest else -torch.inf
        vals = torch.cat([vals, vals.new_full((B, pad), pad_val)], dim=1)
        idx = torch.cat([idx, idx.new_zeros((B, pad))], dim=1)
    return vals, idx


def _scatter_selected(valid: torch.Tensor, idx: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(valid, dtype=torch.bool)
    if idx.numel() == 0:
        return selected
    selected.scatter_(1, idx, keep.to(torch.bool))
    return selected


def fast_quota_based_supernode_sampling(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_supernodes: int,
    state: Optional[torch.Tensor] = None,
    mask_id: Optional[torch.Tensor] = None,
    use_mask_instance_quota: bool = True,
    min_mask_supernodes: int = 0,
    min_gripper_supernodes: int = 0,
    gripper_xyz_state_start: int = 0,
    gripper_sampling_radius: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched approximate supernode sampling.

    This keeps the gripper and mask-id quota structure but replaces the expensive
    iterative FPS fill with batched random valid-point selection.
    """
    if xyz.ndim != 3 or valid.ndim != 2:
        raise ValueError(f"Expected xyz [Bf,N,3] and valid [Bf,N], got {tuple(xyz.shape)}, {tuple(valid.shape)}")
    Bf, N, _ = xyz.shape
    M = int(num_supernodes)
    device = xyz.device
    valid = valid.to(torch.bool)
    row_has_valid = valid.any(dim=1)
    first_valid = torch.argmax(valid.to(torch.long), dim=1)

    out_idx = first_valid.view(Bf, 1).expand(Bf, M).clone()
    out_bucket = torch.full((Bf, M), -1, device=device, dtype=torch.long)
    selected = torch.zeros((Bf, N), device=device, dtype=torch.bool)

    cursor = 0
    k_grip = 0
    if int(min_gripper_supernodes) > 0 and state is not None:
        start = int(gripper_xyz_state_start)
        if int(state.shape[-1]) >= start + 3:
            k_grip = min(int(min_gripper_supernodes), M)
            gripper_xyz = state[:, start:start + 3].to(device=device, dtype=xyz.dtype)
            dist = torch.norm(xyz - gripper_xyz[:, None, :], dim=-1)
            eligible = valid
            if float(gripper_sampling_radius) > 0.0:
                in_radius = dist <= float(gripper_sampling_radius)
                has_radius = (valid & in_radius).any(dim=1, keepdim=True)
                eligible = valid & (in_radius | ~has_radius)
            dist_rank = dist.masked_fill(~eligible, torch.inf)
            grip_vals, grip_idx = _topk_with_pad(dist_rank, k=k_grip, largest=False)
            grip_keep = torch.isfinite(grip_vals) & row_has_valid[:, None]
            out_idx[:, :k_grip] = torch.where(grip_keep, grip_idx, first_valid[:, None])
            out_bucket[:, :k_grip] = torch.where(
                grip_keep,
                torch.full_like(grip_idx, 1),
                torch.full_like(grip_idx, -1),
            )
            selected = selected | _scatter_selected(valid, grip_idx, grip_keep)
            cursor = k_grip

    remaining = M - cursor
    if remaining > 0:
        mask_keep = torch.zeros((Bf, remaining), device=device, dtype=torch.bool)
        mask_idx = first_valid.view(Bf, 1).expand(Bf, remaining).clone()

        if bool(use_mask_instance_quota) and mask_id is not None:
            mask_id = mask_id.to(device=device, dtype=torch.long)
            mask_candidate_valid = valid & ~selected
            sentinel = torch.iinfo(torch.long).max
            mask_key = torch.where(mask_candidate_valid, mask_id, torch.full_like(mask_id, sentinel))
            sort_idx = torch.argsort(mask_key, dim=1)
            sorted_valid = torch.gather(mask_candidate_valid, 1, sort_idx)
            sorted_mask = torch.gather(mask_id, 1, sort_idx)
            first_for_id = sorted_valid.clone()
            if N > 1:
                first_for_id[:, 1:] = first_for_id[:, 1:] & (sorted_mask[:, 1:] != sorted_mask[:, :-1])
            pos = torch.arange(N, device=device, dtype=torch.float32).view(1, N)
            mask_scores = torch.where(first_for_id, -pos, torch.full_like(pos.expand(Bf, N), -torch.inf))
            mask_vals, mask_pos = _topk_with_pad(mask_scores, k=remaining, largest=True)
            mask_idx = torch.gather(sort_idx, 1, mask_pos)
            mask_keep = torch.isfinite(mask_vals) & row_has_valid[:, None]

            selected_after_mask = selected | _scatter_selected(valid, mask_idx, mask_keep)
            k_extra = min(max(0, int(min_mask_supernodes)), remaining)
            if k_extra > 0:
                extra_scores = torch.rand((Bf, N), device=device, dtype=torch.float32)
                extra_scores = extra_scores.masked_fill(~(valid & ~selected_after_mask), -torch.inf)
                extra_vals, extra_idx = _topk_with_pad(extra_scores, k=k_extra, largest=True)
                extra_keep = torch.isfinite(extra_vals) & row_has_valid[:, None]
                use_extra = (~mask_keep[:, :k_extra]) & extra_keep
                mask_idx[:, :k_extra] = torch.where(use_extra, extra_idx, mask_idx[:, :k_extra])
                mask_keep[:, :k_extra] = mask_keep[:, :k_extra] | use_extra

        selected_for_global = selected | _scatter_selected(valid, mask_idx, mask_keep)
        global_scores = torch.rand((Bf, N), device=device, dtype=torch.float32)
        global_scores = global_scores.masked_fill(~(valid & ~selected_for_global), -torch.inf)
        global_vals, global_idx = _topk_with_pad(global_scores, k=remaining, largest=True)
        global_keep = torch.isfinite(global_vals) & row_has_valid[:, None]

        fill_idx = torch.where(mask_keep, mask_idx, torch.where(global_keep, global_idx, first_valid[:, None]))
        fill_bucket = torch.where(
            mask_keep,
            torch.full_like(fill_idx, 2),
            torch.where(global_keep, torch.full_like(fill_idx, 0), torch.full_like(fill_idx, -1)),
        )
        out_idx[:, cursor:] = fill_idx
        out_bucket[:, cursor:] = fill_bucket

    out_valid = torch.gather(valid, 1, out_idx) & row_has_valid[:, None]
    out_bucket = torch.where(out_valid, out_bucket, torch.full_like(out_bucket, -1))
    return out_idx, out_valid, out_bucket


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or idx.ndim != 2:
        raise ValueError(f"Expected x [B,N,C], idx [B,M], got {tuple(x.shape)}, {tuple(idx.shape)}")
    B, _, C = x.shape
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(B, idx.shape[1], C))


def build_knn_neighbors(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    supernode_xyz: torch.Tensor,
    *,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Bf, N, _ = xyz.shape
    M = int(supernode_xyz.shape[1])
    k_eff = max(1, min(int(k), N))
    dist = torch.cdist(supernode_xyz.float(), xyz.float())
    dist = dist.masked_fill(~valid.to(torch.bool).unsqueeze(1), torch.inf)
    vals, idx = torch.topk(dist, k=k_eff, dim=-1, largest=False)
    neighbor_mask = torch.isfinite(vals)
    if k_eff < int(k):
        pad = int(k) - k_eff
        idx = torch.cat([idx, idx.new_zeros((Bf, M, pad))], dim=-1)
        neighbor_mask = torch.cat([neighbor_mask, neighbor_mask.new_zeros((Bf, M, pad))], dim=-1)
    return idx, neighbor_mask


def gather_neighbors(x: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    _, M, K = neighbor_idx.shape
    x_exp = x.unsqueeze(1).expand(B, M, N, C)
    idx_exp = neighbor_idx.unsqueeze(-1).expand(B, M, K, C)
    return torch.gather(x_exp, 2, idx_exp)


class PointToSupernodeMessagePassing(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(int(d) + 4, int(d)),
            nn.SiLU(),
            nn.Linear(int(d), int(d)),
        )
        self.out_norm = nn.LayerNorm(int(d))

    def forward(
        self,
        *,
        point_feat: torch.Tensor,
        point_xyz: torch.Tensor,
        supernode_xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        supernode_init_feat: torch.Tensor,
    ) -> torch.Tensor:
        neigh_feat = gather_neighbors(point_feat, neighbor_idx)
        neigh_xyz = gather_neighbors(point_xyz, neighbor_idx)
        rel = neigh_xyz - supernode_xyz.unsqueeze(2)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        edge_feat = torch.cat([neigh_feat, rel.to(dtype=neigh_feat.dtype), dist.to(dtype=neigh_feat.dtype)], dim=-1)
        msg = self.edge_mlp(edge_feat)
        mask_f = neighbor_mask.to(dtype=msg.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        agg = (msg * mask_f).sum(dim=2) / denom
        return self.out_norm(supernode_init_feat + agg)


class SupernodeFrameTokenizer(nn.Module):
    def __init__(self, *, cfg: SupernodeFrameTokenizerConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)
        self.d_model = d
        self.num_supernodes = int(cfg.num_supernodes)
        self.frame_tokens_out = int(cfg.frame_tokens_out)
        self.neighbors_per_supernode = int(cfg.neighbors_per_supernode)
        self.use_mask_id = bool(cfg.use_mask_id)
        self.use_mask_embedding = bool(cfg.use_mask_embedding)
        self.use_gripper_point_features = bool(cfg.use_gripper_point_features)
        self.gripper_xyz_state_start = int(cfg.gripper_xyz_state_start)
        self.supernode_sampling_mode = str(cfg.supernode_sampling_mode).lower()
        if self.supernode_sampling_mode not in {"fps", "exact_fps", "fast", "fast_random"}:
            raise ValueError(
                "cfg.supernode_sampling_mode must be one of: 'fps', 'exact_fps', 'fast', 'fast_random'. "
                f"Got {cfg.supernode_sampling_mode!r}."
            )

        self.xyz_proj = nn.Linear(3, d, bias=False)
        self.rgb_proj = nn.Linear(3, d, bias=False)
        self.rgb_alpha = nn.Parameter(torch.tensor(float(cfg.rgb_alpha_init), dtype=torch.float32))
        self.mask_embed = (
            nn.Embedding(int(cfg.mask_hash_buckets), d)
            if bool(cfg.use_mask_embedding)
            else None
        )
        self.gripper_proj = (
            nn.Linear(4, d, bias=False) if bool(cfg.use_gripper_point_features) else None
        )
        self.gripper_alpha = (
            nn.Parameter(torch.tensor(float(cfg.gripper_alpha_init), dtype=torch.float32))
            if bool(cfg.use_gripper_point_features)
            else None
        )
        self.state_proj = nn.Sequential(
            nn.Linear(int(state_dim), d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.message_passing = PointToSupernodeMessagePassing(d)
        self.refiner = SupernodeRefiner(
            d=d,
            n_heads=int(cfg.n_heads),
            n_layers=int(cfg.supernode_refine_layers),
            dropout=float(cfg.dropout),
            attention_backend=str(cfg.attention_backend),
        )
        self.pool = (
            FramePerceiverTokenizer(
                d=d,
                m=int(cfg.frame_tokens_out),
                n_heads=int(cfg.n_heads),
                n_layers=int(cfg.supernode_pool_layers),
                dropout=float(cfg.dropout),
                attention_backend=str(cfg.attention_backend),
            )
            if bool(cfg.compress_supernodes)
            else None
        )

    def _hash_mask_ids(self, mask_id: torch.Tensor) -> torch.Tensor:
        if self.mask_embed is None:
            raise RuntimeError("mask embedding requested but mask_embed is not initialized")
        return torch.remainder(mask_id, self.mask_embed.num_embeddings)

    def _gripper_xyz_from_state(self, state: torch.Tensor) -> torch.Tensor:
        start = int(self.gripper_xyz_state_start)
        end = start + 3
        if int(state.shape[-1]) < end:
            raise ValueError(
                f"state_dim={int(state.shape[-1])} is too small for gripper_xyz_state_start={start}."
            )
        return state[..., start:end]

    def build_point_features(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        rgb: Optional[torch.Tensor],
        mask_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = self.xyz_proj(xyz)
        if rgb is not None:
            h = h + self.rgb_alpha.to(dtype=h.dtype) * self.rgb_proj(rgb.to(device=xyz.device, dtype=xyz.dtype))
        if self.use_gripper_point_features:
            if self.gripper_proj is None or self.gripper_alpha is None:
                raise RuntimeError("gripper point feature modules were not initialized")
            gripper_xyz = self._gripper_xyz_from_state(state).to(device=xyz.device, dtype=xyz.dtype)
            rel = xyz - gripper_xyz.unsqueeze(1)
            dist = torch.norm(rel, dim=-1, keepdim=True)
            grip_feat = torch.cat([rel, dist], dim=-1)
            h = h + self.gripper_alpha.to(dtype=h.dtype) * self.gripper_proj(grip_feat)
        if self.use_mask_embedding and mask_id is not None:
            if self.mask_embed is None:
                raise RuntimeError("use_mask_embedding=True but mask_embed is not initialized")
            h = h + self.mask_embed(self._hash_mask_ids(mask_id.to(device=xyz.device)))
        return h

    def forward(
        self,
        *,
        xyz: torch.Tensor,
        valid: torch.Tensor,
        state: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        mask_id: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        valid = valid.to(device=xyz.device, dtype=torch.bool)
        if mask_id is not None:
            mask_id = mask_id.to(device=xyz.device, dtype=torch.long)
        h_points = self.build_point_features(xyz=xyz, state=state, rgb=rgb, mask_id=mask_id)
        sampler = (
            fast_quota_based_supernode_sampling
            if self.supernode_sampling_mode in {"fast", "fast_random"}
            else quota_based_supernode_sampling
        )
        sample_idx, supernode_mask, sample_bucket = sampler(
            xyz,
            valid,
            num_supernodes=self.num_supernodes,
            state=state,
            mask_id=mask_id if self.use_mask_id else None,
            use_mask_instance_quota=bool(self.cfg.use_mask_instance_quota),
            min_mask_supernodes=int(self.cfg.min_mask_supernodes),
            min_gripper_supernodes=int(self.cfg.min_gripper_supernodes),
            gripper_xyz_state_start=int(self.cfg.gripper_xyz_state_start),
            gripper_sampling_radius=float(self.cfg.gripper_sampling_radius),
        )
        supernode_xyz = gather_points(xyz, sample_idx)
        supernode_init_feat = gather_points(h_points, sample_idx)
        neighbor_idx, neighbor_mask = build_knn_neighbors(
            xyz,
            valid,
            supernode_xyz,
            k=self.neighbors_per_supernode,
        )
        h_super = self.message_passing(
            point_feat=h_points,
            point_xyz=xyz,
            supernode_xyz=supernode_xyz,
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            supernode_init_feat=supernode_init_feat,
        )
        h_super = self.refiner(h_super, mask=supernode_mask)
        if self.pool is not None:
            visual_tokens = self.pool(h_super, point_mask=supernode_mask)
        else:
            visual_tokens = h_super
        state_token = self.state_proj(state).unsqueeze(1)
        frame_tokens = torch.cat([visual_tokens, state_token], dim=1)

        if not return_diagnostics:
            return frame_tokens
        diagnostics: Dict[str, Any] = {
            "supernode_idx": sample_idx.detach(),
            "supernode_xyz": supernode_xyz.detach(),
            "supernode_mask": supernode_mask.detach(),
            "sample_bucket": sample_bucket.detach(),
            "neighbor_idx": neighbor_idx.detach(),
            "neighbor_mask": neighbor_mask.detach(),
            "neighbor_coverage": neighbor_mask.detach().sum(dim=-1),
        }
        return frame_tokens, diagnostics
