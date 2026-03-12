from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseVoxelConvTokenizer(nn.Module):
    """
    Voxelize points -> voxel features -> lightweight 3D-conv-like mixing -> select m tokens.

    Contract:
      input:
        point_tokens: [Bf, N, d]   (your per-point embeddings)
        xyz:          [Bf, N, 3]   (world coords used for voxelization)
        point_mask:   [Bf, N] bool True=keep (optional)
      output:
        tokens:       [Bf, m, d]
    Notes:
      - This is "conv-like" without external sparse-conv deps.
      - For true sparse 3D convs (MinkowskiEngine), the interface stays the same.
    """

    def __init__(
        self,
        *,
        d: int,
        m: int,
        voxel_size: float = 0.01,
        n_mix_layers: int = 2,
        dropout: float = 0.0,
        max_voxels: int = 4096,  # cap to keep worst-case bounded
        use_learned_topk: bool = True,
    ):
        super().__init__()
        self.d = int(d)
        self.m = int(m)
        self.voxel_size = float(voxel_size)
        self.max_voxels = int(max_voxels)
        self.use_learned_topk = bool(use_learned_topk)

        # "conv-like" mixing: repeat neighbor aggregation + MLP
        self.mix_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d),
                nn.Linear(self.d, 2 * self.d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * self.d, self.d),
            )
            for _ in range(int(n_mix_layers))
        ])

        # scoring for top-k token selection
        self.score = nn.Linear(self.d, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _hash_coords(c: torch.Tensor) -> torch.Tensor:
        """
        c: [V,3] int64
        returns: [V] int64 hash
        """
        # Large-ish primes; good enough for integer grid hashing
        return c[:, 0] * 73856093 + c[:, 1] * 19349663 + c[:, 2] * 83492791

    def _voxelize_one(
        self,
        xyz: torch.Tensor,        # [N,3]
        feat: torch.Tensor,       # [N,d]
        mask: torch.Tensor,       # [N] bool
    ):
        """
        Returns voxel features and voxel coords:
          vfeat: [V,d]
          vcoord: [V,3] int64
          vcount: [V] int64
        """
        vs = self.voxel_size
        xyz = xyz[mask]
        feat = feat[mask]
        if xyz.numel() == 0:
            return (
                feat.new_zeros((0, self.d)),
                xyz.new_zeros((0, 3), dtype=torch.long),
                torch.zeros((0,), device=xyz.device, dtype=torch.long),
            )

        # voxel coords
        c = torch.floor(xyz / vs).to(torch.long)  # [Nv,3]
        h = self._hash_coords(c)                  # [Nv]

        # unique voxels
        uniq_h, inv = torch.unique(h, return_inverse=True)  # uniq_h [V], inv [Nv]
        V = int(uniq_h.shape[0])

        # optionally cap voxel count (keeps compute bounded)
        if V > self.max_voxels:
            # pick top voxels by occupancy quickly
            counts = torch.bincount(inv, minlength=V)
            top = torch.topk(counts, k=self.max_voxels, largest=True).indices
            keep_vox = torch.zeros((V,), device=xyz.device, dtype=torch.bool)
            keep_vox[top] = True
            keep_pts = keep_vox[inv]  # points whose voxel is kept
            c = c[keep_pts]
            feat = feat[keep_pts]
            h = h[keep_pts]
            uniq_h, inv = torch.unique(h, return_inverse=True)
            V = int(uniq_h.shape[0])

        # aggregate point feats -> voxel feats (mean)
        vfeat = feat.new_zeros((V, self.d))
        vfeat.index_add_(0, inv, feat)
        vcount = torch.bincount(inv, minlength=V).to(torch.long)
        vfeat = vfeat / vcount.clamp_min(1).unsqueeze(-1)

        # recover representative voxel coords per unique hash
        # (pick first point in each voxel)
        # Build index of first occurrence for each voxel id
        first = torch.full((V,), -1, device=xyz.device, dtype=torch.long)
        # scatter_reduce would be nicer but keep compatibility
        for i in range(inv.numel()):
            j = int(inv[i].item())
            if first[j] < 0:
                first[j] = i
        vcoord = c[first]  # [V,3]
        return vfeat, vcoord, vcount

    def _neighbor_aggregate(self, vfeat: torch.Tensor, vcoord: torch.Tensor) -> torch.Tensor:
        """
        Conv-like 6-neighborhood aggregation on sparse voxel set using hashing lookup.
        vfeat: [V,d], vcoord: [V,3]
        returns aggregated feat: [V,d]
        """
        V = vcoord.shape[0]
        if V == 0:
            return vfeat

        device = vcoord.device
        # hash table: coord hash -> voxel index
        h = self._hash_coords(vcoord)
        # sort by hash for searchsorted lookup
        hs, order = torch.sort(h)

        def lookup(qcoord: torch.Tensor) -> torch.Tensor:
            qh = self._hash_coords(qcoord)
            pos = torch.searchsorted(hs, qh)
            pos = torch.clamp(pos, 0, V - 1)
            hit = hs[pos] == qh
            idx = order[pos]
            idx = torch.where(hit, idx, torch.full_like(idx, -1))
            return idx  # [V] with -1 for missing

        # 6 neighbors + self
        offsets = torch.tensor(
            [[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],
            device=device, dtype=vcoord.dtype
        )
        agg = vfeat.new_zeros((V, self.d))
        deg = vfeat.new_zeros((V, 1))
        for off in offsets:
            nbr = vcoord + off.view(1, 3)
            idx = lookup(nbr)  # [V]
            valid = idx >= 0
            if valid.any():
                agg[valid] = agg[valid] + vfeat[idx[valid]]
                deg[valid] = deg[valid] + 1.0
        agg = agg / deg.clamp_min(1.0)
        return agg

    def forward(
        self,
        point_tokens: torch.Tensor,      # [Bf,N,d]
        xyz: torch.Tensor,              # [Bf,N,3]
        point_mask: Optional[torch.Tensor] = None,  # [Bf,N] bool True=keep
    ) -> torch.Tensor:
        Bf, N, d = point_tokens.shape
        if d != self.d:
            raise ValueError(f"point_tokens last dim {d} != d={self.d}")
        if xyz.shape[:2] != (Bf, N) or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [Bf,N,3], got {tuple(xyz.shape)}")

        if point_mask is None:
            point_mask = torch.ones((Bf, N), device=point_tokens.device, dtype=torch.bool)
        else:
            point_mask = point_mask.to(device=point_tokens.device, dtype=torch.bool)

        out = []
        for b in range(Bf):
            vfeat, vcoord, _vcount = self._voxelize_one(xyz[b], point_tokens[b], point_mask[b])

            # conv-like mixing: neighbor aggregation + MLP residual
            h = vfeat
            for mlp in self.mix_mlps:
                nbr = self._neighbor_aggregate(h, vcoord)
                h = h + self.drop(mlp(nbr))

            # choose tokens
            V = h.shape[0]
            if V == 0:
                tok = point_tokens.new_zeros((self.m, self.d))
            else:
                m = min(self.m, V)
                if self.use_learned_topk:
                    scores = self.score(h).squeeze(-1)  # [V]
                    topk = torch.topk(scores, k=m, largest=True).indices
                else:
                    # fallback: take first m voxels
                    topk = torch.arange(m, device=h.device)
                tok = h[topk]  # [m,d]
                if m < self.m:
                    tok = F.pad(tok, (0, 0, 0, self.m - m))
            out.append(tok)

        return torch.stack(out, dim=0)  # [Bf,m,d]
