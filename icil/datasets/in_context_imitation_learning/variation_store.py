from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

@dataclass(frozen=True)
class VariationKey:
    task: str
    variation: int
    path: str  # variation<V>.h5

def build_variation_keys(cache_root: Path, task: str) -> List[VariationKey]:
    keys = []
    task_dir = cache_root / task
    for p in sorted(task_dir.glob("variation*.h5")):
        v = int(p.stem.replace("variation", ""))
        keys.append(VariationKey(task=task, variation=v, path=str(p)))
    return keys

class VariationStore(Dataset):
    """
    Indexable over variations, but you generally won't __getitem__ it.
    You will call methods to sample episodes and load slices.
    Keeps one H5 handle per worker per variation file (optional).
    """
    def __init__(self, variation_keys: Sequence[VariationKey], keep_open_per_worker: bool = True):
        self.keys = list(variation_keys)
        self.keep_open_per_worker = keep_open_per_worker
        # cache: (worker_id, variation_idx) -> h5py.File
        self._handles: Dict[Tuple[int,int], h5py.File] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def _worker_id(self) -> int:
        wi = get_worker_info()
        return 0 if wi is None else wi.id

    def _get_handle(self, vidx: int) -> h5py.File:
        key = (self._worker_id(), vidx)
        h = self._handles.get(key)
        if h is None:
            h = h5py.File(self.keys[vidx].path, "r")
            self._handles[key] = h
        return h

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()

    def list_episode_ids(self, vidx: int) -> np.ndarray:
        if self.keep_open_per_worker:
            h = self._get_handle(vidx)
            return np.asarray(h["episode_ids"][:], dtype=np.int64)
        with h5py.File(self.keys[vidx].path, "r") as h:
            return np.asarray(h["episode_ids"][:], dtype=np.int64)

    def episode_length(self, vidx: int, episode_id: int) -> int:
        if self.keep_open_per_worker:
            h = self._get_handle(vidx)
            return int(h["episodes"][str(int(episode_id))].attrs["T"])
        with h5py.File(self.keys[vidx].path, "r") as h:
            return int(h["episodes"][str(int(episode_id))].attrs["T"])

    def _read_dataset_rows(self, ds: h5py.Dataset, t_idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(t_idx, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return ds[idx]
        # h5py fancy indexing requires strictly increasing indices. This path
        # also supports duplicate indices, which occur when target-action
        # padding repeats the last available timestep.
        if np.all(idx[1:] > idx[:-1]):
            return ds[idx]
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        gathered = ds[unique_idx]
        return gathered[inverse]

    def load_episode_slices(
        self,
        vidx: int,
        episode_id: int,
        t_idx: np.ndarray,
        *,
        load_rgb: bool = True,
        load_mask_id: bool = True,
        load_full_traj: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns tensors with leading dim len(t_idx).
        """
        if self.keep_open_per_worker:
            h = self._get_handle(vidx)
            g = h["episodes"][str(int(episode_id))]

            xyz = torch.from_numpy(self._read_dataset_rows(g["xyz"], t_idx)).float()
            valid = torch.from_numpy(self._read_dataset_rows(g["valid"], t_idx)).bool()
            state = torch.from_numpy(self._read_dataset_rows(g["state"], t_idx)).float()
            action = torch.from_numpy(self._read_dataset_rows(g["action"], t_idx)).float()

            out = {"xyz": xyz, "valid": valid, "state": state, "action": action}
            if load_rgb and "rgb" in g:
                out["rgb"] = torch.from_numpy(self._read_dataset_rows(g["rgb"], t_idx)).float() / 255.0
            if load_mask_id and "mask_id" in g:
                out["mask_id"] = torch.from_numpy(self._read_dataset_rows(g["mask_id"], t_idx)).long()
            if load_full_traj:
                out["traj"] = torch.from_numpy(g["action"][:]).float()
            return out

        with h5py.File(self.keys[vidx].path, "r") as h:
            g = h["episodes"][str(int(episode_id))]

            xyz = torch.from_numpy(self._read_dataset_rows(g["xyz"], t_idx)).float()
            valid = torch.from_numpy(self._read_dataset_rows(g["valid"], t_idx)).bool()
            state = torch.from_numpy(self._read_dataset_rows(g["state"], t_idx)).float()
            action = torch.from_numpy(self._read_dataset_rows(g["action"], t_idx)).float()

            out = {"xyz": xyz, "valid": valid, "state": state, "action": action}
            if load_rgb and "rgb" in g:
                out["rgb"] = torch.from_numpy(self._read_dataset_rows(g["rgb"], t_idx)).float() / 255.0
            if load_mask_id and "mask_id" in g:
                out["mask_id"] = torch.from_numpy(self._read_dataset_rows(g["mask_id"], t_idx)).long()
            if load_full_traj:
                out["traj"] = torch.from_numpy(g["action"][:]).float()
            return out
