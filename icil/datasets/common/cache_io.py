"""Shared caching utilities for RLBench dataset preprocessing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np
import torch


HASH_NAMESPACE = "rlbench_cache_v1"


def to_numpy(array: Any, *, dtype=None) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def write_point_cloud(group: h5py.Group, cloud: Dict[str, torch.Tensor]) -> None:
    group.create_dataset(
        "points",
        data=to_numpy(cloud["points"], dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    group.create_dataset(
        "colors",
        data=to_numpy(cloud["colors"], dtype=np.float32),
        compression="gzip",
        compression_opts=4,
    )
    mask = to_numpy(cloud["masks"], dtype=np.uint8)
    group.create_dataset(
        "masks",
        data=mask,
        compression="gzip",
        compression_opts=4,
    )


def read_point_cloud(group: h5py.Group) -> Dict[str, torch.Tensor]:
    points = torch.from_numpy(group["points"][()]).to(torch.float32)
    colors = torch.from_numpy(group["colors"][()]).to(torch.float32)
    masks = torch.from_numpy(group["masks"][()].astype(np.bool_))
    return {"points": points, "colors": colors, "masks": masks}


def write_point_cloud_sequence(group: h5py.Group, sequence: Iterable[Dict[str, torch.Tensor]]) -> None:
    sequence = list(sequence)
    group.attrs["length"] = len(sequence)
    for idx, cloud in enumerate(sequence):
        frame_group = group.create_group(str(idx))
        write_point_cloud(frame_group, cloud)


def read_point_cloud_sequence(group: h5py.Group) -> List[Dict[str, torch.Tensor]]:
    length = int(group.attrs.get("length", len(group)))
    clouds: List[Dict[str, torch.Tensor]] = []
    for idx in range(length):
        clouds.append(read_point_cloud(group[str(idx)]))
    return clouds


def canonical_config(config: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any):
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, set):
            return sorted(_convert(v) for v in value)
        return value

    return {key: _convert(value) for key, value in sorted(config.items())}


def config_to_key(config: Dict[str, Any]) -> str:
    canonical = canonical_config(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(HASH_NAMESPACE.encode("utf-8") + payload).hexdigest()
    return digest

