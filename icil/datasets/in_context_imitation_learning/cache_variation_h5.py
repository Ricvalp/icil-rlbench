from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, List, Set

import h5py
import numpy as np

from icil.datasets.common import resolve_mask_ids, MASK_LABEL_MAP_FILENAME

MASK_NAMES_TO_IGNORE = [
    "Floor", "Wall1", "Wall2", "Wall3", "Wall4", "Roof",
    "workspace", "diningTable_visible",
]

# Robust fallback for scene-specific naming variants
# (e.g., "ResizableFloor_5_25_visibleElement", "Wall2", etc.).
MASK_NAME_SUBSTRINGS_TO_IGNORE = [
    "floor",
    "wall",
    "roof",
    "workspace",
    "table",
]

@dataclass(frozen=True)
class CacheSpec:
    N: int = 4096
    store_rgb: bool = True
    store_mask_id: bool = True
    xyz_dtype: str = "f2"      # float16
    feat_dtype: str = "f4"     # float32
    compression: Optional[str] = "gzip"
    compression_opts: int = 4

def _load_demo(low_dim_obs_pkl: Path):
    with low_dim_obs_pkl.open("rb") as f:
        return pickle.load(f)

def _build_vector(obs, keys: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for k in keys:
        v = getattr(obs, k)
        if v is None:
            raise ValueError(f"Observation attribute '{k}' is None")
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        parts.append(a)
    return np.concatenate(parts, axis=0).astype(np.float32)

def _get_ignore_ids(variation_dir: Path) -> Tuple[int, ...]:
    matched, missing, error = resolve_mask_ids(
        variation_dir,
        MASK_NAMES_TO_IGNORE,
        map_filename=MASK_LABEL_MAP_FILENAME,
    )
    ignore_ids: Set[int] = set(int(x) for x in matched)

    # Also match by case-insensitive substring to handle task/scene-dependent names.
    map_path = variation_dir / MASK_LABEL_MAP_FILENAME
    if map_path.is_file():
        try:
            with map_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            for handle_raw, name_raw in data.items():
                try:
                    handle = int(handle_raw)
                except (TypeError, ValueError):
                    continue
                name = str(name_raw).lower()
                if any(token in name for token in MASK_NAME_SUBSTRINGS_TO_IGNORE):
                    ignore_ids.add(handle)
        except (OSError, json.JSONDecodeError):
            pass

    return tuple(sorted(ignore_ids))

def _filter_by_ignore_ids(masks: np.ndarray, ignore_ids: Tuple[int, ...]) -> np.ndarray:
    if not ignore_ids:
        return np.ones_like(masks, dtype=bool)
    keep = np.ones_like(masks, dtype=bool)
    for mid in ignore_ids:
        keep &= (masks != mid)
    return keep

def _subsample_fixedN(rng: np.random.Generator, M: int, N: int) -> np.ndarray:
    if M >= N:
        return rng.choice(M, size=N, replace=False)
    return rng.choice(M, size=N, replace=True)

def _ensure_episode_group(
    f: h5py.File,
    episode_id: int,
    T: int,
    N: int,
    S: int,
    A: int,
    spec: CacheSpec,
) -> h5py.Group:
    """
    Create group episodes/<episode_id>/ with datasets. If it exists, raises.
    """
    episodes_grp = f.require_group("episodes")
    ep_key = str(int(episode_id))
    if ep_key in episodes_grp:
        raise RuntimeError(f"Episode {episode_id} already exists in cache file.")
    g = episodes_grp.create_group(ep_key)
    g.attrs["episode_id"] = int(episode_id)
    g.attrs["T"] = int(T)
    g.attrs["N"] = int(N)
    low_dim_chunk_t = max(1, min(T, 64))

    # Point cloud
    g.create_dataset(
        "xyz", shape=(T, N, 3), dtype=spec.xyz_dtype,
        chunks=(1, N, 3), compression=spec.compression, compression_opts=spec.compression_opts
    )
    g.create_dataset(
        "valid", shape=(T, N), dtype="?",
        chunks=(1, N), compression=spec.compression, compression_opts=spec.compression_opts
    )
    if spec.store_rgb:
        g.create_dataset(
            "rgb", shape=(T, N, 3), dtype="u1",
            chunks=(1, N, 3), compression=spec.compression, compression_opts=spec.compression_opts
        )
    if spec.store_mask_id:
        g.create_dataset(
            "mask_id", shape=(T, N), dtype="i4",
            chunks=(1, N), compression=spec.compression, compression_opts=spec.compression_opts
        )

    # Low-dim
    g.create_dataset(
        "state", shape=(T, S), dtype=spec.feat_dtype,
        chunks=(low_dim_chunk_t, S), compression=spec.compression, compression_opts=spec.compression_opts
    )
    g.create_dataset(
        "action", shape=(T, A), dtype=spec.feat_dtype,
        chunks=(low_dim_chunk_t, A), compression=spec.compression, compression_opts=spec.compression_opts
    )
    return g

def _update_episode_id_index(f: h5py.File, episode_id: int) -> None:
    """
    Maintain a 1D dataset 'episode_ids' listing episode ids in this variation file.
    Append-only.
    """
    eid = int(episode_id)
    if "episode_ids" not in f:
        dset = f.create_dataset("episode_ids", shape=(0,), maxshape=(None,), dtype="i8")
    else:
        dset = f["episode_ids"]

    n = dset.shape[0]
    dset.resize((n + 1,))
    dset[n] = eid

def cache_episode_into_variation_file(
    *,
    episode_dir: Path,
    variation_dir: Path,
    out_variation_h5: Path,
    task_name: str,
    variation_id: int,
    episode_id: int,
    proprio_keys: Sequence[str],
    action_keys: Sequence[str],
    spec: CacheSpec,
    seed: int = 0,
) -> None:
    """
    Append one episode to the variation H5 file.
    """
    rng = np.random.default_rng(seed)

    pc_dir = episode_dir / "merged_point_cloud"
    low_dim = episode_dir / "low_dim_obs.pkl"
    pc_files = sorted(pc_dir.glob("*.npz"), key=lambda p: int(p.stem))
    demo = _load_demo(low_dim)

    T = min(len(pc_files), len(demo))
    if T <= 0:
        raise RuntimeError(f"Empty episode {episode_dir}")

    ignore_ids = _get_ignore_ids(variation_dir)

    s0 = _build_vector(demo[0], proprio_keys)
    a0 = _build_vector(demo[0], action_keys)
    S, A = int(s0.shape[0]), int(a0.shape[0])

    out_variation_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_variation_h5, "a") as f:
        # File-level attrs (set once)
        if "task" not in f.attrs:
            f.attrs["task"] = str(task_name)
            f.attrs["variation"] = int(variation_id)
            f.attrs["N"] = int(spec.N)
            f.attrs["proprio_keys"] = json.dumps(list(proprio_keys))
            f.attrs["action_keys"] = json.dumps(list(action_keys))
            f.attrs["ignore_ids"] = np.asarray(ignore_ids, dtype=np.int64)
        else:
            file_task = f.attrs.get("task")
            file_variation = int(f.attrs.get("variation", -1))
            file_n = int(f.attrs.get("N", -1))
            if str(file_task) != str(task_name):
                raise ValueError(
                    f"Cache file {out_variation_h5} already contains task={file_task}, "
                    f"but got task={task_name}."
                )
            if file_variation != int(variation_id):
                raise ValueError(
                    f"Cache file {out_variation_h5} already contains variation={file_variation}, "
                    f"but got variation={variation_id}."
                )
            if file_n != int(spec.N):
                raise ValueError(
                    f"Cache file {out_variation_h5} already contains N={file_n}, "
                    f"but got N={spec.N}."
                )

        g = _ensure_episode_group(f, episode_id, T, spec.N, S, A, spec)

        xyz = g["xyz"]
        valid = g["valid"]
        rgb = g["rgb"] if spec.store_rgb else None
        mask_id = g["mask_id"] if spec.store_mask_id else None
        state = g["state"]
        action = g["action"]

        for t in range(T):
            z = np.load(pc_files[t], allow_pickle=False)
            pts = z["points"].astype(np.float32)
            cols = z["colors"].astype(np.uint8)
            msk = z["masks"].astype(np.int32)

            keep = _filter_by_ignore_ids(msk, ignore_ids)
            pts, cols, msk = pts[keep], cols[keep], msk[keep]
            M = pts.shape[0]

            if M == 0:
                xyz[t] = np.zeros((spec.N, 3), dtype=np.float16)
                valid[t] = np.zeros((spec.N,), dtype=bool)
                if rgb is not None:
                    rgb[t] = np.zeros((spec.N, 3), dtype=np.uint8)
                if mask_id is not None:
                    mask_id[t] = np.zeros((spec.N,), dtype=np.int32)
            else:
                idx = _subsample_fixedN(rng, M, spec.N)
                xyz[t] = pts[idx].astype(np.float16)
                valid[t] = True
                if rgb is not None:
                    rgb[t] = cols[idx]
                if mask_id is not None:
                    mask_id[t] = msk[idx]

            state[t] = _build_vector(demo[t], proprio_keys)
            action[t] = _build_vector(demo[t], action_keys)

        _update_episode_id_index(f, episode_id)
