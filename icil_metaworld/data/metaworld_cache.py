from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np


class MetaWorldEpisodeStore:
    def __init__(self, cache_root: str | Path, *, keep_open_per_worker: bool = True):
        root = Path(cache_root).expanduser().resolve()
        if root.is_file():
            self.cache_path = root
            self.root = root.parent
        else:
            self.root = root
            self.cache_path = root / 'cache.h5'
        self.index_path = self.root / 'index.json'
        if not self.cache_path.is_file():
            raise FileNotFoundError(f'MetaWorld cache file not found: {self.cache_path}')
        if not self.index_path.is_file():
            raise FileNotFoundError(f'MetaWorld index file not found: {self.index_path}')
        with self.index_path.open('r', encoding='utf-8') as f:
            self.index = json.load(f)
        self.keep_open_per_worker = bool(keep_open_per_worker)
        self._h5: Optional[h5py.File] = None

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # h5py handles are process-local and cannot be pickled by spawned
        # DataLoader workers. Workers reopen lazily through _file().
        state['_h5'] = None
        return state

    def _file(self) -> h5py.File:
        if self.keep_open_per_worker:
            if self._h5 is None:
                self._h5 = h5py.File(self.cache_path, 'r')
            return self._h5
        return h5py.File(self.cache_path, 'r')

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __len__(self) -> int:
        return len(self.list_task_instance_keys())

    def list_task_names(self) -> List[str]:
        return sorted(str(k) for k in self.index.get('tasks', {}).keys())

    def list_task_ids(self) -> np.ndarray:
        ids = [int(v.get('task_index', i)) for i, v in enumerate(self.index.get('tasks', {}).values())]
        return np.asarray(ids, dtype=np.int64)

    def list_task_instance_keys(
        self,
        *,
        tasks: Sequence[str] = (),
        exclude_tasks: Sequence[str] = (),
    ) -> List[tuple[str, int]]:
        include = {str(t) for t in tasks if str(t)}
        exclude = {str(t) for t in exclude_tasks if str(t)}
        out: List[tuple[str, int]] = []
        for task_name, task_info in sorted(self.index.get('tasks', {}).items()):
            if include and task_name not in include:
                continue
            if task_name in exclude:
                continue
            for instance_id in sorted(task_info.get('instances', {}).keys(), key=lambda x: int(x)):
                eps = task_info.get('instances', {}).get(str(instance_id), [])
                if eps:
                    out.append((str(task_name), int(instance_id)))
        return out

    def task_index(self, task_name: str) -> int:
        return int(self.index['tasks'][str(task_name)]['task_index'])

    def list_episode_ids(self, task_id_or_name: int | str | tuple[str, int], task_instance_id: Optional[int] = None) -> np.ndarray:
        if isinstance(task_id_or_name, tuple):
            task_name = str(task_id_or_name[0])
            task_instance_id = int(task_id_or_name[1])
        elif isinstance(task_id_or_name, str):
            task_name = task_id_or_name
        else:
            task_name = self._task_name_from_index(int(task_id_or_name))
        task_info = self.index['tasks'][task_name]
        if task_instance_id is None:
            eps: List[int] = []
            for ids in task_info.get('instances', {}).values():
                eps.extend(int(eid) for eid in ids)
            return np.asarray(sorted(eps), dtype=np.int64)
        ids = task_info.get('instances', {}).get(str(int(task_instance_id)), [])
        return np.asarray([int(eid) for eid in ids], dtype=np.int64)

    def _task_name_from_index(self, task_index: int) -> str:
        for name, info in self.index.get('tasks', {}).items():
            if int(info.get('task_index', -1)) == int(task_index):
                return str(name)
        raise KeyError(f'No task with task_index={task_index}.')

    def episode_length(self, task_id_or_name: int | str | tuple[str, int], episode_id: int) -> int:
        del task_id_or_name
        ep = self.index['episodes'][str(int(episode_id))]
        return int(ep['length'])

    def infer_dims(self) -> tuple[int, int]:
        with h5py.File(self.cache_path, 'r') as f:
            obs_dim = int(f.attrs.get('obs_model_dim', -1))
            action_dim = int(f.attrs.get('action_dim', -1))
            if obs_dim > 0 and action_dim > 0:
                return obs_dim, action_dim
            first_key = sorted(f['episodes'].keys(), key=lambda x: int(x))[0]
            ep = f['episodes'][first_key]
            return int(ep['obs_model'].shape[-1]), int(ep['actions'].shape[-1])

    @staticmethod
    def _read_rows(dataset: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f'Index array must be 1D, got shape {idx.shape}.')
        if idx.shape[0] == 0:
            return dataset[idx]
        # h5py fancy indexing requires strictly increasing, duplicate-free
        # indices in several versions. Fall back to per-row reads for padded
        # chunks that may repeat the final timestep.
        if np.any(np.diff(idx) <= 0):
            return np.stack([dataset[int(i)] for i in idx.tolist()], axis=0)
        return dataset[idx]

    def load_episode_slices(
        self,
        task_id_or_name: int | str | tuple[str, int],
        episode_id: int,
        t_idx: Sequence[int] | np.ndarray,
        *,
        load_full_traj: bool = False,
        load_raw_obs: bool = False,
    ) -> Dict[str, Any]:
        del task_id_or_name
        close_after = not self.keep_open_per_worker
        f = self._file()
        try:
            ep = f['episodes'][str(int(episode_id))]
            idx = np.asarray(t_idx, dtype=np.int64)
            obs_model = self._read_rows(ep['obs_model'], idx).astype(np.float32)
            actions = self._read_rows(ep['actions'], idx).astype(np.float32)
            out: Dict[str, Any] = {
                'obs_model': obs_model,
                'state': obs_model,
                'action': actions,
                'success': self._read_rows(ep['success'], idx).astype(bool),
            }
            if load_raw_obs:
                out['obs_raw'] = self._read_rows(ep['obs_raw'], idx).astype(np.float32)
            if load_full_traj:
                if load_raw_obs:
                    out['obs_raw_traj'] = ep['obs_raw'][:].astype(np.float32)
                out['obs_model_traj'] = ep['obs_model'][:].astype(np.float32)
                out['action_traj'] = ep['actions'][:].astype(np.float32)
                out['success_traj'] = ep['success'][:].astype(bool)
            return out
        finally:
            if close_after:
                f.close()

    def episode_metadata(self, episode_id: int) -> Dict[str, Any]:
        return dict(self.index['episodes'][str(int(episode_id))])
