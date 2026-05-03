from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from icil_jax_query_memory.train.config import QueryMemoryMetaConfig

from .metaworld_cache import MetaWorldEpisodeStore


@dataclass(frozen=True)
class MetaWorldICILConfig:
    K: int
    T_obs: int
    H: int
    stride: int = 1
    action_stride: int = 1
    pad_short_chunks: bool = False
    action_representation: str = 'absolute'
    task_sampling: str = 'task_instance_uniform'
    sample_same_task_name: bool = True
    sample_same_task_instance: bool = True
    allow_support_query_same_episode: bool = False

    def __post_init__(self) -> None:
        for name in ('K', 'T_obs', 'H', 'stride', 'action_stride'):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f'{name} must be an integer, got {type(value).__name__}.')
            if int(value) < 1:
                raise ValueError(f'{name} must be >= 1.')
        if str(self.action_representation) != 'absolute':
            raise ValueError('MetaWorld action_representation must be "absolute" for raw action chunks.')
        if not bool(self.sample_same_task_name):
            raise NotImplementedError('MetaWorld training currently samples support/query from the same task family.')
        if not bool(self.sample_same_task_instance):
            raise NotImplementedError('MetaWorld training currently samples support/query from the same task instance/goal.')


@dataclass(frozen=True)
class MetaWorldMAMLTaskSpec:
    task_name: str
    task_index: int
    task_instance_id: int
    support_episode_ids: Tuple[int, ...]
    query_episode_id: int

    @property
    def vidx(self) -> int:
        # Compatibility with the RLBench task specs consumed by the shared JAX
        # host-batch adapter.
        return int(self.task_index)


class MetaWorldQueryMemoryTaskBuilder:
    def __init__(
        self,
        store: MetaWorldEpisodeStore,
        *,
        cfg: MetaWorldICILConfig,
        seed: int = 0,
        num_tries_per_item: int = 50,
        task_names: Sequence[str] = (),
        exclude_tasks: Sequence[str] = (),
    ):
        self.store = store
        self.cfg = cfg
        self.seed = int(seed)
        self.num_tries_per_item = int(num_tries_per_item)
        self._iter_counter = 0
        self.task_instance_keys = store.list_task_instance_keys(tasks=task_names, exclude_tasks=exclude_tasks)
        if not self.task_instance_keys:
            raise RuntimeError('No MetaWorld task instances available after task filtering.')

    def _rng(self) -> np.random.Generator:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        return np.random.default_rng(self.seed + 10007 * wid + 1000003 * self._iter_counter)

    def _worker_batch_range(self, total_batches: int) -> Tuple[int, int]:
        wi = get_worker_info()
        if wi is None:
            return 0, total_batches
        per_worker = int(np.ceil(float(total_batches) / float(wi.num_workers)))
        start = wi.id * per_worker
        end = min(start + per_worker, total_batches)
        return start, end

    def _sample_task_instance(self, rng: np.random.Generator) -> tuple[str, int]:
        idx = int(rng.integers(0, len(self.task_instance_keys)))
        return self.task_instance_keys[idx]

    def build_task_spec(self, rng: np.random.Generator) -> Optional[MetaWorldMAMLTaskSpec]:
        for _ in range(self.num_tries_per_item):
            task_name, instance_id = self._sample_task_instance(rng)
            episode_ids = self.store.list_episode_ids(task_name, task_instance_id=instance_id)
            need = int(self.cfg.K) + (0 if bool(self.cfg.allow_support_query_same_episode) else 1)
            if episode_ids.shape[0] < need:
                continue
            if bool(self.cfg.allow_support_query_same_episode):
                support_ids = rng.choice(episode_ids, size=int(self.cfg.K), replace=episode_ids.shape[0] < int(self.cfg.K))
                query_id = int(rng.choice(episode_ids))
            else:
                chosen = rng.choice(episode_ids, size=int(self.cfg.K) + 1, replace=False).astype(np.int64)
                support_ids = chosen[: int(self.cfg.K)]
                query_id = int(chosen[int(self.cfg.K)])
            return MetaWorldMAMLTaskSpec(
                task_name=str(task_name),
                task_index=int(self.store.task_index(task_name)),
                task_instance_id=int(instance_id),
                support_episode_ids=tuple(int(eid) for eid in np.asarray(support_ids).tolist()),
                query_episode_id=int(query_id),
            )
        return None

    def _valid_t0_bounds(self, *, task_name: str, episode_id: int) -> Optional[tuple[int, int]]:
        T = int(self.store.episode_length(task_name, episode_id))
        min_t0 = (int(self.cfg.T_obs) - 1) * int(self.cfg.stride)
        if bool(self.cfg.pad_short_chunks):
            max_t0 = max(min_t0, T - 2)
        else:
            max_t0 = T - 1 - int(self.cfg.H) * int(self.cfg.action_stride)
        if max_t0 < min_t0:
            return None
        return min_t0, max_t0

    def _indices_for_t0(self, *, t0: int, episode_length: int) -> tuple[np.ndarray, np.ndarray]:
        obs_idx = int(t0) - np.arange((int(self.cfg.T_obs) - 1) * int(self.cfg.stride), -1, -int(self.cfg.stride), dtype=np.int64)
        act_idx = int(t0) + np.arange(1, int(self.cfg.H) + 1, dtype=np.int64) * int(self.cfg.action_stride)
        if bool(self.cfg.pad_short_chunks):
            act_idx = np.minimum(act_idx, int(episode_length) - 1)
        return obs_idx.astype(np.int64), act_idx.astype(np.int64)

    def _build_query_sample_at_t0(
        self,
        *,
        task_name: str,
        episode_id: int,
        t0: int,
        demo_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        T = int(self.store.episode_length(task_name, episode_id))
        obs_idx, act_idx = self._indices_for_t0(t0=int(t0), episode_length=T)
        obs = self.store.load_episode_slices(task_name, int(episode_id), obs_idx)
        act = self.store.load_episode_slices(task_name, int(episode_id), act_idx)
        query_state = torch.from_numpy(obs['obs_model']).float()
        target_action = torch.from_numpy(act['action']).float()
        sample: Dict[str, Any] = {
            'query_xyz': torch.zeros((int(self.cfg.T_obs), 1, 3), dtype=torch.float32),
            'query_state': query_state,
            'query_valid': torch.ones((int(self.cfg.T_obs), 1), dtype=torch.bool),
            'target_action': target_action,
            'meta': {
                'task_name': str(task_name),
                'episode_id': int(episode_id),
                't0': int(t0),
            },
        }
        if demo_id is not None:
            sample['demo_id'] = torch.tensor(int(demo_id), dtype=torch.long)
            sample['support_demo_id'] = torch.tensor(int(demo_id), dtype=torch.long)
            sample['chunk_start'] = torch.tensor(float(t0), dtype=torch.float32)
            sample['support_chunk_start'] = torch.tensor(float(t0), dtype=torch.float32)
            sample['meta']['support_demo_id'] = int(demo_id)
            sample['meta']['chunk_start'] = int(t0)
        return sample

    def _stack_query_samples(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError('samples must be non-empty.')
        keys = ('query_xyz', 'query_state', 'query_valid', 'target_action')
        batch: Dict[str, Any] = {key: torch.stack([sample[key] for sample in samples], 0) for key in keys}
        for key in ('demo_id', 'support_demo_id', 'chunk_start', 'support_chunk_start'):
            if all(key in sample for sample in samples):
                batch[key] = torch.stack([sample[key] for sample in samples], 0)
        batch['meta'] = [sample['meta'] for sample in samples]
        return batch

    def build_support_batch(
        self,
        task: MetaWorldMAMLTaskSpec,
        *,
        count: int,
        rng: np.random.Generator,
        support_episode_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        support_ids = [int(eid) for eid in (support_episode_ids or task.support_episode_ids)]
        if not support_ids:
            raise ValueError('support_episode_ids must be non-empty.')
        episode_order: List[int] = []
        while len(episode_order) < int(count):
            perm = rng.permutation(len(support_ids))
            episode_order.extend(int(support_ids[int(idx)]) for idx in perm.tolist())
        episode_order = episode_order[: int(count)]
        samples: List[Dict[str, Any]] = []
        for episode_id in episode_order:
            bounds = self._valid_t0_bounds(task_name=task.task_name, episode_id=episode_id)
            if bounds is None:
                continue
            t0 = int(rng.integers(bounds[0], bounds[1] + 1))
            demo_id = int(support_ids.index(int(episode_id)))
            sample = self._build_query_sample_at_t0(
                task_name=task.task_name,
                episode_id=int(episode_id),
                t0=t0,
                demo_id=demo_id,
            )
            sample['meta'].update(
                {
                    'support_episode': int(episode_id),
                    'support_episodes': support_ids,
                    'task_query_episode': int(task.query_episode_id),
                }
            )
            samples.append(sample)
        if len(samples) < int(count):
            raise RuntimeError(f'Could not build enough support samples: need={count}, got={len(samples)}.')
        return self._stack_query_samples(samples)

    def build_query_batch(
        self,
        task: MetaWorldMAMLTaskSpec,
        *,
        count: int,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        samples: List[Dict[str, Any]] = []
        tries = 0
        while len(samples) < int(count) and tries < max(50, int(count) * self.num_tries_per_item):
            tries += 1
            bounds = self._valid_t0_bounds(task_name=task.task_name, episode_id=int(task.query_episode_id))
            if bounds is None:
                continue
            t0 = int(rng.integers(bounds[0], bounds[1] + 1))
            sample = self._build_query_sample_at_t0(
                task_name=task.task_name,
                episode_id=int(task.query_episode_id),
                t0=t0,
                demo_id=None,
            )
            sample['meta'].update(
                {
                    'support_episodes': [int(ep_id) for ep_id in task.support_episode_ids],
                    'task_query_episode': int(task.query_episode_id),
                }
            )
            samples.append(sample)
        if len(samples) < int(count):
            raise RuntimeError(f'Could not build enough query samples: need={count}, got={len(samples)}.')
        return self._stack_query_samples(samples)


def _resolve_num_inner_batches(cfg: QueryMemoryMetaConfig) -> int:
    inner_steps = int(cfg.inner_steps)
    configured = int(cfg.num_inner_batches)
    if inner_steps <= 0:
        return 0
    if configured <= 0:
        return inner_steps
    return min(configured, inner_steps)


def prepare_metaworld_query_memory_task_for_meta_step(
    task: MetaWorldMAMLTaskSpec,
    *,
    task_builder: MetaWorldQueryMemoryTaskBuilder,
    cfg: QueryMemoryMetaConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    inner_batches: List[Dict[str, Any]] = []
    num_inner_batches = _resolve_num_inner_batches(cfg)
    for _ in range(num_inner_batches):
        inner_batches.append(
            task_builder.build_support_batch(
                task,
                count=int(cfg.num_queries_per_step),
                rng=rng,
            )
        )
    query_batch = task_builder.build_query_batch(
        task,
        count=max(1, int(cfg.num_query_loss_samples)),
        rng=rng,
    )
    return {
        'task': task,
        'support_ids': [int(ep_id) for ep_id in task.support_episode_ids],
        'query_episode_id': int(task.query_episode_id),
        'inner_batches': inner_batches,
        'query_batch': query_batch,
    }


class PreparedMetaWorldQueryMemoryTaskBatchIterable(IterableDataset):
    def __init__(
        self,
        store: MetaWorldEpisodeStore,
        *,
        cfg: MetaWorldICILConfig,
        memory_cfg: QueryMemoryMetaConfig,
        task_batch_size_B: int,
        num_batches: int,
        seed: int = 0,
        num_tries_per_item: int = 50,
        task_names: Sequence[str] = (),
        exclude_tasks: Sequence[str] = (),
    ):
        IterableDataset.__init__(self)
        self.store = store
        self.cfg = cfg
        self.memory_cfg = memory_cfg
        self.B = int(task_batch_size_B)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.num_tries_per_item = int(num_tries_per_item)
        self._iter_counter = 0
        self.task_names = tuple(str(t) for t in task_names)
        self.exclude_tasks = tuple(str(t) for t in exclude_tasks)
        if self.B < 1:
            raise ValueError('task_batch_size_B must be >= 1.')

    def _rng(self) -> np.random.Generator:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        return np.random.default_rng(self.seed + 10007 * wid + 1000003 * self._iter_counter)

    def _worker_batch_range(self, total_batches: int) -> Tuple[int, int]:
        wi = get_worker_info()
        if wi is None:
            return 0, total_batches
        per_worker = int(np.ceil(float(total_batches) / float(wi.num_workers)))
        start = wi.id * per_worker
        end = min(start + per_worker, total_batches)
        return start, end

    def __iter__(self):
        self._iter_counter += 1
        rng = self._rng()
        start, end = self._worker_batch_range(self.num_batches)
        task_builder = MetaWorldQueryMemoryTaskBuilder(
            self.store,
            cfg=self.cfg,
            seed=self.seed,
            num_tries_per_item=self.num_tries_per_item,
            task_names=self.task_names,
            exclude_tasks=self.exclude_tasks,
        )
        for batch_idx in range(max(0, end - start)):
            tasks: List[MetaWorldMAMLTaskSpec] = []
            tries = 0
            while len(tasks) < self.B and tries < self.num_tries_per_item * self.B:
                tries += 1
                task = task_builder.build_task_spec(rng)
                if task is not None:
                    tasks.append(task)
            if len(tasks) < self.B:
                raise RuntimeError(
                    f'Could not assemble MetaWorld task batch (idx={batch_idx}, need={self.B}, got={len(tasks)}).'
                )
            yield [
                prepare_metaworld_query_memory_task_for_meta_step(
                    task,
                    task_builder=task_builder,
                    cfg=self.memory_cfg,
                    rng=rng,
                )
                for task in tasks
            ]
