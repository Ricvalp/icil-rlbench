from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig, ICILSamplerCore
from icil.models.maml.memory_core import MemoryMAMLConfig
from icil.models.maml.tasks import MAMLTaskSpec


class QueryMemoryTaskBatchIterable(ICILSamplerCore, IterableDataset):
    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        task_batch_size_B: int,
        num_batches: int = 100_000,
        seed: int = 0,
        num_tries_per_item: int = 50,
    ):
        if int(cfg.K) < 1:
            raise ValueError('Query-memory MAML requires cfg.K >= 1 support episodes.')
        IterableDataset.__init__(self)
        ICILSamplerCore.__init__(
            self,
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )
        self.B = int(task_batch_size_B)
        self.num_batches = int(num_batches)
        if self.B < 1:
            raise ValueError('task_batch_size_B must be >= 1.')
        if self.num_batches < 1:
            raise ValueError('num_batches must be >= 1.')

    def _build_task_spec(self, rng: np.random.Generator) -> Optional[MAMLTaskSpec]:
        vidx = self._sample_vidx(rng)
        if vidx is None:
            return None
        episode_ids = self.store.list_episode_ids(vidx)
        if episode_ids.shape[0] < self.cfg.K + 1:
            return None
        chosen = rng.choice(episode_ids, size=self.cfg.K + 1, replace=False).astype(np.int64)
        return MAMLTaskSpec(
            vidx=int(vidx),
            support_episode_ids=tuple(int(ep_id) for ep_id in chosen[: self.cfg.K]),
            query_episode_id=int(chosen[self.cfg.K]),
        )

    def __iter__(self):
        self._iter_counter += 1
        rng = self._rng()
        start, end = self._worker_batch_range(self.num_batches)
        target_batches = max(0, end - start)

        for batch_idx in range(target_batches):
            tasks: List[MAMLTaskSpec] = []
            tries = 0
            while len(tasks) < self.B and tries < self.num_tries_per_item * self.B:
                task = self._build_task_spec(rng)
                tries += 1
                if task is None:
                    continue
                tasks.append(task)
            if len(tasks) < self.B:
                raise RuntimeError(
                    f'Could not assemble a full query-memory task batch (worker_batch_idx={batch_idx}, '
                    f'need={self.B}, got={len(tasks)}).'
                )
            yield tasks


class PreparedQueryMemoryTaskBatchIterable(ICILSamplerCore, IterableDataset):
    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        memory_cfg: MemoryMAMLConfig,
        task_batch_size_B: int,
        num_batches: int = 100_000,
        seed: int = 0,
        num_tries_per_item: int = 50,
        use_mask_id: bool = True,
        load_rgb: bool = True,
    ):
        if int(cfg.K) < 1:
            raise ValueError('Query-memory MAML requires cfg.K >= 1 support episodes.')
        IterableDataset.__init__(self)
        ICILSamplerCore.__init__(
            self,
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )
        self.memory_cfg = memory_cfg
        self.B = int(task_batch_size_B)
        self.num_batches = int(num_batches)
        self.use_mask_id = bool(use_mask_id)
        self.load_rgb = bool(load_rgb)
        if self.B < 1:
            raise ValueError('task_batch_size_B must be >= 1.')
        if self.num_batches < 1:
            raise ValueError('num_batches must be >= 1.')

    def _build_task_spec(self, rng: np.random.Generator) -> Optional[MAMLTaskSpec]:
        vidx = self._sample_vidx(rng)
        if vidx is None:
            return None
        episode_ids = self.store.list_episode_ids(vidx)
        if episode_ids.shape[0] < self.cfg.K + 1:
            return None
        chosen = rng.choice(episode_ids, size=self.cfg.K + 1, replace=False).astype(np.int64)
        return MAMLTaskSpec(
            vidx=int(vidx),
            support_episode_ids=tuple(int(ep_id) for ep_id in chosen[: self.cfg.K]),
            query_episode_id=int(chosen[self.cfg.K]),
        )

    def __iter__(self):
        self._iter_counter += 1
        rng = self._rng()
        start, end = self._worker_batch_range(self.num_batches)
        target_batches = max(0, end - start)
        task_builder = QueryMemoryTaskBuilder(
            self.store,
            cfg=self.cfg,
            seed=self.seed,
            num_tries_per_item=self.num_tries_per_item,
        )
        cpu_device = torch.device('cpu')

        for batch_idx in range(target_batches):
            tasks: List[MAMLTaskSpec] = []
            tries = 0
            while len(tasks) < self.B and tries < self.num_tries_per_item * self.B:
                task = self._build_task_spec(rng)
                tries += 1
                if task is None:
                    continue
                tasks.append(task)
            if len(tasks) < self.B:
                raise RuntimeError(
                    f'Could not assemble a full prepared query-memory task batch (worker_batch_idx={batch_idx}, '
                    f'need={self.B}, got={len(tasks)}).'
                )
            yield prepare_outer_batch_for_query_memory_meta_step(
                tasks,
                task_builder=task_builder,
                cfg=self.memory_cfg,
                device=cpu_device,
                use_mask_id=self.use_mask_id,
                rng=rng,
                load_rgb=self.load_rgb,
            )


class QueryMemoryTaskBuilder(ICILSamplerCore):
    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        seed: int = 0,
        num_tries_per_item: int = 50,
    ):
        if int(cfg.K) < 1:
            raise ValueError('Query-memory MAML requires cfg.K >= 1 support episodes.')
        super().__init__(
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )

    @staticmethod
    def attach_diffusion_inputs(
        batch: Dict[str, Any],
        *,
        noise: Optional[torch.Tensor],
        timesteps: Optional[torch.Tensor],
    ) -> None:
        if noise is not None:
            noise_in = noise
            if noise_in.ndim == 2:
                noise_in = noise_in.unsqueeze(0)
            if noise_in.shape[0] == 1 and batch['target_action'].shape[0] > 1:
                noise_in = noise_in.expand(batch['target_action'].shape[0], -1, -1)
            if noise_in.shape != batch['target_action'].shape:
                raise ValueError(
                    f"noise shape {tuple(noise_in.shape)} must match target_action shape {tuple(batch['target_action'].shape)}"
                )
            batch['noise'] = noise_in
        if timesteps is not None:
            t = timesteps
            if t.ndim == 0:
                t = t.view(1)
            if t.shape[0] == 1 and batch['target_action'].shape[0] > 1:
                t = t.expand(batch['target_action'].shape[0])
            if t.shape != (batch['target_action'].shape[0],):
                raise ValueError(
                    f"timesteps must have shape ({batch['target_action'].shape[0]},), got {tuple(t.shape)}"
                )
            batch['timesteps'] = t

    def _build_query_sample_at_t0(
        self,
        *,
        vidx: int,
        episode_id: int,
        t0: int,
        load_rgb: bool,
        load_mask_id: bool,
    ) -> Dict[str, Any]:
        episode_length = int(self.store.episode_length(int(vidx), int(episode_id)))
        obs_idx, act_idx = self._build_obs_act_indices(int(t0), episode_length=episode_length)
        q_obs = self.store.load_episode_slices(
            int(vidx),
            int(episode_id),
            obs_idx,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
            load_full_traj=False,
        )
        q_act = self.store.load_episode_slices(
            int(vidx),
            int(episode_id),
            act_idx,
            load_rgb=False,
            load_mask_id=False,
            load_full_traj=False,
        )
        sample: Dict[str, Any] = {
            'query_xyz': q_obs['xyz'],
            'query_state': q_obs['state'],
            'query_valid': q_obs['valid'],
            'target_action': self._encode_target_action(q_obs['state'], q_act['action']),
            'meta': {
                'vidx': int(vidx),
                'query_episode': int(episode_id),
                't0': int(t0),
            },
        }
        if load_mask_id and 'mask_id' in q_obs:
            sample['query_mask_id'] = q_obs['mask_id']
        if load_rgb and 'rgb' in q_obs:
            sample['query_rgb'] = q_obs['rgb']
        return sample

    def _stack_query_samples(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError('samples must be non-empty.')
        batch: Dict[str, Any] = {
            'query_xyz': torch.stack([sample['query_xyz'] for sample in samples], 0),
            'query_state': torch.stack([sample['query_state'] for sample in samples], 0),
            'query_valid': torch.stack([sample['query_valid'] for sample in samples], 0),
            'target_action': torch.stack([sample['target_action'] for sample in samples], 0),
            'meta': [sample['meta'] for sample in samples],
        }
        if all('query_mask_id' in sample for sample in samples):
            batch['query_mask_id'] = torch.stack([sample['query_mask_id'] for sample in samples], 0)
        if all('query_rgb' in sample for sample in samples):
            batch['query_rgb'] = torch.stack([sample['query_rgb'] for sample in samples], 0)
        return batch

    def _num_valid_query_t0s(self, *, vidx: int, episode_id: int) -> int:
        T = int(self.store.episode_length(int(vidx), int(episode_id)))
        required = 1 + ((int(self.cfg.T_obs) - 1) * int(self.cfg.stride))
        return max(0, T - required + 1)

    def _sample_balanced_indices(self, total: int, *, count: int, rng: np.random.Generator) -> List[int]:
        if total < 1:
            raise ValueError(f'total must be positive, got {total}.')
        if count < 1:
            raise ValueError(f'count must be positive, got {count}.')
        out: List[int] = []
        while len(out) < int(count):
            perm = rng.permutation(total)
            take = min(total, int(count) - len(out))
            out.extend(int(idx) for idx in perm[:take].tolist())
        return out

    def build_episode_batch(
        self,
        *,
        vidx: int,
        episode_id: int,
        count: int,
        rng: np.random.Generator,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        num_valid = self._num_valid_query_t0s(vidx=int(vidx), episode_id=int(episode_id))
        if num_valid < 1:
            raise RuntimeError(
                f'No valid query windows for vidx={vidx}, episode_id={episode_id} with T_obs={int(self.cfg.T_obs)} '
                f'and stride={int(self.cfg.stride)}.'
            )
        sampled_t0s = self._sample_balanced_indices(num_valid, count=int(count), rng=rng)
        samples = [
            self._build_query_sample_at_t0(
                vidx=int(vidx),
                episode_id=int(episode_id),
                t0=int(t0),
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
            )
            for t0 in sampled_t0s
        ]
        batch = self._stack_query_samples(samples)
        self.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
        return batch

    def build_support_batch(
        self,
        task: MAMLTaskSpec,
        *,
        count: int,
        rng: np.random.Generator,
        support_episode_ids: Optional[Sequence[int]] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        support_ids = [int(ep_id) for ep_id in (support_episode_ids or task.support_episode_ids)]
        if not support_ids:
            raise ValueError('support_episode_ids must be non-empty.')
        episode_order: List[int] = []
        while len(episode_order) < int(count):
            perm = rng.permutation(len(support_ids))
            episode_order.extend(int(support_ids[int(idx)]) for idx in perm.tolist())
        episode_order = episode_order[: int(count)]

        samples: List[Dict[str, Any]] = []
        for episode_id in episode_order:
            num_valid = self._num_valid_query_t0s(vidx=int(task.vidx), episode_id=int(episode_id))
            if num_valid < 1:
                raise RuntimeError(
                    f'No valid support windows for vidx={int(task.vidx)}, episode_id={int(episode_id)}.'
                )
            t0 = int(rng.integers(0, num_valid))
            sample = self._build_query_sample_at_t0(
                vidx=int(task.vidx),
                episode_id=int(episode_id),
                t0=t0,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
            )
            sample['meta'].update(
                {
                    'support_episode': int(episode_id),
                    'support_episodes': support_ids,
                    'task_query_episode': int(task.query_episode_id),
                }
            )
            samples.append(sample)
        batch = self._stack_query_samples(samples)
        self.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
        return batch

    def build_query_batch(
        self,
        task: MAMLTaskSpec,
        *,
        count: int,
        rng: np.random.Generator,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        batch = self.build_episode_batch(
            vidx=int(task.vidx),
            episode_id=int(task.query_episode_id),
            count=int(count),
            rng=rng,
            noise=noise,
            timesteps=timesteps,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        for meta in batch['meta']:
            meta.update(
                {
                    'support_episodes': [int(ep_id) for ep_id in task.support_episode_ids],
                    'task_query_episode': int(task.query_episode_id),
                }
            )
        return batch



def _drop_mask_ids_if_disabled(batch: Dict[str, Any], use_mask_id: bool) -> Dict[str, Any]:
    if use_mask_id:
        return batch
    out = dict(batch)
    out.pop('query_mask_id', None)
    return out



def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out



def _resolve_num_inner_batches(cfg: MemoryMAMLConfig) -> int:
    inner_steps = int(cfg.inner_steps)
    configured = int(cfg.num_inner_batches)
    if inner_steps <= 0:
        return 0
    if configured <= 0:
        return inner_steps
    return min(configured, inner_steps)



def prepare_query_memory_task_for_meta_step(
    task: MAMLTaskSpec,
    *,
    task_builder: QueryMemoryTaskBuilder,
    cfg: MemoryMAMLConfig,
    device: torch.device,
    use_mask_id: bool,
    rng: np.random.Generator,
    load_rgb: bool = True,
) -> Dict[str, Any]:
    shared_noise = None
    shared_timesteps = None

    memory_init_batch = task_builder.build_support_batch(
        task,
        count=1,
        rng=rng,
        noise=None,
        timesteps=None,
        load_rgb=load_rgb,
        load_mask_id=use_mask_id,
    )
    memory_init_batch = _to_device(_drop_mask_ids_if_disabled(memory_init_batch, use_mask_id), device)

    inner_batches: List[Dict[str, Any]] = []
    num_inner_batches = _resolve_num_inner_batches(cfg)
    queries_per_step = int(cfg.num_queries_per_step)
    if queries_per_step < 1:
        raise ValueError(f'num_queries_per_step must be >= 1, got {queries_per_step}.')
    for _ in range(num_inner_batches):
        inner_batch = task_builder.build_support_batch(
            task,
            count=queries_per_step,
            rng=rng,
            noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
            timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
            load_rgb=load_rgb,
            load_mask_id=use_mask_id,
        )
        inner_batches.append(_to_device(_drop_mask_ids_if_disabled(inner_batch, use_mask_id), device))

    query_count = max(1, int(cfg.num_query_loss_samples))
    query_batch = task_builder.build_query_batch(
        task,
        count=query_count,
        rng=rng,
        noise=shared_noise if bool(cfg.reuse_diffusion_noise) else None,
        timesteps=shared_timesteps if bool(cfg.reuse_diffusion_noise) else None,
        load_rgb=load_rgb,
        load_mask_id=use_mask_id,
    )
    query_batch = _to_device(_drop_mask_ids_if_disabled(query_batch, use_mask_id), device)

    return {
        'task': task,
        'support_ids': [int(ep_id) for ep_id in task.support_episode_ids],
        'query_episode_id': int(task.query_episode_id),
        'memory_init_batch': memory_init_batch,
        'inner_batches': inner_batches,
        'query_batch': query_batch,
    }



def prepare_outer_batch_for_query_memory_meta_step(
    tasks: Sequence[MAMLTaskSpec],
    *,
    task_builder: QueryMemoryTaskBuilder,
    cfg: MemoryMAMLConfig,
    device: torch.device,
    use_mask_id: bool,
    rng: np.random.Generator,
    load_rgb: bool = True,
) -> List[Dict[str, Any]]:
    return [
        prepare_query_memory_task_for_meta_step(
            task,
            task_builder=task_builder,
            cfg=cfg,
            device=device,
            use_mask_id=use_mask_id,
            rng=rng,
            load_rgb=load_rgb,
        )
        for task in tasks
    ]
