from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig, ICILSamplerCore


@dataclass(frozen=True)
class MAMLTaskSpec:
    vidx: int
    support_episode_ids: Tuple[int, ...]
    query_episode_id: int


class ICILMAMLTaskBatchIterable(ICILSamplerCore, IterableDataset):
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
        if int(cfg.K) < 2:
            raise ValueError("MAML requires cfg.K >= 2 support episodes.")
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
            raise ValueError("task_batch_size_B must be >= 1.")
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

    def _build_task_spec(self, rng: np.random.Generator) -> Optional[MAMLTaskSpec]:
        vidx = self._sample_vidx(rng)
        if vidx is None:
            return None
        episode_ids = self.store.list_episode_ids(vidx)
        if episode_ids.shape[0] < self.cfg.K + 1:
            return None
        chosen = rng.choice(episode_ids, size=self.cfg.K + 1, replace=False).astype(np.int64)
        support_episode_ids = tuple(int(ep_id) for ep_id in chosen[: self.cfg.K])
        query_episode_id = int(chosen[self.cfg.K])
        return MAMLTaskSpec(
            vidx=int(vidx),
            support_episode_ids=support_episode_ids,
            query_episode_id=query_episode_id,
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
                    f"Could not assemble a full MAML task batch (worker_batch_idx={batch_idx}, "
                    f"need={self.B}, got={len(tasks)}). Increase data/episodes or reduce task batch size."
                )
            yield tasks


class MAMLTaskBuilder(ICILSamplerCore):
    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        seed: int = 0,
        num_tries_per_item: int = 50,
    ):
        if int(cfg.K) < 2:
            raise ValueError("MAML requires cfg.K >= 2 support episodes.")
        super().__init__(
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )

    def build_conditioning_from_support_ids(
        self,
        rng: np.random.Generator,
        *,
        vidx: int,
        support_ids: Sequence[int],
        load_rgb: bool = True,
        load_mask_id: bool = True,
        load_full_traj: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if len(support_ids) < 1:
            raise ValueError("support_ids must be non-empty.")

        items: List[Dict[str, Any]] = []
        for eid in support_ids:
            item = self._build_single_support_conditioning(
                rng,
                vidx=vidx,
                episode_id=int(eid),
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
                load_full_traj=load_full_traj,
            )
            if item is None:
                return None
            items.append(item)
        return self._stack_support_conditioning_items(items)

    def _build_single_support_conditioning(
        self,
        rng: np.random.Generator,
        *,
        vidx: int,
        episode_id: int,
        load_rgb: bool = True,
        load_mask_id: bool = True,
        load_full_traj: bool = True,
    ) -> Optional[Dict[str, Any]]:
        T = self.store.episode_length(vidx, int(episode_id))
        if T <= 0:
            return None
        keyframes = self._sample_keyframes(T, self.cfg.L, rng)
        if keyframes.shape[0] != self.cfg.L:
            return None
        sample = self.store.load_episode_slices(
            vidx,
            int(episode_id),
            keyframes,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
            load_full_traj=load_full_traj,
        )
        out: Dict[str, Any] = {
            "cond_xyz": sample["xyz"],
            "cond_state": sample["state"],
            "cond_valid": sample["valid"],
        }
        if load_mask_id and "mask_id" in sample:
            out["cond_mask_id"] = sample["mask_id"]
        if load_rgb and "rgb" in sample:
            out["cond_rgb"] = sample["rgb"]
        if load_full_traj and "traj" in sample:
            out["cond_traj"] = self._stride_traj(sample["traj"])
        return out

    def _stack_support_conditioning_items(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not items:
            raise ValueError("items must be non-empty.")
        out: Dict[str, Any] = {
            "cond_xyz": torch.stack([item["cond_xyz"] for item in items], 0),
            "cond_state": torch.stack([item["cond_state"] for item in items], 0),
            "cond_valid": torch.stack([item["cond_valid"] for item in items], 0),
        }
        if all("cond_mask_id" in item for item in items):
            out["cond_mask_id"] = torch.stack([item["cond_mask_id"] for item in items], 0)
        if all("cond_rgb" in item for item in items):
            out["cond_rgb"] = torch.stack([item["cond_rgb"] for item in items], 0)
        if all("cond_traj" in item for item in items):
            out["cond_traj"], out["cond_traj_mask"] = self._pack_traj_list(
                [item["cond_traj"] for item in items]
            )
        return out

    def _build_query_sample(
        self,
        *,
        vidx: int,
        episode_id: int,
        rng: np.random.Generator,
        load_rgb: bool,
        load_mask_id: bool,
    ) -> Optional[Dict[str, Any]]:
        T = self.store.episode_length(vidx, int(episode_id))
        t0 = self._sample_t0(T, rng)
        if t0 is None:
            return None
        obs_idx, act_idx = self._build_obs_act_indices(t0, episode_length=T)
        q_obs = self.store.load_episode_slices(
            vidx,
            int(episode_id),
            obs_idx,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
            load_full_traj=False,
        )
        q_act = self.store.load_episode_slices(
            vidx,
            int(episode_id),
            act_idx,
            load_rgb=False,
            load_mask_id=False,
            load_full_traj=False,
        )

        sample: Dict[str, Any] = {
            "query_xyz": q_obs["xyz"],
            "query_state": q_obs["state"],
            "query_valid": q_obs["valid"],
            "target_action": q_act["action"],
            "meta": {
                "vidx": int(vidx),
                "query_episode": int(episode_id),
                "t0": int(t0),
            },
        }
        if load_mask_id and "mask_id" in q_obs:
            sample["query_mask_id"] = q_obs["mask_id"]
        if load_rgb and "rgb" in q_obs:
            sample["query_rgb"] = q_obs["rgb"]
        return sample

    def _stack_samples(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("samples must be non-empty.")
        batch: Dict[str, Any] = {
            "cond_xyz": torch.stack([sample["cond_xyz"] for sample in samples], 0),
            "cond_state": torch.stack([sample["cond_state"] for sample in samples], 0),
            "cond_valid": torch.stack([sample["cond_valid"] for sample in samples], 0),
            "query_xyz": torch.stack([sample["query_xyz"] for sample in samples], 0),
            "query_state": torch.stack([sample["query_state"] for sample in samples], 0),
            "query_valid": torch.stack([sample["query_valid"] for sample in samples], 0),
            "target_action": torch.stack([sample["target_action"] for sample in samples], 0),
            "meta": [sample["meta"] for sample in samples],
        }
        if all("cond_mask_id" in sample for sample in samples):
            batch["cond_mask_id"] = torch.stack([sample["cond_mask_id"] for sample in samples], 0)
        if all("query_mask_id" in sample for sample in samples):
            batch["query_mask_id"] = torch.stack([sample["query_mask_id"] for sample in samples], 0)
        if all("cond_rgb" in sample for sample in samples):
            batch["cond_rgb"] = torch.stack([sample["cond_rgb"] for sample in samples], 0)
        if all("query_rgb" in sample for sample in samples):
            batch["query_rgb"] = torch.stack([sample["query_rgb"] for sample in samples], 0)
        if all("cond_traj" in sample and "cond_traj_mask" in sample for sample in samples):
            batch["cond_traj"], batch["cond_traj_mask"] = self._pad_traj_items(
                [sample["cond_traj"] for sample in samples],
                [sample["cond_traj_mask"] for sample in samples],
            )
        return batch

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
            if noise_in.shape[0] == 1 and batch["target_action"].shape[0] > 1:
                noise_in = noise_in.expand(batch["target_action"].shape[0], -1, -1)
            if noise_in.shape != batch["target_action"].shape:
                raise ValueError(
                    f"noise shape {tuple(noise_in.shape)} must match target_action shape "
                    f"{tuple(batch['target_action'].shape)}"
                )
            batch["noise"] = noise_in

        if timesteps is not None:
            t = timesteps
            if t.ndim == 0:
                t = t.view(1)
            if t.shape[0] == 1 and batch["target_action"].shape[0] > 1:
                t = t.expand(batch["target_action"].shape[0])
            if t.shape != (batch["target_action"].shape[0],):
                raise ValueError(
                    f"timesteps must have shape ({batch['target_action'].shape[0]},), "
                    f"got {tuple(t.shape)}"
                )
            batch["timesteps"] = t

    def build_support_batch_loo(
        self,
        task: MAMLTaskSpec,
        *,
        holdout_indices: Sequence[int],
        rng: np.random.Generator,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        if not holdout_indices:
            raise ValueError("holdout_indices must be non-empty.")

        support_ids = list(task.support_episode_ids)
        samples: List[Dict[str, Any]] = []
        for holdout_idx in holdout_indices:
            if holdout_idx < 0 or holdout_idx >= len(support_ids):
                raise IndexError(
                    f"holdout_idx={holdout_idx} out of range for K={len(support_ids)}"
                )
            heldout_episode_id = support_ids[int(holdout_idx)]
            kept_support_ids = [
                int(ep_id) for idx, ep_id in enumerate(support_ids) if idx != int(holdout_idx)
            ]
            support = self.build_conditioning_from_support_ids(
                rng,
                vidx=int(task.vidx),
                support_ids=kept_support_ids,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
                load_full_traj=True,
            )
            if support is None:
                raise RuntimeError("Failed to build support conditioning for inner-loop adaptation.")
            query = self._build_query_sample(
                vidx=int(task.vidx),
                episode_id=int(heldout_episode_id),
                rng=rng,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
            )
            if query is None:
                raise RuntimeError("Failed to build held-out support query sample.")
            query["meta"].update(
                {
                    "holdout_index": int(holdout_idx),
                    "heldout_support_episode": int(heldout_episode_id),
                    "support_episodes": kept_support_ids,
                    "task_query_episode": int(task.query_episode_id),
                }
            )
            sample = {**support, **query}
            samples.append(sample)

        batch = self._stack_samples(samples)
        self.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
        return batch

    def build_support_batch_loo_cached(
        self,
        task: MAMLTaskSpec,
        *,
        holdout_indices: Sequence[int],
        rng: np.random.Generator,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        if len(holdout_indices) < 1:
            raise ValueError("holdout_indices must be non-empty.")

        support_ids = list(task.support_episode_ids)
        support_items: List[Dict[str, Any]] = []
        for episode_id in support_ids:
            item = self._build_single_support_conditioning(
                rng,
                vidx=int(task.vidx),
                episode_id=int(episode_id),
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
                load_full_traj=True,
            )
            if item is None:
                raise RuntimeError("Failed to build cached support conditioning for inner-loop adaptation.")
            support_items.append(item)

        samples: List[Dict[str, Any]] = []
        for holdout_idx in holdout_indices:
            holdout_idx = int(holdout_idx)
            if holdout_idx < 0 or holdout_idx >= len(support_ids):
                raise IndexError(
                    f"holdout_idx={holdout_idx} out of range for K={len(support_ids)}"
                )
            heldout_episode_id = support_ids[holdout_idx]
            kept_support_ids = [
                int(ep_id) for idx, ep_id in enumerate(support_ids) if idx != holdout_idx
            ]
            support = self._stack_support_conditioning_items(
                [item for idx, item in enumerate(support_items) if idx != holdout_idx]
            )
            query = self._build_query_sample(
                vidx=int(task.vidx),
                episode_id=int(heldout_episode_id),
                rng=rng,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
            )
            if query is None:
                raise RuntimeError("Failed to build held-out support query sample.")
            query["meta"].update(
                {
                    "holdout_index": int(holdout_idx),
                    "heldout_support_episode": int(heldout_episode_id),
                    "support_episodes": kept_support_ids,
                    "task_query_episode": int(task.query_episode_id),
                }
            )
            sample = {**support, **query}
            samples.append(sample)

        batch = self._stack_samples(samples)
        self.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
        return batch

    def build_query_batch(
        self,
        task: MAMLTaskSpec,
        *,
        rng: np.random.Generator,
        num_context_episodes: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        load_rgb: bool = True,
        load_mask_id: bool = True,
    ) -> Dict[str, Any]:
        support_ids = list(task.support_episode_ids)
        if num_context_episodes is not None:
            if num_context_episodes <= 0:
                raise ValueError("num_context_episodes must be positive when provided.")
            if num_context_episodes > len(support_ids):
                raise ValueError(
                    f"Requested {num_context_episodes} context demos but task only has {len(support_ids)}."
                )
            if num_context_episodes < len(support_ids):
                keep = np.sort(rng.choice(len(support_ids), size=num_context_episodes, replace=False))
                support_ids = [support_ids[int(idx)] for idx in keep.tolist()]

        support = self.build_conditioning_from_support_ids(
            rng,
            vidx=int(task.vidx),
            support_ids=support_ids,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
            load_full_traj=True,
        )
        if support is None:
            raise RuntimeError("Failed to build outer-query support conditioning.")
        query = self._build_query_sample(
            vidx=int(task.vidx),
            episode_id=int(task.query_episode_id),
            rng=rng,
            load_rgb=load_rgb,
            load_mask_id=load_mask_id,
        )
        if query is None:
            raise RuntimeError("Failed to build outer query sample.")
        query["meta"].update(
            {
                "support_episodes": [int(ep_id) for ep_id in support_ids],
                "task_query_episode": int(task.query_episode_id),
            }
        )
        batch = self._stack_samples([{**support, **query}])
        self.attach_diffusion_inputs(batch, noise=noise, timesteps=timesteps)
        return batch
