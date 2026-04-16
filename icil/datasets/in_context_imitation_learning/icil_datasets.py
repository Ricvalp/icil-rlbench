from __future__ import annotations

import argparse
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    from .variation_store import VariationStore, build_variation_keys
except ImportError:  # pragma: no cover - allow direct script execution
    from variation_store import VariationStore, build_variation_keys


# Assumed interface of your VariationStore:
# - __len__() -> number of variation files (vidx)
# - list_episode_ids(vidx) -> np.ndarray[int64] shape [E]
# - episode_length(vidx, episode_id) -> int
# - load_episode_slices(vidx, episode_id, t_idx) -> Dict[str, torch.Tensor]
#     returns at least: xyz [len(t_idx),N,3], state [len(t_idx),S], action [len(t_idx),A]
#     optionally: mask_id [len(t_idx),N]
#
# You already have this in your repo; we don't re-define it here.


@dataclass(frozen=True)
class ICILConfig:
    K: int
    L: int
    T_obs: int
    H: int
    stride: int = 1
    task_sampling: str = "variation_power"
    task_sampling_alpha: float = 1.0

    def __post_init__(self) -> None:
        for name in ("K", "L", "T_obs", "H", "stride"):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if self.K < 1:
            raise ValueError("K must be >= 1.")
        if self.L < 1:
            raise ValueError("L must be >= 1.")
        if self.T_obs < 1:
            raise ValueError("T_obs must be >= 1.")
        if self.H < 1:
            raise ValueError("H must be >= 1.")
        if self.stride < 1:
            raise ValueError("stride must be >= 1.")
        if str(self.task_sampling) not in ("variation_power", "variation_uniform", "task_uniform"):
            raise ValueError(
                "task_sampling must be one of: variation_power, variation_uniform, task_uniform."
            )
        if float(self.task_sampling_alpha) < 0.0:
            raise ValueError("task_sampling_alpha must be >= 0.")


class ICILSamplerCore:
    """
    Shared sampling/composition logic. Does not yield batches directly.
    Provides:
      - _build_one_sample(): atomic ICIL sample for pretraining
      - _build_task_bundle(): sample K+1 episodes for a task/variation (used by meta/TTT)
      - _build_leave_one_out_items(): construct J=K+1 query/support items from a bundle
    """

    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        seed: int = 0,
        num_tries_per_item: int = 50,
    ):
        self.store = store
        self.cfg = cfg
        self.seed = seed
        self.num_tries_per_item = num_tries_per_item
        self._iter_counter = 0
        self._task_sampling_task_names: Optional[List[str]] = None
        self._task_sampling_vidx_by_task: Optional[List[np.ndarray]] = None
        self._task_sampling_probs: Optional[np.ndarray] = None
        if self.num_tries_per_item < 1:
            raise ValueError("num_tries_per_item must be >= 1.")

    # ----------------------------
    # RNG / sampling primitives
    # ----------------------------

    def _rng(self) -> np.random.Generator:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        # Include iterator counter so consecutive epochs do not replay identical samples.
        return np.random.default_rng(self.seed + 10007 * wid + 1000003 * self._iter_counter)

    def _worker_batch_range(self, total_batches: int) -> Tuple[int, int]:
        wi = get_worker_info()
        if wi is None:
            return 0, total_batches
        # Split work across workers so total yielded batches ~= total_batches (global).
        per_worker = int(np.ceil(float(total_batches) / float(wi.num_workers)))
        start = wi.id * per_worker
        end = min(start + per_worker, total_batches)
        return start, end

    def _build_task_sampling_index(self) -> bool:
        if self._task_sampling_probs is not None:
            return True
        keys = getattr(self.store, "keys", None)
        if keys is None:
            return False

        task_to_vidx: Dict[str, List[int]] = {}
        for vidx, key in enumerate(keys):
            task = str(getattr(key, "task", ""))
            if not task:
                task = f"vidx:{vidx}"
            task_to_vidx.setdefault(task, []).append(int(vidx))
        if not task_to_vidx:
            return False

        task_names = sorted(task_to_vidx)
        vidx_by_task = [
            np.asarray(task_to_vidx[task], dtype=np.int64)
            for task in task_names
        ]

        mode = str(self.cfg.task_sampling)
        alpha = 0.0 if mode == "task_uniform" else float(self.cfg.task_sampling_alpha)
        counts = np.asarray([len(vidxs) for vidxs in vidx_by_task], dtype=np.float64)
        weights = np.power(counts, alpha)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            weights = np.ones_like(counts)
            weight_sum = float(weights.sum())

        self._task_sampling_task_names = task_names
        self._task_sampling_vidx_by_task = vidx_by_task
        self._task_sampling_probs = weights / weight_sum
        return True

    def task_sampling_probabilities(self) -> Dict[str, float]:
        if str(self.cfg.task_sampling) == "variation_uniform":
            keys = getattr(self.store, "keys", None)
            if keys is None:
                return {}
            counts: Dict[str, int] = {}
            for key in keys:
                task = str(getattr(key, "task", ""))
                counts[task] = counts.get(task, 0) + 1
            total = float(sum(counts.values()))
            return {task: count / total for task, count in sorted(counts.items()) if total > 0.0}

        if not self._build_task_sampling_index():
            return {}
        assert self._task_sampling_task_names is not None
        assert self._task_sampling_probs is not None
        return {
            task: float(prob)
            for task, prob in zip(self._task_sampling_task_names, self._task_sampling_probs)
        }

    def _sample_vidx(self, rng: np.random.Generator) -> Optional[int]:
        V = len(self.store)
        if V <= 0:
            return None
        use_task_sampling = (
            str(self.cfg.task_sampling) != "variation_uniform"
            and self._build_task_sampling_index()
        )
        for _ in range(self.num_tries_per_item):
            if use_task_sampling:
                assert self._task_sampling_probs is not None
                assert self._task_sampling_vidx_by_task is not None
                task_idx = int(rng.choice(len(self._task_sampling_probs), p=self._task_sampling_probs))
                task_vidxs = self._task_sampling_vidx_by_task[task_idx]
                vidx = int(task_vidxs[int(rng.integers(0, len(task_vidxs)))])
            else:
                vidx = int(rng.integers(0, V))
            # need at least K+1 episodes available
            eids = self.store.list_episode_ids(vidx)
            if eids.shape[0] >= self.cfg.K + 1:
                return vidx
        return None

    def _sample_keyframes(self, T: int, L: int, rng: np.random.Generator) -> np.ndarray:
        if T <= 0:
            return np.zeros((0,), dtype=np.int64)
        if T >= L:
            return np.sort(rng.choice(T, size=L, replace=False)).astype(np.int64)
        # keep fixed length by repeating
        return np.sort(rng.choice(T, size=L, replace=True)).astype(np.int64)

    def _sample_t0(self, T: int, rng: np.random.Generator) -> Optional[int]:
        # Causal alignment:
        # obs_idx uses stride spacing:
        #   obs_idx = t0 + [0, stride, ..., (T_obs-1)*stride]
        # act_idx starts strictly AFTER the last observed step:
        #   act_idx = obs_idx[-1] + stride * [1, 2, ..., H]
        # We require the query observation window to be fully real, but allow the
        # target-action window to run past the episode end and repeat the last
        # available action timestep via index clamping.
        # required raw timesteps for the observation window only:
        #   1 + ((T_obs - 1) * stride)
        required = 1 + ((self.cfg.T_obs - 1) * self.cfg.stride)
        max_t0 = T - required
        if max_t0 < 0:
            return None
        return int(rng.integers(0, max_t0 + 1))

    def _build_obs_act_indices(
        self,
        t0: int,
        *,
        episode_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        obs_idx = t0 + np.arange(0, cfg.T_obs * cfg.stride, cfg.stride, dtype=np.int64)
        act_start = int(obs_idx[-1] + cfg.stride)
        act_idx = act_start + np.arange(0, cfg.H * cfg.stride, cfg.stride, dtype=np.int64)
        if episode_length is not None:
            if episode_length < 1:
                raise ValueError(f"episode_length must be >= 1, got {episode_length}.")
            act_idx = np.minimum(act_idx, episode_length - 1)
        return obs_idx, act_idx

    def _stride_traj(self, traj: torch.Tensor) -> torch.Tensor:
        if traj.dim() != 2:
            raise ValueError(f"Expected trajectory tensor [T,D], got shape={tuple(traj.shape)}")
        stride = int(self.cfg.stride)
        if stride <= 1:
            return traj
        return traj[::stride]

    def _pack_traj_list(
        self,
        traj_list: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(traj_list) == 0:
            raise ValueError("traj_list must be non-empty.")
        max_T = max(int(traj.shape[0]) for traj in traj_list)
        traj_dim = int(traj_list[0].shape[-1])

        padded_traj: List[torch.Tensor] = []
        padded_mask: List[torch.Tensor] = []
        for traj in traj_list:
            if traj.dim() != 2:
                raise ValueError(f"Expected trajectory tensor [T,D], got shape={tuple(traj.shape)}")
            T = int(traj.shape[0])
            if int(traj.shape[-1]) != traj_dim:
                raise ValueError("All trajectories in traj_list must share the same feature dimension.")
            traj_pad = traj.new_zeros((max_T, traj_dim))
            traj_pad[:T] = traj
            mask = torch.zeros((max_T,), dtype=torch.bool, device=traj.device)
            mask[:T] = True
            padded_traj.append(traj_pad)
            padded_mask.append(mask)

        return torch.stack(padded_traj, 0), torch.stack(padded_mask, 0)

    def _pad_traj_items(
        self,
        traj_items: Sequence[torch.Tensor],
        mask_items: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(traj_items) == 0 or len(mask_items) == 0:
            raise ValueError("traj_items and mask_items must be non-empty.")
        if len(traj_items) != len(mask_items):
            raise ValueError("traj_items and mask_items must have the same length.")

        max_T = max(int(traj.shape[-2]) for traj in traj_items)
        padded_traj_items: List[torch.Tensor] = []
        padded_mask_items: List[torch.Tensor] = []
        for traj, mask in zip(traj_items, mask_items):
            if mask.shape != traj.shape[:-1]:
                raise ValueError(
                    f"Trajectory/mask shape mismatch: traj={tuple(traj.shape)}, mask={tuple(mask.shape)}"
                )
            T = int(traj.shape[-2])
            if T < max_T:
                traj_pad = traj.new_zeros(*traj.shape[:-2], max_T - T, traj.shape[-1])
                mask_pad = torch.zeros(*mask.shape[:-1], max_T - T, dtype=torch.bool, device=mask.device)
                traj = torch.cat([traj, traj_pad], dim=-2)
                mask = torch.cat([mask, mask_pad], dim=-1)
            padded_traj_items.append(traj)
            padded_mask_items.append(mask)

        return torch.stack(padded_traj_items, 0), torch.stack(padded_mask_items, 0)

    def build_support_conditioning(
        self,
        rng: np.random.Generator,
        *,
        vidx: int,
        support_ids: Sequence[int],
        load_rgb: bool = True,
        load_mask_id: bool = True,
        load_full_traj: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Build only the support-conditioning portion used during training:
          cond_xyz: [K,L,N,3]
          cond_state: [K,L,S]
          cond_valid: [K,L,N]
          optional cond_mask_id: [K,L,N]
          optional cond_rgb: [K,L,N,3]
          optional cond_traj: [K,T,D]
          optional cond_traj_mask: [K,T]
        """
        if self.store is None:
            raise ValueError("build_support_conditioning requires a valid VariationStore.")

        cfg = self.cfg
        if len(support_ids) < cfg.K:
            raise ValueError(f"Need at least K={cfg.K} support episode ids, got {len(support_ids)}.")

        cond_xyz, cond_state, cond_valid, cond_mask, cond_rgb, cond_traj = [], [], [], [], [], []
        cond_has_mask_for_all = bool(load_mask_id)
        cond_has_rgb_for_all = bool(load_rgb)
        cond_has_traj = bool(load_full_traj)

        for eid in support_ids[: cfg.K]:
            eid = int(eid)
            Ti = self.store.episode_length(vidx, eid)
            if Ti <= 0:
                return None
            kf = self._sample_keyframes(Ti, cfg.L, rng)
            if kf.shape[0] != cfg.L:
                return None
            c = self.store.load_episode_slices(
                vidx,
                eid,
                kf,
                load_rgb=load_rgb,
                load_mask_id=load_mask_id,
                load_full_traj=load_full_traj,
            )
            cond_xyz.append(c["xyz"])       # [L,N,3]
            cond_state.append(c["state"])   # [L,S]
            cond_valid.append(c["valid"])   # [L,N]
            if load_mask_id and "mask_id" in c:
                cond_mask.append(c["mask_id"])
            elif load_mask_id:
                cond_has_mask_for_all = False
            if load_rgb and "rgb" in c:
                cond_rgb.append(c["rgb"])
            elif load_rgb:
                cond_has_rgb_for_all = False
            if load_full_traj and "traj" in c:
                cond_traj.append(self._stride_traj(c["traj"]))
            elif load_full_traj:
                cond_has_traj = False

        out: Dict[str, Any] = {
            "cond_xyz": torch.stack(cond_xyz, 0),      # [K,L,N,3]
            "cond_state": torch.stack(cond_state, 0),  # [K,L,S]
            "cond_valid": torch.stack(cond_valid, 0),  # [K,L,N]
        }
        if cond_has_mask_for_all and len(cond_mask) == cfg.K:
            out["cond_mask_id"] = torch.stack(cond_mask, 0)  # [K,L,N]
        if cond_has_rgb_for_all and len(cond_rgb) == cfg.K:
            out["cond_rgb"] = torch.stack(cond_rgb, 0)       # [K,L,N,3]
        if cond_has_traj and len(cond_traj) == cfg.K:
            out["cond_traj"], out["cond_traj_mask"] = self._pack_traj_list(cond_traj)  # [K,T,D], [K,T]
        return out

    # ----------------------------
    # Atomic ICIL sample (pretrain)
    # ----------------------------

    def _build_one_sample(
        self,
        rng: np.random.Generator,
        *,
        vidx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns one ICIL sample dict:
          cond_xyz: [K,L,N,3]
          cond_state: [K,L,S]
          query_xyz: [T_obs,N,3]
          query_state: [T_obs,S]
          target_action: [H,A]
          (+ optional masks)
        """
        cfg = self.cfg

        if vidx is None:
            vidx = self._sample_vidx(rng)
            if vidx is None:
                return None

        episode_ids = self.store.list_episode_ids(vidx)
        if episode_ids.shape[0] < cfg.K + 1:
            return None

        # sample supports + query
        chosen = rng.choice(episode_ids, size=cfg.K + 1, replace=False)
        cond_ids = chosen[: cfg.K]
        query_id = int(chosen[cfg.K])

        Tq = self.store.episode_length(vidx, query_id)
        t0 = self._sample_t0(Tq, rng)
        if t0 is None:
            return None

        obs_idx, act_idx = self._build_obs_act_indices(t0, episode_length=Tq)

        q_obs = self.store.load_episode_slices(vidx, query_id, obs_idx, load_rgb=True, load_mask_id=True, load_full_traj=False)
        q_act = self.store.load_episode_slices(vidx, query_id, act_idx, load_rgb=False, load_mask_id=False, load_full_traj=False)

        support = self.build_support_conditioning(
            rng,
            vidx=vidx,
            support_ids=cond_ids,
            load_rgb=True,
            load_mask_id=True,
            load_full_traj=True,
        )
        if support is None:
            return None

        sample: Dict[str, Any] = {
            "query_xyz": q_obs["xyz"],                     # [T_obs,N,3]
            "query_state": q_obs["state"],                 # [T_obs,S]
            "query_valid": q_obs["valid"],                 # [T_obs,N]
            "target_action": q_act["action"],              # [H,A]
            "meta": {"vidx": vidx, "query_episode": query_id, "t0": t0},
        }
        sample.update(support)
        if "mask_id" in q_obs:
            sample["query_mask_id"] = q_obs["mask_id"]          # [T_obs,N]
        if "rgb" in q_obs:
            sample["query_rgb"] = q_obs["rgb"]                  # [T_obs,N,3]

        return sample

    # ----------------------------
    # Task bundle + leave-one-out (meta/TTT)
    # ----------------------------

    def _build_task_bundle(
        self,
        rng: np.random.Generator,
        *,
        vidx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Sample a task/variation and pick K+1 episodes that will be used for leave-one-out.
        Returns:
          {
            "vidx": int,
            "episode_ids": np.ndarray shape [K+1],
          }
        """
        cfg = self.cfg
        if vidx is None:
            vidx = self._sample_vidx(rng)
            if vidx is None:
                return None
        episode_ids = self.store.list_episode_ids(vidx)
        if episode_ids.shape[0] < cfg.K + 1:
            return None
        chosen = rng.choice(episode_ids, size=cfg.K + 1, replace=False).astype(np.int64)
        return {"vidx": vidx, "episode_ids": chosen}

    def _build_leave_one_out_items(
        self,
        rng: np.random.Generator,
        bundle: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        For a given bundle (vidx + K+1 episodes), construct J=K+1 inner items:
          For j in [0..J-1]:
            query = episode_ids[j]
            support = episode_ids[all != j]  (size K)

        Returns structured tensors:
          cond_xyz: [J, K, L, N, 3]
          cond_state: [J, K, L, S]
          query_xyz: [J, T_obs, N, 3]
          query_state: [J, T_obs, S]
          target_action: [J, H, A]
          (+ masks)
        """
        cfg = self.cfg
        vidx = int(bundle["vidx"])
        ep_ids = np.asarray(bundle["episode_ids"], dtype=np.int64)
        J = ep_ids.shape[0]
        if J != cfg.K + 1:
            return None

        # Pre-sample query windows for each query episode
        query_xyz_list, query_state_list, query_valid_list, target_action_list, query_mask_list, query_rgb_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        query_has_mask_for_all = True
        query_has_rgb_for_all = True
        t0_list: List[int] = []

        for j in range(J):
            qid = int(ep_ids[j])
            Tq = self.store.episode_length(vidx, qid)
            t0 = self._sample_t0(Tq, rng)
            if t0 is None:
                return None
            t0_list.append(t0)
            obs_idx, act_idx = self._build_obs_act_indices(t0, episode_length=Tq)
            q_obs = self.store.load_episode_slices(
                vidx, qid, obs_idx, load_rgb=True, load_mask_id=True, load_full_traj=False
            )
            q_act = self.store.load_episode_slices(
                vidx, qid, act_idx, load_rgb=False, load_mask_id=False, load_full_traj=False
            )

            query_xyz_list.append(q_obs["xyz"])
            query_state_list.append(q_obs["state"])
            query_valid_list.append(q_obs["valid"])
            target_action_list.append(q_act["action"])
            if "mask_id" in q_obs:
                query_mask_list.append(q_obs["mask_id"])
            else:
                query_has_mask_for_all = False
            if "rgb" in q_obs:
                query_rgb_list.append(q_obs["rgb"])
            else:
                query_has_rgb_for_all = False

        # Support for each j: all except j
        cond_xyz_J, cond_state_J, cond_valid_J, cond_mask_J, cond_rgb_J = [], [], [], [], []
        cond_traj_J, cond_traj_mask_J = [], []
        cond_has_mask_for_all_j = True
        cond_has_rgb_for_all_j = True
        cond_has_traj_for_all_j = True
        for j in range(J):
            support_ids = [int(ep_ids[k]) for k in range(J) if k != j]  # length K
            if len(support_ids) != cfg.K:
                return None

            cond_xyz, cond_state, cond_valid, cond_mask, cond_rgb, cond_traj = [], [], [], [], [], []
            cond_has_mask_this_j = True
            cond_has_rgb_this_j = True
            cond_has_traj_this_j = True
            for sid in support_ids:
                Ts = self.store.episode_length(vidx, sid)
                if Ts <= 0:
                    return None
                kf = self._sample_keyframes(Ts, cfg.L, rng)
                if kf.shape[0] != cfg.L:
                    return None
                s = self.store.load_episode_slices(
                    vidx, sid, kf, load_rgb=True, load_mask_id=True, load_full_traj=True
                )
                cond_xyz.append(s["xyz"])       # [L,N,3]
                cond_state.append(s["state"])   # [L,S]
                cond_valid.append(s["valid"])   # [L,N]
                if "mask_id" in s:
                    cond_mask.append(s["mask_id"])
                else:
                    cond_has_mask_this_j = False
                if "rgb" in s:
                    cond_rgb.append(s["rgb"])
                else:
                    cond_has_rgb_this_j = False
                if "traj" in s:
                    cond_traj.append(self._stride_traj(s["traj"]))
                else:
                    cond_has_traj_this_j = False
            cond_xyz_J.append(torch.stack(cond_xyz, 0))       # [K,L,N,3]
            cond_state_J.append(torch.stack(cond_state, 0))   # [K,L,S]
            cond_valid_J.append(torch.stack(cond_valid, 0))   # [K,L,N]
            if cond_has_mask_this_j and len(cond_mask) == cfg.K:
                cond_mask_J.append(torch.stack(cond_mask, 0))  # [K,L,N]
            else:
                cond_has_mask_for_all_j = False
            if cond_has_rgb_this_j and len(cond_rgb) == cfg.K:
                cond_rgb_J.append(torch.stack(cond_rgb, 0))    # [K,L,N,3]
            else:
                cond_has_rgb_for_all_j = False
            if cond_has_traj_this_j and len(cond_traj) == cfg.K:
                traj_padded, traj_mask = self._pack_traj_list(cond_traj)
                cond_traj_J.append(traj_padded)            # [K,T,D]
                cond_traj_mask_J.append(traj_mask)         # [K,T]
            else:
                cond_has_traj_for_all_j = False

        out: Dict[str, Any] = {
            "cond_xyz": torch.stack(cond_xyz_J, 0),            # [J,K,L,N,3]
            "cond_state": torch.stack(cond_state_J, 0),        # [J,K,L,S]
            "cond_valid": torch.stack(cond_valid_J, 0),        # [J,K,L,N]
            "query_xyz": torch.stack(query_xyz_list, 0),       # [J,T_obs,N,3]
            "query_state": torch.stack(query_state_list, 0),   # [J,T_obs,S]
            "query_valid": torch.stack(query_valid_list, 0),   # [J,T_obs,N]
            "target_action": torch.stack(target_action_list, 0),  # [J,H,A]
            "meta": {"vidx": vidx, "episode_ids": ep_ids, "t0": np.asarray(t0_list, dtype=np.int64)},
        }
        if cond_has_mask_for_all_j and len(cond_mask_J) == J:
            out["cond_mask_id"] = torch.stack(cond_mask_J, 0)  # [J,K,L,N]
        if query_has_mask_for_all and len(query_mask_list) == J:
            out["query_mask_id"] = torch.stack(query_mask_list, 0)  # [J,T_obs,N]
        if cond_has_rgb_for_all_j and len(cond_rgb_J) == J:
            out["cond_rgb"] = torch.stack(cond_rgb_J, 0)       # [J,K,L,N,3]
        if query_has_rgb_for_all and len(query_rgb_list) == J:
            out["query_rgb"] = torch.stack(query_rgb_list, 0)  # [J,T_obs,N,3]
        if cond_has_traj_for_all_j and len(cond_traj_J) == J:
            out["cond_traj"], out["cond_traj_mask"] = self._pad_traj_items(cond_traj_J, cond_traj_mask_J)
        return out


class ICILPretrainBatchIterable(ICILSamplerCore, IterableDataset):
    """
    Yields full pretraining batches.
    Each yielded item is already batched; DataLoader should use batch_size=1 and unwrap.

    Output shapes:
      cond_xyz: [B,K,L,N,3]
      query_xyz: [B,T_obs,N,3]
      target_action: [B,H,A]
      etc.
    """

    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        batch_size_B: int,
        num_batches: int = 100_000,
        seed: int = 0,
        num_tries_per_item: int = 50,
    ):
        IterableDataset.__init__(self)
        ICILSamplerCore.__init__(
            self,
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )
        self.B = batch_size_B
        self.num_batches = num_batches
        if self.B < 1:
            raise ValueError("batch_size_B must be >= 1.")
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

    def __iter__(self):
        self._iter_counter += 1
        rng = self._rng()
        start, end = self._worker_batch_range(self.num_batches)
        target_batches = max(0, end - start)

        for batch_idx in range(target_batches):
            samples: List[Dict[str, Any]] = []
            tries = 0
            while len(samples) < self.B and tries < self.num_tries_per_item * self.B:
                s = self._build_one_sample(rng)
                tries += 1
                if s is None:
                    continue
                samples.append(s)

            if len(samples) < self.B:
                raise RuntimeError(
                    f"Could not assemble a full pretrain batch (worker_batch_idx={batch_idx}, "
                    f"need={self.B}, got={len(samples)}). Increase data/episodes or reduce B."
                )

            # stack batch dimension B
            batch: Dict[str, Any] = {
                "cond_xyz": torch.stack([s["cond_xyz"] for s in samples], 0),
                "cond_state": torch.stack([s["cond_state"] for s in samples], 0),
                "cond_valid": torch.stack([s["cond_valid"] for s in samples], 0),
                "query_xyz": torch.stack([s["query_xyz"] for s in samples], 0),
                "query_state": torch.stack([s["query_state"] for s in samples], 0),
                "query_valid": torch.stack([s["query_valid"] for s in samples], 0),
                "target_action": torch.stack([s["target_action"] for s in samples], 0),
                "meta": [s["meta"] for s in samples],
            }
            if all("cond_mask_id" in s for s in samples):
                batch["cond_mask_id"] = torch.stack([s["cond_mask_id"] for s in samples], 0)
            if all("query_mask_id" in s for s in samples):
                batch["query_mask_id"] = torch.stack([s["query_mask_id"] for s in samples], 0)
            if all("cond_rgb" in s for s in samples):
                batch["cond_rgb"] = torch.stack([s["cond_rgb"] for s in samples], 0)
            if all("query_rgb" in s for s in samples):
                batch["query_rgb"] = torch.stack([s["query_rgb"] for s in samples], 0)
            if all("cond_traj" in s and "cond_traj_mask" in s for s in samples):
                batch["cond_traj"], batch["cond_traj_mask"] = self._pad_traj_items(
                    [s["cond_traj"] for s in samples],
                    [s["cond_traj_mask"] for s in samples],
                )

            yield batch


class ICILMetaBatchIterable(ICILSamplerCore, IterableDataset):
    """
    Yields meta/TTT batches with leave-one-out inner items.

    For each "task" (variation file), we sample K+1 episodes once and build J=K+1 inner items.
    We do this for B tasks per batch.

    Output shapes (structured):
      cond_xyz: [B, J, K, L, N, 3]
      query_xyz: [B, J, T_obs, N, 3]
      target_action: [B, J, H, A]

    If flatten_inner=True, we flatten (B,J) into batch dimension B' = B*J to reuse
    the same model forward as pretraining:
      cond_xyz: [B', K, L, N, 3]
      query_xyz: [B', T_obs, N, 3]
      target_action: [B', H, A]
    """

    def __init__(
        self,
        store,
        *,
        cfg: ICILConfig,
        task_batch_size_B: int,
        num_batches: int = 50_000,
        seed: int = 0,
        num_tries_per_item: int = 50,
        flatten_inner: bool = True,
    ):
        IterableDataset.__init__(self)
        ICILSamplerCore.__init__(
            self,
            store,
            cfg=cfg,
            seed=seed,
            num_tries_per_item=num_tries_per_item,
        )
        self.B = task_batch_size_B
        self.num_batches = num_batches
        self.flatten_inner = flatten_inner
        if self.B < 1:
            raise ValueError("task_batch_size_B must be >= 1.")
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1.")

    def __iter__(self):
        self._iter_counter += 1
        rng = self._rng()
        cfg = self.cfg
        J = cfg.K + 1
        start, end = self._worker_batch_range(self.num_batches)
        target_batches = max(0, end - start)

        for batch_idx in range(target_batches):
            items: List[Dict[str, Any]] = []

            tries = 0
            while len(items) < self.B and tries < self.num_tries_per_item * self.B:
                bundle = self._build_task_bundle(rng)
                tries += 1
                if bundle is None:
                    continue
                loo = self._build_leave_one_out_items(rng, bundle)
                if loo is None:
                    continue
                items.append(loo)

            if len(items) < self.B:
                raise RuntimeError(
                    f"Could not assemble a full meta batch (worker_batch_idx={batch_idx}, "
                    f"need={self.B}, got={len(items)}). Increase data/episodes or reduce task batch size."
                )

            # Stack B tasks
            cond_xyz = torch.stack([it["cond_xyz"] for it in items], 0)         # [B,J,K,L,N,3]
            cond_state = torch.stack([it["cond_state"] for it in items], 0)     # [B,J,K,L,S]
            query_xyz = torch.stack([it["query_xyz"] for it in items], 0)       # [B,J,T_obs,N,3]
            query_state = torch.stack([it["query_state"] for it in items], 0)   # [B,J,T_obs,S]
            target_action = torch.stack([it["target_action"] for it in items], 0)  # [B,J,H,A]

            batch: Dict[str, Any] = {
                "cond_xyz": cond_xyz,
                "cond_state": cond_state,
                "cond_valid": torch.stack([it["cond_valid"] for it in items], 0),
                "query_xyz": query_xyz,
                "query_state": query_state,
                "query_valid": torch.stack([it["query_valid"] for it in items], 0),
                "target_action": target_action,
                "meta": [it["meta"] for it in items],
            }

            if all("cond_mask_id" in it for it in items):
                batch["cond_mask_id"] = torch.stack([it["cond_mask_id"] for it in items], 0)  # [B,J,K,L,N]
            if all("query_mask_id" in it for it in items):
                batch["query_mask_id"] = torch.stack([it["query_mask_id"] for it in items], 0)  # [B,J,T_obs,N]
            if all("cond_rgb" in it for it in items):
                batch["cond_rgb"] = torch.stack([it["cond_rgb"] for it in items], 0)  # [B,J,K,L,N,3]
            if all("query_rgb" in it for it in items):
                batch["query_rgb"] = torch.stack([it["query_rgb"] for it in items], 0)  # [B,J,T_obs,N,3]
            if all("cond_traj" in it and "cond_traj_mask" in it for it in items):
                batch["cond_traj"], batch["cond_traj_mask"] = self._pad_traj_items(
                    [it["cond_traj"] for it in items],
                    [it["cond_traj_mask"] for it in items],
                )  # [B,J,K,T,D], [B,J,K,T]

            if self.flatten_inner:
                # Flatten (B,J) -> B'
                B = cond_xyz.shape[0]
                batch["cond_xyz"] = batch["cond_xyz"].reshape(B * J, *batch["cond_xyz"].shape[2:])
                batch["cond_state"] = batch["cond_state"].reshape(B * J, *batch["cond_state"].shape[2:])
                batch["cond_valid"] = batch["cond_valid"].reshape(B * J, *batch["cond_valid"].shape[2:])
                batch["query_xyz"] = batch["query_xyz"].reshape(B * J, *batch["query_xyz"].shape[2:])
                batch["query_state"] = batch["query_state"].reshape(B * J, *batch["query_state"].shape[2:])
                batch["query_valid"] = batch["query_valid"].reshape(B * J, *batch["query_valid"].shape[2:])
                batch["target_action"] = batch["target_action"].reshape(B * J, *batch["target_action"].shape[2:])
                if "cond_mask_id" in batch:
                    batch["cond_mask_id"] = batch["cond_mask_id"].reshape(B * J, *batch["cond_mask_id"].shape[2:])
                if "query_mask_id" in batch:
                    batch["query_mask_id"] = batch["query_mask_id"].reshape(B * J, *batch["query_mask_id"].shape[2:])
                if "cond_rgb" in batch:
                    batch["cond_rgb"] = batch["cond_rgb"].reshape(B * J, *batch["cond_rgb"].shape[2:])
                if "query_rgb" in batch:
                    batch["query_rgb"] = batch["query_rgb"].reshape(B * J, *batch["query_rgb"].shape[2:])
                if "cond_traj" in batch:
                    batch["cond_traj"] = batch["cond_traj"].reshape(B * J, *batch["cond_traj"].shape[2:])
                if "cond_traj_mask" in batch:
                    batch["cond_traj_mask"] = batch["cond_traj_mask"].reshape(B * J, *batch["cond_traj_mask"].shape[2:])
                flat_meta: List[Dict[str, Any]] = []
                for task_meta in batch["meta"]:
                    vidx = int(task_meta["vidx"])
                    episode_ids = np.asarray(task_meta["episode_ids"], dtype=np.int64)
                    t0s = np.asarray(task_meta["t0"], dtype=np.int64)
                    for j in range(J):
                        query_ep = int(episode_ids[j])
                        support_eps = [int(episode_ids[k]) for k in range(J) if k != j]
                        flat_meta.append(
                            {
                                "vidx": vidx,
                                "inner_index": j,
                                "query_episode": query_ep,
                                "support_episodes": support_eps,
                                "t0": int(t0s[j]),
                            }
                        )
                batch["meta_task"] = batch["meta"]
                batch["meta"] = flat_meta
                batch["meta_flattened"] = True
            else:
                batch["meta_flattened"] = False

            yield batch


def _assert_pretrain_batch(
    batch: Dict[str, Any],
    *,
    cfg: ICILConfig,
    B: int,
    N: int,
    S: int,
    A: int,
) -> None:
    assert tuple(batch["cond_xyz"].shape) == (B, cfg.K, cfg.L, N, 3), batch["cond_xyz"].shape
    assert tuple(batch["cond_state"].shape) == (B, cfg.K, cfg.L, S), batch["cond_state"].shape
    assert tuple(batch["cond_valid"].shape) == (B, cfg.K, cfg.L, N), batch["cond_valid"].shape
    assert tuple(batch["query_xyz"].shape) == (B, cfg.T_obs, N, 3), batch["query_xyz"].shape
    assert tuple(batch["query_state"].shape) == (B, cfg.T_obs, S), batch["query_state"].shape
    assert tuple(batch["query_valid"].shape) == (B, cfg.T_obs, N), batch["query_valid"].shape
    assert tuple(batch["target_action"].shape) == (B, cfg.H, A), batch["target_action"].shape
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == B
    if "cond_mask_id" in batch:
        assert tuple(batch["cond_mask_id"].shape) == (B, cfg.K, cfg.L, N), batch["cond_mask_id"].shape
    if "query_mask_id" in batch:
        assert tuple(batch["query_mask_id"].shape) == (B, cfg.T_obs, N), batch["query_mask_id"].shape
    if "cond_rgb" in batch:
        assert tuple(batch["cond_rgb"].shape) == (B, cfg.K, cfg.L, N, 3), batch["cond_rgb"].shape
    if "query_rgb" in batch:
        assert tuple(batch["query_rgb"].shape) == (B, cfg.T_obs, N, 3), batch["query_rgb"].shape
    if "cond_traj" in batch:
        assert batch["cond_traj"].shape[:2] == (B, cfg.K), batch["cond_traj"].shape
        assert batch["cond_traj"].shape[-1] == A, batch["cond_traj"].shape
        assert batch["cond_traj_mask"].shape == batch["cond_traj"].shape[:-1], batch["cond_traj_mask"].shape


def _assert_meta_batch(
    batch: Dict[str, Any],
    *,
    cfg: ICILConfig,
    B: int,
    N: int,
    S: int,
    A: int,
    flatten_inner: bool,
) -> None:
    J = cfg.K + 1
    if flatten_inner:
        bp = B * J
        assert tuple(batch["cond_xyz"].shape) == (bp, cfg.K, cfg.L, N, 3), batch["cond_xyz"].shape
        assert tuple(batch["cond_state"].shape) == (bp, cfg.K, cfg.L, S), batch["cond_state"].shape
        assert tuple(batch["cond_valid"].shape) == (bp, cfg.K, cfg.L, N), batch["cond_valid"].shape
        assert tuple(batch["query_xyz"].shape) == (bp, cfg.T_obs, N, 3), batch["query_xyz"].shape
        assert tuple(batch["query_state"].shape) == (bp, cfg.T_obs, S), batch["query_state"].shape
        assert tuple(batch["query_valid"].shape) == (bp, cfg.T_obs, N), batch["query_valid"].shape
        assert tuple(batch["target_action"].shape) == (bp, cfg.H, A), batch["target_action"].shape
        assert batch.get("meta_flattened", False) is True
        assert isinstance(batch["meta"], list) and len(batch["meta"]) == bp
        assert isinstance(batch.get("meta_task"), list) and len(batch["meta_task"]) == B
        if "cond_mask_id" in batch:
            assert tuple(batch["cond_mask_id"].shape) == (bp, cfg.K, cfg.L, N), batch["cond_mask_id"].shape
        if "query_mask_id" in batch:
            assert tuple(batch["query_mask_id"].shape) == (bp, cfg.T_obs, N), batch["query_mask_id"].shape
        if "cond_rgb" in batch:
            assert tuple(batch["cond_rgb"].shape) == (bp, cfg.K, cfg.L, N, 3), batch["cond_rgb"].shape
        if "query_rgb" in batch:
            assert tuple(batch["query_rgb"].shape) == (bp, cfg.T_obs, N, 3), batch["query_rgb"].shape
        if "cond_traj" in batch:
            assert batch["cond_traj"].shape[:2] == (bp, cfg.K), batch["cond_traj"].shape
            assert batch["cond_traj"].shape[-1] == A, batch["cond_traj"].shape
            assert batch["cond_traj_mask"].shape == batch["cond_traj"].shape[:-1], batch["cond_traj_mask"].shape
    else:
        assert tuple(batch["cond_xyz"].shape) == (B, J, cfg.K, cfg.L, N, 3), batch["cond_xyz"].shape
        assert tuple(batch["cond_state"].shape) == (B, J, cfg.K, cfg.L, S), batch["cond_state"].shape
        assert tuple(batch["cond_valid"].shape) == (B, J, cfg.K, cfg.L, N), batch["cond_valid"].shape
        assert tuple(batch["query_xyz"].shape) == (B, J, cfg.T_obs, N, 3), batch["query_xyz"].shape
        assert tuple(batch["query_state"].shape) == (B, J, cfg.T_obs, S), batch["query_state"].shape
        assert tuple(batch["query_valid"].shape) == (B, J, cfg.T_obs, N), batch["query_valid"].shape
        assert tuple(batch["target_action"].shape) == (B, J, cfg.H, A), batch["target_action"].shape
        assert batch.get("meta_flattened", True) is False
        assert isinstance(batch["meta"], list) and len(batch["meta"]) == B
        if "cond_mask_id" in batch:
            assert tuple(batch["cond_mask_id"].shape) == (B, J, cfg.K, cfg.L, N), batch["cond_mask_id"].shape
        if "query_mask_id" in batch:
            assert tuple(batch["query_mask_id"].shape) == (B, J, cfg.T_obs, N), batch["query_mask_id"].shape
        if "cond_rgb" in batch:
            assert tuple(batch["cond_rgb"].shape) == (B, J, cfg.K, cfg.L, N, 3), batch["cond_rgb"].shape
        if "query_rgb" in batch:
            assert tuple(batch["query_rgb"].shape) == (B, J, cfg.T_obs, N, 3), batch["query_rgb"].shape
        if "cond_traj" in batch:
            assert batch["cond_traj"].shape[:3] == (B, J, cfg.K), batch["cond_traj"].shape
            assert batch["cond_traj"].shape[-1] == A, batch["cond_traj"].shape
            assert batch["cond_traj_mask"].shape == batch["cond_traj"].shape[:-1], batch["cond_traj_mask"].shape


def _discover_cached_tasks(cache_root: Path) -> List[str]:
    if not cache_root.is_dir():
        return []
    tasks: List[str] = []
    for p in sorted(cache_root.iterdir()):
        if p.is_dir() and any(p.glob("variation*.h5")):
            tasks.append(p.name)
    return tasks


def _build_store_from_cache(
    cache_root: Path,
    tasks: Optional[Sequence[str]],
    *,
    keep_open_per_worker: bool,
) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    selected_tasks = list(tasks) if tasks else _discover_cached_tasks(cache_root)
    if not selected_tasks:
        raise RuntimeError(f"No cached tasks found under {cache_root}")

    keys = []
    missing = []
    for task in selected_tasks:
        task_keys = build_variation_keys(cache_root, task)
        if not task_keys:
            missing.append(task)
            continue
        keys.extend(task_keys)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise RuntimeError(f"No variation*.h5 files found for tasks: {missing_csv}")
    if not keys:
        raise RuntimeError(f"No variation*.h5 files found under {cache_root}")

    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected_tasks


def _infer_store_dims(store) -> Tuple[int, int, int]:
    for vidx in range(len(store)):
        episode_ids = store.list_episode_ids(vidx)
        if episode_ids.shape[0] == 0:
            continue
        episode_id = int(episode_ids[0])
        T = int(store.episode_length(vidx, episode_id))
        if T <= 0:
            continue
        one = store.load_episode_slices(
            vidx,
            episode_id,
            np.asarray([0], dtype=np.int64),
            load_rgb=False,
            load_mask_id=False,
        )
        N = int(one["xyz"].shape[1])
        S = int(one["state"].shape[1])
        A = int(one["action"].shape[1])
        return N, S, A
    raise RuntimeError("Unable to infer dimensions from store (no non-empty episodes found).")
