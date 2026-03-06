#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from absl import app
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from torch.profiler import profile, record_function
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from icil.datasets.in_context_imitation_learning.icil_datasets import (
    ICILConfig,
    ICILPretrainBatchIterable,
)
from icil.models import build_policy
from icil.profiling.profile_pretrain_perceiver_encoder_decoder_common import (
    as_bool,
    build_model_cfg,
    build_profiler_activities,
    build_store,
    drop_mask_ids_if_disabled,
    export_memory_timeline_artifacts,
    infer_dims,
    resolve_trace_path,
    resolve_use_mask_id,
    set_seed,
    to_device,
    unwrap_batch,
)

_TRAIN_CONFIG = config_flags.DEFINE_config_file(
    "train_config",
    default="configs/pretrain_perceiver_encoder_decoder.py",
    help_string="Path to the base pretraining ml_collections config file.",
)

_PROFILE_CONFIG = config_flags.DEFINE_config_file(
    "profile_config",
    default="configs/profile_pretrain_perceiver_encoder_decoder.py",
    help_string="Path to the profiling ml_collections config file.",
)


def profile_train(train_cfg: ConfigDict, profile_cfg: ConfigDict) -> Path:
    seed = int(train_cfg.seed)
    set_seed(seed)

    if torch.cuda.is_available() and str(train_cfg.device).startswith("cuda"):
        device = torch.device(str(train_cfg.device))
    else:
        device = torch.device("cpu")

    trace_n_steps = int(profile_cfg.trace_n_steps)
    if trace_n_steps < 1:
        raise ValueError("profile_config.trace_n_steps must be >= 1.")

    cache_root = Path(str(train_cfg.data.cache_root))
    tasks = list(train_cfg.data.tasks) if train_cfg.data.tasks is not None else []
    store, _ = build_store(
        cache_root=cache_root,
        tasks=tasks,
        keep_open_per_worker=as_bool(train_cfg.data.keep_open_per_worker),
    )

    try:
        state_dim, action_dim = infer_dims(store)

        dataset_cfg = ICILConfig(
            K=int(train_cfg.dataset.K),
            L=int(train_cfg.dataset.L),
            T_obs=int(train_cfg.dataset.T_obs),
            H=int(train_cfg.dataset.H),
            stride=int(train_cfg.dataset.stride),
        )

        train_steps = int(train_cfg.train.num_steps)
        grad_accum = int(train_cfg.train.grad_accum_steps)
        if grad_accum < 1:
            raise ValueError("train_config.train.grad_accum_steps must be >= 1.")
        if trace_n_steps > train_steps:
            raise ValueError(
                f"profile trace_n_steps={trace_n_steps} exceeds configured train num_steps={train_steps}."
            )

        total_micro_batches = trace_n_steps * grad_accum
        pretrain_dataset = ICILPretrainBatchIterable(
            store=store,
            cfg=dataset_cfg,
            batch_size_B=int(train_cfg.train.batch_size),
            num_batches=total_micro_batches,
            seed=seed,
            num_tries_per_item=int(train_cfg.dataset.num_tries_per_item),
        )

        num_workers = int(train_cfg.data.num_workers)
        pin_memory = as_bool(train_cfg.data.pin_memory) and device.type == "cuda"
        persistent_workers = as_bool(train_cfg.data.persistent_workers) and num_workers > 0
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=1,
            collate_fn=unwrap_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model = build_policy(
            build_model_cfg(train_cfg.model),
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.train.lr),
            betas=(float(train_cfg.train.beta1), float(train_cfg.train.beta2)),
            weight_decay=float(train_cfg.train.weight_decay),
        )

        use_amp = as_bool(train_cfg.train.use_amp) and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        grad_clip_norm = float(train_cfg.train.grad_clip_norm)
        use_mask_id = resolve_use_mask_id(train_cfg.model)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        step = 0
        micro_count = 0
        data_iter = iter(pretrain_loader)

        activities = build_profiler_activities(device, profile_cfg)
        trace_path = resolve_trace_path(profile_cfg)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with profile(
            activities=activities,
            record_shapes=as_bool(getattr(profile_cfg, "record_shapes", True)),
            profile_memory=as_bool(getattr(profile_cfg, "profile_memory", True)),
            with_stack=as_bool(getattr(profile_cfg, "with_stack", False)),
            with_flops=as_bool(getattr(profile_cfg, "with_flops", False)),
        ) as prof:
            pbar = tqdm(total=trace_n_steps, desc="Profiling train steps", dynamic_ncols=True)
            try:
                while step < trace_n_steps:
                    with record_function("dataloader.next"):
                        try:
                            batch = next(data_iter)
                        except StopIteration as exc:
                            raise RuntimeError(
                                "Dataloader exhausted before reaching trace_n_steps."
                            ) from exc

                    with record_function("batch.to_device"):
                        batch = to_device(batch, device)
                        model_batch = drop_mask_ids_if_disabled(batch, use_mask_id)

                    with record_function("train.forward"):
                        with torch.autocast(device_type=device.type, enabled=use_amp):
                            out = model.forward_loss(model_batch)
                            loss = out["loss"] / grad_accum

                    with record_function("train.backward"):
                        scaler.scale(loss).backward()
                    micro_count += 1

                    if micro_count % grad_accum == 0:
                        with record_function("train.optimizer_step"):
                            if grad_clip_norm > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                        step += 1
                        pbar.update(1)

                    prof.step()
            finally:
                pbar.close()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(trace_path))
        export_memory_timeline_artifacts(prof, trace_path=trace_path, device=device)
        return trace_path
    finally:
        store.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    del argv
    train_cfg = _TRAIN_CONFIG.value
    profile_cfg = _PROFILE_CONFIG.value
    profile_train(train_cfg, profile_cfg)


if __name__ == "__main__":
    app.run(main)
