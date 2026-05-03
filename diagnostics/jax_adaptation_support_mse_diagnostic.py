from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from diagnostics.jax_adaptation_support_common import (
    adapt_memory_for_support,
    as_bool,
    build_cached_support_ids,
    build_target_query_batch,
    build_task_store,
    load_policy_components,
    mean_metric_dict,
    mse_metrics,
    numpy_batch_to_jax,
    set_seed,
)
from icil.models.maml.query_memory_tasks import QueryMemoryTaskBuilder

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_diagnose_adaptation_support_mse.py',
    help_string='Path to ml_collections config file.',
)


def _task_label(task_name: str, variation: int) -> str:
    return f'{task_name}_var{int(variation)}'


def _adaptation_summary(adaptation: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in adaptation.items() if key != 'memory_tokens'}


def diagnose(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    set_seed(seed)
    components = load_policy_components(cfg)
    checkpoint_path = components['checkpoint_path']
    params = components['params']
    dataset_cfg = components['dataset_cfg']
    memory_cfg = components['memory_cfg']
    adapt_with_stats_fn = components['adapt_with_stats_fn']
    predict_fn = components['predict_fn']
    use_mask_id = bool(components['use_mask_id'])
    use_rgb = bool(components['use_rgb'])

    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    different_task_name = str(cfg.adaptation.different_task_name)
    different_variation = int(cfg.adaptation.different_variation)
    if task_name == different_task_name and variation == different_variation:
        raise ValueError('Different-task adaptation must use a different task/variation than the target task.')

    cache_root = Path(str(cfg.conditioning.cache_root)).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f'Cache root not found: {cache_root}')

    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = (
        Path(str(cfg.output.root_dir)).expanduser().resolve()
        / 'jax_adaptation_support_mse'
        / f'{_task_label(task_name, variation)}_adapt_other_{_task_label(different_task_name, different_variation)}_{run_id}'
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    target_store = build_task_store(cache_root, task_name)
    different_store = build_task_store(cache_root, different_task_name)
    try:
        rng = np.random.default_rng(seed + 701)
        target_builder = QueryMemoryTaskBuilder(
            target_store,
            cfg=dataset_cfg,
            seed=seed + 711,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )
        different_builder = QueryMemoryTaskBuilder(
            different_store,
            cfg=dataset_cfg,
            seed=seed + 721,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )

        target_vidx, same_support_ids = build_cached_support_ids(
            store=target_store,
            dataset_cfg=dataset_cfg,
            variation=variation,
            rng=rng,
        )
        different_vidx, different_support_ids = build_cached_support_ids(
            store=different_store,
            dataset_cfg=dataset_cfg,
            variation=different_variation,
            rng=rng,
        )

        same_adaptation = adapt_memory_for_support(
            params=params,
            adapt_with_stats_fn=adapt_with_stats_fn,
            task_builder=target_builder,
            vidx=target_vidx,
            support_ids=same_support_ids,
            memory_cfg=memory_cfg,
            use_mask_id=use_mask_id,
            use_rgb=use_rgb,
            rng=rng,
            run_dir=run_dir,
            stem='same_task_adaptation',
        )
        different_adaptation = adapt_memory_for_support(
            params=params,
            adapt_with_stats_fn=adapt_with_stats_fn,
            task_builder=different_builder,
            vidx=different_vidx,
            support_ids=different_support_ids,
            memory_cfg=memory_cfg,
            use_mask_id=use_mask_id,
            use_rgb=use_rgb,
            rng=rng,
            run_dir=run_dir,
            stem='different_task_adaptation',
        )

        same_metrics: List[Dict[str, float]] = []
        different_metrics: List[Dict[str, float]] = []
        pred_deltas: List[float] = []
        exclude_support = same_support_ids if as_bool(cfg.mse.exclude_same_task_support_episodes) else ()

        for batch_idx in range(int(cfg.mse.num_batches)):
            query_batch = build_target_query_batch(
                task_builder=target_builder,
                store=target_store,
                vidx=target_vidx,
                batch_size=int(cfg.mse.batch_size),
                rng=np.random.default_rng(seed + 1009 + batch_idx),
                use_mask_id=use_mask_id,
                use_rgb=use_rgb,
                exclude_episode_ids=exclude_support,
            )
            query_jax = numpy_batch_to_jax(query_batch)
            pred_same = np.asarray(predict_fn(params, query_jax, same_adaptation['memory_tokens']))
            pred_different = np.asarray(predict_fn(params, query_jax, different_adaptation['memory_tokens']))
            target = np.asarray(query_batch['target_action'])
            same_metrics.append(mse_metrics(pred_same, target))
            different_metrics.append(mse_metrics(pred_different, target))
            pred_deltas.append(float(np.mean(np.square(pred_different - pred_same))))

        same_mean = mean_metric_dict(same_metrics)
        different_mean = mean_metric_dict(different_metrics)
        summary: Dict[str, Any] = {
            'checkpoint_path': str(checkpoint_path),
            'target': {'task': task_name, 'variation': variation},
            'same_task_adaptation': {
                'task': task_name,
                'variation': variation,
                'support_ids': same_support_ids,
                **_adaptation_summary(same_adaptation),
            },
            'different_task_adaptation': {
                'task': different_task_name,
                'variation': different_variation,
                'support_ids': different_support_ids,
                **_adaptation_summary(different_adaptation),
            },
            'dataset': {
                'K': int(dataset_cfg.K),
                'L': int(dataset_cfg.L),
                'T_obs': int(dataset_cfg.T_obs),
                'H': int(dataset_cfg.H),
                'stride': int(dataset_cfg.stride),
                'action_representation': str(dataset_cfg.action_representation),
            },
            'memory_ttt': {
                'inner_steps': int(memory_cfg.inner_steps),
                'inner_lr': float(memory_cfg.inner_lr),
                'max_grad_norm': float(memory_cfg.max_grad_norm),
                'num_queries_per_step': int(memory_cfg.num_queries_per_step),
                'num_inner_batches': int(memory_cfg.num_inner_batches),
                'first_order': bool(memory_cfg.first_order),
            },
            'mse': {
                'num_batches': int(cfg.mse.num_batches),
                'batch_size': int(cfg.mse.batch_size),
                'same_task_adaptation': same_mean,
                'different_task_adaptation': different_mean,
                'different_minus_same': {
                    key: float(different_mean[key] - same_mean[key])
                    for key in sorted(same_mean.keys() & different_mean.keys())
                },
                'pred_different_vs_same_mse': float(np.mean(pred_deltas)) if pred_deltas else None,
                'per_batch_same_task_adaptation': same_metrics,
                'per_batch_different_task_adaptation': different_metrics,
            },
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        with (run_dir / 'mse_summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary['mse'], f, indent=2)
        logging.info('same-task adaptation MSE: %s', same_mean)
        logging.info('different-task adaptation MSE: %s', different_mean)
        logging.info('diagnostics written to %s', run_dir)
    finally:
        target_store.close()
        different_store.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Unexpected positional arguments.')
    diagnose(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
