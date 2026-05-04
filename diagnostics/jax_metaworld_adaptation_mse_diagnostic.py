from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from diagnostics.jax_metaworld_adaptation_common import (
    adapt_memory_for_task,
    build_query_batch_for_task,
    evaluate_query_predictions,
    initial_memory,
    load_metaworld_policy_components,
    load_store,
    make_builder,
    mean_metric_dict,
    resolve_task_name,
    sample_task_spec_for_family,
    sample_wrong_family,
    set_seed,
    strip_predictions,
)

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_metaworld_adaptation_mse_diagnostic.py',
    help_string='Path to ml_collections config file.',
)


def _adaptation_summary(adaptation: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in adaptation.items() if key != 'memory_tokens'}


def diagnose(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    set_seed(seed)
    components = load_metaworld_policy_components(cfg)
    params = components['params']
    data_cfg = components['data_cfg']
    memory_cfg = components['memory_cfg']
    predict_fn = components['predict_fn']
    adapt_with_stats_fn = components['adapt_with_stats_fn']
    checkpoint_path = components['checkpoint_path']

    store = load_store(cfg, components['ckpt'])
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / 'jax_metaworld_adaptation_mse' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    rng = np.random.default_rng(seed + 9001)
    per_batch: List[Dict[str, Any]] = []
    metric_buckets: Dict[str, List[Dict[str, float]]] = {
        'no_adaptation': [],
        'same_family_adaptation': [],
        'wrong_family_adaptation': [],
    }
    deltas: Dict[str, List[float]] = {}
    same_adaptations: List[Dict[str, Any]] = []
    wrong_adaptations: List[Dict[str, Any]] = []

    try:
        for batch_idx in range(int(cfg.mse.num_batches)):
            target_task_name = resolve_task_name(store, str(cfg.task.name), rng)
            wrong_task_name = sample_wrong_family(store, target_task_name, str(cfg.adaptation.different_task_name), rng)
            target_builder = make_builder(
                store,
                data_cfg,
                seed=seed + 100 + batch_idx,
                task_names=(target_task_name,),
                num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
            )
            wrong_builder = make_builder(
                store,
                data_cfg,
                seed=seed + 200 + batch_idx,
                task_names=(wrong_task_name,),
                num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
            )
            target_spec = sample_task_spec_for_family(
                store=store,
                data_cfg=data_cfg,
                task_name=target_task_name,
                seed=seed + 300 + batch_idx,
                rng=rng,
            )
            wrong_spec = sample_task_spec_for_family(
                store=store,
                data_cfg=data_cfg,
                task_name=wrong_task_name,
                seed=seed + 400 + batch_idx,
                rng=rng,
            )
            same_adaptation = adapt_memory_for_task(
                params=params,
                adapt_with_stats_fn=adapt_with_stats_fn,
                builder=target_builder,
                task=target_spec,
                memory_cfg=memory_cfg,
                rng=rng,
                run_dir=run_dir if batch_idx == 0 else None,
                stem='same_family_adaptation',
            )
            wrong_adaptation = adapt_memory_for_task(
                params=params,
                adapt_with_stats_fn=adapt_with_stats_fn,
                builder=wrong_builder,
                task=wrong_spec,
                memory_cfg=memory_cfg,
                rng=rng,
                run_dir=run_dir if batch_idx == 0 else None,
                stem='wrong_family_adaptation',
            )
            query_batch = build_query_batch_for_task(
                builder=target_builder,
                task=target_spec,
                batch_size=int(cfg.mse.batch_size),
                rng=rng,
            )
            result = evaluate_query_predictions(
                params=params,
                predict_fn=predict_fn,
                query_batch=query_batch,
                memories={
                    'no_adaptation': initial_memory(params),
                    'same_family_adaptation': same_adaptation['memory_tokens'],
                    'wrong_family_adaptation': wrong_adaptation['memory_tokens'],
                },
            )
            clean = strip_predictions(result)
            for key in metric_buckets:
                metric_buckets[key].append(clean[key]['metrics'])
            for key, value in clean.get('_prediction_deltas', {}).items():
                deltas.setdefault(key, []).append(float(value))
            same_adaptations.append(_adaptation_summary(same_adaptation))
            wrong_adaptations.append(_adaptation_summary(wrong_adaptation))
            per_batch.append(
                {
                    'batch_idx': int(batch_idx),
                    'target_task_name': str(target_task_name),
                    'wrong_task_name': str(wrong_task_name),
                    'target_query_task_instance_id': int(target_spec.query_task_instance_id),
                    'target_support_task_instance_ids': [int(v) for v in target_spec.support_task_instance_ids],
                    'wrong_support_task_instance_ids': [int(v) for v in wrong_spec.support_task_instance_ids],
                    'metrics': clean,
                }
            )

        mean_metrics = {key: mean_metric_dict(values) for key, values in metric_buckets.items()}
        same = mean_metrics['same_family_adaptation']
        wrong = mean_metrics['wrong_family_adaptation']
        no = mean_metrics['no_adaptation']
        summary: Dict[str, Any] = {
            'checkpoint_path': str(checkpoint_path),
            'cache_root': str(store.root),
            'dataset': {
                'K': int(data_cfg.K),
                'T_obs': int(data_cfg.T_obs),
                'H': int(data_cfg.H),
                'sample_same_task_instance': bool(data_cfg.sample_same_task_instance),
                'support_zero_goal': bool(data_cfg.support_zero_goal),
                'query_zero_goal': bool(data_cfg.query_zero_goal),
            },
            'memory_ttt': {
                'inner_steps': int(memory_cfg.inner_steps),
                'inner_lr': float(memory_cfg.inner_lr),
                'max_grad_norm': float(memory_cfg.max_grad_norm),
                'num_queries_per_step': int(memory_cfg.num_queries_per_step),
                'num_inner_batches': int(memory_cfg.num_inner_batches),
                'first_order': bool(memory_cfg.first_order),
                'inner_loss_mode': str(memory_cfg.inner_loss_mode),
            },
            'mse': {
                'num_batches': int(cfg.mse.num_batches),
                'batch_size': int(cfg.mse.batch_size),
                'mean': mean_metrics,
                'same_minus_no': {k: float(same[k] - no[k]) for k in sorted(same.keys() & no.keys())},
                'wrong_minus_same': {k: float(wrong[k] - same[k]) for k in sorted(wrong.keys() & same.keys())},
                'prediction_deltas_mean_abs': {k: float(np.mean(v)) for k, v in sorted(deltas.items())},
                'per_batch': per_batch,
            },
            'same_family_adaptations': same_adaptations,
            'wrong_family_adaptations': wrong_adaptations,
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        with (run_dir / 'mse_summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary['mse'], f, indent=2)
        logging.info('no adaptation: %s', no)
        logging.info('same-family adaptation: %s', same)
        logging.info('wrong-family adaptation: %s', wrong)
        logging.info('diagnostics written to %s', run_dir)
    finally:
        store.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Unexpected positional arguments.')
    diagnose(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
