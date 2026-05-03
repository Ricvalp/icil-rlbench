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
    LiveConditioningProcessor,
    adapt_memory_for_support,
    build_cached_support_ids,
    build_rlbench_env,
    build_task_store,
    load_policy_components,
    query_stride_mode_from_eval,
    run_eval_episode,
    set_seed,
    summarize_rollout_results,
)
from icil.models.maml.query_memory_tasks import QueryMemoryTaskBuilder

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_eval_adaptation_support_rollout.py',
    help_string='Path to ml_collections config file.',
)


def _task_label(task_name: str, variation: int) -> str:
    return f'{task_name}_var{int(variation)}'


def _adaptation_summary(adaptation: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in adaptation.items() if key != 'memory_tokens'}


def evaluate(cfg: ConfigDict) -> None:
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
    action_dim = int(components['action_dim'])

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
        / 'jax_adaptation_support_rollout'
        / f'{_task_label(task_name, variation)}_adapt_other_{_task_label(different_task_name, different_variation)}_{run_id}'
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    target_store = build_task_store(cache_root, task_name)
    different_store = build_task_store(cache_root, different_task_name)
    env = None
    try:
        rng = np.random.default_rng(seed + 1701)
        target_builder = QueryMemoryTaskBuilder(
            target_store,
            cfg=dataset_cfg,
            seed=seed + 1711,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )
        different_builder = QueryMemoryTaskBuilder(
            different_store,
            cfg=dataset_cfg,
            seed=seed + 1721,
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

        env, task_env = build_rlbench_env(cfg, task_name)
        processor = LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=use_rgb,
            use_mask_id=use_mask_id,
            seed=seed + 1731,
        )
        query_stride_mode = query_stride_mode_from_eval(cfg)
        rollout_results: Dict[str, List[Dict[str, Any]]] = {
            'same_task_adaptation': [],
            'different_task_adaptation': [],
        }
        for support_label, adaptation in (
            ('same_task_adaptation', same_adaptation),
            ('different_task_adaptation', different_adaptation),
        ):
            for episode_index in range(int(cfg.task.num_eval_episodes)):
                result = run_eval_episode(
                    episode_index=episode_index,
                    support_label=support_label,
                    task_env=task_env,
                    variation=variation,
                    params=params,
                    predict_fn=predict_fn,
                    adapted_memory_tokens=adaptation['memory_tokens'],
                    dataset_cfg=dataset_cfg,
                    query_stride_mode=query_stride_mode,
                    processor=processor,
                    cfg=cfg,
                    run_dir=run_dir,
                    action_dim=action_dim,
                    use_mask_id=use_mask_id,
                    use_rgb=use_rgb,
                )
                rollout_results[support_label].append(result)
                logging.info(
                    '%s episode=%d success=%s steps=%d error=%s',
                    support_label,
                    episode_index,
                    result['success'],
                    result['env_steps'],
                    result['error'],
                )

        same_summary = summarize_rollout_results(rollout_results['same_task_adaptation'])
        different_summary = summarize_rollout_results(rollout_results['different_task_adaptation'])
        summary: Dict[str, Any] = {
            'checkpoint_path': str(checkpoint_path),
            'target': {'task': task_name, 'variation': variation},
            'same_task_adaptation': {
                'task': task_name,
                'variation': variation,
                'support_ids': same_support_ids,
                **_adaptation_summary(same_adaptation),
                **same_summary,
            },
            'different_task_adaptation': {
                'task': different_task_name,
                'variation': different_variation,
                'support_ids': different_support_ids,
                **_adaptation_summary(different_adaptation),
                **different_summary,
            },
            'different_minus_same_success_rate': (
                float(different_summary['success_rate']) - float(same_summary['success_rate'])
            ),
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
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        with (run_dir / 'rollout_summary.json').open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'same_task_adaptation': same_summary,
                    'different_task_adaptation': different_summary,
                    'different_minus_same_success_rate': summary['different_minus_same_success_rate'],
                },
                f,
                indent=2,
            )
        logging.info('same-task adaptation success_rate=%.3f', float(same_summary['success_rate']))
        logging.info('different-task adaptation success_rate=%.3f', float(different_summary['success_rate']))
        logging.info('diagnostics written to %s', run_dir)
    finally:
        target_store.close()
        different_store.close()
        if env is not None:
            env.shutdown()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Unexpected positional arguments.')
    evaluate(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
