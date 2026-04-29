from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from absl import app, logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm

from icil.action_representation import decode_action_chunk
from icil.datasets.in_context_imitation_learning.icil_datasets import ICILConfig
from icil.datasets.in_context_imitation_learning.variation_store import (
    VariationStore,
    build_variation_keys,
)
from icil.eval.eval_single_task_perceiver_direct_regression import (
    _LiveConditioningProcessor,
    _build_query_window,
    _build_rlbench_env,
    _extract_rgb_frame,
    _normalize_quaternion_xyzw,
    _sanitize_action,
    _support_cache_root_from_eval_and_checkpoint,
    _warn_if_cached_num_points_mismatch,
    _write_video,
)
from icil.models import (
    QueryMemoryDirectRegressionBuilderConfig,
    build_query_memory_builder_config_from_configdict,
    build_query_memory_direct_regression_policy,
)
from icil.models.maml.memory_core import (
    adapt_memory_tokens_for_prepared_task,
    sample_actions_with_memory_tokens,
)
from icil.models.maml.query_memory_tasks import QueryMemoryTaskBuilder
from icil.models.maml.tasks import MAMLTaskSpec
import icil.models.maml.memory_train as memory_train_lib

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/eval_query_memory_direct_regression.py',
    help_string='Path to ml_collections config file.',
)


def _as_bool(v: Any) -> bool:
    return bool(v)



def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def _resolve_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and str(device_str).startswith('cuda'):
        return torch.device(str(device_str))
    return torch.device('cpu')



def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint object type: {type(checkpoint).__name__}')
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model' payload is not a state_dict dictionary.")
    if state_dict and all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
    return checkpoint, state_dict



def _model_config_from_checkpoint_or_default(
    cfg: ConfigDict,
    ckpt: Dict[str, Any],
) -> QueryMemoryDirectRegressionBuilderConfig:
    model_from_ckpt: Optional[Dict[str, Any]] = None
    if isinstance(ckpt.get('config', None), dict):
        raw = ckpt['config'].get('model', None)
        if isinstance(raw, dict):
            model_from_ckpt = raw
    if model_from_ckpt is not None:
        return build_query_memory_builder_config_from_configdict(
            ConfigDict(model_from_ckpt),
            as_bool=_as_bool,
        )
    return build_query_memory_builder_config_from_configdict(cfg.model, as_bool=_as_bool)



def _dataset_config_from_eval_and_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any]) -> ICILConfig:
    ckpt_dataset = {}
    if isinstance(ckpt.get('config', None), dict):
        ckpt_dataset = ckpt['config'].get('dataset', {}) or {}

    use_ckpt = _as_bool(getattr(cfg.dataset, 'use_checkpoint_dataset_config', True))

    def _ival(name: str, default: int) -> int:
        if use_ckpt and name in ckpt_dataset:
            return int(ckpt_dataset[name])
        return int(getattr(cfg.dataset, name, default))

    action_representation = str(getattr(cfg.dataset, 'action_representation', 'absolute'))
    if use_ckpt and 'action_representation' in ckpt_dataset:
        action_representation = str(ckpt_dataset['action_representation'])

    return ICILConfig(
        K=_ival('K', 1),
        L=_ival('L', 1),
        T_obs=_ival('T_obs', 1),
        H=_ival('H', 1),
        stride=_ival('stride', 1),
        action_representation=action_representation,
    )



def _query_stride_mode_from_eval(cfg: ConfigDict) -> str:
    mode = str(getattr(cfg.dataset, 'query_stride_mode', 'dataset')).lower()
    if mode not in ('dataset', 'consecutive'):
        raise ValueError("cfg.dataset.query_stride_mode must be one of: dataset, consecutive.")
    return mode



def _resolve_use_mask_id(model_cfg: QueryMemoryDirectRegressionBuilderConfig) -> bool:
    if str(model_cfg.query_encoder_name) == 'simple_query_point_encoder':
        return bool(model_cfg.simple_query_point_encoder.use_mask_id)
    if str(model_cfg.query_encoder_name) == 'dp3_query_frame_encoder':
        return bool(model_cfg.dp3_query_frame_encoder.use_mask_id)
    raise ValueError(f'Unknown query_encoder_name={model_cfg.query_encoder_name!r}.')



def _resolve_use_rgb(model_cfg: QueryMemoryDirectRegressionBuilderConfig) -> bool:
    if str(model_cfg.query_encoder_name) == 'simple_query_point_encoder':
        return bool(model_cfg.simple_query_point_encoder.use_rgb)
    if str(model_cfg.query_encoder_name) == 'dp3_query_frame_encoder':
        return bool(model_cfg.dp3_query_frame_encoder.use_rgb)
    raise ValueError(f'Unknown query_encoder_name={model_cfg.query_encoder_name!r}.')



def _infer_state_action_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    try:
        action_dim = int(state_dict['action_out.weight'].shape[0])
    except KeyError as exc:
        raise KeyError("Could not infer action_dim from checkpoint. Expected key 'action_out.weight'.") from exc

    state_dim_candidates = (
        'context_encoder.state_proj.0.weight',
        'context_encoder.state_mlp.0.weight',
    )
    state_dim = None
    for key in state_dim_candidates:
        if key in state_dict:
            state_dim = int(state_dict[key].shape[1])
            break
    if state_dim is None:
        state_dim = 8
        logging.warning('Could not infer state_dim from checkpoint; using fallback state_dim=%d.', state_dim)
    return state_dim, action_dim



def _resolve_memory_eval_cfg(
    cfg: ConfigDict,
    ckpt: Dict[str, Any],
) -> memory_train_lib.MemoryMAMLConfig:
    ckpt_cfg = ckpt.get('config', None) if isinstance(ckpt, dict) else None
    ckpt_maml = ckpt_cfg.get('maml', {}) if isinstance(ckpt_cfg, dict) else {}

    def _ival(local_value: int, key: str, default: int) -> int:
        if int(local_value) >= 0:
            return int(local_value)
        if isinstance(ckpt_maml, dict) and key in ckpt_maml:
            return int(ckpt_maml[key])
        return int(default)

    def _fval(local_value: float, key: str, default: float) -> float:
        if float(local_value) >= 0.0:
            return float(local_value)
        if isinstance(ckpt_maml, dict) and key in ckpt_maml:
            return float(ckpt_maml[key])
        return float(default)

    local_mode = str(getattr(cfg.memory_ttt, 'inner_lr_mode', 'infer'))
    if local_mode == 'infer':
        resolved_inner_lr_mode = memory_train_lib.infer_inner_lr_mode(
            checkpoint=ckpt,
            checkpoint_config=ckpt_cfg,
            local_mode=None,
            legacy_learn_inner_lrs=getattr(cfg.memory_ttt, 'learn_inner_lrs', None),
        )
    else:
        resolved_inner_lr_mode = memory_train_lib.normalize_inner_lr_mode(local_mode)

    reuse_diffusion_noise = getattr(cfg.memory_ttt, 'reuse_diffusion_noise', None)
    if reuse_diffusion_noise is None:
        reuse_diffusion_noise = bool(ckpt_maml.get('reuse_diffusion_noise', False)) if isinstance(ckpt_maml, dict) else False

    return memory_train_lib.MemoryMAMLConfig(
        inner_steps=_ival(getattr(cfg.memory_ttt, 'inner_steps', -1), 'inner_steps', 1),
        inner_lr=_fval(getattr(cfg.memory_ttt, 'inner_lr', -1.0), 'inner_lr', 1e-3),
        inner_lr_mode=resolved_inner_lr_mode,
        outer_lr=0.0,
        weight_decay=0.0,
        max_grad_norm=_fval(getattr(cfg.memory_ttt, 'max_grad_norm', -1.0), 'max_grad_norm', 1.0),
        num_queries_per_step=_ival(getattr(cfg.memory_ttt, 'num_queries_per_step', -1), 'num_queries_per_step', 1),
        num_inner_batches=_ival(getattr(cfg.memory_ttt, 'num_inner_batches', -1), 'num_inner_batches', 0),
        num_query_loss_samples=1,
        holdout_index=-1,
        reuse_diffusion_noise=bool(reuse_diffusion_noise),
        grad_accum_steps=int(getattr(cfg.memory_ttt, 'grad_accum_steps', 1)),
    )



def _build_cached_support_ids(
    *,
    store: VariationStore,
    dataset_cfg: ICILConfig,
    variation: int,
    rng: np.random.Generator,
) -> Tuple[int, List[int]]:
    candidates = [
        idx for idx, key in enumerate(store.keys)
        if variation < 0 or int(key.variation) == int(variation)
    ]
    if not candidates:
        available = sorted({int(key.variation) for key in store.keys})
        raise RuntimeError(
            f"No cached variation found for requested variation={variation}. Available variations: {available}"
        )
    vidx = int(candidates[0] if variation >= 0 else rng.choice(np.asarray(candidates, dtype=np.int64)))
    episode_ids = store.list_episode_ids(vidx)
    if episode_ids.shape[0] < dataset_cfg.K:
        raise RuntimeError(
            f'Need at least K={dataset_cfg.K} cached support episodes, got {episode_ids.shape[0]}.'
        )
    support_ids = rng.choice(episode_ids, size=dataset_cfg.K, replace=False)
    return vidx, [int(eid) for eid in np.asarray(support_ids).tolist()]



def _prepare_support_adaptation(
    *,
    task_builder: QueryMemoryTaskBuilder,
    vidx: int,
    support_ids: Sequence[int],
    memory_cfg: memory_train_lib.MemoryMAMLConfig,
    device: torch.device,
    use_mask_id: bool,
    use_rgb: bool,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    dummy_query_episode = int(support_ids[0])
    task = MAMLTaskSpec(
        vidx=int(vidx),
        support_episode_ids=tuple(int(eid) for eid in support_ids),
        query_episode_id=int(dummy_query_episode),
    )
    memory_init_batch = task_builder.build_support_batch(
        task,
        count=1,
        rng=rng,
        load_rgb=use_rgb,
        load_mask_id=use_mask_id,
    )
    memory_init_batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in memory_init_batch.items()
    }
    if not use_mask_id:
        memory_init_batch.pop('query_mask_id', None)

    inner_batches: List[Dict[str, Any]] = []
    num_inner_batches = int(memory_cfg.inner_steps) if int(memory_cfg.num_inner_batches) <= 0 else min(int(memory_cfg.num_inner_batches), int(memory_cfg.inner_steps))
    for _ in range(max(0, num_inner_batches)):
        batch = task_builder.build_support_batch(
            task,
            count=int(memory_cfg.num_queries_per_step),
            rng=rng,
            load_rgb=use_rgb,
            load_mask_id=use_mask_id,
        )
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if not use_mask_id:
            batch.pop('query_mask_id', None)
        inner_batches.append(batch)
    return {
        'task': task,
        'memory_init_batch': memory_init_batch,
        'inner_batches': inner_batches,
    }



def _run_eval_episode(
    *,
    episode_index: int,
    task_env: Any,
    variation: int,
    model: torch.nn.Module,
    adapted_memory_tokens: torch.Tensor,
    adapted_memory_mask: Optional[torch.Tensor],
    device: torch.device,
    dataset_cfg: ICILConfig,
    query_stride_mode: str,
    processor: _LiveConditioningProcessor,
    cfg: ConfigDict,
    run_dir: Path,
    action_dim: int,
) -> Dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    if variation >= 0:
        task_env.set_variation(int(variation))
    descriptions, obs = task_env.reset()
    del descriptions

    history: List[Dict[str, torch.Tensor]] = [processor.observation_to_frame(obs)]
    frames: List[np.ndarray] = []
    if _as_bool(cfg.video.enable):
        frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))

    success = False
    terminated = False
    error: Optional[str] = None
    env_steps = 0
    execute_actions = max(1, int(cfg.control.execute_actions_per_plan))
    max_env_steps = int(cfg.task.max_env_steps)
    pbar = tqdm(total=max_env_steps, desc=f'Episode {episode_index}', leave=False, unit='step')
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(history, dataset_cfg=dataset_cfg, query_stride_mode=query_stride_mode)
            query_batch: Dict[str, Any] = {
                'query_xyz': query['query_xyz'].to(device, non_blocking=True),
                'query_state': query['query_state'].to(device, non_blocking=True),
                'query_valid': query['query_valid'].to(device, non_blocking=True),
                'target_action': torch.zeros(
                    (1, int(dataset_cfg.H), int(action_dim)),
                    device=device,
                    dtype=query['query_xyz'].dtype,
                ),
            }
            if 'query_mask_id' in query:
                query_batch['query_mask_id'] = query['query_mask_id'].to(device, non_blocking=True)
            if 'query_rgb' in query:
                query_batch['query_rgb'] = query['query_rgb'].to(device, non_blocking=True)

            with torch.no_grad():
                plan = sample_actions_with_memory_tokens(
                    model,
                    query_batch,
                    memory_tokens=adapted_memory_tokens,
                    memory_token_mask=adapted_memory_mask,
                )
            plan = decode_action_chunk(
                plan,
                query_state=query_batch['query_state'],
                representation=str(dataset_cfg.action_representation),
            )
            plan_np = plan[0].detach().cpu().numpy()
            n_exec = int(min(execute_actions, plan_np.shape[0], max_env_steps - env_steps))

            for i in range(n_exec):
                action = _sanitize_action(
                    plan_np[i],
                    normalize_quaternion=_as_bool(cfg.control.normalize_quaternion),
                    discretize_gripper=_as_bool(cfg.control.discretize_gripper),
                )
                try:
                    obs, reward, terminated = task_env.step(action.astype(np.float32))
                except InvalidActionError as exc:
                    error = f'InvalidActionError: {exc}'
                    terminated = True
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    terminated = True
                    break

                env_steps += 1
                pbar.update(1)
                success = bool(float(reward) > 0.5)
                history.append(processor.observation_to_frame(obs))
                if _as_bool(cfg.video.enable):
                    frames.append(_extract_rgb_frame(obs, str(cfg.video.camera)))
                if success or terminated or env_steps >= max_env_steps:
                    break
    finally:
        pbar.close()

    video_path = None
    if _as_bool(cfg.video.enable) and len(frames) > 0:
        video_file = run_dir / 'videos' / f'episode_{episode_index:04d}.{str(cfg.video.format).lower()}'
        video_path = str(_write_video(frames, video_file, fps=int(cfg.video.fps)))

    return {
        'episode_index': int(episode_index),
        'success': bool(success),
        'terminated': bool(terminated),
        'env_steps': int(env_steps),
        'error': error,
        'video_path': video_path,
    }



def evaluate(cfg: ConfigDict) -> None:
    seed = int(cfg.seed)
    _set_seed(seed)
    device = _resolve_device(str(cfg.device))

    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    ckpt, state_dict = _load_checkpoint(checkpoint_path, device)
    model_cfg = _model_config_from_checkpoint_or_default(cfg, ckpt)
    dataset_cfg = _dataset_config_from_eval_and_checkpoint(cfg, ckpt)
    query_stride_mode = _query_stride_mode_from_eval(cfg)
    memory_cfg = _resolve_memory_eval_cfg(cfg, ckpt)
    use_mask_id = _resolve_use_mask_id(model_cfg)
    use_rgb = _resolve_use_rgb(model_cfg)
    state_dim, action_dim = _infer_state_action_dims_from_state_dict(state_dict)
    if int(model_cfg.query_memory_direct_regression.horizon) != int(dataset_cfg.H):
        raise ValueError(
            'Query-memory direct-regression horizon must match the resolved dataset horizon. '
            f'Got model.query_memory_direct_regression.horizon={model_cfg.query_memory_direct_regression.horizon} '
            f'and dataset.H={dataset_cfg.H}.'
        )

    model = build_query_memory_direct_regression_policy(model_cfg, state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    inner_lr_schedule = memory_train_lib._build_inner_lr_schedule_wrapper(memory_cfg)
    if inner_lr_schedule is not None:
        inner_lr_schedule = inner_lr_schedule.to(device)
    memory_train_lib._load_inner_lr_schedule_state(
        inner_lr_schedule,
        checkpoint=ckpt,
        checkpoint_path=checkpoint_path,
    )

    task_name = str(cfg.task.name)
    variation = int(cfg.task.variation)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    run_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f'{task_name}_var{variation}_{run_id}'
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'resolved_eval_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    logging.info('Loading task=%s variation=%d', task_name, variation)
    logging.info('Checkpoint=%s', checkpoint_path)
    logging.info(
        'Model cfg: query_encoder=%s | horizon=%d | memory_tokens=%d | loss_type=%s',
        model_cfg.query_encoder_name,
        int(model_cfg.query_memory_direct_regression.horizon),
        int(model_cfg.query_memory_direct_regression.memory_num_tokens),
        str(model_cfg.query_memory_direct_regression.loss_type),
    )
    logging.info('Conditioning cfg: use_rgb=%s | use_mask_id=%s', use_rgb, use_mask_id)
    logging.info(
        'Dataset cfg: K=%d L=%d T_obs=%d H=%d stride=%d | query_stride_mode=%s',
        dataset_cfg.K,
        dataset_cfg.L,
        dataset_cfg.T_obs,
        dataset_cfg.H,
        dataset_cfg.stride,
        query_stride_mode,
    )
    logging.info(
        'Memory adaptation: inner_steps=%d | inner_lr=%.3e | inner_lr_mode=%s | num_queries_per_step=%d',
        int(memory_cfg.inner_steps),
        float(memory_cfg.inner_lr),
        str(memory_cfg.inner_lr_mode),
        int(memory_cfg.num_queries_per_step),
    )

    env = None
    task_env = None
    support_store: Optional[VariationStore] = None
    results: List[Dict[str, Any]] = []
    adapted_memory_tokens: Optional[torch.Tensor] = None
    adapted_memory_mask: Optional[torch.Tensor] = None
    support_info: Optional[Dict[str, Any]] = None

    try:
        if variation < 0:
            raise ValueError('Cached support conditioning requires cfg.task.variation >= 0.')
        cache_root = _support_cache_root_from_eval_and_checkpoint(cfg, ckpt)
        task_keys = build_variation_keys(cache_root, task_name)
        if not task_keys:
            raise RuntimeError(f"No cached variations found for task '{task_name}' under {cache_root}.")
        _warn_if_cached_num_points_mismatch(
            task_keys=task_keys,
            expected_num_points=int(cfg.conditioning.num_points),
            task_name=task_name,
        )
        support_store = VariationStore(task_keys, keep_open_per_worker=True)
        task_builder = QueryMemoryTaskBuilder(
            store=support_store,
            cfg=dataset_cfg,
            seed=seed + 101,
            num_tries_per_item=int(getattr(cfg.dataset, 'num_tries_per_item', 100)),
        )

        env, task_env = _build_rlbench_env(cfg, task_name)
        processor = _LiveConditioningProcessor(
            task_env=task_env,
            num_points=int(cfg.conditioning.num_points),
            use_rgb=use_rgb,
            use_mask_id=use_mask_id,
            seed=seed + 11,
        )

        rng = np.random.default_rng(seed + 17)
        for ep in range(int(cfg.task.num_eval_episodes)):
            if adapted_memory_tokens is None or _as_bool(cfg.conditioning.regenerate_demos_each_episode):
                vidx, support_ids = _build_cached_support_ids(
                    store=support_store,
                    dataset_cfg=dataset_cfg,
                    variation=variation,
                    rng=rng,
                )
                support_info = {
                    'task': support_store.keys[vidx].task,
                    'variation': int(support_store.keys[vidx].variation),
                    'support_episodes': [int(eid) for eid in support_ids],
                }
                prepared = _prepare_support_adaptation(
                    task_builder=task_builder,
                    vidx=vidx,
                    support_ids=support_ids,
                    memory_cfg=memory_cfg,
                    device=device,
                    use_mask_id=use_mask_id,
                    use_rgb=use_rgb,
                    rng=rng,
                )
                adapted_memory_tokens, adapted_memory_mask = adapt_memory_tokens_for_prepared_task(
                    model,
                    prepared,
                    cfg=memory_cfg,
                    create_graph=False,
                    inner_lr_schedule=inner_lr_schedule,
                )
                logging.info('Adapted memory tokens from variation=%d episodes=%s.', int(support_info['variation']), support_info['support_episodes'])

            res = _run_eval_episode(
                episode_index=ep,
                task_env=task_env,
                variation=variation,
                model=model,
                adapted_memory_tokens=adapted_memory_tokens,
                adapted_memory_mask=adapted_memory_mask,
                device=device,
                dataset_cfg=dataset_cfg,
                query_stride_mode=query_stride_mode,
                processor=processor,
                cfg=cfg,
                run_dir=run_dir,
                action_dim=action_dim,
            )
            results.append(res)
            logging.info(
                'Episode %d | success=%s | steps=%d%s',
                ep,
                res['success'],
                res['env_steps'],
                f" | error={res['error']}" if res['error'] else '',
            )

        n_success = sum(1 for r in results if r['success'])
        success_rate = float(n_success) / float(max(1, len(results)))
        summary = {
            'task': task_name,
            'variation': variation,
            'checkpoint_path': str(checkpoint_path),
            'num_episodes': len(results),
            'num_success': int(n_success),
            'success_rate': success_rate,
            'support_info': support_info,
            'results': results,
        }
        with (run_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logging.info(
            'Evaluation complete | success=%d/%d (%.3f) | outputs=%s',
            n_success,
            len(results),
            success_rate,
            run_dir,
        )
    finally:
        if support_store is not None:
            support_store.close()
        if env is not None:
            env.shutdown()



def main(argv: Sequence[str]) -> None:
    del argv
    cfg = _CONFIG.value
    evaluate(cfg)


if __name__ == '__main__':
    app.run(main)
