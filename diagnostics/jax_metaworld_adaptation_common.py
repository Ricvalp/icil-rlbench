from __future__ import annotations

# Avoid JAX grabbing the full workstation GPU just to run diagnostics.
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict

from diagnostics.jax_adaptation_support_common import mse_metrics, save_inner_loss_artifacts
from icil_jax_query_memory.models.config_utils import build_model_config_from_raw, resolve_dtype
from icil_jax_query_memory.models.query_memory_direct_regression import QueryMemoryDirectRegressionModel
from icil_jax_query_memory.train.config import QueryMemoryMetaConfig
from icil_jax_query_memory.train.step import create_adapt_with_stats_fn, create_predict_fn
from icil_jax_query_memory.utils.checkpoints import load_checkpoint
from icil_metaworld.data.import_utils import import_metaworld
from icil_metaworld.data.metaworld_cache import MetaWorldEpisodeStore
from icil_metaworld.data.metaworld_task_builder import (
    MetaWorldICILConfig,
    MetaWorldMAMLTaskSpec,
    MetaWorldQueryMemoryTaskBuilder,
)
from icil_metaworld.data.observation_filter import ObservationFilterConfig, filter_observation, normalize_env_name


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def numpy_batch_to_jax(batch: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    out: Dict[str, jnp.ndarray] = {}
    for key in (
        'query_xyz',
        'query_state',
        'query_valid',
        'target_action',
        'query_rgb',
        'query_mask_id',
        'demo_id',
        'support_demo_id',
        'chunk_start',
        'support_chunk_start',
    ):
        if key in batch:
            out[key] = jnp.asarray(batch[key])
    return out


def torch_batch_to_numpy(batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in batch.items():
        if key == 'meta':
            continue
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
    return out


def stack_inner_batches(batches: Sequence[Dict[str, Any]], *, inner_steps: int) -> Dict[str, np.ndarray]:
    if int(inner_steps) <= 0:
        return {}
    if not batches:
        raise ValueError('Need at least one support batch when inner_steps > 0.')
    np_batches = [torch_batch_to_numpy(batch) for batch in batches]
    expanded = [np_batches[idx % len(np_batches)] for idx in range(int(inner_steps))]
    keys = sorted(set.intersection(*(set(batch.keys()) for batch in expanded)))
    return {key: np.stack([batch[key] for batch in expanded], axis=0) for key in keys}


def mean_metric_dict(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {key: float(np.mean([float(item[key]) for item in items])) for key in keys}


def _cfg_dict(ckpt: Dict[str, Any], key: str) -> Dict[str, Any]:
    cfg = ckpt.get('config', {}) if isinstance(ckpt.get('config', None), dict) else {}
    value = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    return value if isinstance(value, dict) else {}


def _auto_bool(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ('', 'auto'):
            return 'auto'
        if text in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if text in ('0', 'false', 'no', 'n', 'off'):
            return False
        raise ValueError(f'Expected bool or "auto", got {value!r}.')
    if value is None:
        return 'auto'
    return bool(value)


def _auto_str(value: Any) -> str:
    text = str(value or '').strip()
    return 'auto' if text.lower() in ('', 'auto') else text


def _model_cfg_from_checkpoint(cfg: ConfigDict, ckpt: Dict[str, Any], *, compute_dtype: jnp.dtype):
    raw = None
    ckpt_cfg = ckpt.get('config', {}) if isinstance(ckpt.get('config', None), dict) else {}
    maybe = ckpt_cfg.get('model', None) if isinstance(ckpt_cfg, dict) else None
    if isinstance(maybe, dict):
        raw = ConfigDict(maybe)
    if raw is None:
        raw = cfg.model
    resolved = ckpt_cfg.get('resolved', {}) if isinstance(ckpt_cfg, dict) else {}
    state_dim = int(resolved.get('state_dim', getattr(cfg.model, 'state_dim', 39)))
    action_dim = int(resolved.get('action_dim', getattr(cfg.model, 'action_dim', 4)))

    # Checkpoints trained during the transition to family-level MetaWorld WRITE
    # support observation conditioning may not contain this knob. Default to the
    # corrected family-level semantics unless the checkpoint/config is explicit.
    decoder_raw = raw.query_memory_direct_regression
    if not hasattr(decoder_raw, 'write_use_support_obs'):
        data_cfg = _cfg_dict(ckpt, 'data')
        family_level = not bool(data_cfg.get('sample_same_task_instance', False))
        if family_level and bool(getattr(decoder_raw, 'separate_write_read_heads', False)):
            decoder_raw.write_use_support_obs = True
    return build_model_config_from_raw(raw, state_dim=state_dim, action_dim=action_dim, compute_dtype=compute_dtype)


def resolve_metaworld_data_cfg(cfg: ConfigDict, ckpt: Dict[str, Any]) -> MetaWorldICILConfig:
    ckpt_dataset = _cfg_dict(ckpt, 'dataset')
    ckpt_data = _cfg_dict(ckpt, 'data')
    use_ckpt = bool(getattr(cfg.dataset, 'use_checkpoint_dataset_config', True))

    def _ival(section: Dict[str, Any], name: str, local_default: int) -> int:
        if use_ckpt and name in section:
            return int(section[name])
        return int(getattr(cfg.dataset, name, local_default))

    def _bdata(name: str, default: bool) -> bool:
        local = _auto_bool(getattr(cfg.data, name, 'auto'))
        if local != 'auto':
            return bool(local)
        if use_ckpt and name in ckpt_data:
            return bool(ckpt_data[name])
        return bool(default)

    def _sdata(name: str, default: str) -> str:
        local = _auto_str(getattr(cfg.data, name, 'auto'))
        if local != 'auto':
            return str(local)
        if use_ckpt and name in ckpt_data:
            return str(ckpt_data[name])
        return str(default)

    def _bdataset(name: str, default: bool) -> bool:
        local = _auto_bool(getattr(cfg.dataset, name, 'auto'))
        if local != 'auto':
            return bool(local)
        if use_ckpt and name in ckpt_dataset:
            return bool(ckpt_dataset[name])
        return bool(default)

    def _sdataset(name: str, default: str) -> str:
        local = _auto_str(getattr(cfg.dataset, name, 'auto'))
        if local != 'auto':
            return str(local)
        if use_ckpt and name in ckpt_dataset:
            return str(ckpt_dataset[name])
        return str(default)

    return MetaWorldICILConfig(
        K=_ival(ckpt_dataset, 'K', 4),
        T_obs=_ival(ckpt_dataset, 'T_obs', 2),
        H=_ival(ckpt_dataset, 'H', 8),
        stride=_ival(ckpt_dataset, 'stride', 1),
        action_stride=_ival(ckpt_dataset, 'action_stride', 1),
        pad_short_chunks=_bdataset('pad_short_chunks', False),
        action_representation=_sdataset('action_representation', 'absolute'),
        task_sampling=_sdata('task_sampling', 'task_instance_uniform'),
        sample_same_task_name=_bdata('sample_same_task_name', True),
        sample_same_task_instance=_bdata('sample_same_task_instance', False),
        allow_support_query_same_episode=_bdata('allow_support_query_same_episode', False),
        support_zero_goal=_bdata('support_zero_goal', False),
        query_zero_goal=_bdata('query_zero_goal', False),
    )


def resolve_memory_cfg(cfg: ConfigDict, ckpt: Dict[str, Any]) -> QueryMemoryMetaConfig:
    ckpt_maml = _cfg_dict(ckpt, 'maml')
    model_cfg = _cfg_dict(ckpt, 'model')
    decoder_cfg = model_cfg.get('query_memory_direct_regression', {}) if isinstance(model_cfg, dict) else {}

    def _ival(name: str, default: int) -> int:
        local = getattr(cfg.memory_ttt, name, -1)
        if int(local) >= 0:
            return int(local)
        if name in ckpt_maml:
            return int(ckpt_maml[name])
        return int(default)

    def _fval(name: str, default: float) -> float:
        local = getattr(cfg.memory_ttt, name, -1.0)
        if float(local) >= 0.0:
            return float(local)
        if name in ckpt_maml:
            return float(ckpt_maml[name])
        return float(default)

    inner_loss_mode = str(ckpt_maml.get('inner_loss_mode', getattr(cfg.memory_ttt, 'inner_loss_mode', 'write'))).lower()
    if hasattr(cfg.memory_ttt, 'inner_loss_mode') and str(cfg.memory_ttt.inner_loss_mode):
        inner_loss_mode = str(cfg.memory_ttt.inner_loss_mode).lower()
    if inner_loss_mode not in ('read', 'write'):
        raise ValueError("memory_ttt.inner_loss_mode must be 'read' or 'write'.")
    inner_lr_mode = str(ckpt_maml.get('inner_lr_mode', getattr(cfg.memory_ttt, 'inner_lr_mode', 'fixed')))
    if inner_lr_mode != 'fixed':
        raise ValueError(f'MetaWorld diagnostics only support fixed inner_lr_mode, got {inner_lr_mode!r}.')

    first_order = bool(ckpt_maml.get('first_order', False))
    if hasattr(cfg.memory_ttt, 'first_order'):
        first_order = bool(cfg.memory_ttt.first_order)
    memory_layer_norm_after_update = bool(
        ckpt_maml.get(
            'memory_layer_norm_after_update',
            getattr(
                cfg.memory_ttt,
                'memory_layer_norm_after_update',
                decoder_cfg.get('memory_layer_norm_after_update', False) if isinstance(decoder_cfg, dict) else False,
            ),
        )
    )
    return QueryMemoryMetaConfig(
        inner_steps=_ival('inner_steps', 1),
        inner_lr=_fval('inner_lr', 1e-1),
        inner_lr_mode='fixed',
        outer_lr=0.0,
        weight_decay=0.0,
        max_grad_norm=_fval('max_grad_norm', 1.0),
        num_queries_per_step=_ival('num_queries_per_step', 16),
        num_query_loss_samples=_ival('num_query_loss_samples', 8),
        num_inner_batches=_ival('num_inner_batches', 0),
        holdout_index=-1,
        first_order=first_order,
        reuse_diffusion_noise=False,
        grad_accum_steps=1,
        inner_loss_mode=inner_loss_mode,
        memory_layer_norm_after_update=memory_layer_norm_after_update,
    )


def load_metaworld_policy_components(cfg: ConfigDict) -> Dict[str, Any]:
    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    ckpt = load_checkpoint(checkpoint_path)
    compute_dtype = resolve_dtype(str(getattr(cfg, 'compute_dtype', 'float32')))
    model_cfg = _model_cfg_from_checkpoint(cfg, ckpt, compute_dtype=compute_dtype)
    data_cfg = resolve_metaworld_data_cfg(cfg, ckpt)
    memory_cfg = resolve_memory_cfg(cfg, ckpt)
    model = QueryMemoryDirectRegressionModel(cfg=model_cfg)
    params = ckpt['params']
    adapt_with_stats_fn = create_adapt_with_stats_fn(
        model=model,
        inner_steps=int(memory_cfg.inner_steps),
        inner_lr=float(memory_cfg.inner_lr),
        max_grad_norm=float(memory_cfg.max_grad_norm),
        first_order=bool(memory_cfg.first_order),
        inner_loss_mode=str(memory_cfg.inner_loss_mode),
        memory_layer_norm_after_update=bool(memory_cfg.memory_layer_norm_after_update),
    )
    predict_fn = create_predict_fn(model=model)
    return {
        'checkpoint_path': checkpoint_path,
        'ckpt': ckpt,
        'params': params,
        'model_cfg': model_cfg,
        'data_cfg': data_cfg,
        'memory_cfg': memory_cfg,
        'model': model,
        'adapt_with_stats_fn': adapt_with_stats_fn,
        'predict_fn': predict_fn,
        'state_dim': int(model_cfg.state_dim),
        'action_dim': int(model_cfg.action_dim),
    }


def load_store(cfg: ConfigDict, ckpt: Optional[Dict[str, Any]] = None) -> MetaWorldEpisodeStore:
    cache_root = _auto_str(getattr(cfg.data, 'cache_root', ''))
    if cache_root == 'auto':
        cache_root = ''
    if not cache_root and ckpt is not None:
        cache_root = str(_cfg_dict(ckpt, 'data').get('cache_root', ''))
    if not cache_root:
        cache_root = os.environ.get('ICIL_METAWORLD_MT10_GOAL_CACHE_ROOT') or os.environ.get('ICIL_METAWORLD_MT10_CACHE_ROOT', '')
    root = Path(cache_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'MetaWorld cache root not found: {root}')
    return MetaWorldEpisodeStore(
        root,
        keep_open_per_worker=False,
        preload_to_memory=bool(getattr(cfg.data, 'preload_to_memory', False)),
    )


def make_builder(
    store: MetaWorldEpisodeStore,
    data_cfg: MetaWorldICILConfig,
    *,
    seed: int,
    task_names: Sequence[str] = (),
    exclude_tasks: Sequence[str] = (),
    num_tries_per_item: int = 100,
) -> MetaWorldQueryMemoryTaskBuilder:
    return MetaWorldQueryMemoryTaskBuilder(
        store,
        cfg=data_cfg,
        seed=int(seed),
        num_tries_per_item=int(num_tries_per_item),
        task_names=task_names,
        exclude_tasks=exclude_tasks,
    )


def _normalised_task_filter(values: Sequence[str] | Any) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    return tuple(normalize_env_name(str(v)) for v in raw if str(v).strip())


def resolve_task_name(
    store: MetaWorldEpisodeStore,
    requested: str,
    rng: np.random.Generator,
    *,
    task_names: Sequence[str] = (),
    exclude_tasks: Sequence[str] = (),
) -> str:
    value = str(requested or '').strip()
    include = set(_normalised_task_filter(task_names))
    exclude = set(_normalised_task_filter(exclude_tasks))
    if value:
        task_name = normalize_env_name(value)
        if include and task_name not in include:
            raise ValueError(f'Requested task {task_name!r} is not in cfg.data.tasks={sorted(include)}.')
        if task_name in exclude:
            raise ValueError(f'Requested task {task_name!r} is excluded by cfg.data.exclude_tasks.')
        return task_name
    names = [name for name in store.list_task_names() if (not include or name in include) and name not in exclude]
    if not names:
        raise RuntimeError('MetaWorld store has no tasks after cfg.data.tasks/exclude_tasks filtering.')
    return str(names[int(rng.integers(0, len(names)))])


def sample_task_spec_for_family(
    *,
    store: MetaWorldEpisodeStore,
    data_cfg: MetaWorldICILConfig,
    task_name: str,
    seed: int,
    rng: np.random.Generator,
) -> MetaWorldMAMLTaskSpec:
    builder = make_builder(store, data_cfg, seed=int(seed), task_names=(task_name,))
    spec = builder.build_task_spec(rng)
    if spec is None:
        raise RuntimeError(f'Could not sample MetaWorld task spec for {task_name}.')
    return spec


def sample_wrong_family(store: MetaWorldEpisodeStore, target_task_name: str, requested: str, rng: np.random.Generator) -> str:
    if str(requested or '').strip():
        wrong = normalize_env_name(str(requested))
        if wrong == target_task_name:
            raise ValueError('wrong task family must differ from target task family.')
        return wrong
    names = [name for name in store.list_task_names() if str(name) != str(target_task_name)]
    if not names:
        raise RuntimeError('Need at least two task families for wrong-family adaptation.')
    return str(names[int(rng.integers(0, len(names)))])


def prepare_support_inner(
    *,
    builder: MetaWorldQueryMemoryTaskBuilder,
    task: MetaWorldMAMLTaskSpec,
    memory_cfg: QueryMemoryMetaConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if int(memory_cfg.inner_steps) <= 0:
        return {}
    num_inner_batches = int(memory_cfg.inner_steps) if int(memory_cfg.num_inner_batches) <= 0 else min(
        int(memory_cfg.num_inner_batches), int(memory_cfg.inner_steps)
    )
    batches = [
        builder.build_support_batch(task, count=int(memory_cfg.num_queries_per_step), rng=rng)
        for _ in range(max(1, num_inner_batches))
    ]
    return stack_inner_batches(batches, inner_steps=int(memory_cfg.inner_steps))


def adapt_memory_for_task(
    *,
    params: Any,
    adapt_with_stats_fn: Any,
    builder: MetaWorldQueryMemoryTaskBuilder,
    task: MetaWorldMAMLTaskSpec,
    memory_cfg: QueryMemoryMetaConfig,
    rng: np.random.Generator,
    run_dir: Optional[Path] = None,
    stem: str = 'adaptation',
) -> Dict[str, Any]:
    support_inner = prepare_support_inner(builder=builder, task=task, memory_cfg=memory_cfg, rng=rng)
    adapted_memory, inner_losses, inner_grad_norms = adapt_with_stats_fn(params, numpy_batch_to_jax(support_inner))
    inner_losses_np = np.asarray(inner_losses, dtype=np.float32)
    inner_grads_np = np.asarray(inner_grad_norms, dtype=np.float32)
    artifacts: Dict[str, str] = {}
    if run_dir is not None:
        artifacts = save_inner_loss_artifacts(
            inner_losses=[float(v) for v in inner_losses_np.tolist()],
            inner_grad_norms=[float(v) for v in inner_grads_np.tolist()],
            run_dir=run_dir,
            stem=stem,
        )
    return {
        'memory_tokens': adapted_memory,
        'inner_losses': [float(v) for v in inner_losses_np.tolist()],
        'inner_grad_norms': [float(v) for v in inner_grads_np.tolist()],
        'avg_inner_loss': float(np.mean(inner_losses_np)) if inner_losses_np.size else 0.0,
        'support_ids': [int(v) for v in task.support_episode_ids],
        'support_task_instance_ids': [int(v) for v in task.support_task_instance_ids],
        'query_task_instance_id': int(task.query_task_instance_id),
        **artifacts,
    }


def build_query_batch_for_task(
    *,
    builder: MetaWorldQueryMemoryTaskBuilder,
    task: MetaWorldMAMLTaskSpec,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    batch = builder.build_query_batch(task, count=int(batch_size), rng=rng)
    return torch_batch_to_numpy(batch)


def initial_memory(params: Any) -> jnp.ndarray:
    return jnp.asarray(params['memory_token_init'])


def evaluate_query_predictions(
    *,
    params: Any,
    predict_fn: Any,
    query_batch: Dict[str, np.ndarray],
    memories: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    query_jax = numpy_batch_to_jax(query_batch)
    target = np.asarray(query_batch['target_action'], dtype=np.float32)
    out: Dict[str, Dict[str, Any]] = {}
    for name, memory in memories.items():
        pred = np.asarray(predict_fn(params, query_jax, memory), dtype=np.float32)
        metrics = mse_metrics(pred, target)
        xyz_dim = min(3, int(target.shape[-1]))
        metrics['xyz_l1'] = float(np.mean(np.abs(pred[..., :xyz_dim] - target[..., :xyz_dim])))
        if target.shape[-1] > 3:
            metrics['gripper_l1'] = float(np.mean(np.abs(pred[..., 3:] - target[..., 3:])))
        out[name] = {'pred': pred, 'metrics': metrics}
    names = list(memories.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            delta = float(np.mean(np.abs(out[a]['pred'] - out[b]['pred'])))
            out.setdefault('_prediction_deltas', {})[f'{a}_vs_{b}_mean_abs'] = delta
    return out


def strip_predictions(result: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in result.items():
        if key == '_prediction_deltas':
            clean[key] = value
        else:
            clean[key] = {'metrics': value['metrics']}
    return clean


def make_metaworld_env_for_instance(
    *,
    task_name: str,
    task_instance_id: int,
    benchmark_name: str,
    split: str,
    benchmark_seed: int,
    camera_name: str = 'corner',
    width: int = 320,
    height: int = 240,
    force_goal_observable: bool = False,
) -> Any:
    metaworld = import_metaworld()
    benchmark_name = str(benchmark_name).upper()
    if benchmark_name == 'MT1':
        benchmark = metaworld.MT1(normalize_env_name(task_name), seed=int(benchmark_seed))
    elif benchmark_name == 'ML1':
        benchmark = metaworld.ML1(normalize_env_name(task_name), seed=int(benchmark_seed))
    else:
        benchmark = getattr(metaworld, benchmark_name)(seed=int(benchmark_seed))
    classes = benchmark.train_classes if str(split).lower() == 'train' else benchmark.test_classes
    tasks = benchmark.train_tasks if str(split).lower() == 'train' else benchmark.test_tasks
    task_name = normalize_env_name(task_name)
    matching = [task for task in tasks if str(task.env_name) == task_name]
    if task_name not in classes:
        raise KeyError(f'{task_name} not found in {benchmark_name}/{split}.')
    if int(task_instance_id) >= len(matching):
        raise IndexError(f'task_instance_id={task_instance_id} but {task_name} has {len(matching)} instances.')
    env = classes[task_name](
        render_mode='rgb_array',
        camera_name=str(camera_name) if str(camera_name) else None,
        width=int(width),
        height=int(height),
    )
    env.set_task(matching[int(task_instance_id)])
    if bool(force_goal_observable):
        env._partially_observable = False
        try:
            del env.sawyer_observation_space
        except Exception:
            pass
    return env


def reset_env(env: Any, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        out = env.reset(seed=int(seed)) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return np.asarray(obs, dtype=np.float32), dict(info or {})


def step_env(env: Any, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    out = env.step(np.asarray(action, dtype=np.float32))
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    else:
        raise RuntimeError(f'Unsupported MetaWorld step return format: {type(out).__name__} len={len(out)}')
    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), dict(info or {})


def success_from_info(info: Dict[str, Any]) -> bool:
    value = info.get('success', False)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def obs_to_model_state(obs: np.ndarray, *, state_dim: int) -> np.ndarray:
    raw = np.asarray(obs, dtype=np.float32).reshape(-1)
    if int(state_dim) == raw.shape[0]:
        return raw.astype(np.float32)
    if int(state_dim) == raw.shape[0] - 3:
        model, _ = filter_observation(
            raw,
            ObservationFilterConfig(variant='no_task_no_goal', remove_task_id=True, remove_goal=True, normalize=False),
        )
        return model.astype(np.float32)
    raise ValueError(f'Cannot map raw MetaWorld obs dim {raw.shape[0]} to model state_dim={state_dim}.')


def render_frame(env: Any, *, flip_vertical: bool = True) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f'Unexpected MetaWorld render frame shape: {frame.shape}')
    if bool(flip_vertical):
        frame = np.flipud(frame)
    return frame.astype(np.uint8)


def write_gif(path: Path, frames: Sequence[np.ndarray], *, fps: int = 20) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio

    imageio.mimsave(path, [np.asarray(f, dtype=np.uint8) for f in frames], duration=1.0 / max(1, int(fps)))
    return str(path)
