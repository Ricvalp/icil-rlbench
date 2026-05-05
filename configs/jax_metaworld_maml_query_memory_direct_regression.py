import os

from configs.jax_maml_query_memory_direct_regression import get_config as get_base_config


def _metaworld_env(name, default, generic_name=None):
    value = os.environ.get(f'ICIL_METAWORLD_{name}', '')
    if value:
        return value
    if generic_name:
        return os.environ.get(generic_name, default)
    return default


def _metaworld_output_parent(default):
    return _metaworld_env('OUTPUT_PARENT_DIR', default, 'ICIL_OUTPUT_PARENT_DIR')


def _metaworld_checkpoint_parent(default):
    return _metaworld_env('CHECKPOINT_PARENT_DIR', default, 'ICIL_CHECKPOINT_PARENT_DIR')


def _metaworld_cache_root(default):
    return _metaworld_env('CACHE_ROOT', default)


def _metaworld_wandb_project(default):
    return os.environ.get('ICIL_METAWORLD_WANDB_PROJECT', os.environ.get('WANDB_PROJECT', default))


def _apply_metaworld_overrides(cfg):
    cfg.output_parent_dir = _metaworld_output_parent(
        os.path.join('output_data_playground_v3', '.experiments', 'jax_metaworld_query_memory_direct_regression_maml', 'runs'),
    )
    cfg.train.checkpoint_parent_dir = _metaworld_checkpoint_parent(
        os.path.join('output_data_playground_v3', '.experiments', 'jax_metaworld_query_memory_direct_regression_maml', 'checkpoints'),
    )

    cfg.data.source = 'metaworld'
    cfg.data.cache_root = _metaworld_cache_root(
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'button_press_ml1_train'),
    )
    cfg.data.tasks = ('button-press-v3',)
    cfg.data.exclude_tasks = ()
    cfg.data.keep_open_per_worker = True
    cfg.data.preload_to_memory = False
    # MetaWorld cache items are low-dimensional. Keeping this at 0 avoids
    # spawned PyTorch workers importing the JAX runtime and consuming multiple
    # GB each on local workstations.
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.prefetch_factor = 2
    cfg.data.task_sampling = 'task_instance_uniform'
    cfg.data.sample_same_task_name = True
    cfg.data.sample_same_task_instance = True
    cfg.data.allow_support_query_same_episode = False
    cfg.data.support_zero_goal = False
    cfg.data.query_zero_goal = False

    cfg.dataset.K = 4
    cfg.dataset.L = 0
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 8
    cfg.dataset.stride = 1
    cfg.dataset.action_stride = 1
    cfg.dataset.pad_short_chunks = False
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.num_tries_per_item = 100

    decoder = cfg.model.query_memory_direct_regression
    decoder.d_model = 256
    decoder.n_heads = 4
    decoder.decoder_layers = 4
    decoder.decoder_mlp_mult = 4
    decoder.horizon = int(cfg.dataset.H)
    decoder.memory_num_tokens = 64
    decoder.context_attention_mode = 'two_ctx'
    decoder.loss_type = 'l1'

    encoder = cfg.model.simple_query_point_encoder
    encoder.d_model = 256
    encoder.use_rgb = False
    encoder.use_mask_id = False
    encoder.use_gripper_point_features = False
    encoder.max_T_obs = int(cfg.dataset.T_obs)
    encoder.add_state_token = True

    obj = cfg.model.object_centric_state
    obj.d_model = 256
    obj.max_T_obs = int(cfg.dataset.T_obs)
    obj.goal_available = True

    support_encoder = cfg.model.support_encoder_memory
    support_encoder.d_model = 256
    support_encoder.n_heads = 4
    support_encoder.memory_num_tokens = int(decoder.memory_num_tokens)
    support_encoder.max_demo_id = int(decoder.write_max_demo_id) if hasattr(decoder, 'write_max_demo_id') else 16
    support_encoder.max_time_bins = int(decoder.write_max_time_bins) if hasattr(decoder, 'write_max_time_bins') else 512

    cfg.train.num_steps = 100000
    cfg.train.batch_size = 8
    cfg.train.weight_decay = 1e-4
    cfg.train.log_every = 10
    cfg.train.ckpt_every = 1000
    cfg.train.use_amp = False
    cfg.train.amp_dtype = 'bf16'

    cfg.maml.first_order = False
    cfg.maml.inner_loss_mode = 'read'
    cfg.maml.inner_steps = 2
    cfg.maml.inner_lr = 1e-2
    cfg.maml.inner_lr_mode = 'fixed'
    cfg.maml.outer_lr = 1e-4
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.num_queries_per_step = 16
    cfg.maml.num_inner_batches = 0
    cfg.maml.num_query_loss_samples = 8
    cfg.maml.holdout_index = -1
    cfg.maml.reuse_diffusion_noise = False
    cfg.maml.grad_accum_steps = 1
    cfg.maml.training_mode_metrics_only = True

    cfg.wandb.project = _metaworld_wandb_project('icil-jax-metaworld-query-memory-direct-maml')
    cfg.wandb.tags = ('metaworld', 'button-press-v3', 'read_inner')
    cfg.wandb.n_sample_steps = 0
    return cfg


def get_config():
    cfg = get_base_config()
    return _apply_metaworld_overrides(cfg)
