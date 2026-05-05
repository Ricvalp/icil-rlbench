import os

from ml_collections import ConfigDict

from configs.jax_metaworld_maml_query_memory_write_read_direct_regression import get_config as get_write_read_base_config


PROJECT = 'icil-metaworld-objectcentric-maml'


def _ml45_goal_cache_root() -> str:
    return os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT',
        os.path.join('/mnt', 'external_storage', 'robotics', 'metaworld', 'icil_metaworld', 'ml45_goal_train_50x1'),
    )


def _ml45_same_instance_cache_root() -> str:
    return os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_SAME_INSTANCE_CACHE_ROOT',
        os.path.join('/mnt', 'external_storage', 'robotics', 'metaworld', 'icil_metaworld', 'ml45_goal_train_50x5'),
    )


def _set_object_centric_model(cfg: ConfigDict) -> None:
    d_model = 256
    memory_tokens = 64
    cfg.model.query_encoder_name = 'object_centric_state'
    cfg.model.query_goal_hidden = bool(cfg.data.query_zero_goal)
    cfg.model.support_goal_hidden = bool(cfg.data.support_zero_goal)

    obj = cfg.model.object_centric_state
    obj.d_model = d_model
    obj.max_T_obs = int(cfg.dataset.T_obs)
    obj.hand_pos_slice = (0, 3)
    obj.gripper_slice = (3, 4)
    obj.obj1_pos_slice = (4, 7)
    obj.obj2_pos_slice = (11, 14)
    obj.goal_pos_slice = (36, 39)
    obj.has_obj2 = True
    obj.goal_available = True
    obj.hidden_goal_token_policy = 'mask'

    dec = cfg.model.query_memory_direct_regression
    dec.d_model = d_model
    dec.n_heads = 4
    dec.decoder_layers = 4
    dec.decoder_mlp_mult = 4
    dec.horizon = int(cfg.dataset.H)
    dec.memory_num_tokens = memory_tokens
    dec.memory_conditioning_mode = 'cross_attn_plus_film'
    dec.memory_conditioning_strength = 1.0
    dec.separate_write_read_heads = True
    dec.shared_write_read_head = False
    dec.write_num_query_tokens = 4
    dec.write_use_demo_id_embed = True
    dec.write_use_time_embed = True
    dec.write_max_demo_id = 16
    dec.write_max_time_bins = 512
    dec.write_time_embed_type = 'continuous_sinusoidal'
    dec.write_query_mlp_mult = 2
    dec.write_use_support_obs = False
    dec.use_decoder_mode_embed = True
    dec.memory_layer_norm_after_update = False
    dec.memory_update_clip_norm = 1.0
    dec.action_loss_type = 'l1'
    dec.position_loss_weight = 1.0
    dec.rotation_loss_weight = 1.0
    dec.gripper_loss_weight = 1.0
    dec.chunk_decay = 0.0

    support = cfg.model.support_encoder_memory
    support.d_model = d_model
    support.n_heads = 4
    support.memory_num_tokens = memory_tokens
    support.support_encoder_layers = 2
    support.memory_self_attn_layers = 1
    support.mlp_mult = 2
    support.max_support_chunks = 256
    support.max_demo_id = 16
    support.max_time_bins = 512
    support.dropout = 0.0
    support.goal_visible = not bool(cfg.data.support_zero_goal)


def _set_common_training(cfg: ConfigDict) -> None:
    cfg.data.source = 'metaworld'
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = ()
    cfg.data.preload_to_memory = True
    cfg.data.keep_open_per_worker = True
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.prefetch_factor = 2

    cfg.dataset.K = 4
    cfg.dataset.L = 0
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 8
    cfg.dataset.stride = 1
    cfg.dataset.action_stride = 1
    cfg.dataset.pad_short_chunks = False
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.num_tries_per_item = 200

    cfg.train.num_steps = 100000
    cfg.train.batch_size = 64
    cfg.train.weight_decay = 1e-4
    cfg.train.log_every = 10
    cfg.train.ckpt_every = 1000
    cfg.train.use_amp = False
    cfg.train.amp_dtype = 'bf16'

    cfg.maml.inner_loss_mode = 'write'
    cfg.maml.inner_lr = 3e-2
    cfg.maml.outer_lr = 1e-4
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.num_queries_per_step = 32
    cfg.maml.num_inner_batches = 0
    cfg.maml.num_query_loss_samples = 8
    cfg.maml.memory_layer_norm_after_update = False
    cfg.maml.use_read_improvement_margin = False
    cfg.maml.read_improvement_margin = 0.0
    cfg.maml.read_improvement_margin_weight = 0.0
    cfg.maml.use_wrong_support_margin = False
    cfg.maml.wrong_support_margin = 0.01
    cfg.maml.wrong_support_margin_weight = 1.0
    cfg.maml.wrong_support_strategy = 'same_family_wrong_goal'
    cfg.maml.use_memory_contrast = False
    cfg.maml.log_output_delta = True
    cfg.maml.training_mode_metrics_only = False
    cfg.maml.log_attention_metrics = True
    cfg.maml.goal_prediction_loss_weight = 0.0

    cfg.wandb.project = os.environ.get('ICIL_OBJECTCENTRIC_WANDB_PROJECT', PROJECT)
    cfg.wandb.tags = ('metaworld', 'object_centric', 'icil')
    cfg.wandb.n_sample_steps = 0


def _apply_setting(cfg: ConfigDict, setting: str) -> None:
    cfg.experiment = ConfigDict()
    cfg.experiment.setting = str(setting)
    setting = str(setting)
    if setting == 'a_clean_same_instance_hidden_goal':
        cfg.data.cache_root = _ml45_same_instance_cache_root()
        cfg.data.sample_same_task_name = True
        cfg.data.sample_same_task_instance = True
        cfg.data.allow_support_query_same_episode = False
        cfg.data.support_zero_goal = True
        cfg.data.query_zero_goal = True
    elif setting == 'a_debug_support_goal_same_instance_hidden_query_goal':
        cfg.data.cache_root = _ml45_same_instance_cache_root()
        cfg.data.sample_same_task_name = True
        cfg.data.sample_same_task_instance = True
        cfg.data.allow_support_query_same_episode = False
        cfg.data.support_zero_goal = False
        cfg.data.query_zero_goal = True
    elif setting == 'a_oracle_same_instance_query_goal_visible':
        cfg.data.cache_root = _ml45_same_instance_cache_root()
        cfg.data.sample_same_task_name = True
        cfg.data.sample_same_task_instance = True
        cfg.data.allow_support_query_same_episode = False
        cfg.data.support_zero_goal = False
        cfg.data.query_zero_goal = False
    elif setting == 'b_same_family_different_goal_visible_query_goal':
        cfg.data.cache_root = _ml45_goal_cache_root()
        cfg.data.sample_same_task_name = True
        cfg.data.sample_same_task_instance = False
        cfg.data.allow_support_query_same_episode = False
        cfg.data.support_zero_goal = False
        cfg.data.query_zero_goal = False
    elif setting == 'c_heldout_family_visible_query_goal':
        cfg.data.cache_root = _ml45_goal_cache_root()
        cfg.data.sample_same_task_name = True
        cfg.data.sample_same_task_instance = False
        cfg.data.allow_support_query_same_episode = False
        cfg.data.support_zero_goal = False
        cfg.data.query_zero_goal = False
    else:
        raise ValueError(f'Unknown object-centric ICIL setting: {setting!r}')


def _apply_variant(cfg: ConfigDict, variant: str) -> None:
    cfg.experiment.model_variant = str(variant)
    variant = str(variant)
    dec = cfg.model.query_memory_direct_regression
    if variant == 'query_only':
        dec.memory_initialization_mode = 'base_only'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 0
        cfg.maml.inner_lr = 0.0
        cfg.maml.first_order = True
    elif variant == 'no_encoder_gradmem_fomaml':
        dec.memory_initialization_mode = 'base_only'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 2
        cfg.maml.inner_lr = 3e-2
        cfg.maml.first_order = True
    elif variant == 'support_encoder_no_update':
        dec.memory_initialization_mode = 'additive'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 1
        cfg.maml.inner_lr = 0.0
        cfg.maml.first_order = True
    elif variant == 'support_encoder_fomaml':
        dec.memory_initialization_mode = 'additive'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 2
        cfg.maml.inner_lr = 3e-2
        cfg.maml.first_order = True
    elif variant == 'support_encoder_full_maml':
        dec.memory_initialization_mode = 'additive'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 2
        cfg.maml.inner_lr = 3e-2
        cfg.maml.first_order = False
    elif variant == 'support_encoder_base_only_ablation':
        dec.memory_initialization_mode = 'base_only'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 1
        cfg.maml.inner_lr = 0.0
        cfg.maml.first_order = True
    elif variant == 'support_encoder_pure_encoder':
        dec.memory_initialization_mode = 'pure_encoder'
        dec.write_use_support_obs = False
        cfg.maml.inner_steps = 2
        cfg.maml.inner_lr = 3e-2
        cfg.maml.first_order = True
    else:
        raise ValueError(f'Unknown object-centric ICIL model variant: {variant!r}')


def build_config(*, setting: str, variant: str, run_name: str) -> ConfigDict:
    cfg = get_write_read_base_config()
    _set_common_training(cfg)
    _apply_setting(cfg, setting)
    _set_object_centric_model(cfg)
    _apply_variant(cfg, variant)
    cfg.model.query_goal_hidden = bool(cfg.data.query_zero_goal)
    cfg.model.support_goal_hidden = bool(cfg.data.support_zero_goal)
    cfg.model.support_encoder_memory.goal_visible = not bool(cfg.data.support_zero_goal)
    cfg.wandb.name = run_name
    cfg.wandb.tags = tuple(cfg.wandb.tags) + (str(setting), str(variant))
    return cfg
