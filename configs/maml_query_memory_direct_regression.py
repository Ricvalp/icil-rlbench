import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.device = 'cuda'
    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'query_memory_direct_regression_maml',
            'runs',
        ),
    )

    cfg.data = ConfigDict()
    cfg.data.cache_root = os.environ.get(
        'ICIL_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.rlbench_cache_dense'),
    )
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = ()
    cfg.data.keep_open_per_worker = True
    cfg.data.num_workers = 16
    cfg.data.pin_memory = True
    cfg.data.persistent_workers = True
    cfg.data.task_sampling = 'variation_uniform'
    cfg.data.task_sampling_alpha = 0.5

    cfg.dataset = ConfigDict()
    cfg.dataset.K = 4
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.num_tries_per_item = 100

    cfg.model = ConfigDict()
    cfg.model.query_encoder_name = 'simple_query_point_encoder'
    cfg.model.query_memory_direct_regression = ConfigDict()
    cfg.model.query_memory_direct_regression.d_model = 512
    cfg.model.query_memory_direct_regression.n_heads = 8
    cfg.model.query_memory_direct_regression.decoder_layers = 8
    cfg.model.query_memory_direct_regression.decoder_mlp_mult = 4
    cfg.model.query_memory_direct_regression.dropout = 0.0
    cfg.model.query_memory_direct_regression.grad_checkpoint_decoder = False
    cfg.model.query_memory_direct_regression.context_attention_mode = 'two_ctx'
    cfg.model.query_memory_direct_regression.attention_backend = 'manual'
    cfg.model.query_memory_direct_regression.loss_type = 'l1'
    cfg.model.query_memory_direct_regression.horizon = int(cfg.dataset.H)
    cfg.model.query_memory_direct_regression.conditioner_mlp_mult = 2
    cfg.model.query_memory_direct_regression.conditioner_dropout = 0.0
    cfg.model.query_memory_direct_regression.memory_num_tokens = 128

    cfg.model.simple_query_point_encoder = ConfigDict()
    cfg.model.simple_query_point_encoder.d_model = 512
    cfg.model.simple_query_point_encoder.use_rgb = True
    cfg.model.simple_query_point_encoder.use_mask_id = False
    cfg.model.simple_query_point_encoder.mask_hash_buckets = 2048
    cfg.model.simple_query_point_encoder.use_gripper_point_features = False
    cfg.model.simple_query_point_encoder.gripper_xyz_state_start = 0
    cfg.model.simple_query_point_encoder.max_T_obs = 16
    cfg.model.simple_query_point_encoder.add_state_token = True

    cfg.model.dp3_query_frame_encoder = ConfigDict()
    cfg.model.dp3_query_frame_encoder.d_model = 512
    cfg.model.dp3_query_frame_encoder.pointcloud_out_channels = 256
    cfg.model.dp3_query_frame_encoder.pointcloud_use_layernorm = True
    cfg.model.dp3_query_frame_encoder.pointcloud_final_norm = 'layernorm'
    cfg.model.dp3_query_frame_encoder.use_rgb = True
    cfg.model.dp3_query_frame_encoder.use_mask_id = False
    cfg.model.dp3_query_frame_encoder.mask_hash_buckets = 2048
    cfg.model.dp3_query_frame_encoder.mask_embed_dim = 8
    cfg.model.dp3_query_frame_encoder.use_gripper_point_features = False
    cfg.model.dp3_query_frame_encoder.gripper_xyz_state_start = 0
    cfg.model.dp3_query_frame_encoder.state_mlp_hidden_dims = (64,)
    cfg.model.dp3_query_frame_encoder.state_feat_dim = 64
    cfg.model.dp3_query_frame_encoder.max_T_obs = 16

    cfg.train = ConfigDict()
    cfg.train.num_steps = 100000
    cfg.train.batch_size = 4
    cfg.train.weight_decay = 1e-4
    cfg.train.log_every = 10
    cfg.train.ckpt_every = 500
    cfg.train.resume_path = ''
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'query_memory_direct_regression_maml',
            'checkpoints',
        ),
    )

    cfg.maml = ConfigDict()
    cfg.maml.first_order = False
    cfg.maml.inner_steps = 2
    cfg.maml.inner_lr = 1e-2
    cfg.maml.inner_lr_mode = 'fixed'
    cfg.maml.outer_lr = 1e-4
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.num_queries_per_step = 32
    cfg.maml.num_inner_batches = 0
    cfg.maml.num_query_loss_samples = 1
    cfg.maml.holdout_index = -1
    cfg.maml.reuse_diffusion_noise = False
    cfg.maml.grad_accum_steps = 1

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-query-memory-direct-maml')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', 'ricvalp')
    cfg.wandb.group = ''
    cfg.wandb.name = ''
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.tags = ()
    cfg.wandb.n_loss_steps = 1
    cfg.wandb.n_sample_steps = 1000
    cfg.wandb.sample_batch_items = 16
    cfg.wandb.include_query_pointcloud_in_x0_pred_vs_gt_3d = False
    cfg.wandb.query_pointcloud_max_points = 4096

    return cfg
