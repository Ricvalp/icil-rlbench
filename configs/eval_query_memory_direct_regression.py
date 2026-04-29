import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.device = 'cuda'
    cfg.checkpoint_path = ''

    cfg.task = ConfigDict()
    cfg.task.name = 'close_drawer'
    cfg.task.variation = 0
    cfg.task.num_eval_episodes = 10
    cfg.task.max_env_steps = 80

    cfg.dataset = ConfigDict()
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 1
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = 'absolute'
    cfg.dataset.query_stride_mode = 'dataset'
    cfg.dataset.num_tries_per_item = 100

    cfg.conditioning = ConfigDict()
    cfg.conditioning.cache_root = os.environ.get('ICIL_CACHE_ROOT', '')
    cfg.conditioning.regenerate_demos_each_episode = False
    cfg.conditioning.num_points = 1024

    cfg.memory_ttt = ConfigDict()
    cfg.memory_ttt.inner_steps = -1
    cfg.memory_ttt.inner_lr = -1.0
    cfg.memory_ttt.inner_lr_mode = 'infer'
    cfg.memory_ttt.max_grad_norm = -1.0
    cfg.memory_ttt.num_queries_per_step = -1
    cfg.memory_ttt.grad_accum_steps = 1
    cfg.memory_ttt.num_inner_batches = -1
    cfg.memory_ttt.reuse_diffusion_noise = None

    cfg.sim = ConfigDict()
    cfg.sim.headless = True
    cfg.sim.renderer = 'opengl'
    cfg.sim.image_size = (128, 128)
    cfg.sim.arm_max_velocity = 1.0
    cfg.sim.arm_max_acceleration = 4.0
    cfg.sim.collision_checking = False

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 2
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera = 'front'
    cfg.video.fps = 10
    cfg.video.format = 'mp4'

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get(
        'ICIL_EVAL_OUTPUT_DIR',
        os.path.join('output', '.experiments', 'query_memory_direct_regression_eval'),
    )

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
    cfg.model.query_memory_direct_regression.horizon = 16
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

    return cfg
