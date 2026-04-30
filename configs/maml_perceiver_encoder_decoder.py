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
            'perceiver_encoder_decoder_maml',
            'runs',
        ),
    )

    cfg.data = ConfigDict()
    cfg.data.cache_root = os.environ.get(
        'ICIL_CACHE_ROOT',
        os.path.join('output_data_playground_v4', '.rlbench_cache_dense'),
    )
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = ('put_item_in_drawer', 'lamp_on')
    cfg.data.keep_open_per_worker = True
    cfg.data.num_workers = 8
    cfg.data.pin_memory = True
    cfg.data.persistent_workers = True
    # Sample tasks with probability proportional to num_variations ** alpha.
    # alpha=1.0 is variation-uniform; alpha=0.0 is task-uniform.
    cfg.data.task_sampling = 'variation_power'
    cfg.data.task_sampling_alpha = 0.5

    cfg.dataset = ConfigDict()
    cfg.dataset.K = 0  # 0 => infer from pretrained checkpoint as K_pretrain + 1
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = 'absolute'  # 'absolute' | 'delta_xyz'
    cfg.dataset.num_tries_per_item = 100

    cfg.model = ConfigDict()
    cfg.model.encoder_name = 'traj_perceiver'  # also supports "*_v2" and "*_supernode_v2" encoders

    cfg.model.policy = ConfigDict()
    cfg.model.policy.d_model = 512
    cfg.model.policy.n_heads = 8
    cfg.model.policy.denoiser_layers = 10
    cfg.model.policy.denoiser_mlp_mult = 4
    cfg.model.policy.dropout = 0.0
    cfg.model.policy.grad_checkpoint_dit = False
    cfg.model.policy.context_attention_mode = 'single'
    cfg.model.policy.attention_backend = 'manual'  # 'manual' | 'sdpa' ('flash' alias)
    cfg.model.policy.num_train_timesteps = 1000
    cfg.model.policy.beta_start = 1e-4
    cfg.model.policy.beta_end = 2e-2
    cfg.model.policy.beta_schedule = 'squaredcos_cap_v2'
    cfg.model.policy.prediction_type = 'v_prediction'
    cfg.model.policy.set_alpha_to_one = True
    cfg.model.policy.steps_offset = 0
    cfg.model.policy.num_inference_steps = None

    cfg.model.perceiver_demo_query = ConfigDict()
    cfg.model.perceiver_demo_query.d_model = 512
    cfg.model.perceiver_demo_query.n_heads = 8
    cfg.model.perceiver_demo_query.m_frame_tokens = 128
    cfg.model.perceiver_demo_query.frame_tokenizer_layers = 2
    cfg.model.perceiver_demo_query.M_demo_latents = 256
    cfg.model.perceiver_demo_query.demo_perceiver_layers = 3
    cfg.model.perceiver_demo_query.ignore_demos = False
    cfg.model.perceiver_demo_query.mask_hash_buckets = 1
    cfg.model.perceiver_demo_query.use_mask_id = False
    cfg.model.perceiver_demo_query.role_embed_max_K = 4
    cfg.model.perceiver_demo_query.role_embed_max_L = 16
    cfg.model.perceiver_demo_query.role_embed_max_Tobs = 2
    cfg.model.perceiver_demo_query.rgb_alpha_init = 1.0
    cfg.model.perceiver_demo_query.dropout = 0.0
    cfg.model.perceiver_demo_query.compress_demo_latents = True
    cfg.model.perceiver_demo_query.checkpoint_demo_memory = False
    cfg.model.perceiver_demo_query.checkpoint_build_demo_memory = False
    cfg.model.perceiver_demo_query.checkpoint_frame_tokenizer = False
    cfg.model.perceiver_demo_query.tokenize_frames_chunked = False
    cfg.model.perceiver_demo_query.chunk_frames = 32

    cfg.model.traj_perceiver = ConfigDict()
    cfg.model.traj_perceiver.d_model = 512
    cfg.model.traj_perceiver.n_heads = 8
    cfg.model.traj_perceiver.dropout = 0.0
    cfg.model.traj_perceiver.m_frame_tokens = 128
    cfg.model.traj_perceiver.frame_tokenizer_layers = 2
    cfg.model.traj_perceiver.M_demo_latents = 256
    cfg.model.traj_perceiver.demo_perceiver_layers = 3
    cfg.model.traj_perceiver.mask_hash_buckets = 1
    cfg.model.traj_perceiver.use_mask_id = False
    cfg.model.traj_perceiver.role_embed_max_K = 4
    cfg.model.traj_perceiver.role_embed_max_L = 16
    cfg.model.traj_perceiver.role_embed_max_Tobs = 2
    cfg.model.traj_perceiver.rgb_alpha_init = 1.0
    cfg.model.traj_perceiver.ignore_demos = False
    cfg.model.traj_perceiver.compress_demo_latents = True
    cfg.model.traj_perceiver.checkpoint_demo_memory = False
    cfg.model.traj_perceiver.checkpoint_build_demo_memory = False
    cfg.model.traj_perceiver.checkpoint_frame_tokenizer = False
    cfg.model.traj_perceiver.tokenize_frames_chunked = False
    cfg.model.traj_perceiver.chunk_frames = 8
    cfg.model.traj_perceiver.m_traj_tokens = 16
    cfg.model.traj_perceiver.traj_perceiver_layers = 2
    cfg.model.traj_perceiver.traj_dim = 8
    cfg.model.traj_perceiver.use_demo_id_embed = True
    cfg.model.traj_perceiver.include_traj_tokens = True
    cfg.model.traj_perceiver.use_cond_state_as_traj_fallback = False

    cfg.model.perceiver_demo_query_v2 = ConfigDict()
    cfg.model.perceiver_demo_query_v2.d_model = 512
    cfg.model.perceiver_demo_query_v2.n_heads = 8
    cfg.model.perceiver_demo_query_v2.dropout = 0.0
    cfg.model.perceiver_demo_query_v2.demo_m_frame_tokens = 128
    cfg.model.perceiver_demo_query_v2.demo_frame_tokenizer_layers = 2
    cfg.model.perceiver_demo_query_v2.demo_n_heads = 8
    cfg.model.perceiver_demo_query_v2.query_m_frame_tokens = 64
    cfg.model.perceiver_demo_query_v2.query_frame_tokenizer_layers = 2
    cfg.model.perceiver_demo_query_v2.query_n_heads = 8
    cfg.model.perceiver_demo_query_v2.M_demo_latents = 256
    cfg.model.perceiver_demo_query_v2.demo_perceiver_layers = 3
    cfg.model.perceiver_demo_query_v2.mask_hash_buckets = 1
    cfg.model.perceiver_demo_query_v2.use_mask_id = False
    cfg.model.perceiver_demo_query_v2.role_embed_max_K = 4
    cfg.model.perceiver_demo_query_v2.role_embed_max_L = 16
    cfg.model.perceiver_demo_query_v2.role_embed_max_Tobs = 2
    cfg.model.perceiver_demo_query_v2.ignore_demos = False
    cfg.model.perceiver_demo_query_v2.compress_demo_latents = True
    cfg.model.perceiver_demo_query_v2.demo_rgb_alpha_init = 1.0
    cfg.model.perceiver_demo_query_v2.query_rgb_alpha_init = 1.0
    cfg.model.perceiver_demo_query_v2.use_gripper_point_features = False
    cfg.model.perceiver_demo_query_v2.gripper_xyz_state_start = 0
    cfg.model.perceiver_demo_query_v2.gripper_alpha_init = 1.0
    cfg.model.perceiver_demo_query_v2.demo_post_self_attn_layers = 1
    cfg.model.perceiver_demo_query_v2.query_post_self_attn_layers = 1
    cfg.model.perceiver_demo_query_v2.post_self_attn_mlp_mult = 4
    cfg.model.perceiver_demo_query_v2.checkpoint_demo_memory = False
    cfg.model.perceiver_demo_query_v2.checkpoint_build_demo_memory = False
    cfg.model.perceiver_demo_query_v2.checkpoint_frame_tokenizer = False
    cfg.model.perceiver_demo_query_v2.tokenize_frames_chunked = False
    cfg.model.perceiver_demo_query_v2.chunk_frames = 32

    cfg.model.traj_perceiver_v2 = ConfigDict()
    cfg.model.traj_perceiver_v2.d_model = 512
    cfg.model.traj_perceiver_v2.n_heads = 8
    cfg.model.traj_perceiver_v2.dropout = 0.0
    cfg.model.traj_perceiver_v2.demo_m_frame_tokens = 128
    cfg.model.traj_perceiver_v2.demo_frame_tokenizer_layers = 2
    cfg.model.traj_perceiver_v2.demo_n_heads = 8
    cfg.model.traj_perceiver_v2.query_m_frame_tokens = 64
    cfg.model.traj_perceiver_v2.query_frame_tokenizer_layers = 2
    cfg.model.traj_perceiver_v2.query_n_heads = 8
    cfg.model.traj_perceiver_v2.M_demo_latents = 256
    cfg.model.traj_perceiver_v2.demo_perceiver_layers = 3
    cfg.model.traj_perceiver_v2.mask_hash_buckets = 1
    cfg.model.traj_perceiver_v2.use_mask_id = False
    cfg.model.traj_perceiver_v2.role_embed_max_K = 4
    cfg.model.traj_perceiver_v2.role_embed_max_L = 16
    cfg.model.traj_perceiver_v2.role_embed_max_Tobs = 2
    cfg.model.traj_perceiver_v2.ignore_demos = False
    cfg.model.traj_perceiver_v2.compress_demo_latents = True
    cfg.model.traj_perceiver_v2.demo_rgb_alpha_init = 1.0
    cfg.model.traj_perceiver_v2.query_rgb_alpha_init = 1.0
    cfg.model.traj_perceiver_v2.use_gripper_point_features = False
    cfg.model.traj_perceiver_v2.gripper_xyz_state_start = 0
    cfg.model.traj_perceiver_v2.gripper_alpha_init = 1.0
    cfg.model.traj_perceiver_v2.demo_post_self_attn_layers = 1
    cfg.model.traj_perceiver_v2.query_post_self_attn_layers = 1
    cfg.model.traj_perceiver_v2.post_self_attn_mlp_mult = 4
    cfg.model.traj_perceiver_v2.checkpoint_demo_memory = False
    cfg.model.traj_perceiver_v2.checkpoint_build_demo_memory = False
    cfg.model.traj_perceiver_v2.checkpoint_frame_tokenizer = False
    cfg.model.traj_perceiver_v2.tokenize_frames_chunked = False
    cfg.model.traj_perceiver_v2.chunk_frames = 8
    cfg.model.traj_perceiver_v2.m_traj_tokens = 16
    cfg.model.traj_perceiver_v2.traj_perceiver_layers = 2
    cfg.model.traj_perceiver_v2.traj_dim = 8
    cfg.model.traj_perceiver_v2.use_demo_id_embed = True
    cfg.model.traj_perceiver_v2.include_traj_tokens = True
    cfg.model.traj_perceiver_v2.use_cond_state_as_traj_fallback = False


    # Supernode Perceiver V2 demo/query encoder.
    # Opt in with --config.model.encoder_name=perceiver_demo_query_supernode_v2.
    cfg.model.perceiver_demo_query_supernode_v2 = ConfigDict()
    cfg.model.perceiver_demo_query_supernode_v2.d_model = 512
    cfg.model.perceiver_demo_query_supernode_v2.n_heads = 4
    cfg.model.perceiver_demo_query_supernode_v2.dropout = 0.0
    cfg.model.perceiver_demo_query_supernode_v2.demo_n_heads = 4
    cfg.model.perceiver_demo_query_supernode_v2.query_n_heads = 4
    cfg.model.perceiver_demo_query_supernode_v2.M_demo_latents = 256
    cfg.model.perceiver_demo_query_supernode_v2.demo_perceiver_layers = 3
    cfg.model.perceiver_demo_query_supernode_v2.mask_hash_buckets = 1
    cfg.model.perceiver_demo_query_supernode_v2.use_mask_id = True
    cfg.model.perceiver_demo_query_supernode_v2.use_mask_embedding = False
    cfg.model.perceiver_demo_query_supernode_v2.use_mask_instance_quota = True
    cfg.model.perceiver_demo_query_supernode_v2.supernode_sampling_mode = "fps"  # "fps" | "fast_random"
    cfg.model.perceiver_demo_query_supernode_v2.role_embed_max_K = 4
    cfg.model.perceiver_demo_query_supernode_v2.role_embed_max_L = 16
    cfg.model.perceiver_demo_query_supernode_v2.role_embed_max_Tobs = 2
    cfg.model.perceiver_demo_query_supernode_v2.ignore_demos = False
    cfg.model.perceiver_demo_query_supernode_v2.compress_demo_latents = True
    cfg.model.perceiver_demo_query_supernode_v2.demo_rgb_alpha_init = 1.0
    cfg.model.perceiver_demo_query_supernode_v2.query_rgb_alpha_init = 1.0
    cfg.model.perceiver_demo_query_supernode_v2.use_gripper_point_features = True
    cfg.model.perceiver_demo_query_supernode_v2.gripper_xyz_state_start = 0
    cfg.model.perceiver_demo_query_supernode_v2.gripper_alpha_init = 1.0
    cfg.model.perceiver_demo_query_supernode_v2.demo_post_self_attn_layers = 1
    cfg.model.perceiver_demo_query_supernode_v2.query_post_self_attn_layers = 2
    cfg.model.perceiver_demo_query_supernode_v2.post_self_attn_mlp_mult = 4
    cfg.model.perceiver_demo_query_supernode_v2.demo_supernodes = 128
    cfg.model.perceiver_demo_query_supernode_v2.query_supernodes = 128
    cfg.model.perceiver_demo_query_supernode_v2.demo_frame_tokens_out = 64
    cfg.model.perceiver_demo_query_supernode_v2.query_frame_tokens_out = 128
    cfg.model.perceiver_demo_query_supernode_v2.neighbors_per_supernode = 32
    cfg.model.perceiver_demo_query_supernode_v2.demo_supernode_refine_layers = 1
    cfg.model.perceiver_demo_query_supernode_v2.query_supernode_refine_layers = 2
    cfg.model.perceiver_demo_query_supernode_v2.compress_supernodes_demo = True
    cfg.model.perceiver_demo_query_supernode_v2.compress_supernodes_query = True
    cfg.model.perceiver_demo_query_supernode_v2.supernode_pool_layers = 1
    cfg.model.perceiver_demo_query_supernode_v2.min_gripper_supernodes = 2
    cfg.model.perceiver_demo_query_supernode_v2.min_mask_supernodes = 4
    cfg.model.perceiver_demo_query_supernode_v2.gripper_sampling_radius = 0.10
    cfg.model.perceiver_demo_query_supernode_v2.checkpoint_demo_memory = False
    cfg.model.perceiver_demo_query_supernode_v2.checkpoint_build_demo_memory = False
    cfg.model.perceiver_demo_query_supernode_v2.checkpoint_frame_tokenizer = False
    cfg.model.perceiver_demo_query_supernode_v2.tokenize_frames_chunked = True
    cfg.model.perceiver_demo_query_supernode_v2.chunk_frames = 256

    # Trajectory Supernode Perceiver V2 encoder.
    # Opt in with --config.model.encoder_name=traj_supernode_perceiver_v2.
    cfg.model.traj_supernode_perceiver_v2 = ConfigDict()
    cfg.model.traj_supernode_perceiver_v2.d_model = 512
    cfg.model.traj_supernode_perceiver_v2.n_heads = 4
    cfg.model.traj_supernode_perceiver_v2.dropout = 0.0
    cfg.model.traj_supernode_perceiver_v2.demo_n_heads = 4
    cfg.model.traj_supernode_perceiver_v2.query_n_heads = 4
    cfg.model.traj_supernode_perceiver_v2.M_demo_latents = 256
    cfg.model.traj_supernode_perceiver_v2.demo_perceiver_layers = 3
    cfg.model.traj_supernode_perceiver_v2.mask_hash_buckets = 1
    cfg.model.traj_supernode_perceiver_v2.use_mask_id = True
    cfg.model.traj_supernode_perceiver_v2.use_mask_embedding = False
    cfg.model.traj_supernode_perceiver_v2.use_mask_instance_quota = True
    cfg.model.traj_supernode_perceiver_v2.supernode_sampling_mode = "fps"  # "fps" | "fast_random"
    cfg.model.traj_supernode_perceiver_v2.role_embed_max_K = 4
    cfg.model.traj_supernode_perceiver_v2.role_embed_max_L = 16
    cfg.model.traj_supernode_perceiver_v2.role_embed_max_Tobs = 2
    cfg.model.traj_supernode_perceiver_v2.ignore_demos = False
    cfg.model.traj_supernode_perceiver_v2.compress_demo_latents = True
    cfg.model.traj_supernode_perceiver_v2.demo_rgb_alpha_init = 1.0
    cfg.model.traj_supernode_perceiver_v2.query_rgb_alpha_init = 1.0
    cfg.model.traj_supernode_perceiver_v2.use_gripper_point_features = True
    cfg.model.traj_supernode_perceiver_v2.gripper_xyz_state_start = 0
    cfg.model.traj_supernode_perceiver_v2.gripper_alpha_init = 1.0
    cfg.model.traj_supernode_perceiver_v2.demo_post_self_attn_layers = 1
    cfg.model.traj_supernode_perceiver_v2.query_post_self_attn_layers = 2
    cfg.model.traj_supernode_perceiver_v2.post_self_attn_mlp_mult = 4
    cfg.model.traj_supernode_perceiver_v2.demo_supernodes = 128
    cfg.model.traj_supernode_perceiver_v2.query_supernodes = 128
    cfg.model.traj_supernode_perceiver_v2.demo_frame_tokens_out = 64
    cfg.model.traj_supernode_perceiver_v2.query_frame_tokens_out = 128
    cfg.model.traj_supernode_perceiver_v2.neighbors_per_supernode = 32
    cfg.model.traj_supernode_perceiver_v2.demo_supernode_refine_layers = 1
    cfg.model.traj_supernode_perceiver_v2.query_supernode_refine_layers = 2
    cfg.model.traj_supernode_perceiver_v2.compress_supernodes_demo = True
    cfg.model.traj_supernode_perceiver_v2.compress_supernodes_query = True
    cfg.model.traj_supernode_perceiver_v2.supernode_pool_layers = 1
    cfg.model.traj_supernode_perceiver_v2.min_gripper_supernodes = 2
    cfg.model.traj_supernode_perceiver_v2.min_mask_supernodes = 4
    cfg.model.traj_supernode_perceiver_v2.gripper_sampling_radius = 0.10
    cfg.model.traj_supernode_perceiver_v2.checkpoint_demo_memory = False
    cfg.model.traj_supernode_perceiver_v2.checkpoint_build_demo_memory = False
    cfg.model.traj_supernode_perceiver_v2.checkpoint_frame_tokenizer = False
    cfg.model.traj_supernode_perceiver_v2.tokenize_frames_chunked = True
    cfg.model.traj_supernode_perceiver_v2.chunk_frames = 256
    cfg.model.traj_supernode_perceiver_v2.m_traj_tokens = 32
    cfg.model.traj_supernode_perceiver_v2.traj_perceiver_layers = 2
    cfg.model.traj_supernode_perceiver_v2.traj_dim = 8
    cfg.model.traj_supernode_perceiver_v2.use_demo_id_embed = True
    cfg.model.traj_supernode_perceiver_v2.include_traj_tokens = True
    cfg.model.traj_supernode_perceiver_v2.use_cond_state_as_traj_fallback = False

    cfg.model.conv3d_demo_query = ConfigDict()
    cfg.model.conv3d_demo_query.d_model = 512
    cfg.model.conv3d_demo_query.n_heads = 8
    cfg.model.conv3d_demo_query.m_frame_tokens = 128
    cfg.model.conv3d_demo_query.max_voxels = 4096
    cfg.model.conv3d_demo_query.voxel_size = 0.01
    cfg.model.conv3d_demo_query.use_learned_topk = True
    cfg.model.conv3d_demo_query.n_mix_layers = 2
    cfg.model.conv3d_demo_query.M_demo_latents = 256
    cfg.model.conv3d_demo_query.demo_perceiver_layers = 3
    cfg.model.conv3d_demo_query.mask_hash_buckets = 1
    cfg.model.conv3d_demo_query.use_mask_id = False
    cfg.model.conv3d_demo_query.role_embed_max_K = 4
    cfg.model.conv3d_demo_query.role_embed_max_L = 16
    cfg.model.conv3d_demo_query.role_embed_max_Tobs = 2
    cfg.model.conv3d_demo_query.rgb_alpha_init = 1.0
    cfg.model.conv3d_demo_query.dropout = 0.0
    cfg.model.conv3d_demo_query.ignore_demos = False

    cfg.model.traj_conv3d = ConfigDict()
    cfg.model.traj_conv3d.d_model = 512
    cfg.model.traj_conv3d.n_heads = 8
    cfg.model.traj_conv3d.dropout = 0.0
    cfg.model.traj_conv3d.m_frame_tokens = 128
    cfg.model.traj_conv3d.n_mix_layers = 2
    cfg.model.traj_conv3d.max_voxels = 4096
    cfg.model.traj_conv3d.voxel_size = 0.01
    cfg.model.traj_conv3d.use_learned_topk = True
    cfg.model.traj_conv3d.M_demo_latents = 256
    cfg.model.traj_conv3d.demo_perceiver_layers = 3
    cfg.model.traj_conv3d.mask_hash_buckets = 1
    cfg.model.traj_conv3d.use_mask_id = False
    cfg.model.traj_conv3d.role_embed_max_K = 4
    cfg.model.traj_conv3d.role_embed_max_L = 16
    cfg.model.traj_conv3d.role_embed_max_Tobs = 2
    cfg.model.traj_conv3d.rgb_alpha_init = 1.0
    cfg.model.traj_conv3d.ignore_demos = False
    cfg.model.traj_conv3d.m_traj_tokens = 16
    cfg.model.traj_conv3d.traj_perceiver_layers = 2
    cfg.model.traj_conv3d.traj_dim = 8
    cfg.model.traj_conv3d.use_demo_id_embed = True
    cfg.model.traj_conv3d.include_traj_tokens = True
    cfg.model.traj_conv3d.use_cond_state_as_traj_fallback = False

    cfg.finetune = ConfigDict()
    cfg.finetune.pretrained_checkpoint = ''
    cfg.finetune.strict_load = True

    cfg.maml = ConfigDict()
    cfg.maml.inner_steps = 4
    cfg.maml.inner_lr = 1e-4
    cfg.maml.inner_lr_mode = 'fixed'
    cfg.maml.outer_lr = 1e-4
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.last_frac_fast = 0.25
    cfg.maml.include_decoder_mlp_fast = True
    cfg.maml.include_ada_fast = True
    cfg.maml.include_final_norm_fast = True
    cfg.maml.include_input_projections_fast = False
    cfg.maml.include_output_head_fast = False
    cfg.maml.include_diffusion_conditioning_fast = False
    cfg.maml.num_loo_per_task = 8
    cfg.maml.outer_context_size = 0
    cfg.maml.reuse_diffusion_noise = False

    cfg.outer = ConfigDict()
    cfg.outer.train_encoder = False
    cfg.outer.train_decoder = True
    cfg.outer.train_input_projections = True
    cfg.outer.train_output_head = True
    cfg.outer.train_diffusion_conditioning = True

    cfg.train = ConfigDict()
    cfg.train.num_steps = 100000
    cfg.train.batch_size = 2  # outer batch size in tasks
    cfg.train.weight_decay = 1e-4
    cfg.train.log_every = 20
    cfg.train.ckpt_every = 100
    cfg.train.resume_path = ''
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_maml',
            'checkpoints',
        ),
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-maml')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', 'ricvalp')
    cfg.wandb.group = ''
    cfg.wandb.name = ''
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.tags = ()
    cfg.wandb.n_loss_steps = 20
    cfg.wandb.n_sample_steps = 200
    cfg.wandb.n_inner_loss_steps = 200
    cfg.wandb.sample_batch_items = 8
    cfg.wandb.sample_mse_items = 8
    cfg.wandb.sample_inference_steps = 100
    cfg.wandb.sample_trace_frames = 0
    cfg.wandb.sample_eta = 0.0
    cfg.wandb.include_query_pointcloud_in_x0_pred_vs_gt_3d = False
    cfg.wandb.query_pointcloud_max_points = 4096

    return cfg
