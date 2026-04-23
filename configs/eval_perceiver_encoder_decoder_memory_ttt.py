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
    cfg.dataset.K = 0  # 0 => pretrain K+1, or stored K for MAML checkpoints.
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = 'absolute'  # 'absolute' | 'delta_xyz'; checkpoint overrides when enabled
    cfg.dataset.query_stride_mode = 'consecutive'  # 'dataset' | 'consecutive'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.support_source = 'cache'  # Memory TTT currently supports cached support only.
    cfg.conditioning.cache_root = os.environ.get(
        'ICIL_CACHE_ROOT',
        '',
    )  # empty => use checkpoint['config']['data']['cache_root']
    cfg.conditioning.regenerate_demos_each_episode = False
    cfg.conditioning.use_rgb = True
    cfg.conditioning.use_mask_id = False  # fallback only if checkpoint model config does not specify it
    cfg.conditioning.num_points = 1024

    cfg.memory_ttt = ConfigDict()
    cfg.memory_ttt.inner_steps = -1  # <0 => infer from checkpoint maml/memory_ttt config if available.
    cfg.memory_ttt.inner_lr = -1.0  # <0 => infer from checkpoint maml/memory_ttt config if available.
    cfg.memory_ttt.inner_lr_mode = 'infer'  # 'infer' | 'fixed' | 'shared_learned' | 'per_step_learned'
    cfg.memory_ttt.optimizer = 'infer'  # 'infer' | 'adam' | 'sgd'
    cfg.memory_ttt.sgd_momentum = -1.0  # <0 => infer if possible, else 0.0.
    cfg.memory_ttt.max_grad_norm = -1.0  # <0 => infer from checkpoint maml/memory_ttt config if available.
    cfg.memory_ttt.num_queries_per_step = -1  # <0 => infer from checkpoint maml/memory_ttt config if available.
    cfg.memory_ttt.grad_accum_steps = 1  # Split each inner-loop query batch into this many microbatches.
    cfg.memory_ttt.num_inner_batches = -1  # <0 => infer from checkpoint; 0 => build inner_steps batches.
    cfg.memory_ttt.reuse_diffusion_noise = None  # None => infer from checkpoint if available.
    cfg.memory_ttt.preload_support_batches_to_device = False
    cfg.memory_ttt.holdout_index = None  # None => infer from checkpoint if available, else random holdout.
    cfg.memory_ttt.log_query_loss = True
    cfg.memory_ttt.num_query_loss_samples = 16  # <0 => infer from checkpoint if available, else inner batch size.
    cfg.memory_ttt.log_query_sample_mse = True  # If True, sample actions on the extra query episode and log MSE to GT.
    cfg.memory_ttt.num_tries_per_item = 100

    # Optional: also adapt decoder params. Off by default so the experiment is memory-token-only.
    cfg.memory_ttt.optimize_decoder = None  # None => infer from checkpoint if available, else False.
    cfg.memory_ttt.decoder_lr = -1.0  # <0 => infer from checkpoint if available, else inner_lr.
    cfg.memory_ttt.decoder_param_prefixes = ('denoiser.', 'action_in.', 'action_out.', 't_mlp.')

    cfg.sim = ConfigDict()
    cfg.sim.headless = True
    cfg.sim.renderer = 'opengl'  # 'opengl' | 'opengl3'
    cfg.sim.image_size = (256, 256)
    cfg.sim.arm_max_velocity = 1.0
    cfg.sim.arm_max_acceleration = 4.0
    cfg.sim.collision_checking = False

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 2
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True

    cfg.inference = ConfigDict()
    cfg.inference.inference_steps = 100
    cfg.inference.eta = 0.0

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.cameras = ('front', 'left_shoulder', 'overhead')
    cfg.video.camera = 'front'
    cfg.video.fps = 10
    cfg.video.formats = ('mp4', 'gif')
    cfg.video.format = 'mp4'

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-eval-memory-ttt')
    cfg.wandb.entity = os.environ.get('WANDB_ENTITY', 'ricvalp')
    cfg.wandb.group = ''
    cfg.wandb.name = ''
    cfg.wandb.mode = os.environ.get('WANDB_MODE', 'online')
    cfg.wandb.tags = ()

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get(
        'ICIL_EVAL_OUTPUT_DIR',
        os.path.join(
            'output',
            '.experiments',
            'perceiver_encoder_decoder_eval_memory_ttt',
        ),
    )

    return cfg
