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
    cfg.task.num_eval_episodes = 50
    cfg.task.max_env_steps = 220

    cfg.dataset = ConfigDict()
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 0  # 0 => infer from checkpoint['config']['dataset']['K'] used during MAML training
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = 'absolute'  # 'absolute' | 'delta_xyz'; checkpoint overrides when enabled
    cfg.dataset.query_stride_mode = 'consecutive'  # 'dataset' | 'consecutive'

    cfg.conditioning = ConfigDict()
    cfg.conditioning.support_source = 'cache'  # MAML eval currently supports cached support only.
    cfg.conditioning.cache_root = os.environ.get(
        'ICIL_CACHE_ROOT',
        '',
    )  # empty => use checkpoint['config']['data']['cache_root']
    cfg.conditioning.regenerate_demos_each_episode = False
    cfg.conditioning.use_rgb = True
    cfg.conditioning.use_mask_id = False  # fallback only if checkpoint model config does not specify it
    cfg.conditioning.num_points = 1024

    cfg.maml_eval = ConfigDict()
    cfg.maml_eval.inner_steps_override = -1  # -1 => use checkpoint; 0 => no fast-param updates.
    cfg.maml_eval.grad_accum_steps = 1  # Split each inner-loop LOO batch into this many microbatches.
    cfg.maml_eval.num_support_batches_loo = 128  # 0 => build inner_steps support batches, else reuse min(this, inner_steps).
    cfg.maml_eval.preload_support_batches_to_device = False
    cfg.maml_eval.log_query_loss = True  # If True, evaluate loss on one extra episode not used for MAML adaptation.
    cfg.maml_eval.num_query_loss_samples = 1  # Number of t0 windows averaged for the extra query/eval loss.
    cfg.maml_eval.log_query_sample_mse = True  # If True, sample actions on the extra query episode and log MSE to GT.
    cfg.maml_eval.num_tries_per_item = 100

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
    cfg.video.cameras = ('front', 'left_shoulder', 'overhead')  # subset of: left_shoulder, right_shoulder, overhead, wrist, front
    cfg.video.camera = 'front'  # left_shoulder | right_shoulder | overhead | wrist | front
    cfg.video.fps = 10
    cfg.video.formats = ('mp4', 'gif')  # subset of: mp4, gif
    cfg.video.format = 'mp4'  # mp4 | gif

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-eval-maml')
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
            'perceiver_encoder_decoder_eval_maml',
        ),
    )

    return cfg
