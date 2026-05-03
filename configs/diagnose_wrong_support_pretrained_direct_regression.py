import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.device = "cuda"
    cfg.checkpoint_path = ""

    cfg.task = ConfigDict()
    cfg.task.name = "open_drawer"
    cfg.task.variation = 0
    cfg.task.max_env_steps = 80
    # Set to 0 to run only the offline MSE diagnostic.
    cfg.task.num_rollout_episodes = 0

    cfg.wrong_support = ConfigDict()
    cfg.wrong_support.task_name = "slide_block_to_target"
    cfg.wrong_support.variation = 0

    cfg.dataset = ConfigDict()
    # If True, K/L/T_obs/H/stride/action_representation are read from checkpoint["config"]["dataset"].
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 4
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    cfg.dataset.action_representation = "absolute"
    # Controls how query windows are built during RLBench rollouts.
    cfg.dataset.query_stride_mode = "consecutive"  # "dataset" | "consecutive"

    cfg.conditioning = ConfigDict()
    cfg.conditioning.cache_root = os.environ.get("ICIL_CACHE_ROOT", "")
    cfg.conditioning.use_rgb = True
    # Fallback for old checkpoints that do not store mask-id usage in model config.
    cfg.conditioning.use_mask_id = False
    cfg.conditioning.num_points = 1024
    cfg.conditioning.regenerate_support_each_rollout = False

    cfg.mse = ConfigDict()
    cfg.mse.enable = True
    cfg.mse.num_batches = 16
    cfg.mse.batch_size = 16
    cfg.mse.num_tries_per_item = 100

    cfg.sim = ConfigDict()
    cfg.sim.headless = True
    cfg.sim.renderer = "opengl"  # "opengl" | "opengl3"
    cfg.sim.image_size = (128, 128)
    cfg.sim.arm_max_velocity = 1.0
    cfg.sim.arm_max_acceleration = 4.0
    cfg.sim.collision_checking = False

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 8
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera = "front"
    cfg.video.fps = 10
    cfg.video.format = "mp4"

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get(
        "ICIL_EVAL_OUTPUT_DIR",
        os.path.join("output", ".experiments", "wrong_support_diagnostics"),
    )

    return cfg
