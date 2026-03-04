import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.device = "cuda"
    cfg.checkpoint_path = ""

    cfg.task = ConfigDict()
    cfg.task.name = "put_item_in_drawer"
    cfg.task.variation = 0
    cfg.task.num_eval_episodes = 10
    cfg.task.max_env_steps = 220

    cfg.dataset = ConfigDict()
    # If True, these values are read from checkpoint["config"]["dataset"].
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 4
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2
    # Controls how query windows are built from eval history:
    # - "dataset": use dataset stride spacing (default)
    # - "consecutive": use adjacent history frames (no extra striding)
    cfg.dataset.query_stride_mode = "consecutive"

    cfg.conditioning = ConfigDict()
    cfg.conditioning.regenerate_demos_each_episode = False
    cfg.conditioning.use_rgb = True
    # Fallback for old checkpoints that do not store model.use_mask_id.
    # New checkpoints use checkpoint["config"]["model"]["use_mask_id"].
    cfg.conditioning.use_mask_id = False
    cfg.conditioning.num_points = 4096

    cfg.sim = ConfigDict()
    cfg.sim.headless = True
    cfg.sim.renderer = "opengl"  # "opengl" | "opengl3"
    cfg.sim.image_size = (128, 128)
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
    cfg.video.camera = "front"  # left_shoulder | right_shoulder | overhead | wrist | front
    cfg.video.fps = 20
    cfg.video.format = "mp4"  # mp4 | gif

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get(
        "ICIL_EVAL_OUTPUT_DIR",
        os.path.join(
            "output",
            ".experiments",
            "perceiver_encoder_decoder_eval",
        ),
    )

    return cfg
