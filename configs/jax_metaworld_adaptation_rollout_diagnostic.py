import os

from ml_collections import ConfigDict

from configs.jax_metaworld_adaptation_mse_diagnostic import get_config as get_mse_config


def get_config():
    cfg = get_mse_config()

    cfg.rollout = ConfigDict()
    cfg.rollout.num_episodes = 10
    cfg.rollout.max_steps = 200
    cfg.rollout.execute_actions_per_plan = 4

    cfg.sim = ConfigDict()
    cfg.sim.benchmark = 'MT10'
    cfg.sim.split = 'train'
    cfg.sim.benchmark_seed = int(os.environ.get('ICIL_METAWORLD_BENCHMARK_SEED', '0'))

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera_name = 'corner'
    cfg.video.width = 320
    cfg.video.height = 240
    cfg.video.frame_stride = 2
    cfg.video.fps = 20
    return cfg
