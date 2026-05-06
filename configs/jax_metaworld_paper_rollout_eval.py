import os

from ml_collections import ConfigDict

from configs.jax_metaworld_adaptation_mse_diagnostic import get_config as get_mse_config


def get_config():
    cfg = get_mse_config()

    cfg.rollout = ConfigDict()
    cfg.rollout.instances_per_task = 15
    cfg.rollout.require_instances_per_task = True
    cfg.rollout.max_steps = 200
    cfg.rollout.execute_actions_per_plan = 4

    cfg.sim = ConfigDict()
    # Empty means infer from the selected cache index.json.
    cfg.sim.benchmark = ''
    cfg.sim.split = ''
    # "auto" forces goal-observable ML envs when the selected cache stores
    # goal-observable 39D states and query_zero_goal=False.
    cfg.sim.force_goal_observable = 'auto'
    cfg.sim.benchmark_seed = int(os.environ.get('ICIL_METAWORLD_BENCHMARK_SEED', '0'))

    cfg.adaptation.include_wrong_family = False

    cfg.video = ConfigDict()
    cfg.video.enable = False
    cfg.video.max_videos_per_condition = 0
    cfg.video.camera_name = 'corner'
    cfg.video.width = 320
    cfg.video.height = 240
    cfg.video.frame_stride = 2
    cfg.video.fps = 20
    return cfg
