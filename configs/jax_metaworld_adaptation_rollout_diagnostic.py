import os

from ml_collections import ConfigDict

from configs.jax_metaworld_adaptation_mse_diagnostic import get_config as get_mse_config


def get_config():
    cfg = get_mse_config()

    cfg.rollout = ConfigDict()
    cfg.rollout.num_episodes = 10
    cfg.rollout.max_steps = 200
    cfg.rollout.execute_actions_per_plan = 4
    cfg.rollout.log_initial_plan_deltas = True

    cfg.sim = ConfigDict()
    # Empty means infer from the selected cache index.json.
    cfg.sim.benchmark = ''
    cfg.sim.split = ''
    # "auto" forces goal-observable ML envs when the selected cache stores
    # goal-observable 39D states and query_zero_goal=False.
    cfg.sim.force_goal_observable = 'auto'
    cfg.sim.benchmark_seed = int(os.environ.get('ICIL_METAWORLD_BENCHMARK_SEED', '0'))
    # Setting A only: preserve the MetaWorld task goal while resampling starts
    # at evaluation time, matching the fixed-goal/random-start cache generator.
    cfg.sim.fixed_goal_random_start = False
    cfg.sim.fixed_goal_random_start_goal_slice = 'auto'
    cfg.sim.fixed_goal_random_start_goal_dims = 3
    cfg.sim.fixed_goal_random_start_validate_goal = True
    cfg.sim.fixed_goal_random_start_goal_tolerance = 1e-5
    cfg.sim.fixed_goal_random_start_max_resample_calls = 256

    cfg.video = ConfigDict()
    cfg.video.enable = True
    cfg.video.camera_name = 'corner'
    cfg.video.width = 320
    cfg.video.height = 240
    cfg.video.frame_stride = 2
    cfg.video.fps = 20
    return cfg
