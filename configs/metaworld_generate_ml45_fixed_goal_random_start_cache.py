import os

from configs.metaworld_generate_ml45_goal_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML45_FIXED_GOAL_RANDOM_START_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml45_fixed_goal_random_start_train_50x5'),
    )
    cfg.metaworld.num_task_instances_per_task = 50
    cfg.metaworld.num_successful_episodes_per_instance = 5
    cfg.metaworld.max_attempts_per_instance = 600
    cfg.metaworld.skip_failed_task_instances = True
    cfg.metaworld.fixed_goal_random_start = True
    cfg.metaworld.fixed_goal_random_start_goal_slice = 'auto'
    cfg.metaworld.fixed_goal_random_start_goal_dims = 3
    cfg.metaworld.fixed_goal_random_start_validate_goal = True
    return cfg
