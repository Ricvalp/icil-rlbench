import os

from configs.metaworld_generate_ml45_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml45_goal_train_50x1'),
    )
    cfg.metaworld.num_task_instances_per_task = 50
    cfg.metaworld.num_successful_episodes_per_instance = 1
    cfg.metaworld.max_attempts_per_instance = 200
    cfg.metaworld.skip_failed_task_instances = True
    cfg.metaworld.force_goal_observable = True
    cfg.obs.variant = 'raw'
    cfg.obs.remove_task_id = False
    cfg.obs.remove_goal = False
    return cfg
