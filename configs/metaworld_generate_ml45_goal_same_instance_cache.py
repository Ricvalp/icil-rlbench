import os

from configs.metaworld_generate_ml45_goal_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_SAME_INSTANCE_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml45_goal_train_50x5'),
    )
    cfg.metaworld.num_task_instances_per_task = 50
    # Setting A needs K support episodes plus one held-out query episode from
    # the same task instance. The object-centric ICIL configs default to K=4.
    cfg.metaworld.num_successful_episodes_per_instance = 5
    cfg.metaworld.max_attempts_per_instance = 600
    cfg.metaworld.skip_failed_task_instances = True
    return cfg
