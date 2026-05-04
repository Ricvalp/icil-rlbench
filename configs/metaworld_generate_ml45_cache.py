import os

from configs.metaworld_generate_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML45_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml45_train_20x8'),
    )
    cfg.metaworld.benchmark = 'ML45'
    cfg.metaworld.task_names = ()  # all ML45 train task families
    cfg.metaworld.train_or_test = 'train'
    cfg.metaworld.num_task_instances_per_task = 20
    cfg.metaworld.num_successful_episodes_per_instance = 8
    cfg.metaworld.max_attempts_per_instance = 80
    cfg.debug.limit_tasks = 0
    cfg.debug.limit_instances = 0
    cfg.debug.limit_episodes = 0
    return cfg
