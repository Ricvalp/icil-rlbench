import os

from configs.metaworld_generate_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_MT50_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'mt50_train_20x8'),
    )
    cfg.metaworld.benchmark = 'MT50'
    cfg.metaworld.task_names = ()  # all MT50 task families
    cfg.metaworld.train_or_test = 'train'
    cfg.metaworld.num_task_instances_per_task = 20
    cfg.metaworld.num_successful_episodes_per_instance = 8
    cfg.metaworld.max_attempts_per_instance = 80
    cfg.metaworld.skip_failed_task_instances = True
    cfg.obs.variant = 'no_task_no_goal'
    cfg.obs.remove_task_id = True
    cfg.obs.remove_goal = True
    cfg.debug.limit_tasks = 0
    cfg.debug.limit_instances = 0
    cfg.debug.limit_episodes = 0
    return cfg
