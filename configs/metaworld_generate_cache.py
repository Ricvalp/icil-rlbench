import os

from ml_collections import ConfigDict


def _metaworld_cache_root(default):
    return os.environ.get('ICIL_METAWORLD_CACHE_ROOT', default)


def get_config():
    cfg = ConfigDict()
    cfg.seed = 0
    cfg.require_scripted_policy = True

    cfg.output = ConfigDict()
    cfg.output.cache_root = _metaworld_cache_root(
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'button_press_ml1_train'),
    )
    cfg.output.overwrite = False

    cfg.metaworld = ConfigDict()
    cfg.metaworld.benchmark = 'ML1'
    cfg.metaworld.task_names = ('button-press-v3',)
    cfg.metaworld.train_or_test = 'train'
    cfg.metaworld.num_task_instances_per_task = 10
    cfg.metaworld.num_successful_episodes_per_instance = 8
    cfg.metaworld.max_attempts_per_instance = 80
    cfg.metaworld.max_path_length = 200
    cfg.metaworld.render = False
    cfg.metaworld.keep_successful_only = True

    cfg.obs = ConfigDict()
    cfg.obs.variant = 'no_task_no_goal'
    cfg.obs.remove_task_id = True
    cfg.obs.remove_goal = True
    cfg.obs.normalize = False

    cfg.action = ConfigDict()
    cfg.action.store_raw_action = True
    cfg.action.clip = False

    cfg.debug = ConfigDict()
    cfg.debug.limit_tasks = 0
    cfg.debug.limit_instances = 0
    cfg.debug.limit_episodes = 0

    return cfg
