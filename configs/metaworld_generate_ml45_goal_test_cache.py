import os

from configs.metaworld_generate_ml45_goal_cache import get_config as get_train_config


def get_config():
    cfg = get_train_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_TEST_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml45_goal_test_50x1'),
    )
    cfg.metaworld.train_or_test = 'test'
    return cfg
