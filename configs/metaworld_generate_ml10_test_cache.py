import os

from configs.metaworld_generate_ml10_cache import get_config as get_train_config


def get_config():
    cfg = get_train_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_ML10_TEST_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml10_test_50x8'),
    )
    cfg.metaworld.train_or_test = 'test'
    return cfg
