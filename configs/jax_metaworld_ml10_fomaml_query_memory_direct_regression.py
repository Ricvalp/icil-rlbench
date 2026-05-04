import os

from configs.jax_metaworld_fomaml_query_memory_direct_regression import get_config as get_base_config


def _ml10_cache_root():
    return os.environ.get(
        'ICIL_METAWORLD_ML10_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'ml10_train_50x8'),
    )


def get_config():
    cfg = get_base_config()
    cfg.data.cache_root = _ml10_cache_root()
    cfg.data.tasks = ()
    cfg.dataset.K = 4
    cfg.dataset.H = 8
    cfg.model.query_memory_direct_regression.horizon = int(cfg.dataset.H)
    cfg.wandb.tags = ('metaworld', 'ml10', 'read_inner', 'fomaml')
    cfg.wandb.name = os.environ.get('WANDB_NAME', 'ml10-read-inner-fomaml')
    return cfg
