from configs.jax_metaworld_maml_query_memory_write_read_memory_contrast_direct_regression import (
    get_config as get_maml_config,
)


def get_config():
    cfg = get_maml_config()
    cfg.maml.first_order = True
    cfg.wandb.name = 'ml45-query-hidden-memory-delta-contrast-fomaml'
    cfg.wandb.tags = ('metaworld', 'ml45', 'write_read', 'memory_delta_contrast', 'query_goal_hidden', 'fomaml')
    return cfg
