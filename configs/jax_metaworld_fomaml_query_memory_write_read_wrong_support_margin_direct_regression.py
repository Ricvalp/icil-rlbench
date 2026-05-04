from configs.jax_metaworld_maml_query_memory_write_read_wrong_support_margin_direct_regression import (
    get_config as get_maml_config,
)


def get_config():
    cfg = get_maml_config()
    cfg.maml.first_order = True
    cfg.wandb.name = 'ml45-query-hidden-wrong-support-margin-fomaml'
    cfg.wandb.tags = ('metaworld', 'ml45', 'write_read', 'wrong_support_margin', 'query_goal_hidden', 'fomaml')
    return cfg
