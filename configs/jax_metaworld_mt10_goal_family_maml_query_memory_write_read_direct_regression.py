import os

from configs.jax_metaworld_mt10_maml_query_memory_write_read_direct_regression import get_config as get_base_config


def _mt10_goal_cache_root():
    return os.environ.get(
        'ICIL_METAWORLD_MT10_GOAL_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'mt10_goal_train_50x1'),
    )


def get_config():
    cfg = get_base_config()
    cfg.data.cache_root = _mt10_goal_cache_root()
    cfg.data.tasks = ()
    cfg.data.sample_same_task_name = True
    cfg.data.sample_same_task_instance = False
    cfg.data.allow_support_query_same_episode = False
    cfg.data.support_zero_goal = False
    cfg.data.query_zero_goal = False
    cfg.model.query_memory_direct_regression.write_use_support_obs = True
    cfg.wandb.tags = ('metaworld', 'mt10', 'goal', 'family_instances', 'write_read', 'gradmem', 'maml')
    cfg.wandb.name = os.environ.get('WANDB_NAME', 'mt10-goal-family-write-read-maml')
    return cfg
