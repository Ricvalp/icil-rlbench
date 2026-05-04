import os

from configs.jax_metaworld_mt10_goal_family_maml_query_memory_write_read_direct_regression import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    # Ablation: WRITE/support sees a zeroed final 3D goal slot, READ/query keeps
    # the goal. Dimensionality stays 39D so the same model can be used.
    cfg.data.support_zero_goal = True
    cfg.data.query_zero_goal = False
    cfg.wandb.tags = ('metaworld', 'mt10', 'goal_query_only', 'family_instances', 'write_read', 'gradmem', 'maml')
    cfg.wandb.name = os.environ.get('WANDB_NAME', 'mt10-goal-family-support-no-goal-write-read-maml')
    return cfg
