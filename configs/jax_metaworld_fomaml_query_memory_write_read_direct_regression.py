import os

from configs.jax_metaworld_maml_query_memory_direct_regression import (
    _metaworld_checkpoint_parent,
    _metaworld_output_parent,
    _metaworld_wandb_project,
)
from configs.jax_metaworld_maml_query_memory_write_read_direct_regression import get_config as get_maml_config


def get_config():
    cfg = get_maml_config()
    cfg.output_parent_dir = _metaworld_output_parent(
        os.path.join('output_data_playground_v3', '.experiments', 'jax_metaworld_query_memory_write_read_direct_regression_fomaml', 'runs'),
    )
    cfg.train.checkpoint_parent_dir = _metaworld_checkpoint_parent(
        os.path.join('output_data_playground_v3', '.experiments', 'jax_metaworld_query_memory_write_read_direct_regression_fomaml', 'checkpoints'),
    )
    cfg.maml.first_order = True
    cfg.train.use_amp = True
    cfg.train.amp_dtype = 'bf16'
    cfg.wandb.project = _metaworld_wandb_project('icil-jax-metaworld-query-memory-write-read-fomaml')
    cfg.wandb.tags = ('metaworld', 'button-press-v3', 'write_read', 'gradmem', 'fomaml')
    return cfg
