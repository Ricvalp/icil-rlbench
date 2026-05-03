import os

from configs.jax_eval_query_memory_direct_regression import get_config as _get_eval_config
from ml_collections import ConfigDict


def get_config():
    cfg = _get_eval_config()

    cfg.adaptation = ConfigDict()
    cfg.adaptation.different_task_name = 'slide_block_to_target'
    cfg.adaptation.different_variation = 0

    cfg.mse = ConfigDict()
    cfg.mse.num_batches = 16
    cfg.mse.batch_size = 16
    cfg.mse.exclude_same_task_support_episodes = True

    cfg.output.root_dir = os.environ.get(
        'ICIL_EVAL_OUTPUT_DIR',
        os.path.join('output', '.experiments', 'jax_adaptation_support_mse_diagnostic'),
    )

    return cfg
