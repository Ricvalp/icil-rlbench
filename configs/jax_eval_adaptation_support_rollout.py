import os

from configs.jax_eval_query_memory_direct_regression import get_config as _get_eval_config
from ml_collections import ConfigDict


def get_config():
    cfg = _get_eval_config()

    cfg.adaptation = ConfigDict()
    cfg.adaptation.different_task_name = 'slide_block_to_target'
    cfg.adaptation.different_variation = 0

    cfg.task.num_eval_episodes = 10
    cfg.video.format = 'gif'
    cfg.output.root_dir = os.environ.get(
        'ICIL_EVAL_OUTPUT_DIR',
        os.path.join('output', '.experiments', 'jax_adaptation_support_rollout_diagnostic'),
    )

    return cfg
