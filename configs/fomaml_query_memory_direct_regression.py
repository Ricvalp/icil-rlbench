import os

from configs.maml_query_memory_direct_regression import get_config as get_maml_config


def get_config():
    cfg = get_maml_config()
    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'query_memory_direct_regression_fomaml',
            'runs',
        ),
    )
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'query_memory_direct_regression_fomaml',
            'checkpoints',
        ),
    )
    cfg.maml.first_order = True
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-query-memory-direct-fomaml')
    return cfg
