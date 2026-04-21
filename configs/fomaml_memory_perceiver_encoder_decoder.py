import os

from configs.maml_memory_perceiver_encoder_decoder import get_config as get_memory_maml_config


def get_config():
    cfg = get_memory_maml_config()

    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_memory_fomaml',
            'runs',
        ),
    )
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_memory_fomaml',
            'checkpoints',
        ),
    )
    cfg.maml.first_order = True
    cfg.maml.inner_lr_mode = 'per_step_learned'
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-memory-fomaml')

    return cfg
