import os

from configs.maml_perceiver_encoder_decoder import get_config as get_maml_config



def get_config():
    cfg = get_maml_config()

    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_fomaml',
            'runs',
        ),
    )
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_fomaml',
            'checkpoints',
        ),
    )
    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-fomaml')
    cfg.maml.include_input_projections_fast = True
    cfg.maml.include_output_head_fast = True
    cfg.maml.include_diffusion_conditioning_fast = True

    return cfg
