import os

from configs.maml_perceiver_encoder_decoder import get_config as get_maml_config


def get_config():
    cfg = get_maml_config()

    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_memory_maml',
            'runs',
        ),
    )
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join(
            'output_data_playground_v3',
            '.experiments',
            'perceiver_encoder_decoder_memory_maml',
            'checkpoints',
        ),
    )

    # Memory-token MAML uses support tokens as the fast variables. With K resolved
    # as K_pretrain + 1, the inner loop holds out one support episode and encodes
    # the remaining K-1 episodes, matching the number of demos seen in pretraining.
    cfg.maml.first_order = False
    cfg.maml.inner_steps = 8
    cfg.maml.inner_lr = 1e-4
    cfg.maml.outer_lr = 1e-4
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.num_queries_per_step = 8
    cfg.maml.num_inner_batches = 0  # 0 => prepare one batch per inner step.
    cfg.maml.num_query_loss_samples = 1
    cfg.maml.holdout_index = -1
    cfg.maml.reuse_diffusion_noise = False
    cfg.maml.grad_accum_steps = 1

    # Default to learning the token producer while keeping the decoder fixed.
    # The decoder can still be enabled as a slow component from the command line.
    cfg.outer.train_encoder = True
    cfg.outer.train_decoder = False
    cfg.outer.train_input_projections = False
    cfg.outer.train_output_head = False
    cfg.outer.train_diffusion_conditioning = False

    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-perceiver-memory-maml')

    return cfg
