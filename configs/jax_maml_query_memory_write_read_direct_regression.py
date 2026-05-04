import os

from configs.jax_maml_query_memory_direct_regression import get_config as get_base_config


def get_config():
    cfg = get_base_config()

    cfg.output_parent_dir = os.environ.get(
        'ICIL_OUTPUT_PARENT_DIR',
        os.path.join('output_data_playground_v3', '.experiments', 'jax_query_memory_write_read_direct_regression_maml', 'runs'),
    )
    cfg.train.checkpoint_parent_dir = os.environ.get(
        'ICIL_CHECKPOINT_PARENT_DIR',
        os.path.join('output_data_playground_v3', '.experiments', 'jax_query_memory_write_read_direct_regression_maml', 'checkpoints'),
    )

    decoder = cfg.model.query_memory_direct_regression
    decoder.separate_write_read_heads = True
    decoder.shared_write_read_head = False
    decoder.write_num_query_tokens = 4
    decoder.write_use_demo_id_embed = True
    decoder.write_use_time_embed = True
    decoder.write_max_demo_id = 16
    decoder.write_max_time_bins = 512
    decoder.write_time_embed_type = 'continuous_sinusoidal'
    decoder.write_query_mlp_mult = 2
    decoder.write_use_support_obs = False
    decoder.use_decoder_mode_embed = True
    decoder.memory_layer_norm_after_update = True
    decoder.memory_update_clip_norm = 1.0
    decoder.action_loss_type = 'l1'
    decoder.position_loss_weight = 1.0
    decoder.rotation_loss_weight = 1.0
    decoder.gripper_loss_weight = 1.0
    decoder.chunk_decay = 0.0

    cfg.maml.first_order = False
    cfg.maml.inner_loss_mode = 'write'
    cfg.maml.inner_steps = 1
    cfg.maml.inner_lr = 3e-3
    cfg.maml.inner_lr_mode = 'fixed'
    cfg.maml.max_grad_norm = 1.0
    cfg.maml.memory_layer_norm_after_update = True
    cfg.maml.use_read_improvement_margin = False
    cfg.maml.read_improvement_margin = 0.0
    cfg.maml.read_improvement_margin_weight = 0.0
    cfg.maml.log_output_delta = False
    cfg.maml.training_mode_metrics_only = True

    cfg.wandb.project = os.environ.get('WANDB_PROJECT', 'icil-jax-query-memory-write-read-maml')
    cfg.wandb.tags = ('write_read', 'gradmem')

    return cfg
