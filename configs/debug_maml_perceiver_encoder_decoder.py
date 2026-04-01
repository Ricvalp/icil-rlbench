from configs.maml_perceiver_encoder_decoder import get_config as get_base_config


def get_config():
    cfg = get_base_config()

    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False

    # Keep this runnable without a pretrained checkpoint.
    cfg.dataset.K = 3
    cfg.dataset.num_tries_per_item = 32

    cfg.maml.outer_context_size = 2

    cfg.train.num_steps = 5
    cfg.train.batch_size = 1
    cfg.train.log_every = 1
    cfg.train.ckpt_every = 0
    cfg.train.resume_path = ""

    cfg.wandb.enable = False
    cfg.wandb.n_loss_steps = 1
    cfg.wandb.n_sample_steps = 1
    cfg.wandb.sample_batch_items = 1
    cfg.wandb.sample_mse_items = 2
    cfg.wandb.sample_inference_steps = 10
    cfg.wandb.sample_trace_frames = 4

    return cfg
