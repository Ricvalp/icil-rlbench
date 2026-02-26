import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    cfg.seed = 0
    cfg.device = "cuda"
    cfg.output_parent_dir = os.environ.get(
        "ICIL_OUTPUT_PARENT_DIR",
        os.path.join(
            "output_data_playground_v3",
            ".experiments",
            "perceiver_encoder_decoder_pretrain",
            "runs",
        ),
    )

    cfg.data = ConfigDict()
    cfg.data.cache_root = os.environ.get(
        "ICIL_CACHE_ROOT",
        os.path.join("output_data_playground_v3", ".rlbench_cache_dense"),
    )
    cfg.data.tasks = ()  # () => use all tasks in cache_root
    cfg.data.keep_open_per_worker = True
    cfg.data.num_workers = 8
    cfg.data.pin_memory = True
    cfg.data.persistent_workers = True

    cfg.dataset = ConfigDict()
    cfg.dataset.K = 4
    cfg.dataset.L = 16
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 8
    cfg.dataset.stride = 3
    cfg.dataset.num_tries_per_item = 100

    cfg.model = ConfigDict()
    cfg.model.d_model = 512
    cfg.model.n_heads = 8
    cfg.model.m_frame_tokens = 64
    cfg.model.frame_tokenizer_layers = 2
    cfg.model.M_demo_latents = 256
    cfg.model.demo_perceiver_layers = 3
    cfg.model.denoiser_layers = 10
    cfg.model.denoiser_mlp_mult = 4
    cfg.model.dropout = 0.0
    cfg.model.mask_hash_buckets = 2048
    cfg.model.role_embed_max_K = 32
    cfg.model.role_embed_max_L = 64
    cfg.model.role_embed_max_Tobs = 16
    cfg.model.rgb_alpha_init = 1.0
    cfg.model.diffusion_T = 1000

    cfg.train = ConfigDict()
    cfg.train.num_steps = 10_000
    cfg.train.batch_size = 16
    cfg.train.grad_accum_steps = 1
    cfg.train.lr = 1e-4
    cfg.train.beta1 = 0.9
    cfg.train.beta2 = 0.95
    cfg.train.weight_decay = 1e-4
    cfg.train.grad_clip_norm = 1.0
    cfg.train.use_amp = True
    cfg.train.log_every = 20
    cfg.train.ckpt_every = 500
    cfg.train.resume_path = ""
    cfg.train.checkpoint_parent_dir = os.environ.get(
        "ICIL_CHECKPOINT_PARENT_DIR",
        os.path.join(
            "output_data_playground_v3",
            ".experiments",
            "perceiver_encoder_decoder_pretrain",
            "checkpoints",
        ),
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get("WANDB_PROJECT", "icil-perceiver-pretrain")
    cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "ricvalp")
    cfg.wandb.group = ""
    cfg.wandb.name = ""
    cfg.wandb.mode = os.environ.get("WANDB_MODE", "online")  # online | offline | disabled
    cfg.wandb.tags = ()
    cfg.wandb.n_loss_steps = 20
    cfg.wandb.n_sample_steps = 200
    cfg.wandb.sample_batch_items = 4
    cfg.wandb.sample_inference_steps = 50
    cfg.wandb.sample_trace_frames = 8
    cfg.wandb.sample_eta = 0.0
    cfg.wandb.sample_clip_x0 = 1.0

    return cfg
