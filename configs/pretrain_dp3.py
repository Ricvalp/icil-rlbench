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
            "dp3_pretrain",
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
    cfg.dataset.K = 1
    cfg.dataset.L = 1
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 8
    cfg.dataset.stride = 3
    cfg.dataset.num_tries_per_item = 100

    cfg.model = ConfigDict()
    # Temporal fields below are kept in sync at runtime by icil/pretrain_dp3.py:
    # horizon <- dataset.H, n_action_steps <- dataset.H, n_obs_steps <- dataset.T_obs
    cfg.model.horizon = 8
    cfg.model.n_action_steps = 8
    cfg.model.n_obs_steps = 2
    cfg.model.scheduler_type = "ddim"  # "ddim" | "ddpm"
    cfg.model.num_train_timesteps = 100
    cfg.model.beta_start = 1e-4
    cfg.model.beta_end = 2e-2
    cfg.model.beta_schedule = "squaredcos_cap_v2"
    cfg.model.prediction_type = "sample"  # "epsilon" | "sample" | "v_prediction"
    cfg.model.clip_sample = True
    cfg.model.set_alpha_to_one = True
    cfg.model.steps_offset = 0
    cfg.model.num_inference_steps = 10
    cfg.model.obs_as_global_cond = True
    cfg.model.diffusion_step_embed_dim = 128
    cfg.model.down_dims = (512, 1024, 2048)
    cfg.model.kernel_size = 5
    cfg.model.n_groups = 8
    cfg.model.condition_type = "film"
    cfg.model.use_down_condition = True
    cfg.model.use_mid_condition = True
    cfg.model.use_up_condition = True
    cfg.model.encoder_output_dim = 64
    cfg.model.use_pc_color = False
    cfg.model.pointnet_type = "pointnet"
    cfg.model.pointcloud_in_channels = 3
    cfg.model.pointcloud_use_layernorm = True
    cfg.model.pointcloud_final_norm = "layernorm"  # "layernorm" | "none"
    cfg.model.pointcloud_use_projection = True
    cfg.model.pointcloud_out_channels = 64
    cfg.model.state_mlp_size = (64, 64)

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
            "dp3_pretrain",
            "checkpoints",
        ),
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = False
    cfg.wandb.project = os.environ.get("WANDB_PROJECT", "icil-dp3-pretrain")
    cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "ricvalp")
    cfg.wandb.group = ""
    cfg.wandb.name = ""
    cfg.wandb.mode = os.environ.get("WANDB_MODE", "online")  # online | offline | disabled
    cfg.wandb.tags = ()
    cfg.wandb.n_loss_steps = 20
    cfg.wandb.n_sample_steps = 200
    cfg.wandb.sample_batch_items = 16
    cfg.wandb.sample_inference_steps = 50
    cfg.wandb.sample_trace_frames = 8
    cfg.wandb.sample_eta = 0.0
    cfg.wandb.sample_clip_x0 = 10.0
    cfg.wandb.include_query_pointcloud_in_x0_pred_vs_gt_3d = False
    cfg.wandb.query_pointcloud_max_points = 4096

    return cfg
