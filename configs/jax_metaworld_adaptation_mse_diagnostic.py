import os

from ml_collections import ConfigDict

from configs.jax_metaworld_mt10_goal_family_maml_query_memory_write_read_direct_regression import (
    get_config as get_train_config,
)


def _output_root():
    return os.environ.get(
        'ICIL_METAWORLD_OUTPUT_PARENT_DIR',
        os.path.join('/mnt', 'external_storage', 'robotics', 'metaworld', 'icil_runs', 'outputs'),
    )


def _cache_root():
    return ''


def get_config():
    train_cfg = get_train_config()
    cfg = ConfigDict()
    cfg.seed = 0
    cfg.checkpoint_path = '/mnt/external_storage/robotics/metaworld/icil_runs/checkpoints/3dvo03xs/step_0056000.pkl'
    cfg.compute_dtype = 'float32'
    cfg.model = train_cfg.model

    cfg.data = ConfigDict()
    cfg.data.cache_root = _cache_root()
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = ()
    # None means infer from the checkpoint config. Pass true/false or a
    # concrete string on the CLI to override for diagnostics. The default
    # support sampling is family-level: same task name, different goal instance.
    cfg.data.task_sampling = None
    cfg.data.sample_same_task_name = True
    cfg.data.sample_same_task_instance = False
    cfg.data.allow_support_query_same_episode = None
    cfg.data.support_zero_goal = None
    cfg.data.query_zero_goal = None
    cfg.data.preload_to_memory = False

    cfg.dataset = ConfigDict()
    cfg.dataset.use_checkpoint_dataset_config = True
    cfg.dataset.K = 4
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 8
    cfg.dataset.stride = 1
    cfg.dataset.action_stride = 1
    cfg.dataset.pad_short_chunks = None
    cfg.dataset.action_representation = None
    cfg.dataset.num_tries_per_item = 100

    # Negative numeric values mean "read the value from the checkpoint".
    cfg.memory_ttt = ConfigDict()
    cfg.memory_ttt.inner_steps = -1
    cfg.memory_ttt.inner_lr = -1.0
    cfg.memory_ttt.max_grad_norm = -1.0
    cfg.memory_ttt.num_queries_per_step = -1
    cfg.memory_ttt.num_query_loss_samples = -1
    cfg.memory_ttt.num_inner_batches = -1
    cfg.memory_ttt.inner_loss_mode = ''

    cfg.task = ConfigDict()
    cfg.task.name = ''

    cfg.adaptation = ConfigDict()
    cfg.adaptation.different_task_name = ''

    cfg.mse = ConfigDict()
    cfg.mse.num_batches = 32
    cfg.mse.batch_size = 64

    cfg.output = ConfigDict()
    cfg.output.root_dir = _output_root()
    return cfg
