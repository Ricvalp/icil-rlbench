import os

from configs.jax_metaworld_maml_query_memory_write_read_direct_regression import get_config as get_base_config


def _ml45_goal_cache_root():
    return os.environ.get(
        'ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT',
        os.path.join('/mnt', 'external_storage', 'robotics', 'metaworld', 'icil_metaworld', 'ml45_goal_train_50x1'),
    )


def get_config():
    cfg = get_base_config()

    cfg.data.cache_root = _ml45_goal_cache_root()
    cfg.data.tasks = ()
    cfg.data.exclude_tasks = ()
    cfg.data.sample_same_task_name = True
    cfg.data.sample_same_task_instance = False
    cfg.data.support_zero_goal = False
    cfg.data.query_zero_goal = True
    cfg.data.preload_to_memory = True
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False

    decoder = cfg.model.query_memory_direct_regression
    decoder.write_use_support_obs = True
    decoder.memory_layer_norm_after_update = False

    cfg.maml.inner_steps = 2
    cfg.maml.inner_lr = 1e-1
    cfg.maml.memory_layer_norm_after_update = False
    cfg.maml.use_read_improvement_margin = False
    cfg.maml.read_improvement_margin = 0.0
    cfg.maml.read_improvement_margin_weight = 0.0
    cfg.maml.use_wrong_support_margin = True
    cfg.maml.wrong_support_margin = 0.01
    cfg.maml.wrong_support_margin_weight = 1.0
    cfg.maml.wrong_support_strategy = 'random_different_task'
    cfg.maml.use_memory_contrast = False
    cfg.maml.training_mode_metrics_only = False

    cfg.train.batch_size = 64
    cfg.wandb.name = 'ml45-query-hidden-wrong-support-margin-maml'
    cfg.wandb.tags = ('metaworld', 'ml45', 'write_read', 'wrong_support_margin', 'query_goal_hidden')
    return cfg
