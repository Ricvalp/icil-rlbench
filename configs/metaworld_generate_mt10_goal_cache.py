import os

from configs.metaworld_generate_mt10_cache import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.output.cache_root = os.environ.get(
        'ICIL_METAWORLD_MT10_GOAL_CACHE_ROOT',
        os.path.join('output_data_playground_v3', '.metaworld_cache', 'mt10_goal_train_50x1'),
    )
    # Task instances already provide different goals/configurations. The
    # scripted expert is deterministic, so one successful rollout per instance
    # avoids storing byte-identical duplicate episodes.
    cfg.metaworld.num_successful_episodes_per_instance = 1
    cfg.metaworld.max_attempts_per_instance = 80
    cfg.obs.variant = 'raw'
    cfg.obs.remove_goal = False
    return cfg
