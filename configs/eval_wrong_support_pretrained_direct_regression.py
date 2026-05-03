from configs.diagnose_wrong_support_pretrained_direct_regression import get_config as _get_base_config


def get_config():
    cfg = _get_base_config()

    # Simulation-only wrong-support eval. The offline correct-vs-wrong MSE
    # diagnostic remains available in diagnose_wrong_support_pretrained_direct_regression.py.
    cfg.mse.enable = False
    cfg.task.num_rollout_episodes = 20

    return cfg
