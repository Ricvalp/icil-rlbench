from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from icil.models.maml.query_memory_train import train

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/maml_query_memory_direct_regression.py',
    help_string='Path to a ml_collections config file.',
)


def main(argv):
    del argv
    cfg = _CONFIG.value
    cfg.maml.first_order = False
    train(cfg)


if __name__ == '__main__':
    app.run(main)
