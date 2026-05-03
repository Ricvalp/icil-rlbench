from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from icil_jax_query_memory.train.runner import train

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_fomaml_query_memory_write_read_direct_regression.py',
    help_string='Path to ml_collections config file.',
)


def main(argv=None):
    del argv
    train(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
