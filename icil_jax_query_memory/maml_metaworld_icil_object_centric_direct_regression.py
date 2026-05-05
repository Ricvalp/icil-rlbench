from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from icil_jax_query_memory.train.runner import train

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/jax_metaworld_icil_a3_support_encoder_fomaml.py',
    help_string='Object-centric MetaWorld ICIL/MAML config.',
)


def main(argv=None):
    del argv
    train(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
