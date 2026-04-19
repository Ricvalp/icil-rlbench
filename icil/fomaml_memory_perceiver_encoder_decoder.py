from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from icil.models.maml.memory_train import train_memory_maml

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default='configs/fomaml_memory_perceiver_encoder_decoder.py',
    help_string='Path to a ml_collections config file.',
)


def main(argv=None):
    del argv
    train_memory_maml(_CONFIG.value)


if __name__ == '__main__':
    app.run(main)
